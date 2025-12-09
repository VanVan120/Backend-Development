import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
import base64
from io import BytesIO
from stain_utils import MacenkoNormalizer

# --- 1. Model Architecture (Must match Training Notebook) ---

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.upsample(self.conv(x))

class OSCCMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.densenet169(pretrained=False) # Pretrained not needed for inference loading
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.head_tvnt = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))
        self.head_poi = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5))
        self.head_pni = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))
        self.head_tb = nn.Sequential(nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head_mi = nn.Sequential(nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Linear(128, 1))
        
        self.decoder = nn.Sequential(
            UpsampleBlock(num_ftrs, 512), UpsampleBlock(512, 256),
            UpsampleBlock(256, 128), UpsampleBlock(128, 64),
            UpsampleBlock(64, 32), nn.Conv2d(32, 1, kernel_size=1)
        )

        # --- NEW: PNI Segmentation Decoder ---
        self.decoder_pni = nn.Sequential(
            UpsampleBlock(num_ftrs, 512), UpsampleBlock(512, 256),
            UpsampleBlock(256, 128), UpsampleBlock(128, 64),
            UpsampleBlock(64, 32), nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        pooled = F.relu(features, inplace=False)
        pooled = F.adaptive_avg_pool2d(pooled, (1, 1))
        pooled = torch.flatten(pooled, 1)
        
        return {
            'tvnt': self.head_tvnt(pooled),
            'poi': self.head_poi(pooled),
            'pni': self.head_pni(pooled),
            'tb': self.head_tb(pooled),
            'mi': self.head_mi(pooled),
            'doi': self.decoder(features),
            'pni_seg': self.decoder_pni(features)
        }

# --- 2. Inference Wrapper Class ---

class ModelAInference:
    def __init__(self, model_path="model_a.pth", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Model A from {model_path} on {self.device}...")
        
        self.model = OSCCMultiTaskModel()
        if os.path.exists(model_path):
            # strict=False allows loading weights even if we added new heads (like pni_seg)
            # weights_only=False is required for PyTorch 2.6+ when loading full models or older checkpoints
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False), strict=False)
        else:
            print(f"WARNING: Model file {model_path} not found. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Stain Normalizer
        self.normalizer = MacenkoNormalizer()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calculate_doi(self, mask_tensor):
        """Calculates DOI (mm) from segmentation mask."""
        mask = torch.sigmoid(mask_tensor).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        
        if np.sum(mask) == 0: return 0.0
        
        y_indices, _ = np.where(mask > 0)
        if len(y_indices) == 0: return 0.0
        
        pixel_depth = np.max(y_indices) - np.min(y_indices)
        return float(pixel_depth * 0.00025) # 0.25 microns/pixel -> mm

    def generate_heatmap(self, img_tensor, pred_score):
        """Generates Grad-CAM heatmap."""
        # Hook to capture feature maps and gradients
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        def forward_hook(module, input, output):
            activations.append(output)
            
        # Hook into the last dense block of DenseNet169
        target_layer = self.model.backbone.features.norm5 
        
        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)
        
        # Zero grads
        self.model.zero_grad()
        
        # We need to re-run forward pass to ensure hooks capture this specific image
        # (The previous forward pass in predict() was inside no_grad context)
        # Enable grad temporarily for Grad-CAM
        with torch.enable_grad():
            output = self.model(img_tensor)
            score = output['tvnt']
            target_class = torch.argmax(score)
            score[:, target_class].backward()
        
        # Generate Heatmap
        grads = gradients[0].cpu().data.numpy().squeeze()
        fmaps = activations[0].cpu().data.numpy().squeeze()
        
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmaps[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        # Cleanup
        handle_f.remove()
        handle_b.remove()
        
        return cam

    def predict(self, image_path):
        """
        Runs inference on a single image.
        Returns a dictionary of predictions.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # --- NEW: Apply Stain Normalization ---
            # Convert PIL -> Numpy
            img_np = np.array(image)
            try:
                img_norm_np = self.normalizer.normalize(img_np)
                image_norm = Image.fromarray(img_norm_np)
            except Exception as e:
                print(f"Warning: Stain normalization failed ({e}). Using original image.")
                image_norm = image
            # --------------------------------------

            img_tensor = self.transform(image_norm).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                
            # Process Outputs
            tvnt_prob = F.softmax(outputs['tvnt'], dim=1)[0]
            is_tumour = tvnt_prob[1].item() > 0.5
            
            poi_probs = F.softmax(outputs['poi'], dim=1)[0]
            poi_class = torch.argmax(poi_probs).item()
            
            pni_prob = F.softmax(outputs['pni'], dim=1)[0]
            has_pni = pni_prob[1].item() > 0.5
            
            tb_count = max(0, int(outputs['tb'].item()))
            mi_count = max(0, int(outputs['mi'].item()))
            
            doi_mm = self.calculate_doi(outputs['doi'])
            
            # Generate Heatmap (Only if tumour is detected to save resources)
            heatmap_b64 = None
            original_b64 = None
            if is_tumour:
                heatmap = self.generate_heatmap(img_tensor, outputs['tvnt'])
                
                # Create Overlay
                img_resized = image.resize((224, 224))
                img_np = np.array(img_resized)
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
                
                # Convert Overlay to Base64 string for API response
                pil_img = Image.fromarray(overlay)
                buff = BytesIO()
                pil_img.save(buff, format="JPEG")
                heatmap_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

                # Convert Original Resized to Base64 (for consistent display)
                buff_orig = BytesIO()
                img_resized.save(buff_orig, format="JPEG")
                original_b64 = base64.b64encode(buff_orig.getvalue()).decode("utf-8")

            return {
                "status": "success",
                "predictions": {
                    "tumour_detected": is_tumour,
                    "tumour_probability": float(tvnt_prob[1].item()),
                    "pattern_of_invasion": poi_class, # 0-4
                    "depth_of_invasion_mm": round(doi_mm, 4),
                    "tumour_buds_count": tb_count,
                    "perineural_invasion": has_pni,
                    "mitotic_figures_count": mi_count,
                    "heatmap_overlay": heatmap_b64, # Base64 string or None
                    "original_resized": original_b64 # Base64 string or None
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Test run
    inference = ModelAInference()
    # Create a dummy image for testing
    dummy_img_path = "test_patch.jpg"
    Image.new('RGB', (224, 224), color='red').save(dummy_img_path)
    
    result = inference.predict(dummy_img_path)
    print("Test Prediction:", result)
    os.remove(dummy_img_path)
