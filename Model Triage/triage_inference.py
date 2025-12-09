import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

class TriageRouter:
    """
    API Inference Class for the Smart Triage Router.
    Classifies images as 'Clinical' or 'Histopathological'.
    """
    def __init__(self, model_path='triage_router.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Classes must match the alphabetical order of ImageFolder used during training
        # ['Clinical', 'Histopathological']
        self.classes = ['Clinical', 'Histopathological']
        
        print(f"Initializing TriageRouter on {self.device}...")
        
        # Load Model Architecture (ResNet18)
        # We do not need pretrained weights here, as we will load our own state_dict
        self.model = models.resnet18(pretrained=False)
        
        # Modify the final layer to match our 2 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Load Trained Weights
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded model weights from '{model_path}'")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")
            
        self.model = self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        
        # Define Transforms (Must match training validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, threshold=0.60):
        """
        Predicts the class of a single image.
        Args:
            image_path (str): Path to the image file.
            threshold (float): Minimum confidence score required for a valid prediction.
        Returns:
            str: Predicted class label ('Clinical' or 'Histopathological') or 'Unknown'.
        """
        if not os.path.exists(image_path):
            return "Error: Image file not found."

        try:
            # Open image and convert to RGB (handles greyscale or alpha channels)
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess
            image_tensor = self.transform(image).unsqueeze(0) # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            if confidence.item() < threshold:
                return "Unknown"
                
            predicted_label = self.classes[predicted_idx.item()]
            return predicted_label
            
        except Exception as e:
            return f"Error during inference: {str(e)}"

if __name__ == "__main__":
    # Simple test if run directly
    # Ensure you have a dummy image or handle the error gracefully
    print("TriageRouter Inference Module")
    # router = TriageRouter()
    # print(router.predict("test_image.jpg"))
