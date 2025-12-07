import os
import cv2
import numpy as np
from ultralytics import YOLO  # Import Standard YOLO
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    raise ImportError("Please run: pip install sahi ultralytics")

class OralHygieneModel:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model_path = model_path
        self.device = "cuda:0" 
        
        # 1. LOAD STANDARD YOLO (For Low Res)
        self.standard_model = YOLO(model_path)
        
        # 2. LOAD SAHI MODEL (For High Res)
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.15,
            device=self.device
        )
        
        self.class_names = {
            0: 'Caries', 1: 'Calculus', 2: 'Gingivitis',
            3: 'Tooth Discoloration', 4: 'Ulcers', 5: 'Hypodontia'
        }

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"Image file not found at {image_path}"}

        try:
            # --- STEP 1: CHECK RESOLUTION ---
            img = cv2.imread(image_path)
            if img is None: return {"error": "Could not read image."}
            
            # Fix Color (BGR to RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            height, width = img.shape[:2]
            quality_note = "High Quality (SAHI Mode)"
            
            detections = []
            findings = set()
            counts = {name: 0 for name in self.class_names.values()}

            # --- STEP 2: CHOOSE STRATEGY ---
            
            # STRATEGY A: LOW RES -> STANDARD YOLO
            # If image is small, SAHI fails. We use standard YOLO which understands "context" better.
            if width < 640 or height < 640:
                quality_note = "Low Resolution. Switched to Standard Mode for better stability."
                
                # Standard Inference (let YOLO handle resizing internally)
                results = self.standard_model.predict(
                    image_path, 
                    conf=0.15, 
                    imgsz=640, # Standard size
                    save=False, 
                    verbose=False
                )
                
                if results:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        self._process_detection(cls_id, conf, bbox, detections, findings, counts)

            # STRATEGY B: HIGH RES -> SAHI (GOLD STANDARD)
            # If image is big, we slice it to find tiny details.
            else:
                sahi_result = get_sliced_prediction(
                    img_rgb, # Use the RGB numpy array
                    self.sahi_model,
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    verbose=0
                )
                
                for pred in sahi_result.object_prediction_list:
                    cls_id = pred.category.id
                    conf = pred.score.value
                    bbox = pred.bbox.to_xyxy()
                    self._process_detection(cls_id, conf, bbox, detections, findings, counts)

            # --- STEP 3: FINAL LOGIC (Shared) ---
            if len(detections) == 0:
                 return {
                    "screening_result": "No Issues Detected",
                    "findings": [],
                    "hygiene_score": "High",
                    "quality_note": quality_note,
                    "detections": []
                }

            # Hygiene Logic
            screening_result = "Normal"
            if len(detections) > 0: screening_result = "Issues Detected"
            if "Ulcers" in findings or "Caries" in findings: screening_result = "Refer to Dentist"

            if counts['Caries'] > 0 or counts['Gingivitis'] > 1 or counts['Calculus'] > 1:
                hygiene_score = "Low"
            elif counts['Tooth Discoloration'] > 0 or counts['Ulcers'] > 0:
                hygiene_score = "Medium"
            else:
                hygiene_score = "High"

            return {
                "screening_result": screening_result,
                "findings": list(findings),
                "hygiene_score": hygiene_score,
                "quality_note": quality_note,
                "detections": detections
            }

        except Exception as e:
            return {"error": str(e)}

    def _process_detection(self, cls_id, conf, bbox, detections, findings, counts):
        """Helper function to process boxes from either method"""
        
        # TUNED THRESHOLDS
        confidence_thresholds = {
            'Ulcers': 0.75,              
            'Tooth Discoloration': 0.40, 
            'Caries': 0.35,              
            'Hypodontia': 0.60,          
            'Calculus': 0.25,            
            'Gingivitis': 0.30           
        }
        
        class_name = self.class_names.get(cls_id, "Unknown")
        min_conf = confidence_thresholds.get(class_name, 0.25)
        
        if conf >= min_conf:
            findings.add(class_name)
            counts[class_name] += 1
            detections.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": [round(x, 2) for x in bbox]
            })