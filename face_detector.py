import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],  # Changed from CoreML to CPU
            allowed_modules=['detection']
        )
        # Adjusted parameters for better detection
        self.app.prepare(
            ctx_id=-1,  # Use CPU
            det_size=(640, 640),  # Reduced size for better CPU performance
            det_thresh=0.15
        )

    def detect_faces(self, image):
        # Simplify detection to improve performance
        enhanced = self._enhance_image(image)
        faces = self.app.get(enhanced)
        
        # Filter detections
        filtered_faces = []
        for face in faces:
            bbox = face.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height
            
            if 0.3 < aspect_ratio < 3.0 and width > 15 and height > 15:
                if face.det_score > 0.15:
                    filtered_faces.append(face)
        
        return filtered_faces

    def _enhance_image(self, image):
        # Enhanced image processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Stronger CLAHE for better detail in shadows
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Enhance contrast
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def _merge_overlapping_detections(self, faces, iou_thresh=0.3):
        if not faces:
            return faces
            
        # Sort by confidence
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        kept_faces = []
        
        for face in faces:
            should_keep = True
            for kept_face in kept_faces:
                if self._calculate_iou(face.bbox, kept_face.bbox) > iou_thresh:
                    should_keep = False
                    break
            if should_keep:
                kept_faces.append(face)
        
        return kept_faces

    def _calculate_iou(self, box1, box2):
        # Calculate intersection over union
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box2[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
