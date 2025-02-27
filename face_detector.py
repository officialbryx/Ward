import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        # Optimized for drone view (25m range)
        self.app.prepare(
            ctx_id=-1,
            det_size=(1280, 1280),  # Larger size for distant faces
            det_thresh=0.1  # More sensitive detection
        )
        self.face_recognizer = None  # Will be set from main.py

    def detect_faces(self, image):
        faces = self.app.get(image)
        filtered_faces = []
        
        for face in faces:
            bbox = face.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Filter based on expected face size at 25m
            min_face_size = 15  # Minimum face size in pixels
            if width >= min_face_size and height >= min_face_size:
                # Add name if recognizer is available
                if self.face_recognizer and hasattr(face, 'embedding'):
                    face.name = self.face_recognizer.identify_face(face.embedding)
                filtered_faces.append(face)
        
        return filtered_faces
