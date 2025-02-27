import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

__all__ = ['FaceRecognizer']

class FaceRecognizer:
    def __init__(self, database_path="known_faces"):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        # Configure with valid parameters only
        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )
        self.database_path = database_path
        self.known_embeddings = {}
        self.recognition_threshold = 0.55  # ArcFace recognition threshold
        print("Initializing ArcFace recognition system...")
        self._load_known_faces()
    
    def _load_known_faces(self):
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            
        # Load each person's folder
        for lastname in os.listdir(self.database_path):
            person_path = os.path.join(self.database_path, lastname)
            if os.path.isdir(person_path):
                embeddings = []
                # Process each image in person's folder
                for img_name in os.listdir(person_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(person_path, img_name)
                        img = cv2.imread(img_path)
                        faces = self.app.get(img)
                        if faces:
                            embeddings.append(faces[0].embedding)
                
                if embeddings:
                    self.known_embeddings[lastname] = np.mean(embeddings, axis=0)
    
    def identify_face(self, face_embedding, threshold=None):
        """
        Identify face using ArcFace embedding comparison
        threshold: Optional override for recognition threshold
        """
        if threshold is None:
            threshold = self.recognition_threshold

        if not self.known_embeddings:
            return None
            
        best_match = None
        best_score = float('inf')
        
        # Compare using L2 normalized embeddings (ArcFace standard)
        norm_embedding = face_embedding / np.linalg.norm(face_embedding)
        for lastname, known_embedding in self.known_embeddings.items():
            norm_known = known_embedding / np.linalg.norm(known_embedding)
            # Calculate cosine similarity
            similarity = np.dot(norm_embedding, norm_known)
            distance = 1 - similarity
            
            if distance < threshold and distance < best_score:
                best_score = distance
                best_match = lastname
                
        return best_match

# Ensure the class is available when imported
if __name__ == '__main__':
    pass
