import cv2
import numpy as np
from PIL import ImageGrab
import os
from insightface.app import FaceAnalysis
from screeninfo import get_monitors
import time

class ScreenCapture:
    def __init__(self, region=None):
        self.region = region  # (left, top, right, bottom)

    def capture(self):
        screenshot = ImageGrab.grab(bbox=self.region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

class FaceRecognizer:
    def __init__(self, database_path="known_faces"):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )
        self.database_path = database_path
        self.known_embeddings = {}
        self.recognition_threshold = 0.55
        print("Initializing ArcFace recognition system...")
        self._load_known_faces()
    
    def _load_known_faces(self):
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            
        for lastname in os.listdir(self.database_path):
            person_path = os.path.join(self.database_path, lastname)
            if os.path.isdir(person_path):
                embeddings = []
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
        if threshold is None:
            threshold = self.recognition_threshold

        if not self.known_embeddings:
            return None
            
        best_match = None
        best_score = float('inf')
        
        norm_embedding = face_embedding / np.linalg.norm(face_embedding)
        for lastname, known_embedding in self.known_embeddings.items():
            norm_known = known_embedding / np.linalg.norm(known_embedding)
            similarity = np.dot(norm_embedding, norm_known)
            distance = 1 - similarity
            
            if distance < threshold and distance < best_score:
                best_score = distance
                best_match = lastname
                
        return best_match

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(
            ctx_id=-1,
            det_size=(1280, 1280),
            det_thresh=0.1
        )
        self.face_recognizer = None

    def detect_faces(self, image):
        faces = self.app.get(image)
        filtered_faces = []
        
        for face in faces:
            bbox = face.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            min_face_size = 15
            if width >= min_face_size and height >= min_face_size:
                if self.face_recognizer and hasattr(face, 'embedding'):
                    face.name = self.face_recognizer.identify_face(face.embedding)
                filtered_faces.append(face)
        
        return filtered_faces

def main():
    # Initialize face recognition
    recognizer = FaceRecognizer(database_path="known_faces")
    
    # Get primary monitor dimensions
    monitor = get_monitors()[0]
    width, height = monitor.width, monitor.height
    
    # Calculate region - example: capture middle 50% of screen
    margin_x = width // 4  # 25% margin from each side
    margin_y = height // 4  # 25% margin from each side
    region = (margin_x, margin_y, width - margin_x, height - margin_y)
    
    print(f"Capturing region: {region}")
    
    # Initialize detector with recognition
    detector = FaceDetector()
    detector.face_recognizer = recognizer
    
    screen_cap = ScreenCapture(region=region)

    while True:
        # Capture screen
        frame = screen_cap.capture()
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw faces on frame
        for face in faces:
            bbox = face.bbox.astype(int)
            # Draw face rectangle
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Display name if recognized
            if hasattr(face, 'name') and face.name:
                # Draw background rectangle for text
                name_text = f"ID: {face.name}"
                (text_width, text_height), _ = cv2.getTextSize(
                    name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[3]),
                    (bbox[0] + text_width, bbox[3] + text_height + 5),
                    (0, 255, 0),
                    -1  # Fill rectangle
                )
                
                # Draw name text below the face box
                cv2.putText(
                    frame,
                    name_text,
                    (bbox[0], bbox[3] + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),  # Black text
                    2
                )

        # Display result
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

