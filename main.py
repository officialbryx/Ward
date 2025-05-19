import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import time
import logging
import sys
from tqdm import tqdm

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class ScreenCapture:
    def __init__(self, camera_id=0):
        logger.info("Initializing camera capture...")
        self.camera_id = camera_id
        self.cap = None
        self.last_frame = None
        
        # First try file-based devices
        available_devices = self._get_available_devices()
        logger.info(f"Found camera devices: {available_devices}")
        
        # Try each available device or default webcam
        devices_to_try = available_devices if available_devices else [0]  # Use default webcam if no devices found
        
        for device in devices_to_try:
            try:
                logger.info(f"Trying camera device: {device}")
                self.cap = cv2.VideoCapture(device)
                
                # Test if camera is working
                if self.cap.isOpened():
                    # Wait for camera to initialize
                    time.sleep(1)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"Successfully initialized camera: {device}")
                        # Configure camera
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        return
                    
                self.cap.release()
                self.cap = None
                    
            except Exception as e:
                logger.error(f"Error initializing camera {device}: {str(e)}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
        
        raise RuntimeError("Failed to initialize any camera. Please check connections.")
    
    def _get_available_devices(self):
        devices = []
        # Check common device paths
        for i in range(10):  # Check first 10 possible devices
            device = f"/dev/video{i}"
            if os.path.exists(device):
                devices.append(device)
        if not devices:
            logger.info("No video devices found in /dev, will try default webcam")
        return devices

    def capture(self):
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not properly initialized")
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame
                return frame
        except Exception as e:
            logger.error(f"Frame capture error: {str(e)}")
            
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

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
        logger.info("Initializing ArcFace recognition system...")
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
        logger.info("Initializing face detector...")
        try:
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            logger.info("Preparing face detection models...")
            self.app.prepare(
                ctx_id=-1,
                det_size=(1280, 1280),
                det_thresh=0.1
            )
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            raise

        self.face_recognizer = None

    def detect_faces(self, image):
        if image is None:
            logger.warning("Received None image for face detection")
            return []
            
        try:
            faces = self.app.get(image)
            logger.debug(f"Detected {len(faces)} faces")
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
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []

def main():
    screen_cap = None
    try:
        # Define headless mode at start of main
        headless_mode = not os.environ.get('DISPLAY')
        logger.info(f"Running in {'headless' if headless_mode else 'display'} mode")
        
        logger.info("Starting face recognition system...")
        recognizer = FaceRecognizer(database_path="known_faces")
        
        logger.info("Setting up camera...")
        screen_cap = ScreenCapture(camera_id=0)
        
        logger.info("Initializing face detector...")
        detector = FaceDetector()
        detector.face_recognizer = recognizer

        logger.info("Starting main loop...")
        frame_count = 0
        fps_time = time.time()

        while True:
            frame = screen_cap.capture()
            if frame is None:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
                
            logger.debug("Processing frame...")
            faces = detector.detect_faces(frame)
            
            # Draw faces on frame
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                if hasattr(face, 'name') and face.name:
                    name_text = f"ID: {face.name}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[3]),
                        (bbox[0] + text_width, bbox[3] + text_height + 5),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        frame,
                        name_text,
                        (bbox[0], bbox[3] + text_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2
                    )

            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                logger.info(f"FPS: {fps:.2f}")

            if not headless_mode:
                try:
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    logger.error(f"Display error: {e}")
                    headless_mode = True
                    logger.info("Switching to headless mode")
            else:
                # In headless mode, just log detections
                if faces:
                    logger.info(f"Detected {len(faces)} faces")
                    for face in faces:
                        if hasattr(face, 'name') and face.name:
                            logger.info(f"Identified: {face.name}")

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        if screen_cap is not None:
            logger.info("Cleaning up camera resources...")
            screen_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

