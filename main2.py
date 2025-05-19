import cv2
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from screen_capture import ScreenCapture
from screeninfo import get_monitors
import time

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
