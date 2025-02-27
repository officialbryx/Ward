import cv2
from face_detector import FaceDetector
from screen_capture import ScreenCapture
from screeninfo import get_monitors

def main():
    # Get primary monitor dimensions
    monitor = get_monitors()[0]
    width, height = monitor.width, monitor.height
    
    # Calculate region - example: capture middle 50% of screen
    margin_x = width // 4  # 25% margin from each side
    margin_y = height // 4  # 25% margin from each side
    region = (margin_x, margin_y, width - margin_x, height - margin_y)
    
    print(f"Capturing region: {region}")
    
    detector = FaceDetector()
    screen_cap = ScreenCapture(region=region)

    while True:
        # Capture screen
        frame = screen_cap.capture()
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw faces on frame
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Display result
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
