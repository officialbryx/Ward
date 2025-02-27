import cv2
from face_detector import FaceDetector
from screen_capture import ScreenCapture
from screeninfo import get_monitors
import time

def main():
    # Get primary monitor dimensions
    monitor = get_monitors()[0]
    width, height = monitor.width, monitor.height
    
    # Calculate region - example: capture middle 50% of screen
    margin_x = width // 4
    margin_y = height // 4
    region = (margin_x, margin_y, width - margin_x, height - margin_y)
    
    print(f"Capturing region: {region}")
    
    detector = FaceDetector()
    screen_cap = ScreenCapture(region=region)

    # Create window with specific properties
    window_name = 'Face Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # Position window in bottom right corner
    window_width = 400
    window_height = 300
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.moveWindow(window_name, width - window_width - 50, height - window_height - 50)

    # Small delay to ensure window is created
    time.sleep(0.5)

    while True:
        frame = screen_cap.capture()
        faces = detector.detect_faces(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Resize frame to fit window
        frame = cv2.resize(frame, (window_width, window_height))
        cv2.imshow(window_name, frame)
        
        # Bring window to front periodically
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Check for both 'q' and ESC key (27)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC key
            print("Stopping recording...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
