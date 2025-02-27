import cv2
from face_detector import FaceDetector
from screen_capture import ScreenCapture
from region_selector import RegionSelector
import time

def main():
    # Initialize region selector
    selector = RegionSelector()
    print("Select a region by clicking and dragging. Press Enter to confirm or Esc to cancel.")
    region = selector.select_region()
    
    if region is None:
        print("No region selected. Exiting...")
        return
        
    print(f"Selected region: {region}")
    
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
    
    # Small delay to ensure window is created
    time.sleep(0.5)

    while True:
        frame = screen_cap.capture()
        faces = detector.detect_faces(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            # Always use green for face detections
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        frame = cv2.resize(frame, (window_width, window_height))
        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Stopping recording...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
