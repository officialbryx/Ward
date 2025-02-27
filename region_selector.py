import cv2
import numpy as np
from PIL import ImageGrab

class RegionSelector:
    def __init__(self):
        self.selecting = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        self.region = None
        
    def select_region(self):
        # Capture full screen
        screen = np.array(ImageGrab.grab())
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        
        # Create window and set mouse callback
        cv2.namedWindow('Select Region')
        cv2.setMouseCallback('Select Region', self._mouse_callback)
        
        clone = screen.copy()
        
        while True:
            image = clone.copy()
            if self.selecting:
                cv2.rectangle(image, (self.x0, self.y0), (self.x1, self.y1), (0, 255, 0), 2)
            
            cv2.imshow('Select Region', image)
            key = cv2.waitKey(1)
            
            # Press Enter to confirm selection or Esc to cancel
            if key == 13:  # Enter key
                if self.x0 != self.x1 and self.y0 != self.y1:
                    self.region = (
                        min(self.x0, self.x1),
                        min(self.y0, self.y1),
                        max(self.x0, self.x1),
                        max(self.y0, self.y1)
                    )
                break
            elif key == 27:  # Esc key
                self.region = None
                break
                
        cv2.destroyWindow('Select Region')
        return self.region
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.x0, self.y0 = x, y
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.x1, self.y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.x1, self.y1 = x, y
