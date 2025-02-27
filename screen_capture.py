import numpy as np
from PIL import ImageGrab
import cv2

__all__ = ['ScreenCapture']

class ScreenCapture:
    def __init__(self, region=None):
        self.region = region  # (left, top, right, bottom)

    def capture(self):
        screenshot = ImageGrab.grab(bbox=self.region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

# Ensure the class is available when imported
if __name__ == '__main__':
    pass
