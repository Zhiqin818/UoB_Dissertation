import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # Create mask for green color
        self.low_green = np.array([35, 50, 50])
        self.high_green = np.array([85, 255, 255])

    def detect(self, frame):
        # Convert BGR to HSV
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply Gaussian blur to reduce noise
        hsv_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)

        # Create mask with green color range
        mask = cv2.inRange(hsv_img, self.low_green, self.high_green)

        # Apply morphological operations to remove small noise and fill gaps
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # print("no contours")
            return (0, 0, 0, 0)

        # Sort contours by area in descending order
        filtered_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for cnt in filtered_contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            break

        return box