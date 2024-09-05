import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # Process noise covariance
        self.kf.processNoiseCov = np.array([[1e-2, 0, 0, 0],
                                            [0, 1e-2, 0, 0],
                                            [0, 0, 1e-5, 0],
                                            [0, 0, 0, 1e-5]], np.float32)

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.array([[1e-5, 0],
                                                [0, 1e-5]], np.float32)

        # Initial state
        self.kf.statePre = np.zeros((4, 1), np.float32)
        self.kf.statePost = np.zeros((4, 1), np.float32)

    def predict(self, x, y, t=1):

        predicted = [x, y]
        for _ in range(t):
            measured = np.array([[np.float32(predicted[0])], [np.float32(predicted[1])]])
            self.kf.correct(measured)
            predicted = self.kf.predict()

        predicted_x, predicted_y = int(predicted[0]), int(predicted[1])

        # if the prediction lcoation is out of frame
        if predicted_x <= 0 or predicted_x >= 640:
            predicted_x = x
        if predicted_y <= 0 or predicted_y >= 480:
            predicted_y = y

        return predicted_x, predicted_y

