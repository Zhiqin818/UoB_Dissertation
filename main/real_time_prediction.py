import pyrealsense2 as rs
import numpy as np
import cv2
import time
from object_detection import ObjectDetector
from kalman_filter import KalmanFilter

class Camera:
    def __init__(self):
        # initialize realsense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 分别是宽、高、数据格式、帧率
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

        # Start streaming
        self.profile = self.pipeline.start(config)

    def getting_frame(self):
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        # get depth frame
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.219), cv2.COLORMAP_JET)

        # display color frame
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_frame, depth_colormap

if __name__ == "__main__":
    # initialize realsense camera
    camera = Camera()
    # initialize object detector
    object_detection = ObjectDetector()
    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()

    time_steps = 4

    history_location = []
    predicted_location = []
    history_trajectory = []
    predicted_trajectory = []

    try:
        while True:

            color_image, depth_frame, depth_colormap = camera.getting_frame()

            # detect the moving object
            object_bbox = object_detection.detect(color_image)
            x, y, x2, y2 = object_bbox
            cx = int((x + x2) / 2)
            cy = int((y + y2) / 2)

            if (640 > cx > 0 and 480 > cy > 0):

                print(f"Detected object center: ({cx}, {cy})")

                cv2.circle(color_image, (cx, cy), 10, (0, 0, 255), -1)

                history_trajectory.append([cx, cy])

                # if the object barely move
                if (history_location == []):
                    predicted_x, predicted_y = cx, cy
                elif (history_location[0] == cx and history_location[1] == cy):
                    predicted_x, predicted_y = cx, cy
                else:
                    # Predict the future location
                    predicted_x, predicted_y = kf.predict(cx, cy, t=time_steps)
                    # predicted_x, predicted_y = ekf.step(np.array([cx, cy]), 1, time_steps)

                    print(f"Predicted position: {predicted_x}, {predicted_y}")

                cv2.circle(color_image, (predicted_x, predicted_y), 10, (255, 0, 0), -1)

                predicted_trajectory.append([predicted_x, predicted_y])

                history_location = [cx, cy]
                history_trajectory = [predicted_x, predicted_y]

            else:
                print("No object detected")

            # display the colour image
            cv2.imshow('Frame', color_image)

            time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        camera.pipeline.stop()
