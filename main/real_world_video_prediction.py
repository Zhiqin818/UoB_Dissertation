import pyrealsense2 as rs
import numpy as np
import cv2
import time
from object_detection import ObjectDetector
# from kalman_filter_function import kalman_filter_2d
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter


# Create a pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file
config.enable_device_from_file("circle.bag")

# Start streaming from file
pipeline_profile = pipeline.start(config)

# Get the stream profiles and check available streams
stream_profiles = pipeline_profile.get_streams()
available_streams = {stream.stream_type(): stream for stream in stream_profiles}

# Check for available streams
depth_stream = available_streams.get(rs.stream.depth)
color_stream = available_streams.get(rs.stream.color)

history_location = []
history_trajectory = []
predicted_trajectory = []
errors = []
timestep = 4

object_detection = ObjectDetector()
kf = KalmanFilter()
ekf = ExtendedKalmanFilter()

# Get the duration of the recorded file
device = pipeline_profile.get_device()
playback = device.as_playback()
duration = playback.get_duration()

start_time = time.time()

try:
    while True:
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Check if the elapsed time is greater than the duration of the recorded file
        if elapsed_time >= duration.total_seconds():
            break

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Get frames if available
        depth_frame = frames.get_depth_frame() if depth_stream else None
        color_frame = frames.get_color_frame() if color_stream else None

        # Convert images to numpy arrays
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            # Convert color image to BGR format (default for OpenCV)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Stack and show images if both frames are available
        if depth_frame and color_frame:
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
        elif depth_frame:
            cv2.imshow('Depth', depth_colormap)
        elif color_frame:
            cv2.imshow('Color', color_image)

        # detect the moving object
        object_bbox = object_detection.detect(color_image)

        if object_bbox is not None:
            x, y, x2, y2 = object_bbox
            cx = int((x + x2) / 2)
            cy = int((y + y2) / 2)

            cv2.circle(color_image, (cx, cy), 10, (0, 0, 255), -1)

            history_trajectory.append([cx, cy])

            # Predict the object's future location if it moved
            if history_location == [] or (abs(history_location[0] - cx) <= 2 and abs(history_location[1] - cy) <= 2):
                predicted_x, predicted_y = cx, cy
            else:
                # predicted_x, predicted_y = kf.predict(cx, cy, timestep)
                predicted_x, predicted_y = ekf.step(np.array([cx, cy]), 1, timestep)

            cv2.circle(color_image, (predicted_x, predicted_y), 10, (255, 0, 0), -1)

            predicted_trajectory.append([predicted_x, predicted_y])

            history_location = [cx, cy]

        else:
            print("No object detected")

        # display the colour image
        cv2.imshow('Color', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

    # calculate
    for gt, pred in zip(history_trajectory, predicted_trajectory):
        error = np.linalg.norm(np.array(gt) - np.array(pred))
        errors.append(error)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.square(errors)))
    mape = np.mean(np.abs(np.array(errors) / np.linalg.norm(history_trajectory, axis=1))) * 100


    import matplotlib.pyplot as plt

    history_trajectory = np.array(history_trajectory)
    predicted_trajectory = np.array(predicted_trajectory)

    plt.figure(figsize=(10, 6))
    plt.plot(history_trajectory[:, 0], history_trajectory[:, 1], 'g', label='Ground Truth')
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r--', label='Prediction')
    plt.title('Object Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Error plot
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b', label='Prediction Error')
    plt.title('Prediction Error Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Error')
    plt.legend()
    plt.show()





