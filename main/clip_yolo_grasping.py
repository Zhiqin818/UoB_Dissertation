import torch
import cv2
from PIL import Image
import clip
from camera import Camera
import threading
import queue  # Import the queue module
from dobot import Calibration
from object_detection import ObjectDetector
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from camera import Camera
from dmp import dmp_discrete
import numpy as np
import time

def dmp_setup():
    static_point = [250, 0, 50]
    goal_location = [200, 20, 0]

    t = np.linspace(0, 1, data_len)
    y_demo = np.zeros((3, data_len))

    # Create a simple curved trajectory using sine and cosine functions
    for i in range(3):
        y_demo[i, :] = np.linspace(static_point[i], goal_location[i], data_len)

    # Add a sine curve to create curvature
    y_demo[1, :] += np.sin(2 * np.pi * t)  # Larger amplitude for more pronounced curve
    y_demo[2, :] += np.cos(2 * np.pi * t)  # Larger amplitude for more pronounced curve

    # DMP learning
    dmp = dmp_discrete(n_dmps=y_demo.shape[0], n_bfs=400, dt=1.0 / data_len)
    dmp.learning(y_demo, plot=False)

    return dmp

def real_time_trajectory_generation(dobot, dmp, predicted_arm_pos_queue, object_catched):

    # while True:
    while not object_catched:
        try:
            # Wait for a new position and empty the queue to get the most recent value
            predicted_arm_pos = predicted_arm_pos_queue.get()  # Get the first available value
            while True:
                try:
                    # Keep getting the latest position until the queue is empty
                    next_predicted_arm_pos = predicted_arm_pos_queue.get_nowait()
                    if next_predicted_arm_pos is None:
                        return  # Exit if a None is received
                    predicted_arm_pos = next_predicted_arm_pos  # Update to the most recent position
                except queue.Empty:
                    break  # Exit loop once the queue is empty and proceed with the latest position

            if predicted_arm_pos is None:
                continue  # Skip if the final position is None

            # Request an interrupt to stop any ongoing movement
            dobot.request_interrupt()

            current_end_effector = dobot.check_dobot_Arm_location()

            distance = np.linalg.norm(np.array(current_end_effector) - np.array(predicted_arm_pos))

            if (distance < 30):
                dobot.move_dobot_arm_wait([predicted_arm_pos[0], predicted_arm_pos[1], predicted_arm_pos[2]])
            else:
                desired_elements = max(2, int(distance // step_size))
                y_reproduce, dy_reproduce, ddy_reproduce = dmp.reproduce(initial=current_end_effector,
                                                                         goal=predicted_arm_pos)
                # Execute the new movement trajectory
                dobot.move_dobot_arm_dmp(y_reproduce, desired_elements, data_len)

        except queue.Empty:
            # If the queue is empty, continue with the most recent data
            pass


if __name__ == "__main__":
    # Load YOLOv5 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load CLIP model
    clip_model, preprocess = clip.load('ViT-B/32', device=device)

    # Define text prompt for CLIP
    text_prompt = input("Enter a target description: ")
    text_features = clip_model.encode_text(clip.tokenize(text_prompt).to(device))

    # Initialize Dobot robotic arm
    dobot = Calibration()
    # Initialize Realsense camera
    camera = Camera()
    # Initialize object detector
    object_detection = ObjectDetector()
    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()
    ekf = ExtendedKalmanFilter()

    time_steps = 4
    data_len = 40
    step_size = 20
    desired_elements = 10
    history_location = []
    object_arm_pos = [600, 600, 600]
    predicted_arm_pos = None
    pre_predicted_arm_pos = None
    object_catched = False
    cx = 0
    cy = 0

    dmp = dmp_setup()


    print("######################### Starting")

    # Queue to communicate between main thread and DMP thread
    predicted_arm_pos_queue = queue.Queue()

    # Start the DMP thread
    dmp_thread = threading.Thread(target=real_time_trajectory_generation, args=(dobot, dmp, predicted_arm_pos_queue, object_catched))
    dmp_thread.start()

    try:
        while True:
            frame, depth_frame, depth_colormap = camera.getting_frame()

            # Convert frame to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform object detection with YOLOv5
            results = model(pil_img)  # YOLOv5 expects a PIL Image or a path to an image file

            # Parse results
            detections = results.pandas().xyxy[0]  # Get detection results as a pandas dataframe

            # Process detections
            for _, row in detections.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                detected_object_name = row['name']

                # Extract the object region from the frame
                object_img = frame[y1:y2, x1:x2]
                object_pil_img = Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))
                object_pil_img = preprocess(object_pil_img).unsqueeze(0).to(device)

                # Get the object features from CLIP
                with torch.no_grad():
                    object_features = clip_model.encode_image(object_pil_img)

                # Compute similarity between detected object and text prompt
                similarity = torch.cosine_similarity(object_features, text_features)
                if similarity.item() > 0.25:  # Adjust threshold as needed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{detected_object_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

            if 640 > cx > 0 and 480 > cy > 0:
                # cv2.circle(color_image, (cx, cy), 10, (0, 0, 255), -1)
                object_arm_pos = camera.convert_to_arm_pose(cx, cy, depth_frame)

                # Check if the end effector is at the object's location
                reaching_state = dobot.check_dobot_Arm_reached_object(object_arm_pos)
                if reaching_state:
                    print("Caught the moving object")
                    object_catched = True
                    predicted_arm_pos_queue.put(None)  # Stop the DMP thread
                    dobot.press_dobot_arm()
                    dobot.suction_cup_suck()
                    time.sleep(1)
                    dobot.move_dobot_arm_wait([200, 0, 50])
                    dobot.move_dobot_arm_wait([250, 0, -40])
                    dobot.suction_cup_release()
                    dobot.move_dobot_arm_wait([200, 0, 50])
                    time.sleep(1)
                    continue

                # Predict the object's future location if it moved
                if history_location == [] or (
                        abs(history_location[0] - cx) <= 2 and abs(history_location[1] - cy) <= 2):
                    predicted_x, predicted_y = cx, cy
                else:
                    # predicted_x, predicted_y = kf.predict(cx, cy, time_steps)
                    predicted_x, predicted_y = ekf.step(np.array([cx, cy]), 1, time_steps)

                # cv2.circle(frame, (predicted_x, predicted_y), 10, (255, 0, 0), -1)

                predicted_arm_pos = camera.convert_to_arm_pose(predicted_x, predicted_y, depth_frame)

                if np.sqrt(predicted_arm_pos[0] ** 2 + predicted_arm_pos[1] ** 2) > 320:
                    print("Cannot reach!!!!!!!!!!!!!!!")
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                predicted_arm_pos_queue.put(predicted_arm_pos)

                history_location = [cx, cy]

                pre_predicted_arm_pos = predicted_arm_pos

            else:
                print("No object detected")

                reaching_state = dobot.check_dobot_Arm_reached_object(object_arm_pos)
                if reaching_state:
                    print("Caught the moving object")
                    object_catched = True
                    predicted_arm_pos_queue.put(None)  # Stop the DMP thread
                    dobot.press_dobot_arm()
                    dobot.suction_cup_suck()
                    time.sleep(1)
                    dobot.move_dobot_arm_wait([200, 0, 50])
                    dobot.move_dobot_arm_wait([250, 0, -40])
                    dobot.suction_cup_release()
                    dobot.move_dobot_arm_wait([200, 0, 50])
                    time.sleep(1)
                    continue
                else:
                    predicted_arm_pos_queue.put(predicted_arm_pos)

            # Display the frame
            cv2.imshow("Frame", frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the DMP thread
        predicted_arm_pos_queue.put(None)
        dmp_thread.join()
        camera.pipeline.stop()
        cv2.destroyAllWindows()
