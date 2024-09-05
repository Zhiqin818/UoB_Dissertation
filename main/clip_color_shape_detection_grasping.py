import torch
import clip
from PIL import Image
import cv2
import numpy as np
from camera import Camera

import numpy as np
import threading
import queue  # Import the queue module
from dobot import Calibration
from object_detection import ObjectDetector
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from camera import Camera
import cv2
import time
from dmp import dmp_discrete


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


def get_color_and_shape(text_prompt):
    colors = ["red", "green", "blue", "yellow", "black", "purple", "orange", "pink", "gray", "brown"]
    shapes = ["cube", "cuboid", "sphere", "pyramid", "cylinder", "cone", "torus", "prism", "circle", "square", "triangle"]

    words = text_prompt.lower().split()
    target_color = None
    target_shape = None

    for word in words:
        if word in colors:
            target_color = word
        if word in shapes:
            target_shape = word

    return target_color, target_shape

# Function to convert color name to HSV range
def get_hsv_range(color_name):
    color_ranges = {
        "red": ([0, 50, 50], [10, 255, 255]),
        "green": ([35, 50, 50], [85, 255, 255]),
        "blue": ([90, 50, 50], [130, 255, 255]),
        "yellow": ([20, 100, 100], [30, 255, 255]),
        "orange": ([10, 100, 100], [20, 255, 255]),
        "purple": ([130, 50, 50], [160, 255, 255]),
        "pink": ([160, 50, 50], [180, 255, 255]),
    }
    return color_ranges.get(color_name.lower(), ([0, 0, 0], [180, 255, 255]))


def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    # print(len(approx))
    if len(approx) == 3:
        return "triangle prism"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        # print(aspect_ratio)
        if aspect_ratio >= 0.7 and aspect_ratio <= 1.3:
            return "cube"
        else:
            return "cuboid"
    elif len(approx) > 10:
        return "sphere"
    return "unknown"


if __name__ == "__main__":
    # Load the CLIP model and the preprocessing method
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Define the target color and shape
    text_prompt = input("Enter a target description: ")
    target_color, target_shape = get_color_and_shape(text_prompt)
    # Tokenize the text prompt
    text = clip.tokenize([text_prompt]).to(device)

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
            # get the color frame
            color_image, depth_frame, depth_colormap = camera.getting_frame()

            if not object_catched:

                # Convert the frame to a PIL image
                image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

                # Preprocess the image
                image_input = preprocess(image).unsqueeze(0).to(device)

                # Perform inference with CLIP
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text)

                    # Calculate similarity between image and text
                    logits_per_image, logits_per_text = model(image_input, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                if (probs[0][0] > 0.5):

                    # Convert frame to HSV color space
                    hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                    lower_hsv, upper_hsv = get_hsv_range(target_color)
                    mask = cv2.inRange(hsv_frame, np.array(lower_hsv), np.array(upper_hsv))

                    # Find contours of the masked objects
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # Initialize a new mask to draw filtered contours
                    filtered_mask = np.zeros_like(mask)

                    # Define the minimum area threshold
                    min_area = 200

                    # Loop over contours and filter by area
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area >= min_area:
                            # Draw the contour on the filtered mask
                            cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)

                    mask = filtered_mask

                    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty("mask", cv2.WND_PROP_TOPMOST, 1)
                    cv2.imshow("mask", mask)

                    # Find contours of the masked objects
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    box = (0, 0, 0, 0)
                    # If the model detects a high probability for the color description
                    if probs[0][0] > 0.5:  # You can adjust this threshold
                        for contour in contours:
                            if cv2.contourArea(contour) > 100:  # Filter small contours
                                shape = detect_shape(contour)
                                # print("-----------")
                                # print(shape)
                                # print(target_shape)
                                if shape in target_shape or target_shape in shape:
                                    print(shape)
                                    x, y, w, h = cv2.boundingRect(contour)
                                    box = (x, y, x + w, y + h)
                                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(color_image, text_prompt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)

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

                        cv2.circle(color_image, (predicted_x, predicted_y), 10, (255, 0, 0), -1)

                        predicted_arm_pos = camera.convert_to_arm_pose(predicted_x, predicted_y, depth_frame)

                        if np.sqrt(predicted_arm_pos[0] ** 2 + predicted_arm_pos[1] ** 2) > 320:
                            print("Cannot reach!!!!!!!!!!!!!!!")
                            cv2.imshow("Frame", color_image)
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
                            if predicted_arm_pos is not None:
                                # Send the predicted arm position to the DMP thread
                                predicted_arm_pos_queue.put(predicted_arm_pos)

                # Display the frame
                cv2.imshow("Frame", color_image)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Stop the DMP thread
        predicted_arm_pos_queue.put(None)
        dmp_thread.join()
        camera.pipeline.stop()
        cv2.destroyAllWindows()