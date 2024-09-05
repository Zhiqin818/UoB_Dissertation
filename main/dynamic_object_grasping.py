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
    y_demo[1, :] += 2 * np.sin(2 * np.pi * t)  # Larger amplitude for more pronounced curve
    y_demo[2, :] += 2 * np.cos(2 * np.pi * t)  # Larger amplitude for more pronounced curve

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

            print("predicted_arm_pos 2", predicted_arm_pos)

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

    dmp = dmp_setup()

    print("######################### Starting")

    # Queue to communicate between main thread and DMP thread
    predicted_arm_pos_queue = queue.Queue()

    # Start the DMP thread
    dmp_thread = threading.Thread(target=real_time_trajectory_generation, args=(dobot, dmp, predicted_arm_pos_queue, object_catched))
    dmp_thread.start()

    try:
        while True:
            # Get the color frame
            color_image, depth_frame, depth_colormap = camera.getting_frame()

            if not object_catched:

                # Detect the moving object
                object_bbox = object_detection.detect(color_image)
                x, y, x2, y2 = object_bbox
                cx = int((x + x2) / 2)
                cy = int((y + y2) / 2)

                if 640 > cx > 0 and 480 > cy > 0:
                    cv2.circle(color_image, (cx, cy), 10, (0, 0, 255), -1)

                    object_arm_pos = camera.convert_to_arm_pose(cx, cy, depth_frame)
                    # print("object_arm_pos", object_arm_pos)
                    cv2.putText(color_image, f'Current: {object_arm_pos}', (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)

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
                        # object_catched = False
                        continue

                    # Predict the object's future location if it moved
                    if history_location == [] or (abs(history_location[0] - cx) <= 2 and abs(history_location[1] - cy) <= 2):
                        predicted_x, predicted_y = cx, cy
                    else:
                        predicted_x, predicted_y = kf.predict(cx, cy, time_steps)
                        # predicted_x, predicted_y = ekf.step(np.array([cx, cy]), 1, time_steps)

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
                        # object_catched = False
                        continue
                    else:
                        if predicted_arm_pos is not None:
                            # Send the predicted arm position to the DMP thread
                            predicted_arm_pos_queue.put(predicted_arm_pos)

            cv2.imshow("Frame", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the DMP thread
        predicted_arm_pos_queue.put(None)
        dmp_thread.join()

        cv2.destroyAllWindows()
