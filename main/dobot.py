from pydobot import Dobot
import time
import threading


class Calibration:
    def __init__(self):
        self.stop = False
        port = "COM9"
        self.device = Dobot(port=port, verbose=False)
        (x, y, z, r, j1, j2, j3, j4) = self.device.pose()
        print("#################dobot pose")
        print("x: {}, y: {}, z: {}, r: {}, j1: {}, j2: {}, j3: {}, j4: {}".format(x, y, z, r, j1, j2, j3, j4))
        # move to calibration position
        self.device.move_to(x=250, y=0, z=50, r=0, wait=True)
        (x, y, z, r, j1, j2, j3, j4) = self.device.pose()
        print("#################recent pose")
        print("x: {}, y: {}, z: {}, r: {}, j1: {}, j2: {}, j3: {}, j4: {}".format(x, y, z, r, j1, j2, j3, j4))

        self.device.speed(500, 500)

        self.interrupt_event = threading.Event()
        self.currently_moving = False
        self.movement_lock = threading.Lock()

    def check_dobot_Arm_location(self):
        (x, y, z, r, j1, j2, j3, j4) = self.device.pose()
        return [x, y, z]

    def check_dobot_Arm_reached_object(self, object_arm_pos):
        (x, y, z, r, j1, j2, j3, j4) = self.device.pose()
        # print("current_position: ", x, y, z)
        if (abs(object_arm_pos[0] - int(x)) <= 5 and abs(object_arm_pos[1] - int(y)) <= 5 and abs(object_arm_pos[2] - int(z)) <= 5):
            return True
        return False

    def move_dobot_arm_no_wait(self, arm_pos):
        self.device.move_to(arm_pos[0], arm_pos[1], arm_pos[2], 0, wait=False)
        time.sleep(0.001)

    def move_dobot_arm_wait(self, arm_pos):
        self.device.move_to(arm_pos[0], arm_pos[1], arm_pos[2], 0, wait=True)
        time.sleep(0.001)

    def move_dobot_arm_dmp(self, trajectory, desired_elements, data_len):
        if trajectory != []:
            with self.movement_lock:
                gap = (data_len - 1) // (desired_elements - 1)
                x_tra = trajectory[:, 0][::gap][:desired_elements]
                y_tra = trajectory[:, 1][::gap][:desired_elements]
                z_tra = trajectory[:, 2][::gap][:desired_elements]

                for i in range(len(x_tra)):
                    self.move_dobot_arm_wait([x_tra[i], y_tra[i], z_tra[i]])
                    if self.interrupt_event.is_set():
                        # print("Movement interrupted!")
                        self.interrupt_event.clear()
                        # time.sleep(1)
                        break

    def press_dobot_arm(self):
        current_end_effector = self.check_dobot_Arm_location()
        current_end_effector[2] -= 8
        self.move_dobot_arm_wait(current_end_effector)

    def suction_cup_suck(self):
        self.device.suck(enable=True)

    def suction_cup_release(self):
        self.device.suck(enable=False)

    def request_interrupt(self):
        self.interrupt_event.set()

