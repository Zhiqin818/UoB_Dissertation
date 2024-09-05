import numpy as np
import pyrealsense2 as rs
import cv2
import os


class Camera:
    def __init__(self):
        # initialize realsense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

        # Start streaming
        self.profile = self.pipeline.start(config)

        # get intelrealsense parameters
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.intr = depth_profile.get_intrinsics()

        if os.path.exists("../dobot/save_parms/image_to_arm.npy"):
            self.image_to_arm = np.load("../dobot/save_parms/image_to_arm.npy")

        self.pre_object_cam_center = [0, 0, 0]

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

    def convert_to_arm_pose(self, cx, cy, depth_frame):

        # get middle pixel distance
        dist_to_center = depth_frame.get_distance(int(cx), int(cy))

        # convert the pixel coordinates into camera coordinates
        x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], dist_to_center)

        if (abs(x_cam) == 0.0 and abs(y_cam) == 0.0 and abs(z_cam) == 0.0):
            object_cam_center = self.pre_object_cam_center
        else:
            object_cam_center = [x_cam, y_cam, z_cam]

        # convert camera coordinates into the image position
        img_pos = np.ones(4)
        img_pos[0:3] = object_cam_center

        arm_pos = np.dot(self.image_to_arm, np.array(img_pos))

        self.pre_object_cam_center = object_cam_center

        return list(map(int, arm_pos[:3]))

