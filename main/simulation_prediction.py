import time

import matplotlib
# Set the backend
matplotlib.use('MacOSX')  # Or 'TkAgg', 'MacOSX', etc.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter

# Load Kalman filter to predict the trajectory
kf = KalmanFilter()
ekf = ExtendedKalmanFilter()

errors = []

circle_trajectory = np.array([

    # [260, 0, -20],
    # [260, 5, -20],
    # [259, 10, -20],
    # [257, 15, -20],
    # [255, 20, -20],
    # [252, 24, -20],
    # [249, 28, -20],
    # [245, 31, -20],
    # [241, 34, -20],
    # [236, 37, -20],
    # [231, 38, -20],
    # [226, 39, -20],
    # [221, 40, -20],
    # [216, 40, -20],
    # [211, 39, -20],
    # [206, 38, -20],
    # [201, 35, -20],
    # [197, 33, -20],
    # [193, 30, -20],
    # [190, 26, -20],
    # [186, 22, -20],
    # [184, 17, -20],
    # [182, 13, -20],
    # [181, 8, -20],
    # [180, 3, -20],
    # [180, -3, -20],
    # [181, -8, -20],
    # [182, -13, -20],
    # [184, -17, -20],
    # [186, -22, -20],
    # [190, -26, -20],
    # [193, -30, -20],
    # [197, -33, -20],
    # [201, -35, -20],
    # [206, -38, -20],
    # [211, -39, -20],
    # [216, -40, -20],
    # [221, -40, -20],
    # [226, -39, -20],
    # [231, -38, -20],
    # [236, -37, -20],
    # [241, -34, -20],
    # [245, -31, -20],
    # [249, -28, -20],
    # [252, -24, -20],
    # [255, -20, -20],
    # [257, -15, -20],
    # [259, -10, -20],
    # [260, -5, -20],
    # [260, 0, -20]

    [260, 0, -20],
    [260, 2.5, -20],
    [260, 5, -20],
    [259.5, 7.5, -20],
    [259, 10, -20],
    [258, 12.5, -20],
    [257, 15, -20],
    [256, 17.5, -20],
    [255, 20, -20],
    [253.5, 22, -20],
    [252, 24, -20],
    [250.5, 26, -20],
    [249, 28, -20],
    [247, 29.5, -20],
    [245, 31, -20],
    [243, 32.5, -20],
    [241, 34, -20],
    [238.5, 35.5, -20],
    [236, 37, -20],
    [233.5, 37.5, -20],
    [231, 38, -20],
    [228.5, 38.5, -20],
    [226, 39, -20],
    [223.5, 39.5, -20],
    [221, 40, -20],
    [218.5, 39.5, -20],
    [216, 40, -20],
    [213.5, 39.5, -20],
    [211, 39, -20],
    [208.5, 38.5, -20],
    [206, 38, -20],
    [203.5, 36.5, -20],
    [201, 35, -20],
    [199, 34, -20],
    [197, 33, -20],
    [195, 31.5, -20],
    [193, 30, -20],
    [191.5, 28, -20],
    [190, 26, -20],
    [188, 24, -20],
    [186, 22, -20],
    [185, 19.5, -20],
    [184, 17, -20],
    [183, 15, -20],
    [182, 13, -20],
    [181.5, 10.5, -20],
    [181, 8, -20],
    [180.5, 5.5, -20],
    [180, 3, -20],
    [180, 0, -20],
    [180, -3, -20],
    [180.5, -5.5, -20],
    [181, -8, -20],
    [181.5, -10.5, -20],
    [182, -13, -20],
    [183, -15, -20],
    [184, -17, -20],
    [185, -19.5, -20],
    [186, -22, -20],
    [188, -24, -20],
    [190, -26, -20],
    [191.5, -28, -20],
    [193, -30, -20],
    [195, -31.5, -20],
    [197, -33, -20],
    [199, -34, -20],
    [201, -35, -20],
    [203.5, -36.5, -20],
    [206, -38, -20],
    [208.5, -38.5, -20],
    [211, -39, -20],
    [213.5, -39.5, -20],
    [216, -40, -20],
    [218.5, -39.5, -20],
    [221, -40, -20],
    [223.5, -39.5, -20],
    [226, -39, -20],
    [228.5, -38.5, -20],
    [231, -38, -20],
    [233.5, -37.5, -20],
    [236, -37, -20],
    [238.5, -35.5, -20],
    [241, -34, -20],
    [243, -32.5, -20],
    [245, -31, -20],
    [247, -29.5, -20],
    [249, -28, -20],
    [250.5, -26, -20],
    [252, -24, -20],
    [253.5, -22, -20],
    [255, -20, -20],
    [256, -17.5, -20],
    [257, -15, -20],
    [258, -12.5, -20],
    [259, -10, -20],
    [259.5, -7.5, -20],
    [260, -5, -20],
    [260, -2.5, -20],
    [260, 0, -20]

])

cuboid_trajectory = np.array([
    # Edge 1: From (170, -30) to (170, 30)
    [170, -30, -20],
    [170, -24, -20],
    [170, -18, -20],
    [170, -12, -20],
    [170, -6, -20],
    [170, 0, -20],
    [170, 6, -20],
    [170, 12, -20],
    [170, 18, -20],
    [170, 24, -20],
    [170, 30, -20],

    # Edge 2: From (170, 30) to (230, 30)
    [170, 30, -20],
    [176, 30, -20],
    [182, 30, -20],
    [188, 30, -20],
    [194, 30, -20],
    [200, 30, -20],
    [206, 30, -20],
    [212, 30, -20],
    [218, 30, -20],
    [224, 30, -20],
    [230, 30, -20],

    # Edge 3: From (230, 30) to (230, -30)
    [230, 30, -20],
    [230, 24, -20],
    [230, 18, -20],
    [230, 12, -20],
    [230, 6, -20],
    [230, 0, -20],
    [230, -6, -20],
    [230, -12, -20],
    [230, -18, -20],
    [230, -24, -20],
    [230, -30, -20],

    # Edge 4: From (230, -30) to (170, -30)
    [230, -30, -20],
    [224, -30, -20],
    [218, -30, -20],
    [212, -30, -20],
    [206, -30, -20],
    [200, -30, -20],
    [194, -30, -20],
    [188, -30, -20],
    [182, -30, -20],
    [176, -30, -20],
    [170, -30, -20]
])

triangle_trajectory = np.array([
    [250, -30, -20],
    [250, -25, -20],
    [250, -20, -20],
    [250, -15, -20],
    [250, -10, -20],
    [250, -5, -20],
    [250, 0, -20],
    [250, 5, -20],
    [250, 10, -20],
    [250, 15, -20],
    [250, 20, -20],
    [250, 25, -20],
    [250, 30, -20],

    [244, 28, -20],
    [238, 26, -20],
    [232, 24, -20],
    [226, 22, -20],
    [220, 20, -20],
    [214, 18, -20],
    [208, 16, -20],
    [202, 14, -20],
    [196, 12, -20],
    [190, 10, -20],
    [184, 8, -20],
    [178, 6, -20],
    [172, 4, -20],
    [166, 2, -20],
    [160, 0, -20],

    [166, -2, -20],
    [172, -4, -20],
    [178, -6, -20],
    [184, -8, -20],
    [190, -10, -20],
    [196, -12, -20],
    [202, -14, -20],
    [208, -16, -20],
    [214, -18, -20],
    [220, -20, -20],
    [226, -22, -20],
    [232, -24, -20],
    [238, -26, -20],
    [244, -28, -20],
    [250, -30, -20],
])

straight_trajectory = np.array([
    [150, 0, -20],
    [152, 0, -20],
    [154, 0, -20],
    [156, 0, -20],
    [158, 0, -20],
    [160, 0, -20],
    [162, 0, -20],
    [164, 0, -20],
    [166, 0, -20],
    [168, 0, -20],
    [170, 0, -20],
    [172, 0, -20],
    [174, 0, -20],
    [176, 0, -20],
    [178, 0, -20],
    [180, 0, -20],
    [182, 0, -20],
    [184, 0, -20],
    [186, 0, -20],
    [188, 0, -20],
    [190, 0, -20],
    [192, 0, -20],
    [194, 0, -20],
    [196, 0, -20],
    [198, 0, -20],
    [200, 0, -20],
    [202, 0, -20],
    [204, 0, -20],
    [206, 0, -20],
    [208, 0, -20],
    [210, 0, -20],
    [212, 0, -20],
    [214, 0, -20],
    [216, 0, -20],
    [218, 0, -20],
    [220, 0, -20],
    [222, 0, -20],
    [224, 0, -20],
    [226, 0, -20],
    [228, 0, -20],
    [230, 0, -20],
    [232, 0, -20],
    [234, 0, -20],
    [236, 0, -20],
    [238, 0, -20],
    [240, 0, -20],
    [242, 0, -20],
    [244, 0, -20],
    [246, 0, -20],
    [248, 0, -20],
    [250, 0, -20]
])

history_trajectory = circle_trajectory

predicted_trajectory = []

for i in range(len(history_trajectory)):

    # use kalman filter to predict future locations
    predicted_x, predicted_y = kf.predict(history_trajectory[i][0], history_trajectory[i][1], 4)

    # use extended kalman filter to predict future locations
    # predicted_x, predicted_y = ekf.step(np.array([history_trajectory[i][0], history_trajectory[i][1]]), 1, 4)

    predicted_trajectory.append([predicted_x, predicted_y, -20])

predicted_trajectory = np.array(predicted_trajectory)
# print(predicted_square_trajectory)

# Extract coordinates
x1, y1, z1 = zip(*history_trajectory)
x2, y2, z2 = zip(*predicted_trajectory)

# Create a figure and a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits
ax.set_xlim([100, 300])
ax.set_ylim([-80, 80])
ax.set_zlim([-60, 100])

# Ensure the aspect ratio is equal
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1 for x:y:z

# Initialize the scatter plot for the moving points
scat1 = ax.scatter([], [], [], c='r', marker='o', label='Current Object Location')
scat2 = ax.scatter([], [], [], c='g', marker='o', label='Predicted Object Location')

# Add legend
ax.legend()

# Update function for animation
def update(frame):

    # Update the positions of both moving points
    scat1._offsets3d = ([x1[frame]], [y1[frame]], [z1[frame]])
    scat2._offsets3d = ([x2[frame]], [y2[frame]], [z2[frame]])

    # Stop animation after the last frame
    if frame == len(x1) - 1:
        plt.close(fig)  # Close the figure window

    return scat1, scat2


# Create the animation
ani = FuncAnimation(fig, update, frames=max(len(history_trajectory), len(predicted_trajectory)), interval=500, blit=False)

# Show plot
plt.show()

# calculate
for gt, pred in zip(history_trajectory, predicted_trajectory):
    error = np.linalg.norm(np.array(gt) - np.array(pred))
    errors.append(error)

mae = np.mean(errors)

trajectory_fig = plt.figure(figsize=(10, 6))
plt.plot(history_trajectory[:, 0], history_trajectory[:, 1], 'g', label='Ground Truth')
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r--', label='Prediction')
plt.title('Object Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Error plot
prediction_error = plt.figure(figsize=(10, 6))
plt.plot(errors, 'b', label='Prediction Error')
plt.title('Prediction Error Over Time')
plt.xlabel('Timestep')
plt.ylabel('Error')
plt.legend()
plt.show()




