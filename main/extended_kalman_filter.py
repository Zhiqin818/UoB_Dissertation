import numpy as np

class ExtendedKalmanFilter:
    def __init__(self):
        # State vector [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 500  # Initial covariance matrix

        # Process noise covariance matrix
        self.processNoiseCov = np.eye(4) * 0.01

        # Measurement noise covariance matrix
        self.measurementNoiseCov = np.eye(2) * 0.01


    def f(self, state, dt):
        """ State transition function for non-linear movement with dt time step. """
        x, y, vx, vy = state

        ax = 0.1 * vx
        ay = 0.1 * vy

        # Update velocity with acceleration
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        # Update position with new velocity
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt

        return np.array([x_new, y_new, vx_new, vy_new])

    def h(self, state):
        """ Measurement function (direct observation of x and y). """
        x, y, vx, vy = state
        return np.array([x, y])

    def F_jacobian(self, state, dt):
        """ Jacobian of the state transition function. """
        return np.array([[1, 0, dt + 0.1 * dt ** 2, 0],
                         [0, 1, 0, dt + 0.1 * dt ** 2],
                         [0, 0, 1 + 0.1 * dt, 0],
                         [0, 0, 0, 1 + 0.1 * dt]])

    def H_jacobian(self, state):
        """ Jacobian of the measurement function. """
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    def predict(self, dt):
        """ Prediction step of EKF. """
        # Predict state and covariance
        F = self.F_jacobian(self.state, dt)
        self.state = self.f(self.state, dt)
        self.P = F @ self.P @ F.T + self.processNoiseCov

    def update(self, measurement):
        """ Update step of EKF. """
        H = self.H_jacobian(self.state)
        z_pred = self.h(self.state)
        y = measurement - z_pred  # Measurement residual

        S = H @ self.P @ H.T + self.measurementNoiseCov  # Residual covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state and covariance
        self.state += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def step(self, measurement, dt, timestep):
        """ Single step of predict-update in EKF. """
        for _ in range(timestep):
            self.predict(dt)
            self.update(measurement)
            measurement = int(self.state[0]), int(self.state[1])

        predicted_x = int(self.state[0])
        predicted_y = int(self.state[1])

        # if the prediction lcoation is out of frame
        if predicted_x <= 0 or predicted_x >= 640:
            predicted_x = measurement[0]
        if predicted_y <= 0 or predicted_y >= 480:
            predicted_y = measurement[1]

        return predicted_x, predicted_y

