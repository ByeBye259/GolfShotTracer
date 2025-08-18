from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Kalman2D:
    """Constant-velocity Kalman for 2D position."""
    process_noise: float = 1.0
    measurement_noise: float = 1.0

    def __post_init__(self):
        # State: [x, y, vx, vy, ax, ay]  # Added acceleration
        self.dt = 1.0 / 30.0  # Will be updated per frame
        self.kf = cv2.KalmanFilter(6, 2, 0)
        
        # Measurement matrix - only measures position
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        # State transition matrix with acceleration
        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt**2, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt**2],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Tuned noise parameters
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * self.process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0  # Initial uncertainty

    def predict(self, dt: float = None):
        if dt is not None:
            self.dt = max(1e-3, dt)  # Avoid division by zero
            # Update transition matrix with new dt
            self.kf.transitionMatrix[0, 2] = self.dt
            self.kf.transitionMatrix[1, 3] = self.dt
            self.kf.transitionMatrix[0, 4] = 0.5 * self.dt**2
            self.kf.transitionMatrix[1, 5] = 0.5 * self.dt**2
            self.kf.transitionMatrix[2, 4] = self.dt
            self.kf.transitionMatrix[3, 5] = self.dt
        return self.kf.predict()[:2]  # Only return position

    def update(self, z: np.ndarray):
        if z is None:
            return self.kf.predict()
        # Ensure z is float32 and has the right shape
        measurement = np.array(z, dtype=np.float32).reshape(2, 1)
        self.kf.correct(measurement)
        return self.kf.predict()
