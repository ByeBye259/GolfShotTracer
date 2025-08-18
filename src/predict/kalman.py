from dataclasses import dataclass
import numpy as np


@dataclass
class Kalman2D:
    """Constant-velocity Kalman for 2D position."""
    process_noise: float = 5.0
    measurement_noise: float = 2.0

    def __post_init__(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1e3
        self.F = np.eye(4)
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q_base = np.eye(4)
        self.R = np.eye(2) * (self.measurement_noise ** 2)
        self.initialized = False

    def predict(self, dt: float):
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        q = self.process_noise
        G = np.array([[0.5 * dt ** 2], [0.5 * dt ** 2], [dt], [dt]])
        Q = (G @ G.T) * q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        if z is None:
            return self.x.copy()
        if not self.initialized:
            self.x[0, 0], self.x[1, 0] = z[0], z[1]
            self.initialized = True
        y = z.reshape(2, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()
