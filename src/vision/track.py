from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.predict.kalman import Kalman2D


@dataclass
class TrackPoint:
    t_ms: float
    cx: float
    cy: float
    r: float
    conf: float
    vx: float
    vy: float


class SingleBallTracker:
    def __init__(self, process_noise: float = 5.0, measurement_noise: float = 2.0, initial_pos: Optional[Tuple[float, float]] = None):
        self.kf = Kalman2D(process_noise=process_noise, measurement_noise=measurement_noise)
        self.last_t: Optional[float] = None
        self.track: List[TrackPoint] = []
        self.raw_trajectory: List[Tuple[float, float, float]] = []  # (t_ms, x, y)
        self.poly_trajectory: Optional[np.ndarray] = None  # Polynomial-fitted trajectory
        
        # Initialize with provided position if available
        if initial_pos and initial_pos != (-1, -1):
            x, y = initial_pos
            # Initialize state with position, zero velocity, and zero acceleration
            self.kf.x = np.array([[x], [y], [0], [0], [0], [0]])
            # Add initial point to track
            self.track.append(TrackPoint(0.0, x, y, 5.0, 1.0, 0.0, 0.0))
            self.raw_trajectory.append((0.0, x, y))

    def fit_polynomial_trajectory(self, degree: int = 3) -> None:
        """Fit a polynomial to the raw trajectory points.
        
        Args:
            degree: Degree of the polynomial to fit (2 or 3 recommended)
        """
        if len(self.raw_trajectory) < degree + 1:
            return  # Not enough points for polynomial fitting
            
        # Extract x, y coordinates and timestamps
        ts = np.array([t for t, _, _ in self.raw_trajectory])
        xs = np.array([x for _, x, _ in self.raw_trajectory])
        ys = np.array([y for _, _, y in self.raw_trajectory])
        
        # Normalize time to avoid numerical issues
        t_min, t_max = ts.min(), ts.max()
        if t_max > t_min:
            ts_norm = (ts - t_min) / (t_max - t_min)
        else:
            ts_norm = ts - t_min
            
        # Fit polynomial to x and y coordinates separately
        try:
            coeffs_x = np.polyfit(ts_norm, xs, deg=degree)
            coeffs_y = np.polyfit(ts_norm, ys, deg=degree)
            
            # Generate smooth trajectory
            t_smooth = np.linspace(0, 1, num=100)
            x_smooth = np.polyval(coeffs_x, t_smooth)
            y_smooth = np.polyval(coeffs_y, t_smooth)
            
            # Store the smooth trajectory
            self.poly_trajectory = np.column_stack((x_smooth, y_smooth))
        except (np.linalg.LinAlgError, TypeError):
            # Fallback to raw trajectory if polynomial fitting fails
            self.poly_trajectory = np.column_stack((xs, ys))

    def step(self, t_ms: float, detections: List[Tuple[float, float, float, float]]) -> Optional[TrackPoint]:
        dt = 0.0 if self.last_t is None else max(1e-3, (t_ms - self.last_t) / 1000.0)
        self.kf.predict(dt)
        z = None
        if detections:
            # Take highest-confidence detection
            cx, cy, r, conf = detections[0]
            z = np.array([cx, cy])
        
        # Get updated state (x, y, vx, vy, ax, ay)
        x = self.kf.update(z)
        
        # Store the raw position from Kalman filter
        self.raw_trajectory.append((t_ms, float(x[0, 0]), float(x[1, 0])))
        
        # Create track point with position and velocity (discard acceleration for now)
        tp = TrackPoint(
            t_ms,
            float(x[0, 0]),  # x
            float(x[1, 0]),  # y
            detections[0][2] if detections else 2.0,  # radius
            detections[0][3] if detections else 0.0,  # confidence
            float(x[2, 0]),  # vx
            float(x[3, 0])   # vy
        )
        self.track.append(tp)
        self.last_t = t_ms
        return tp

    def get_track(self) -> List[TrackPoint]:
        return self.track

    def get_smooth_trajectory(self, degree: int = 3) -> np.ndarray:
        """Get the polynomial-fitted trajectory.
        
        Args:
            degree: Degree of the polynomial to fit (2 or 3 recommended)
            
        Returns:
            Nx2 array of (x, y) points of the smooth trajectory
        """
        if self.poly_trajectory is None:
            self.fit_polynomial_trajectory(degree=degree)
        return self.poly_trajectory if self.poly_trajectory is not None else np.array([])
