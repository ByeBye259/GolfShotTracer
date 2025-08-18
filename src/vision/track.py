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
        
        # Initialize with provided position if available
        if initial_pos and initial_pos != (-1, -1):
            x, y = initial_pos
            # Initialize state with position and zero velocity
            self.kf.x = np.array([[x], [y], [0], [0]])
            # Add initial point to track
            self.track.append(TrackPoint(0.0, x, y, 5.0, 1.0, 0.0, 0.0))

    def step(self, t_ms: float, detections: List[Tuple[float, float, float, float]]) -> Optional[TrackPoint]:
        dt = 0.0 if self.last_t is None else max(1e-3, (t_ms - self.last_t) / 1000.0)
        self.kf.predict(dt)
        z = None
        if detections:
            # Take highest-confidence detection
            cx, cy, r, conf = detections[0]
            z = np.array([cx, cy])
        x = self.kf.update(z)
        self.last_t = t_ms
        tp = TrackPoint(t_ms, float(x[0, 0]), float(x[1, 0]), detections[0][2] if detections else 2.0, detections[0][3] if detections else 0.0, float(x[2, 0]), float(x[3, 0]))
        self.track.append(tp)
        return tp

    def get_track(self) -> List[TrackPoint]:
        return self.track
