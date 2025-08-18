from dataclasses import dataclass
import numpy as np


@dataclass
class Ballistics2D:
    g: float = 9.80665

    def position(self, p0: np.ndarray, v0: np.ndarray, t: float) -> np.ndarray:
        # p = p0 + v0*t + 0.5*a*t^2; a = (0, -g)
        ax = 0.0
        ay = -self.g
        return p0 + v0 * t + np.array([0.5 * ax * t * t, 0.5 * ay * t * t])

    def trajectory(self, p0: np.ndarray, v0: np.ndarray, ts: np.ndarray) -> np.ndarray:
        pts = [self.position(p0, v0, float(t)) for t in ts]
        return np.stack(pts, axis=0)
