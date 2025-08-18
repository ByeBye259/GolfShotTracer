from typing import Dict, List
import numpy as np
from .model import Ballistics2D
from src.vision.track import TrackPoint


def fit_ballistics_2d(track: List[TrackPoint], m_per_px: float, g: float = 9.80665) -> Dict:
    """Fit 2D ballistic parameters (no drag) to the tracked centroids.

    Returns dict with p0 (m), v0 (m/s), launch angles, and residuals.
    """
    if len(track) < 5:
        raise ValueError("Track too short")
    t = (np.array([p.t_ms for p in track]) - track[0].t_ms) / 1000.0
    x = (np.array([p.cx for p in track]) - track[0].cx) * m_per_px
    y = (np.array([p.cy for p in track]) - track[0].cy) * m_per_px
    y = -y  # image y-down to world y-up

    # Fit quadratic separately for x and y: x = vx*t + cx0; y = vy*t - 0.5*g*t^2 + cy0
    A = np.vstack([t, np.ones_like(t)]).T
    vx, cx0 = np.linalg.lstsq(A, x, rcond=None)[0]
    # y fit with known -0.5*g*t^2 term
    A_y = np.vstack([t, np.ones_like(t)]).T
    y_adj = y + 0.5 * g * t ** 2
    vy, cy0 = np.linalg.lstsq(A_y, y_adj, rcond=None)[0]

    p0 = np.array([0.0, 0.0])
    v0 = np.array([vx, vy])
    model = Ballistics2D(g=g)
    traj = model.trajectory(p0, v0, t)

    res = np.sqrt((traj[:, 0] - x) ** 2 + (traj[:, 1] - y) ** 2)
    launch_speed = float(np.linalg.norm(v0))
    elevation_deg = float(np.degrees(np.arctan2(v0[1], max(1e-6, np.hypot(v0[0], 0.0)))))
    azimuth_deg = 0.0  # Unknown from 2D; set 0 and report low confidence

    return {
        "p0_m": p0.tolist(),
        "v0_mps": v0.tolist(),
        "launch_speed_mps": launch_speed,
        "elevation_deg": elevation_deg,
        "azimuth_deg": azimuth_deg,
        "times_s": t.tolist(),
        "traj_m": traj.tolist(),
        "residual_rmse_m": float(np.sqrt(np.mean(res ** 2))),
    }
