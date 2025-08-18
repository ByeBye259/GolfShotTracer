from typing import List, Tuple
import numpy as np

from src.vision.track import TrackPoint


def fit_scale_from_vertical_accel(track: List[TrackPoint], g: float = 9.80665):
    """Estimate pixel-to-meter scale using vertical quadratic fit.

    Assumes approximately constant scale over the arc and small perspective change.
    y_pix(t) ~ a t^2 + b t + c, with a_pix ≈ g / s (sign inverted due to image y-down).
    Returns meters_per_pixel.
    """
    if len(track) < 5:
        return 0.01  # fallback
    t = np.array([p.t_ms for p in track]) / 1000.0
    y = np.array([p.cy for p in track])
    # Fit quadratic
    A = np.vstack([t ** 2, t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_pix = coef[0]
    if abs(a_pix) < 1e-6:
        return 0.01
    meters_per_pixel = abs(g / (2.0 * a_pix))  # factor 0.5 since s= g / (2*a) when using y = at^2 + bt + c
    meters_per_pixel = float(np.clip(meters_per_pixel, 0.001, 0.1))
    return meters_per_pixel


def initial_velocity_2d(track: List[TrackPoint], m_per_px: float):
    if len(track) < 2:
        return 0.0, 0.0
    t0 = track[0].t_ms / 1000.0
    t1 = track[1].t_ms / 1000.0
    dt = max(1e-3, t1 - t0)
    vx_pix = (track[1].cx - track[0].cx) / dt
    vy_pix = (track[1].cy - track[0].cy) / dt
    vx = vx_pix * m_per_px
    vy = -vy_pix * m_per_px  # image y-down -> world y-up
    return float(vx), float(vy)
