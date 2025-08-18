from typing import List
import numpy as np

from .track import TrackPoint


def find_launch_index(track: List[TrackPoint], speed_jump_thresh_px_per_ms: float = 2.0) -> int:
    """Find launch by largest speed jump near beginning."""
    if not track:
        return 0
    speeds = []
    for i in range(1, len(track)):
        dt = max(1e-3, (track[i].t_ms - track[i - 1].t_ms) / 1000.0)
        vx = (track[i].cx - track[i - 1].cx) / dt
        vy = (track[i].cy - track[i - 1].cy) / dt
        speeds.append((i, np.hypot(vx, vy)))
    if not speeds:
        return 0
    # Find first index exceeding threshold
    for i, s in speeds:
        if s / 1000.0 > speed_jump_thresh_px_per_ms:  # speeds are in px/s; convert to px/ms
            return i
    # fallback: first good point
    return 0
