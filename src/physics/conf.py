from typing import Dict
import numpy as np


def estimate_confidence(residual_rmse_m: float, track_len: int) -> Dict:
    # Simple heuristic: higher residuals -> lower confidence
    base = max(0.05, min(0.95, 1.0 / (1.0 + 10.0 * residual_rmse_m)))
    n_factor = min(1.0, track_len / 30.0)
    conf = base * (0.5 + 0.5 * n_factor)
    return {
        "overall": float(conf),
        "speed": float(conf),
        "elevation": float(conf),
        "azimuth": 0.2,  # low confidence in azimuth with single-view simplification
        "spin_axis": 0.0,
    }
