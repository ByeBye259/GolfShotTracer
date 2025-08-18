from typing import Dict, List, Tuple
import numpy as np


def carry_error_pct(pred_carry: float, true_carry: float) -> float:
    if true_carry <= 1e-6:
        return 0.0
    return abs(pred_carry - true_carry) / true_carry * 100.0


def apex_error_m(pred_apex: float, true_apex: float) -> float:
    return abs(pred_apex - true_apex)


def launch_angle_error_deg(pred_deg: float, true_deg: float) -> float:
    return abs(pred_deg - true_deg)
