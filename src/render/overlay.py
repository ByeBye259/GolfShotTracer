from typing import Dict, List, Tuple
import cv2
import numpy as np


def draw_tracer(frame: np.ndarray, points_px: List[Tuple[float, float]], color=(20, 160, 255), thickness=2, halo_color=(200, 200, 255), halo_alpha=0.25):
    overlay = frame.copy()
    pts = [(int(x), int(y)) for x, y in points_px]
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i - 1], pts[i], color, thickness + 6)
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i - 1], pts[i], color, thickness)
    return cv2.addWeighted(overlay, halo_alpha, frame, 1 - halo_alpha, 0)


def draw_hud(frame: np.ndarray, metrics: Dict, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    lines = [
        f"Speed: {metrics.get('launch_speed_mps', 0.0)*3.6:.1f} km/h",
        f"Launch: {metrics.get('elevation_deg', 0.0):.1f} deg",
        f"Apex: {metrics.get('apex_m', 0.0):.1f} m",
        f"Carry: {metrics.get('carry_m', 0.0):.1f} m",
        f"TOF: {metrics.get('time_of_flight_s', 0.0):.2f} s",
    ]
    y = 30
    for text in lines:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28
    return frame
