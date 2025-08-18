from typing import Dict, List, Tuple
import cv2
import numpy as np


def draw_tracer(frame: np.ndarray, points_px: List[Tuple[float, float]], color=(0, 0, 255), thickness=3):
    """Draw a solid red tracer line on the frame.
    
    Args:
        frame: Input frame to draw on
        points_px: List of (x,y) coordinates for the tracer
        color: BGR color tuple (default: bright red)
        thickness: Line thickness in pixels
    """
    if len(points_px) < 2:
        return frame
        
    result = frame.copy()
    pts = np.array([(int(x), int(y)) for x, y in points_px if x > 0 and y > 0], np.int32)
    
    if len(pts) > 1:
        # Draw a solid red line
        cv2.polylines(result, [pts], isClosed=False, color=color, 
                     thickness=thickness, lineType=cv2.LINE_AA)
    
    return result
