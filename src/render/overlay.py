from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


def draw_tracer(
    frame: np.ndarray,
    points: List[Tuple[float, float]],
    smooth_points: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (0, 0, 255),  # Red
    thickness: int = 2,
) -> np.ndarray:
    """Draw a tracer line on the frame using either raw or smoothed points.

    Args:
        frame: Input frame (BGR format)
        points: List of (x, y) points to draw (raw points)
        smooth_points: Optional Nx2 array of pre-smoothed points
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Frame with tracer line drawn
    """
    if smooth_points is not None and len(smooth_points) >= 2:
        # Use the pre-smoothed polynomial curve
        pts = smooth_points.astype(np.int32).reshape((-1, 1, 2))
    elif len(points) >= 2:
        # Fall back to raw points if no smoothing available
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
    else:
        return frame  # Not enough points to draw
    
    # Draw the line with anti-aliasing
    cv2.polylines(
        frame, 
        [pts], 
        isClosed=False, 
        color=color, 
        thickness=thickness, 
        lineType=cv2.LINE_AA
    )
    
    # Draw a circle at the current position (last point)
    if len(pts) > 0:
        last_point = tuple(pts[-1][0])
        cv2.circle(
            frame, 
            last_point, 
            radius=thickness * 2, 
            color=color, 
            thickness=-1,  # Filled circle
            lineType=cv2.LINE_AA
        )
    
    return frame
