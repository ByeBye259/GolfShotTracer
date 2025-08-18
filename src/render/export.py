from typing import List, Tuple
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
import os

from .overlay import draw_tracer


def write_overlay_video(
    input_video: str,
    output_path: str,
    tracker,
    codec: str = "mp4v",
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Write video with trajectory overlay using polynomial-fitted trajectory.

    Args:
        input_video: Path to input video
        output_path: Path to save output video
        tracker: SingleBallTracker instance with trajectory data
        codec: Video codec to use (default: mp4v)
        ffmpeg_path: Path to ffmpeg executable
    """
    # Get the smooth polynomial trajectory
    smooth_trajectory = tracker.get_smooth_trajectory(degree=3)
    raw_points = [(p.cx, p.cy) for p in tracker.get_track()]
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    point_idx = 0
    points_so_far: List[Tuple[float, float]] = []
    
    # Calculate how many points to show in the tracer (for progressive drawing)
    max_points = len(smooth_trajectory) if smooth_trajectory is not None else len(raw_points)
    points_per_frame = max(1, max_points // total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Add points up to current frame (progressive drawing)
        target_points = min(max_points, (frame_idx + 1) * points_per_frame)
        while point_idx < len(raw_points) and point_idx <= target_points:
            points_so_far = raw_points[:point_idx + 1]
            point_idx += 1

        # Draw trajectory
        if points_so_far:
            # Get the corresponding portion of the smooth trajectory
            smooth_pts = None
            if smooth_trajectory is not None:
                smooth_pts = smooth_trajectory[:min(len(smooth_trajectory), target_points)]
            
            frame = draw_tracer(
                frame, 
                points_so_far,
                smooth_points=smooth_pts,
                color=(0, 0, 255),  # Red
                thickness=3
            )

        # Write frame
        out.write(frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

    # Check if ffmpeg is available
    try:
        # Try to find ffmpeg in the system PATH
        ffmpeg_cmd = shutil.which(ffmpeg_path) or ffmpeg_path
        
        # Remux with ffmpeg for better compatibility
        temp_path = f"{output_path}.temp.mp4"
        os.rename(output_path, temp_path)
        
        cmd = [
            ffmpeg_cmd,
            "-i", temp_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-y",  # Overwrite output file if it exists
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
        except (subprocess.CalledProcessError, OSError) as e:
            # If ffmpeg fails, keep the original file
            if os.path.exists(temp_path):
                os.rename(temp_path, output_path)
            print(f"Warning: FFmpeg remuxing failed, using direct output: {e}")
    except Exception as e:
        print(f"Warning: FFmpeg not found, using direct output: {e}")
