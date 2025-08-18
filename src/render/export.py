from typing import List, Tuple
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np

from .overlay import draw_tracer


def write_overlay_video(input_video: str, output_video: str, points_px: List[Tuple[float, float]], codec: str = "h264", ffmpeg_path: str = "ffmpeg"):
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")
        
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    aspect_ratio = w / h
    
    # Set up temporary output
    tmp_out = str(Path(output_video).with_suffix(".tmp.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID for intermediate (better quality than mp4v)
    
    # Create writer with original dimensions to maintain aspect ratio
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h), isColor=True)

    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(points_px)
    pts_for_frame = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use prefix of points up to current index for nice trail
        if idx < len(points_px):
            pts_for_frame.append(points_px[idx])
        frame = draw_tracer(frame, pts_for_frame)
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()

    # Try to remux audio and encode with proper settings
    out = Path(output_video)
    if shutil.which(ffmpeg_path) and codec.lower() in ("h264", "libx264"):
        try:
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", tmp_out,
                "-i", input_video,
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",  # Lower CRF = better quality (18-28 is good)
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",  # For web streaming
                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                str(out)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            Path(tmp_out).unlink(missing_ok=True)
        except Exception:
            shutil.move(tmp_out, str(out))
    else:
        shutil.move(tmp_out, str(out))
