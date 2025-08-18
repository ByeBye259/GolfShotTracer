from typing import Callable, Dict, List, Tuple, Optional
from pathlib import Path
import yaml
import cv2
import numpy as np
import sys
import os

# Add src to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import get_logger
from src.vision.detect import BallDetector, DetectorConfig
from src.vision.track import SingleBallTracker
from src.vision.launch_find import find_launch_index
from src.render.export import write_overlay_video
from src.utils.ball_selector import BallSelector


logger = get_logger("pipeline")


def process_video(
    input_video: str, 
    output_dir: str, 
    config_path: str, 
    progress_cb: Callable[[float, str], None] = lambda p, m: None,
    interactive: bool = False
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get first frame for interactive selection if needed
    initial_pos = None
    if interactive:
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read first frame from video")
            
        selector = BallSelector()
        progress_cb(0.0, "Click on the ball in the first frame")
        initial_pos = selector.select_ball(first_frame)
        
        if initial_pos == (-1, -1):
            progress_cb(0.0, "Skipping interactive selection, using auto-detect")
        else:
            progress_cb(0.0, f"Selected initial position: {initial_pos}")
            
        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    det_cfg = DetectorConfig(
        backend=cfg.get("detection", {}).get("backend", "auto"),
        min_conf=float(cfg.get("detection", {}).get("min_conf", 0.1)),
        min_radius_px=int(cfg.get("detection", {}).get("min_radius_px", 1)),
        max_radius_px=int(cfg.get("detection", {}).get("max_radius_px", 30)),
    )
    detector = BallDetector(det_cfg)
    tracker = SingleBallTracker(
        process_noise=float(cfg.get("tracking", {}).get("process_noise", 5.0)),
        measurement_noise=float(cfg.get("tracking", {}).get("measurement_noise", 2.0)),
        initial_pos=initial_pos  # Pass the user-selected position if any
    )

    points: List[Tuple[float, float]] = []
    times_ms: List[float] = []

    progress_cb(0.05, "detect+track")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        dets = detector.detect_frame(frame, t_ms)
        det_tuples = [(d.cx, d.cy, d.r, d.conf) for d in dets]
        tp = tracker.step(t_ms, det_tuples)
        points.append((tp.cx, tp.cy))
        times_ms.append(t_ms)
        frame_idx += 1
        if total > 0 and frame_idx % 30 == 0:
            progress_cb(0.05 + 0.4 * (frame_idx / total), "detect+track")

    cap.release()

    if len(points) < 5:
        raise RuntimeError("Ball not detected reliably. Try better lighting/contrast or different clip.")

    # Launch detection: trim early frames before launch to focus the overlay
    launch_idx = find_launch_index(tracker.get_track(), float(cfg.get("launch", {}).get("speed_jump_thresh", 2.0)))
    points = points[launch_idx:]
    times_ms = [t - times_ms[launch_idx] for t in times_ms[launch_idx:]]

    # Fit polynomial to the trajectory
    tracker.fit_polynomial_trajectory(degree=3)
    
    # Write output video with overlay
    progress_cb(0.8, "render")
    out_video = str(Path(output_dir) / "tracer.mp4")
    write_overlay_video(
        input_video=input_video,
        output_path=out_video,
        tracker=tracker,
        codec=cfg.get("export", {}).get("codec", "mp4v"),
        ffmpeg_path=cfg.get("export", {}).get("ffmpeg_path", "ffmpeg")
    )
    
    progress_cb(1.0, "done")
    return out_video
