from typing import Callable, Dict, List, Tuple, Optional
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch
import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.roboflow_detector import RoboflowBallDetector
from src.vision.super_resolution import SuperResolution
from src.vision.deep_sort_tracker import DeepSortTracker, TrackedObject
from src.physics.trajectory_refiner import TrajectoryRefiner, TrajectoryPoint
from src.render.export import write_overlay_video
from src.utils.ball_selector import BallSelector
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def process_video(
    input_video: str, 
    output_dir: str, 
    config_path: str, 
    progress_cb: Callable[[float, str], None] = lambda p, m: None,
    interactive: bool = False
):
    """Process a video to track golf ball and generate trajectory overlay.
    
    Args:
        input_video: Path to input video file
        output_dir: Directory to save output files
        config_path: Path to config file
        progress_cb: Callback for progress updates (progress: float, message: str)
        interactive: Whether to show interactive selection of initial ball position
        
    Returns:
        Path to output video file
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps if fps > 0 else 1/30.0
    
    # Initialize components
    # Initialize super-resolution
    sr_model = SuperResolution(
        model_path=cfg.get('super_resolution', {}).get('model_path'),
        scale=cfg.get('super_resolution', {}).get('scale', 2),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize Roboflow detector
    detector = RoboflowBallDetector(
        api_key=cfg['detector'].get('api_key'),
        model_name=cfg['detector'].get('model_name', 'golf-ball-detection-hii2e'),
        version=cfg['detector'].get('version', 2),
        conf_thresh=cfg['detector'].get('conf_thresh', 0.2)
    )
    
    # Initialize tracker
    tracker = DeepSortTracker(
        max_age=cfg['tracker'].get('max_age', 30),
        max_iou_distance=cfg['tracker'].get('max_iou_distance', 0.7),
        max_cosine_distance=cfg['tracker'].get('max_cosine_distance', 0.2),
        nms_max_overlap=cfg['tracker'].get('nms_max_overlap', 1.0)
    )
    
    # Initialize trajectory refiner
    trajectory_refiner = TrajectoryRefiner(
        g=cfg['physics'].get('gravity', 9.81) * (frame_height / 100),  # Scale gravity to pixels
        dt=dt
    )
    
    # For interactive ball selection
    initial_pos = None
    if interactive:
        ret, frame = cap.read()
        if ret:
            # Enhance frame with super-resolution for better selection
            enhanced_frame = sr_model.enhance(frame)
            
            selector = BallSelector()
            initial_pos = selector.select_ball(enhanced_frame)
            
            # Scale position back to original frame size if needed
            if enhanced_frame.shape != frame.shape[:2]:
                h_ratio = frame.shape[0] / enhanced_frame.shape[0]
                w_ratio = frame.shape[1] / enhanced_frame.shape[1]
                initial_pos = (int(initial_pos[0] * w_ratio), int(initial_pos[1] * h_ratio))
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
    
    # Process video
    frame_count = 0
    tracked_objects = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get timestamp
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0  # Convert to seconds
        
        # Apply super-resolution
        enhanced_frame = sr_model.enhance(frame)
        
        # Detect ball using Roboflow
        detections = detector.detect(enhanced_frame)
        
        # Convert detections to format expected by DeepSORT
        # Roboflow detections are already in (x, y, radius, confidence) format
        dets_for_tracker = []
        for (x, y, r, conf) in detections:
            # Convert circle to bbox [x1, y1, x2, y2, conf]
            x1, y1 = int(x - r), int(y - r)
            x2, y2 = int(x + r), int(y + r)
            dets_for_tracker.append([x1, y1, x2, y2, conf])
        
        # Update tracker
        tracked_objs = tracker.update(dets_for_tracker, enhanced_frame)
        
        # Add to trajectory refiner
        for obj in tracked_objs:
            x1, y1, x2, y2 = obj.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            trajectory_refiner.add_point(cx, cy, t, obj.confidence)
        
        # Refine trajectory
        if frame_count % 5 == 0:  # Don't refine every frame for performance
            refined_trajectory = trajectory_refiner.refine()
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_cb(progress * 0.8, "detect+track")
    
    cap.release()
    
    if not trajectory_refiner.trajectory:
        raise RuntimeError("Ball not detected reliably. Try better lighting/contrast or different clip.")
    
    # Final refinement of the complete trajectory
    refined_trajectory = trajectory_refiner.refine(window_size=min(15, len(trajectory_refiner.trajectory) // 2))
    
    # Extract points for visualization
    points = [(p.x, p.y) for p in refined_trajectory]
    
    # Write output video with overlay
    progress_cb(0.8, "render")
    out_video = str(Path(output_dir) / "tracer.mp4")
    
    # Create a custom tracker object with the refined trajectory for rendering
    class TrajectoryWrapper:
        def get_track(self):
            return [type('obj', (), {'cx': p.x, 'cy': p.y}) for p in refined_trajectory]
        
        def get_smooth_trajectory(self, degree=3):
            # We already have a refined trajectory, just return it
            return np.array([(p.x, p.y) for p in refined_trajectory])
    
    write_overlay_video(
        input_video=input_video,
        output_path=out_video,
        tracker=TrajectoryWrapper(),
        codec=cfg.get("export", {}).get("codec", "mp4v"),
        ffmpeg_path=cfg.get("export", {}).get("ffmpeg_path", "ffmpeg")
    )
    
    progress_cb(1.0, "done")
    return out_video
