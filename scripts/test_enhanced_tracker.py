import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.yolo_detector import YOLOBallDetector
from src.vision.super_resolution import SuperResolution
from src.vision.deep_sort_tracker import DeepSortTracker
from src.physics.trajectory_refiner import TrajectoryRefiner
from src.utils.ball_selector import BallSelector
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("test_enhanced_tracker")

def test_enhanced_tracker(input_video: str, output_video: str, config: Dict[str, Any]):
    """Test the enhanced tracker on a video file."""
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Initialize components
    sr_model = SuperResolution(
        model_path=os.path.join('weights', config['super_resolution'].get('model_path', 'EDSR_x2.pb')),
        scale=config['super_resolution'].get('scale', 2),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Ensure the model path exists for YOLO
    yolo_model_path = os.path.join('weights', config['detector'].get('model_path', 'yolov8n-golf.pt'))
    if not os.path.exists(yolo_model_path):
        logger.warning(f"YOLO model not found at {yolo_model_path}, using default YOLOv8n")
        yolo_model_path = 'yolov8n.pt'
    
    detector = YOLOBallDetector(
        model_path=yolo_model_path,
        conf_thresh=config['detector'].get('conf_thresh', 0.25),
        iou_thresh=config['detector'].get('iou_thresh', 0.45)
    )
    
    tracker = DeepSortTracker(
        max_age=config['tracker'].get('max_age', 30),
        max_iou_distance=config['tracker'].get('max_iou_distance', 0.7),
        max_cosine_distance=config['tracker'].get('max_cosine_distance', 0.2),
        nms_max_overlap=config['tracker'].get('nms_max_overlap', 1.0)
    )
    
    trajectory_refiner = TrajectoryRefiner(
        g=config['physics'].get('gravity', 9.81) * (height / 100),  # Scale gravity to pixels
        dt=1.0/fps if fps > 0 else 1/30.0
    )
    
    # For visualization
    trajectory = []
    
    # Process video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get timestamp
        t = frame_count / fps
        
        # Apply super-resolution (only if needed)
        if frame_count % 5 == 0:  # Process every 5th frame with super-resolution
            enhanced_frame = sr_model.enhance(frame)
        else:
            enhanced_frame = frame.copy()
        
        # Detect ball
        detections = detector.detect(enhanced_frame)
        
        # Convert detections to format expected by DeepSORT
        dets_for_tracker = []
        for (x, y, r, conf) in detections:
            # Convert circle to bbox [x1, y1, x2, y2, conf]
            x1, y1 = int(x - r), int(y - r)
            x2, y2 = int(x + r), int(y + r)
            dets_for_tracker.append([x1, y1, x2, y2, conf])
        
        # Update tracker
        tracked_objs = tracker.update(dets_for_tracker, enhanced_frame)
        
        # Add to trajectory refiner and get current trajectory
        current_pos = None
        for obj in tracked_objs:
            x1, y1, x2, y2 = obj.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            trajectory_refiner.add_point(cx, cy, t, obj.confidence)
            current_pos = (int(cx), int(cy))
        
        # Refine trajectory periodically
        if frame_count % 5 == 0 and len(trajectory_refiner.trajectory) > 5:
            refined_trajectory = trajectory_refiner.refine()
            if refined_trajectory:
                trajectory = [(int(p.x), int(p.y)) for p in refined_trajectory]
        
        # Draw trajectory
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
        
        # Draw current position
        if current_pos:
            cv2.circle(frame, current_pos, 5, (0, 255, 0), -1)
        
        # Draw frame count
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Show progress
        frame_count += 1
        if frame_count % 10 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Test completed. Output saved to: {output_video}")

if __name__ == "__main__":
    import yaml
    import torch
    
    # Load config
    config_path = "configs/enhanced.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Input and output paths
    input_video = "test_videos/synthetic_golf_shot.mp4"
    output_video = "outputs/enhanced_tracking_output.mp4"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Run test
    test_enhanced_tracker(input_video, output_video, config)
