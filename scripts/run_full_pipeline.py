import os
import sys
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.yolo_detector import YOLOBallDetector
from src.vision.roboflow_detector import RoboflowBallDetector
from src.vision.deep_sort_tracker import DeepSortTracker
from src.physics.trajectory_refiner import TrajectoryRefiner
from src.utils.logger import get_logger

logger = get_logger("full_pipeline")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_video(input_video, config):
    """Process a video file and generate trajectory."""
    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize detector based on config
    detector_type = config['detector'].get('type', 'yolo').lower()
    
    if detector_type == 'roboflow':
        detector = RoboflowBallDetector(
            api_key=config['detector'].get('api_key', ''),
            model_name=config['detector'].get('model_name', 'golf-ball-detection-hii2e'),
            version=config['detector'].get('version', 2),
            conf_thresh=config['detector'].get('conf_thresh', 0.2)
        )
        logger.info("Using Roboflow detector")
    else:  # Default to YOLO
        detector = YOLOBallDetector(
            model_path=config['detector'].get('model_path', 'weights/yolov8n-golf.pt'),
            conf_thresh=config['detector'].get('conf_thresh', 0.2),
            iou_thresh=config['detector'].get('iou_thresh', 0.4)
        )
        logger.info("Using YOLO detector")
    
    tracker = DeepSortTracker(
        max_age=config['tracker']['max_age'],
        max_iou_distance=config['tracker']['max_iou_distance'],
        max_cosine_distance=config['tracker']['max_cosine_distance'],
        nms_max_overlap=config['tracker']['nms_max_overlap']
    )
    
    # Initialize trajectory refiner with physics parameters
    trajectory_refiner = TrajectoryRefiner(
        g=config['physics']['gravity'] * (frame_height / 100),  # Scale gravity to pixels
        dt=1.0/fps,
        max_missing_frames=config['physics'].get('max_missing_frames', 3),
        min_velocity=config['physics'].get('min_velocity', 1.0),
        max_acceleration=config['physics'].get('max_acceleration', 100.0)
    )
    
    # Initialize video writer
    output_path = output_dir / config['output']['video_output']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    # Process video
    frame_count = 0
    trajectory = []
    
    print(f"Processing video: {input_video}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_count % config['processing']['frame_skip'] != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                vis_frame = frame.copy()
                
                # Debug: Save a frame for inspection
                if frame_count == 0:
                    debug_frame_path = output_dir / 'debug_first_frame.jpg'
                    cv2.imwrite(str(debug_frame_path), frame)
                    print(f"Debug: Saved first frame to {debug_frame_path}")
                
                # Detect balls
                print(f"\nProcessing frame {frame_count}")
                detections = detector.detect(frame)
                print(f"Found {len(detections)} detections")
                
                # Convert detections to tracker format [x1, y1, x2, y2, conf]
                bboxes = []
                confidences = []
                for i, (x, y, r, conf) in enumerate(detections):
                    x1, y1 = int(x - r), int(y - r)
                    x2, y2 = int(x + r), int(y + r)
                    bboxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
                    print(f"  Detection {i+1}: center=({x:.1f}, {y:.1f}), radius={r:.1f}, conf={conf:.2f}")
                
                # Update trajectory
                if bboxes and len(bboxes) > 0:
                    bboxes = np.array(bboxes)
                    confidences = np.array(confidences)
                    tracked_objects = tracker.update(bboxes, confidences, frame)
                    
                    # Update trajectory with detections
                    for obj in tracked_objects:
                        if hasattr(obj, 'bbox') and len(obj.bbox) >= 4:
                            x1, y1, x2, y2 = obj.bbox[:4]  # Ensure we have at least 4 values
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            confidence = getattr(obj, 'confidence', 1.0)
                            
                            # Add point to trajectory refiner
                            trajectory_refiner.add_point(cx, cy, frame_count/fps, confidence)
                            
                            # Draw current detection
                            if config['output']['show_detections']:
                                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                            config['visualization']['ball_color'], 2)
                
                # Get and draw trajectory
                if config['output']['show_trajectory'] and hasattr(trajectory_refiner, 'trajectory'):
                    # Get the current trajectory points
                    trajectory = trajectory_refiner.trajectory
                    
                    if trajectory and len(trajectory) > 1:
                        # Convert trajectory points to pixel coordinates
                        points = []
                        for point in trajectory:
                            if hasattr(point, 'x') and hasattr(point, 'y'):
                                points.append((int(point.x), int(point.y)))
                        
                        # Draw the trajectory
                        if len(points) > 1:
                            for i in range(1, len(points)):
                                if points[i-1] is not None and points[i] is not None:
                                    cv2.line(vis_frame, points[i-1], points[i],
                                            config['visualization']['trajectory_color'],
                                            config['visualization']['line_thickness'])
                
                # Add FPS counter
                if config['output']['show_fps']:
                    cv2.putText(vis_frame, f'Frame: {frame_count}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               config['visualization']['text_color'], 2)
                
                # Write frame to output
                out.write(vis_frame)
                frame_count += 1
                pbar.update(1)
                
                # Check if we've reached max frames
                if 0 < config['processing']['max_frames'] <= frame_count:
                    break
                    
    except Exception as e:
        logger.error(f"Error processing frame {frame_count}: {str(e)}")
    finally:
        # Release resources
        cap.release()
        out.release()
        logger.info(f"Processing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run full shot tracer pipeline with Roboflow detector')
    parser.add_argument('--video', type=str, 
                      default=r'C:\Users\Foso\Downloads\input (online-video-cutter.com).mp4',
                      help='Path to input video')
    parser.add_argument('--config', type=str, 
                      default='configs/full_pipeline_config.yaml',
                      help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Process video
    process_video(args.video, config)

if __name__ == "__main__":
    main()
