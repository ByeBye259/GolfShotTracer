import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.chdir(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_detection.log')
    ]
)
logger = logging.getLogger(__name__)

def process_video(video_path: str, output_dir: Path, max_frames: int = 100):
    """Process video and save frames with detections."""
    from src.vision.yolo_detector import YOLOBallDetector
    
    # Initialize detector with optimized parameters
    model_path = Path("weights/yolov8n-golf.pt")
    detector = YOLOBallDetector(
        model_path=model_path,
        conf_thresh=0.2,    # Lower threshold for more detections
        iou_thresh=0.4      # Slightly more aggressive NMS
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output video writer with MP4V codec
    output_path = output_dir / 'detections.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Create debug directory for frames
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    total_time = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            logger.info("Reached end of video")
            break
            
        logger.info(f"\nProcessing frame {frame_count}")
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = (time.time() - start_time) * 1000  # ms
        total_time += inference_time
        
        logger.info(f"Inference time: {inference_time:.2f}ms")
        logger.info(f"Detected {len(detections)} objects")
        
        # Draw detections with enhanced visualization
        for i, (x, y, r, conf) in enumerate(detections, 1):
            logger.info(f"  Detection {i}: center=({x:.1f}, {y:.1f}), radius={r:.1f}, conf={conf:.3f}")
            
            # Draw circle with alpha blending for better visibility
            overlay = frame.copy()
            cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 0), -1)  # Filled circle
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw circle outline
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 200, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # Draw confidence score with background for better readability
            label = f"{conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, 
                        (int(x) + 5, int(y) - label_h - 5), 
                        (int(x) + label_w + 10, int(y) + 5), 
                        (0, 0, 0), -1)
            cv2.putText(frame, label, (int(x) + 10, int(y) - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add status bar at the top
        status_bar = np.zeros((40, frame.shape[1], 3), dtype=np.uint8)
        cv2.rectangle(status_bar, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
        
        # Add FPS counter
        fps_text = f"FPS: {1000/max(inference_time, 1):.1f}"
        cv2.putText(status_bar, fps_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame counter
        frame_text = f"Frame: {frame_count}"
        cv2.putText(status_bar, frame_text, (200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(status_bar, det_text, (400, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add status bar to frame
        frame[0:40, 0:frame.shape[1]] = status_bar
        
        # Save frame
        out.write(frame)
        
        # Save frame as image for debugging
        if frame_count % 5 == 0:  # Save every 5th frame
            img_path = frames_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(img_path), frame)
            
            # Save detection details to text file
            with open(frames_dir / f"detections_{frame_count:04d}.txt", 'w') as f:
                f.write(f"Frame: {frame_count}\n")
                f.write(f"Detections: {len(detections)}\n\n")
                for i, (x, y, r, conf) in enumerate(detections, 1):
                    f.write(f"Detection {i}:\n")
                    f.write(f"  Center: ({x:.1f}, {y:.1f})\n")
                    f.write(f"  Radius: {r:.1f}px\n")
                    f.write(f"  Confidence: {conf:.3f}\n\n")
        
        frame_count += 1
    
    # Calculate and log average FPS
    avg_fps = 1000 / (total_time / frame_count) if frame_count > 0 else 0
    logger.info(f"\nAverage FPS: {avg_fps:.2f}")
    logger.info(f"Total frames processed: {frame_count}")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"Debugging completed. Results saved to: {output_dir}")

def main():
    # Paths
    video_path = r"C:\Users\Foso\Documents\input.mp4"
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process video
    process_video(video_path, output_dir, max_frames=100)

if __name__ == "__main__":
    try:
        main()
        print("Debugging complete. Check debug_output/ for results.")
    except Exception as e:
        logger.exception("An error occurred during debugging")
        print(f"Error: {e}. Check debug_detection.log for details.")
