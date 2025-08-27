import os
import sys
import cv2
import numpy as np
import time
import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.roboflow_detector import RoboflowBallDetector

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

class DetectionVisualizer:
    def __init__(self, max_history: int = 30):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('Golf Ball Detection Debug', fontsize=16)
        
        # Detection visualization
        self.img_plot = self.ax1.imshow(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.ax1.set_title('Detections')
        
        # Performance metrics
        self.times = deque(maxlen=max_history)
        self.frame_nums = deque(maxlen=max_history)
        self.detection_counts = deque(maxlen=max_history)
        
        # Initialize performance plot
        self.ax2.set_title('Performance Metrics')
        self.ax2.set_xlabel('Frame')
        self.ax2.set_ylabel('Inference Time (ms)', color='tab:blue')
        self.ax2.set_ylim(0, 500)  # 500ms upper limit
        
        self.perf_ax2 = self.ax2.twinx()
        self.perf_ax2.set_ylabel('Detection Count', color='tab:red')
        self.perf_ax2.set_ylim(0, 10)  # Max 10 detections
        
        self.time_line, = self.ax2.plot([], [], 'b-', label='Inference Time')
        self.det_line, = self.perf_ax2.plot([], [], 'r-', label='Detections')
        
        # Add legend
        lines1, labels1 = self.ax2.get_legend_handles_labels()
        lines2, labels2 = self.perf_ax2.get_legend_handles_labels()
        self.ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
    
    def update_plot(self, 
                   frame: np.ndarray, 
                   detections: List[Tuple[float, float, float, float]],
                   inference_time: float,
                   frame_num: int):
        """Update the visualization with new detection results."""
        # Update detection visualization
        vis_frame = frame.copy()
        
        # Draw detections
        for i, (x, y, r, conf) in enumerate(detections, 1):
            # Draw circle
            cv2.circle(vis_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # Draw confidence score
            label = f"{conf:.2f}"
            cv2.putText(vis_frame, label, (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Update image
        self.img_plot.set_array(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        
        # Update performance metrics
        self.times.append(inference_time)
        self.frame_nums.append(frame_num)
        self.detection_counts.append(len(detections))
        
        # Update performance plot
        self.time_line.set_data(self.frame_nums, self.times)
        self.det_line.set_data(self.frame_nums, self.detection_counts)
        
        # Adjust axes
        if len(self.frame_nums) > 1:
            self.ax2.set_xlim(min(self.frame_nums), max(self.frame_nums))
            self.perf_ax2.set_xlim(min(self.frame_nums), max(self.frame_nums))
        
        return self.img_plot, self.time_line, self.det_line

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Debug golf ball detection with Roboflow')
    parser.add_argument('--video', type=str, 
                      default=r'C:\Users\Foso\Downloads\input (online-video-cutter.com).mp4',
                      help='Path to input video file')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory for debug frames')
    parser.add_argument('--max-frames', type=int, default=100,
                      help='Maximum number of frames to process')
    parser.add_argument('--api-key', type=str, required=True,
                      help='Roboflow API key')
    parser.add_argument('--conf', type=float, default=0.2,
                      help='Confidence threshold (0-1)')
    args = parser.parse_args()

    # Initialize Roboflow detector
    detector = RoboflowBallDetector(
        api_key=args.api_key,
        model_name="golf-ball-detection-hii2e",
        conf_thresh=args.conf
    )

    # Process video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Could not open video: {args.video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = Path(args.output) / 'detections.mp4'
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output) / f"debug_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments
    with open(output_dir / 'config.txt', 'w') as f:
        f.write(f"Video: {args.video}\n")
        f.write(f"Model: golf-ball-detection-hii2e\n")
        f.write(f"Confidence Threshold: {args.conf}\n")
        f.write(f"Max Frames: {args.max_frames}\n")

    # Initialize visualizer
    visualizer = DetectionVisualizer()

    # Process frames
    frame_count = 0
    start_time = time.time()

    while frame_count < min(args.max_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for faster processing (process every 2nd frame)
        if frame_count % 2 != 0:
            frame_count += 1
            continue

        frame_count += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing frame {frame_count}")

        try:
            # Detect balls
            detections = detector.detect(frame)

            # Visualize detections with more details
            vis = frame.copy()

            # Draw detections
            for x, y, r, conf in detections:
                # Draw circle
                cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)

                # Draw center point
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

                # Draw confidence with background for better visibility
                label = f"{conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(vis, 
                           (int(x) - 5, int(y) - label_h - 10), 
                           (int(x) + label_w + 5, int(y) - 5), 
                           (0, 0, 0), -1)
                cv2.putText(vis, label, (int(x), int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add status bar
            status_bar = np.zeros((40, vis.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(status_bar, (0, 0), (vis.shape[1], 40), (50, 50, 50), -1)

            # Add frame info
            fps_text = f"Frame: {frame_count}/{min(args.max_frames, total_frames)}"
            det_text = f"Detections: {len(detections)}"
            conf_text = f"Confidence: {args.conf:.2f}"

            cv2.putText(status_bar, fps_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_bar, det_text, (200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_bar, conf_text, (400, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Add status bar to frame
            vis[0:40, 0:vis.shape[1]] = status_bar

            # Save frame to video
            out.write(vis)

            # Save frame as image every 10 frames
            if frame_count % 10 == 0:
                frame_path = output_dir / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_path), vis)

                # Save detection details
                with open(output_dir / f"detections_{frame_count:04d}.txt", 'w') as f:
                    f.write(f"Frame: {frame_count}\n")
                    f.write(f"Detections: {len(detections)}\n\n")
                    for i, (x, y, r, conf) in enumerate(detections, 1):
                        f.write(f"Detection {i}:\n")
                        f.write(f"  Position: ({x:.1f}, {y:.1f})\n")
                        f.write(f"  Radius: {r:.1f}px\n")
                        f.write(f"  Confidence: {conf:.3f}\n\n")

        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")
            vis = frame.copy()

        # Update image
        visualizer.img_plot.set_array(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        # Update performance metrics
        visualizer.times.append(0)  # Inference time not measured
        visualizer.frame_nums.append(frame_count)
        visualizer.detection_counts.append(len(detections))

        # Update performance plot
        visualizer.time_line.set_data(visualizer.frame_nums, visualizer.times)
        visualizer.det_line.set_data(visualizer.frame_nums, visualizer.detection_counts)

        # Adjust axes
        if len(visualizer.frame_nums) > 1:
            visualizer.ax2.set_xlim(min(visualizer.frame_nums), max(visualizer.frame_nums))
            visualizer.perf_ax2.set_xlim(min(visualizer.frame_nums), max(visualizer.frame_nums))

        if frame_count >= 30:
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate and log performance
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    logger.info(f"\n--- Processing Complete ---")
    logger.info(f"Total frames processed: {frame_count}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Output saved to: {output_dir.absolute()}")
    logger.info("Debugging session completed")
    
    cap.release()
    print("\nDebugging complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
