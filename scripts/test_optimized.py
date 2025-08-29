import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.combined_detector import CombinedDetector

def process_video(input_path, output_path, max_frames=None):
    # Load video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize detector with optimized settings
    config = {
        'yolo': {
            'model_path': "weights/yolov8n-golf.pt",
            'conf_threshold': 0.1,  # Lower threshold for better detection
            'iou_threshold': 0.4,
            'img_size': 640,
            'device': 'cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        },
        'tracker': {
            'max_age': 30,
            'n_init': 3,
            'max_iou_distance': 0.7
        },
        'detector': {
            'model_path': "weights/yolov8n-golf.pt",
            'conf_threshold': 0.1,
            'iou_threshold': 0.4,
            'img_size': 640,
            'device': 'cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        }
    }
    
    detector = CombinedDetector(config)
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    pbar = tqdm(total=min(max_frames, total_frames) if max_frames else total_frames)
    
    while cap.isOpened() and (max_frames is None or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame_start_time = time.time()
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            # Handle different detection formats
            if isinstance(det, dict) and 'bbox' in det:
                # Format: {'bbox': [x,y,w,h], 'confidence': float, ...}
                x, y, w, h = map(int, det['bbox'])
                conf = det.get('confidence', 0)
            elif isinstance(det, (list, tuple)) and len(det) >= 4:
                # Format: [x, y, w, h, confidence, ...]
                x, y, w, h = map(int, det[:4])
                conf = det[4] if len(det) > 4 else 0
            else:
                continue
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS for this frame
        fps = 1.0 / (time.time() - frame_start_time)
        # Only add FPS text if we have a valid frame
        if frame is not None and frame.size > 0:
            try:
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except:
                pass  # Skip if we can't add text to the frame
        
        # Write frame
        out.write(frame)
        
        # Show progress in console
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}/{min(max_frames, total_frames) if max_frames else total_frames}")
        
        frame_count += 1
        pbar.update(1)
    
    # Clean up
    pbar.close()
    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass  # Ignore errors when trying to close windows in headless mode
    print(f"\nProcessing complete. Output saved to {output_path}")
    print(f"Processed {frame_count} frames at an average of {frame_count/(time.time() - start_time):.1f} FPS")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process golf swing video with optimized settings')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='outputs/optimized_output.mp4', 
                       help='Output video path')
    parser.add_argument('--max_frames', type=int, default=30, 
                       help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    process_video(
        args.input, 
        str(output_path), 
        max_frames=args.max_frames
    )
