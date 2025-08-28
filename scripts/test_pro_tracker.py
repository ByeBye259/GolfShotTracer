import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.pro_tracker import ProGolfTracker

def process_video(input_path, output_path, config_path, max_frames=None):
    # Initialize tracker
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tracker = ProGolfTracker(config)
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with tracker
            output, tracks = tracker.process_frame(frame)
            
            # Add debug info
            debug_text = [
                f"Frame: {frame_count}",
                f"Tracks: {len(tracks)}",
                ""
            ]
            
            for i, (track_id, track) in enumerate(tracks.items(), 1):
                x, y, r = track['predicted']
                debug_text.append(f"Track {track_id}: ({x}, {y}) r={r}")
                
                # Draw track history
                if len(track['positions']) > 1:
                    points = np.array(track['positions'], np.int32)
                    cv2.polylines(debug_overlay, [points], False, (0, 255, 0), 1)
            
            # Create debug overlay
            debug_overlay = np.zeros_like(frame, dtype=np.uint8)
            for i, text in enumerate(debug_text):
                y_pos = 30 + i * 25
                cv2.putText(debug_overlay, text, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Combine debug overlay with output
            debug_output = cv2.addWeighted(output, 0.7, debug_overlay, 0.3, 0)
            
            # Write frame to output video
            out.write(debug_output)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\nProcessing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video with professional golf ball tracker')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_pro_tracker.mp4',
                      help='Path to output video')
    parser.add_argument('--config', type=str, default='configs/detection_config.yaml',
                      help='Path to config file')
    parser.add_argument('--max_frames', type=int, default=None,
                      help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.config, args.max_frames)

if __name__ == "__main__":
    main()
