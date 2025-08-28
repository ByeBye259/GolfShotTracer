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

from src.vision.combined_detector import CombinedDetector

def preprocess_frame(frame):
    """Apply preprocessing to enhance frame for better detection"""
    # Convert to LAB color space to handle lighting separately from color
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the original a and b channel
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening kernel
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Increase contrast
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 0      # Brightness control (0-100)
    contrast_enhanced = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    
    return contrast_enhanced

def process_video(input_path, output_path, config_path, max_frames=None, output_dir='output_frames', preprocess=True):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize detector
    detector = CombinedDetector(config)
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    # Define the codec and create VideoWriter object
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
            
            # Preprocess frame if enabled
            if preprocess:
                processed_frame = preprocess_frame(frame)
            else:
                processed_frame = frame
            
            # Process frame with detector
            detections = detector.detect(processed_frame)
            
            # Draw detections on original frame
            for x, y, r, conf in detections:
                # Draw circle
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                # Draw center point
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                # Draw confidence
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (int(x) + 10, int(y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Write frame to output video
            out.write(frame)
            
                    # Save frame to output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
                cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    if output_dir:
        print(f"\nFrames saved to {os.path.abspath(output_dir)}")
    
    print(f"\nProcessing complete. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process video with combined YOLO and Roboflow detector')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_combined.mp4',
                       help='Path to output video')
    parser.add_argument('--config', type=str, default='configs/detection_config.yaml',
                       help='Path to config file')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--output_dir', type=str, default='output_frames',
                       help='Directory to save output frames')
    parser.add_argument('--no_preprocess', action='store_true',
                       help='Disable frame preprocessing')
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.config, 
                 args.max_frames, args.output_dir, 
                 preprocess=not args.no_preprocess)

if __name__ == "__main__":
    main()
