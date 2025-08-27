import os
import sys
import cv2
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vision.roboflow_detector import RoboflowBallDetector

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Roboflow ball detection')
    parser.add_argument('--video', type=str, 
                      default=r'C:\Users\Foso\Downloads\input (online-video-cutter.com).mp4',
                      help='Path to input video')
    parser.add_argument('--output', type=str, default='roboflow_output.mp4',
                      help='Output video path')
    parser.add_argument('--api-key', type=str, required=True,
                      help='Roboflow API key')
    parser.add_argument('--conf', type=float, default=0.2,
                      help='Confidence threshold (0-1)')
    args = parser.parse_args()

    # Initialize detector
    detector = RoboflowBallDetector(
        api_key=args.api_key,
        conf_thresh=args.conf
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
    print(f"Processing video: {args.video}")
    print(f"Output will be saved to: {args.output}")
    print("Press Ctrl+C to stop processing...")
    
    frame_count = 0
    
    try:
        # Process video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}", end='\r')

            # Detect balls
            detections = detector.detect(frame)

            # Draw detections
            for (x, y, r, conf) in detections:
                # Draw circle
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                # Draw center
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                # Draw confidence
                cv2.putText(frame, f'{conf:.2f}', (int(x), int(y-10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(frame)
            
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    
    # Release resources
    cap.release()
    out.release()
    print(f"\nProcessing complete. Output saved to {args.output}")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
