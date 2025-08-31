import cv2
import numpy as np
import os
import time

def detect_golf_balls(video_path, output_path, max_trail=50, min_radius=1, max_radius=50):
    """
    Detect golf balls in a video and draw a tracer line showing their path.
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path to save output video
        max_trail (int): Maximum length of the tracer trail
        min_radius (int): Minimum ball radius to detect
        max_radius (int): Maximum ball radius to detect
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
    
    # Initialize trail points and colors
    trail_points = []
    trail_colors = [(0, 255, 255), (0, 200, 200), (0, 150, 150), (0, 100, 100), (0, 50, 50)]
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best ball candidate
        best_ball = None
        max_circularity = 0
        
        for contour in contours:
            if cv2.contourArea(contour) < 5:
                continue
                
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            if radius < min_radius or radius > max_radius:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if 0.4 < circularity < 1.2 and circularity > max_circularity:
                max_circularity = circularity
                best_ball = (center, radius, circularity)
        
        # Draw trail and ball if found
        if best_ball:
            center, radius, circularity = best_ball
            trail_points.append(center)
            
            if len(trail_points) > max_trail:
                trail_points.pop(0)
            
            # Draw trail
            for i in range(1, len(trail_points)):
                if trail_points[i-1] is None or trail_points[i] is None:
                    continue
                color_idx = min(i // (max_trail // len(trail_colors)), len(trail_colors)-1)
                cv2.line(frame, trail_points[i-1], trail_points[i], trail_colors[color_idx], 2)
            
            # Draw ball
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), -1)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed {frame_count}/{total_frames} frames ({fps:.1f} FPS)", end='\r')
    
    # Cleanup
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    print(f"\nProcessing complete. Output saved to {output_path}")
    print(f"Processed {frame_count} frames in {elapsed:.1f} seconds ({frame_count/elapsed:.1f} FPS)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Golf Ball Tracer')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', default='outputs/tracer_output.mp4', help='Output video file')
    parser.add_argument('--trail', type=int, default=50, help='Maximum trail length')
    parser.add_argument('--min-radius', type=int, default=1, help='Minimum ball radius')
    parser.add_argument('--max-radius', type=int, default=50, help='Maximum ball radius')
    
    args = parser.parse_args()
    
    detect_golf_balls(
        video_path=args.input,
        output_path=args.output,
        max_trail=args.trail,
        min_radius=args.min_radius,
        max_radius=args.max_radius
    )
