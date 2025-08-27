import cv2
import numpy as np
import os
from pathlib import Path

def create_golf_shot_video(output_path, width=1280, height=720, fps=30, duration=5):
    """Generate a synthetic golf shot video for testing."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a green background (golf course)
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:, :] = (50, 120, 50)  # Green color
    
    # Add some noise to make it look more natural
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.uint8)
    background = cv2.add(background, noise)
    
    # Define ball trajectory (parabola)
    def get_ball_position(t, total_frames):
        # Normalized time (0 to 1)
        t_norm = t / total_frames
        
        # Parabolic trajectory
        x = int(width * 0.9 - t_norm * width * 0.8)  # Move from right to left
        y = int(height * 0.8 - (4 * t_norm * (1 - t_norm)) * height * 0.6)  # Parabola
        
        # Add some noise to make it more realistic
        if 0 < t_norm < 0.9:  # Only add noise during flight
            x += np.random.randint(-3, 3)
            y += np.random.randint(-3, 3)
            
        return x, y
    
    # Generate frames
    total_frames = int(fps * duration)
    for t in range(total_frames):
        frame = background.copy()
        
        # Draw ball
        x, y = get_ball_position(t, total_frames)
        if 0 <= x < width and 0 <= y < height:
            # Draw ball (white circle with shadow)
            cv2.circle(frame, (x+2, y+2), 10, (20, 20, 20), -1)  # Shadow
            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)    # Ball
            
            # Add motion blur effect
            if t > 0 and t < total_frames - 1:
                prev_x, prev_y = get_ball_position(t-1, total_frames)
                next_x, next_y = get_ball_position(t+1, total_frames)
                dx = next_x - prev_x
                dy = next_y - prev_y
                for i in range(1, 4):
                    blur_x = x - dx * i // 5
                    blur_y = y - dy * i // 5
                    if 0 <= blur_x < width and 0 <= blur_y < height:
                        cv2.circle(frame, (blur_x, blur_y), 10 - i*2, 
                                 (200, 200, 200), -1)
        
        # Add some grass texture
        for _ in range(20):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(height//2, height)
            length = np.random.randint(5, 15)
            angle = np.random.uniform(-0.5, 0.5)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 80, 0), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {t}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Generated test video: {output_path}")

if __name__ == "__main__":
    # Generate test video
    test_video_path = "test_videos/synthetic_golf_shot.mp4"
    create_golf_shot_video(test_video_path, width=1280, height=720, fps=30, duration=5)
    
    print("Test video generation complete!")
