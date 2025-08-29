import cv2
import numpy as np
from pathlib import Path
import sys
import time
from ultralytics import YOLO
import torch
from tqdm import tqdm

class EnhancedBallDetector:
    def __init__(self, model_path='weights/yolov8n-golf.pt', pose_model='yolov8x-pose.pt'):
        """Initialize the enhanced ball detector with YOLO models."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load ball detection model
        self.ball_model = YOLO(model_path).to(self.device)
        
        # Load pose estimation model for golfer analysis
        self.pose_model = YOLO(pose_model).to(self.device)
        
        # Detection parameters
        self.ball_conf_thresh = 0.1  # Lower threshold for better detection
        self.iou_thresh = 0.4
        self.min_ball_radius = 1     # More sensitive to small balls
        self.max_ball_radius = 50
        
        # Tracking and prediction
        self.last_ball_pos = None
        self.last_ball_vel = None
        self.ball_track = []
        self.occlusion_frames = 0
        self.max_occlusion_frames = 3  # Max frames to predict during occlusion
        
        # Region of interest (ROI) parameters
        self.roi_padding = 0.2  # Padding around golfer's stance
        
    def analyze_golfer_stance(self, frame):
        """Analyze golfer's stance to determine ROI for ball detection."""
        # Run pose estimation
        results = self.pose_model(frame, verbose=False, conf=0.4)
        
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            print("Could not detect golfer's stance. Using full frame.")
            return None
        
        # Get keypoints for the first detected person
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Get bounding box around all keypoints
        valid_kps = [kp for kp in keypoints if kp[0] > 0 and kp[1] > 0]
        if not valid_kps:
            return None
            
        kps_array = np.array(valid_kps)
        x_min, y_min = np.min(kps_array, axis=0)
        x_max, y_max = np.max(kps_array, axis=0)
        
        # Get feet positions (ankles)
        left_ankle = keypoints[15]  # COCO keypoint index for left ankle
        right_ankle = keypoints[16]  # COCO keypoint index for right ankle
        
        # Calculate ROI in front of the golfer
        if left_ankle[0] > 0 and right_ankle[0] > 0:
            # Calculate center between feet
            feet_center = (left_ankle + right_ankle) / 2
            
            # Calculate width between feet
            feet_width = abs(left_ankle[0] - right_ankle[0])
            
            # Determine ball position based on stance (right-handed golfer)
            ball_x = feet_center[0]
            ball_y = min(left_ankle[1], right_ankle[1])  # Higher (lower y) of the two feet
            
            # Adjust ball position based on stance (slightly forward and right for right-handed)
            ball_x += feet_width * 0.5
            ball_y -= feet_width * 0.3
            
            # Create ROI around estimated ball position
            roi_size = feet_width * 1.5
            x1 = max(0, int(ball_x - roi_size/2))
            y1 = max(0, int(ball_y - roi_size/2))
            x2 = min(frame.shape[1], int(ball_x + roi_size/2))
            y2 = min(frame.shape[0], int(ball_y + roi_size/2))
            
            return {
                'roi': (x1, y1, x2, y2),
                'ball_estimate': (int(ball_x), int(ball_y)),
                'confidence': 0.8  # Confidence in ROI estimation
            }
        
        return None
    
    def predict_ball_position(self):
        """Predict ball position based on previous motion."""
        if not self.last_ball_pos or not self.last_ball_vel:
            return None
            
        # Simple linear prediction
        predicted_x = self.last_ball_pos[0] + self.last_ball_vel[0]
        predicted_y = self.last_ball_pos[1] + self.last_ball_vel[1]
        
        return (int(predicted_x), int(predicted_y))
    
    def update_ball_tracking(self, ball_pos):
        """Update ball tracking with new position and calculate velocity."""
        if ball_pos:
            current_pos = (ball_pos['x'], ball_pos['y'])
            
            # Update velocity if we have a previous position
            if self.last_ball_pos:
                dx = current_pos[0] - self.last_ball_pos[0]
                dy = current_pos[1] - self.last_ball_pos[1]
                self.last_ball_vel = (dx, dy)
            
            self.last_ball_pos = current_pos
            self.ball_track.append(current_pos)
            self.occlusion_frames = 0  # Reset occlusion counter
            
            # Keep only recent positions for tracking
            if len(self.ball_track) > 5:
                self.ball_track.pop(0)
        else:
            self.occlusion_frames += 1
    
    def detect_ball(self, frame, roi=None):
        """Detect golf ball in the frame, optionally within an ROI."""
        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2]
            if roi_frame.size == 0:
                roi_frame = frame  # Fallback to full frame if ROI is invalid
        else:
            roi_frame = frame
            x1, y1 = 0, 0
        
        # Run ball detection with increased confidence for the ball class
        results = self.ball_model(
            roi_frame,
            conf=self.ball_conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
            classes=[0]  # Focus on ball class if model has multiple classes
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                conf = box.conf[0].item()
                
                # Adjust coordinates if using ROI
                x += x1
                y += y1
                
                # Filter by size with more lenient constraints
                radius = max(w, h) / 2
                if not (self.min_ball_radius <= radius <= self.max_ball_radius):
                    continue
                
                # If we have a previous position, prioritize detections near it
                if self.last_ball_pos:
                    dist = np.sqrt((x - self.last_ball_pos[0])**2 + 
                                 (y - self.last_ball_pos[1])**2)
                    # Give a small boost to detections near the last known position
                    if dist < 50:  # Within 50 pixels
                        conf *= 1.2
                
                detections.append({
                    'x': x,
                    'y': y,
                    'radius': radius,
                    'confidence': conf,
                    'bbox': [x - w/2, y - h/2, x + w/2, y + h/2]
                })
        
        return detections
    
    def process_video(self, input_path, output_path, max_frames=None):
        """Process video with enhanced ball detection."""
        # Open video
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
        
        frame_count = 0
        start_time = time.time()
        pbar = tqdm(total=min(max_frames, total_frames) if max_frames else total_frames)
        
        # Read first frame to analyze golfer's stance
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame")
            return
        
        # Analyze golfer's stance from first frame
        roi_info = self.analyze_golfer_stance(first_frame)
        
        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while cap.isOpened() and (max_frames is None or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect ball using ROI if available
            if roi_info and frame_count < int(fps * 2):  # Use ROI for first 2 seconds
                detections = self.detect_ball(frame, roi_info['roi'])
                if not detections:  # Fallback to full frame if no detections in ROI
                    detections = self.detect_ball(frame)
            else:
                detections = self.detect_ball(frame)
            
            # Get the best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x['confidence']) if detections else None
            
            # Handle occlusion using tracking
            if best_detection:
                self.update_ball_tracking(best_detection)
            elif self.occlusion_frames < self.max_occlusion_frames and self.last_ball_pos:
                # Predict ball position during occlusion
                predicted_pos = self.predict_ball_position()
                if predicted_pos:
                    best_detection = {
                        'x': predicted_pos[0],
                        'y': predicted_pos[1],
                        'radius': 5,  # Default radius for predicted position
                        'confidence': 0.8,  # Slightly lower confidence for predictions
                        'predicted': True
                    }
            
            # Draw detections
            frame_with_detections = frame.copy()
            
            # Draw ROI if available
            if roi_info and frame_count < int(fps * 2):
                x1, y1, x2, y2 = roi_info['roi']
                cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.circle(frame_with_detections, roi_info['ball_estimate'], 3, (0, 255, 255), -1)
            
            # Draw ball detections
            if best_detection:
                x, y, r = int(best_detection['x']), int(best_detection['y']), int(best_detection['radius'])
                color = (0, 255, 0)  # Green for detected, blue for predicted
                if best_detection.get('predicted', False):
                    color = (255, 165, 0)  # Orange for predicted positions
                    cv2.circle(frame_with_detections, (x, y), r, color, 1, lineType=cv2.LINE_AA)
                    cv2.putText(frame_with_detections, f"Predicted: {best_detection['confidence']:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.circle(frame_with_detections, (x, y), r, color, 2)
                    cv2.putText(frame_with_detections, f"Ball: {best_detection['confidence']:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball track
            for i in range(1, len(self.ball_track)):
                cv2.line(frame_with_detections, 
                        (int(self.ball_track[i-1][0]), int(self.ball_track[i-1][1])),
                        (int(self.ball_track[i][0]), int(self.ball_track[i][1])),
                        (0, 255, 255), 1)
            
            # Draw tracking info
            if self.last_ball_pos and self.last_ball_vel:
                vel_text = f"Vel: ({self.last_ball_vel[0]:.1f}, {self.last_ball_vel[1]:.1f})"
                cv2.putText(frame_with_detections, vel_text, (20, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Add FPS counter
            fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
            cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write frame
            out.write(frame_with_detections)
            
            frame_count += 1
            pbar.update(1)
        
        # Clean up
        pbar.close()
        cap.release()
        out.release()
        
        print(f"\nProcessing complete. Output saved to {output_path}")
        print(f"Processed {frame_count} frames at an average of {frame_count/(time.time() - start_time):.1f} FPS")

def main():
    if len(sys.argv) < 2:
        print("Usage: python enhanced_ball_detector.py <input_video> [output_video] [max_frames]")
        print("Example: python enhanced_ball_detector.py input.mp4 output.mp4 100")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/enhanced_detection.mp4"
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run detector
    detector = EnhancedBallDetector()
    detector.process_video(input_path, output_path, max_frames)

if __name__ == "__main__":
    main()
