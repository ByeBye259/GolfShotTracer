import cv2
import numpy as np
from pathlib import Path
import sys
from ultralytics import YOLO

class GolferStanceAnalyzer:
    def __init__(self, model_path='yolov8x-pose.pt'):  # Using larger model for better accuracy
        """Initialize the golfer stance analyzer with YOLO model."""
        try:
            self.model = YOLO(model_path)
            self.keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            self.model = None
    
    def analyze_first_frame(self, video_path):
        """Analyze the first frame of the video to determine golfer's position and stance."""
        if not self.model:
            print("YOLO model not available. Cannot analyze stance.")
            return None
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Read the first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read the first frame")
            return None
        
        # Resize frame if it's too large (for faster processing)
        height, width = frame.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # Run pose estimation with increased confidence threshold
        results = self.model(frame, conf=0.4, verbose=False)
        
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            print("No keypoints detected in the frame. Trying with a lower confidence threshold...")
            # Try again with lower confidence threshold
            results = self.model(frame, conf=0.2, verbose=False)
            
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            print("Still no keypoints detected. The model might be having trouble detecting the golfer.")
            print("Troubleshooting tips:")
            print("1. Ensure the golfer is clearly visible in the first frame")
            print("2. Try with a different video where the golfer is more centered")
            print("3. The lighting conditions might be affecting detection")
            return None
        
        # Get keypoints for the first detected person
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Print confidence scores for debugging
        print("\nKeypoint detection confidence:")
        for i, (name, kp) in enumerate(zip(self.keypoint_names, keypoints)):
            if kp[0] > 0 and kp[1] > 0:  # Only show detected keypoints
                print(f"- {name}: ({kp[0]:.1f}, {kp[1]:.1f})")
        
        # Calculate important points for stance analysis
        analysis = {
            'shoulder_center': self._get_shoulder_center(keypoints),
            'hip_center': self._get_hip_center(keypoints),
            'feet_position': self._get_feet_position(keypoints),
            'shoulder_angle': self._calculate_shoulder_angle(keypoints),
            'is_right_handed': self._is_right_handed(keypoints),
            'frame': frame,
            'keypoints': keypoints
        }
        
        return analysis
    
    def _get_shoulder_center(self, keypoints):
        """Calculate the center point between shoulders."""
        left_shoulder = keypoints[5]  # COCO keypoint index for left shoulder
        right_shoulder = keypoints[6]  # COCO keypoint index for right shoulder
        return (left_shoulder + right_shoulder) / 2
    
    def _get_hip_center(self, keypoints):
        """Calculate the center point between hips."""
        left_hip = keypoints[11]  # COCO keypoint index for left hip
        right_hip = keypoints[12]  # COCO keypoint index for right hip
        return (left_hip + right_hip) / 2
    
    def _get_feet_position(self, keypoints):
        """Get the position of both feet."""
        left_ankle = keypoints[15]  # COCO keypoint index for left ankle
        right_ankle = keypoints[16]  # COCO keypoint index for right ankle
        return {
            'left': left_ankle,
            'right': right_ankle,
            'width': abs(left_ankle[0] - right_ankle[0]),
            'stance': 'open' if left_ankle[0] < right_ankle[0] else 'closed' if left_ankle[0] > right_ankle[0] else 'square'
        }
    
    def _calculate_shoulder_angle(self, keypoints):
        """Calculate the angle of the shoulder line relative to horizontal."""
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        return np.degrees(np.arctan2(dy, dx))
    
    def _is_right_handed(self, keypoints):
        """Determine if the golfer is right-handed based on shoulder position."""
        # For a right-handed golfer, the right shoulder will be further from the camera
        # This is a simplification and might need adjustment based on camera angle
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        return right_shoulder[0] > left_shoulder[0]
    
    def visualize_analysis(self, analysis, output_path=None):
        """Visualize the golfer's stance with keypoints and analysis."""
        if not analysis:
            return None
        
        frame = analysis['frame'].copy()
        keypoints = analysis['keypoints']
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:  # Only draw if keypoint is detected
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x), int(y-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw shoulder line
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        cv2.line(frame, tuple(left_shoulder.astype(int)), 
                tuple(right_shoulder.astype(int)), (255, 0, 0), 2)
        
        # Draw hip line
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        cv2.line(frame, tuple(left_hip.astype(int)), 
                tuple(right_hip.astype(int)), (0, 255, 0), 2)
        
        # Draw feet
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        cv2.circle(frame, tuple(left_ankle.astype(int)), 8, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_ankle.astype(int)), 8, (0, 0, 255), -1)
        
        # Add analysis text
        text_y = 30
        cv2.putText(frame, f"Handedness: {'Right' if analysis['is_right_handed'] else 'Left'}-handed", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 30
        cv2.putText(frame, f"Shoulder Angle: {analysis['shoulder_angle']:.1f}°", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 30
        cv2.putText(frame, f"Stance: {analysis['feet_position']['stance'].capitalize()}", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save or return the visualization
        if output_path:
            cv2.imwrite(str(output_path), frame)
            print(f"Analysis visualization saved to {output_path}")
        
        return frame

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_golfer_stance.py <video_path> [output_image_path]")
        return
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/stance_analysis.jpg"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Analyze the golfer's stance
    analyzer = GolferStanceAnalyzer()
    analysis = analyzer.analyze_first_frame(video_path)
    
    if analysis:
        print("\nGolfer Stance Analysis:")
        print(f"- Handedness: {'Right' if analysis['is_right_handed'] else 'Left'}-handed")
        print(f"- Shoulder Angle: {analysis['shoulder_angle']:.1f}°")
        print(f"- Stance: {analysis['feet_position']['stance'].capitalize()}")
        print(f"- Feet Width: {analysis['feet_position']['width']:.1f} pixels")
        
        # Visualize and save the analysis
        analyzer.visualize_analysis(analysis, output_path)
        print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    main()
