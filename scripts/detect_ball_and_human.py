import cv2
import numpy as np
from pathlib import Path
import sys
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import YOLO detector (assuming it's already in the project)
from src.vision.yolo_detector import YOLOBallDetector

class GolfSceneDetector:
    def __init__(self, ball_model_path='weights/yolov8n-golf.pt', 
                 object_model_path='yolov8s.pt',  # YOLOv8 small model for general object detection
                 conf_thresh=0.3, iou_thresh=0.4):
        """Initialize detectors for golf scene analysis."""
        # Initialize ball detector
        self.ball_detector = YOLOBallDetector(
            model_path=ball_model_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )
        
        # Initialize YOLO for person and club detection
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(object_model_path)
            # COCO class names for reference
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            # We're interested in person and sports equipment that might include golf clubs
            self.target_classes = {
                'person': 0,
                'sports ball': 32,
                'baseball bat': 34,  # Sometimes detects golf clubs as baseball bats
                'tennis racket': 39,  # Sometimes detects golf clubs as tennis rackets
            }
        except ImportError:
            print("Warning: ultralytics not found. Advanced detections will be disabled.")
            self.yolo_model = None
    
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO."""
        if self.yolo_model is None:
            return [], []
            
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        human_detections = []
        club_detections = []
        
        for result in results:
            for box in result.boxes:
                # Get class ID and confidence
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # For person detection
                if cls_id == self.target_classes['person'] and conf > 0.5:
                    human_detections.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': 'person'
                    })
                
                # For golf club detection (might be detected as baseball bat or tennis racket)
                elif (cls_id in [self.target_classes.get('baseball bat', -1), 
                               self.target_classes.get('tennis racket', -1)]) and conf > 0.4:
                    # Filter by aspect ratio (golf clubs are long and thin)
                    w, h = x2 - x1, y2 - y1
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                    
                    if aspect_ratio > 3.0:  # Golf clubs are typically long and thin
                        club_detections.append({
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': 'golf_club',
                            'aspect_ratio': aspect_ratio
                        })
        
        # Sort club detections by confidence and aspect ratio
        club_detections.sort(key=lambda x: (x['confidence'], x['aspect_ratio']), reverse=True)
        
        # If we have multiple club detections, keep only the most confident one near the person
        if len(club_detections) > 1 and human_detections:
            # Get the first person's position
            person_center = [
                human_detections[0]['bbox'][0] + human_detections[0]['bbox'][2]/2,
                human_detections[0]['bbox'][1] + human_detections[0]['bbox'][3]/2
            ]
            
            # Find the club detection closest to the person
            def distance_to_person(club):
                club_center = [
                    club['bbox'][0] + club['bbox'][2]/2,
                    club['bbox'][1] + club['bbox'][3]/2
                ]
                return ((club_center[0] - person_center[0])**2 + 
                        (club_center[1] - person_center[1])**2)**0.5
            
            # Keep only the club closest to the person
            closest_club = min(club_detections, key=distance_to_person)
            club_detections = [closest_club]
        
        return human_detections, club_detections
    
    def detect_balls(self, frame):
        """Detect golf balls in the frame."""
        return self.ball_detector.detect(frame)
    
    def draw_detections(self, frame, ball_detections, human_detections, club_detections):
        """Draw detections on the frame."""
        # Draw ball detections in green
        for det in ball_detections:
            x, y, r, conf = det
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(frame, f"Ball: {conf:.2f}", (int(x), int(y-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw human detections in red
        for det in human_detections:
            x, y, w, h = map(int, det['bbox'])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"Person: {det['confidence']:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw club detections in blue
        for det in club_detections:
            x, y, w, h = map(int, det['bbox'])
            # Draw a thicker rectangle for the club
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            # Draw a line to show the club's orientation
            cv2.line(frame, (x + w//2, y), (x + w//2, y + h), (255, 255, 0), 2)
            cv2.putText(frame, f"Club: {det['confidence']:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame

def process_video(input_path, output_path, max_frames=None):
    """Process video to detect golf balls, humans, and clubs."""
    # Initialize detector
    detector = GolfSceneDetector()
    
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
    
    while cap.isOpened() and (max_frames is None or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect balls, humans, and clubs
        ball_detections = detector.detect_balls(frame)
        human_detections, club_detections = detector.detect_objects(frame)
        
        # Draw all detections
        frame_with_detections = detector.draw_detections(
            frame.copy(), ball_detections, human_detections, club_detections
        )
        
        # Calculate and display FPS
        fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
        cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame_with_detections)
        
        # Show progress
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}/{min(max_frames, total_frames) if max_frames else total_frames}")
        
        frame_count += 1
        pbar.update(1)
    
    # Clean up
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\nProcessing complete. Output saved to {output_path}")
    print(f"Processed {frame_count} frames at an average of {frame_count/(time.time() - start_time):.1f} FPS")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect golf balls, humans, and clubs in a video')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='outputs/detection_output.mp4',
                      help='Output video path')
    parser.add_argument('--max_frames', type=int, default=30,
                      help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    process_video(args.input, str(output_path), args.max_frames)
