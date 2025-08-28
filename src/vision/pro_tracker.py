import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

class ProGolfTracker:
    def __init__(self, config):
        import time  # Add time import at the top of the file
        
        # Background subtraction with improved settings
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,          # Reduced history for faster adaptation
            varThreshold=32,      # Increased threshold to reduce noise
            detectShadows=False
        )
        
        # Frame counter for periodic background model reset
        self.frame_count = 0
        self.bg_reset_interval = 150  # Reset background every 150 frames
        
        # Kalman filter for trajectory prediction
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]], np.float32)
        self.kf.transitionMatrix = np.array([
            [1,0,1,0,0,0],
            [0,1,0,1,0,0],
            [0,0,1,0,1,0],
            [0,0,0,1,0,1],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]], np.float32)
        
        self.tracked_objects = {}
        self.next_id = 0
        self.max_disappeared = 5
        
        # Golf ball specific parameters
        self.min_radius = 2
        self.max_radius = 50
        self.min_circularity = 0.6
        self.ball_color_lower = np.array([0, 0, 200], dtype="uint8")
        self.ball_color_upper = np.array([180, 50, 255], dtype="uint8")
        
    def preprocess_frame(self, frame):
        # Reset background model periodically
        self.frame_count += 1
        if self.frame_count % self.bg_reset_interval == 0:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, 
                varThreshold=32,
                detectShadows=False
            )
        
        # Convert to LAB color space for better lighting invariance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE for better contrast in the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge back and convert to BGR
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply color thresholding for white/yellow golf balls in HSV
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        
        # Apply background subtraction with learning rate
        fg_mask = self.bg_subtractor.apply(enhanced_bgr, learningRate=0.01)
        
        # Combine masks with bitwise AND
        combined_mask = cv2.bitwise_and(color_mask, fg_mask)
        
        # Advanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove small noise
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes in the foreground
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small remaining noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_ERODE, kernel, iterations=1)
        
        return cleaned
    
    def detect_balls(self, frame, mask):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        for contour in contours:
            # Skip small contours
            area = cv2.contourArea(contour)
            if area < 5 or area > 1000:  # More strict area filtering
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Filter by aspect ratio (should be close to 1 for circles)
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
                
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)
            
            # More strict radius filtering based on frame size
            min_radius = max(1, int(frame.shape[1] * 0.002))  # 0.2% of frame width
            max_radius = min(100, int(frame.shape[1] * 0.05))  # 5% of frame width
            
            if not (min_radius <= radius <= max_radius):
                continue
                
            # Check if the region is too bright (likely reflection)
            roi = frame[max(0,y-radius):min(frame.shape[0],y+radius), 
                       max(0,x-radius):min(frame.shape[1],x+radius)]
            if roi.size == 0:
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:  # Increased from 0.6 for more circular shapes
                continue
                
            # Check solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
                
            solidity = float(area) / hull_area
            if solidity < 0.8:  # Require more solid shapes
                continue
                
            # If we have a previous frame, check for motion
            if hasattr(self, 'prev_frame'):
                # Simple motion check (can be enhanced with optical flow)
                prev_roi = self.prev_frame[max(0,y-radius):min(self.prev_frame.shape[0],y+radius), 
                                         max(0,x-radius):min(self.prev_frame.shape[1],x+radius)]
                if prev_roi.size > 0 and roi.size > 0 and prev_roi.shape == roi.shape:
                    diff = cv2.absdiff(roi, prev_roi)
                    if np.mean(diff) < 10:  # Not enough motion
                        continue
            
            ball_detections.append((x, y, radius))
            
        # Store current frame for next iteration
        self.prev_frame = frame.copy()
            
        return ball_detections
    
    def update_tracks(self, detections):
        # Update existing tracks with new detections
        if len(self.tracked_objects) == 0:
            for i, (x, y, r) in enumerate(detections):
                # Additional validation for new detections
                if r < 1 or r > 100:  # Skip invalid radii
                    continue
                    
                # Check if this is a duplicate of an existing detection
                is_duplicate = False
                for obj_id, obj in self.tracked_objects.items():
                    ox, oy, orad = obj['predicted']
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                    if distance < max(r, orad) * 1.5:  # Overlapping detections
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    self.tracked_objects[self.next_id] = {
                        'positions': deque(maxlen=30),
                        'predicted': (x, y, r),
                        'disappeared': 0,
                        'color': tuple(np.random.randint(0, 255, 3).tolist()),
                        'age': 0,  # Track how long this object has been tracked
                        'total_visible': 1,  # Total frames this object was detected
                        'consecutive_invisible': 0  # Frames since last detection
                    }
                    self.tracked_objects[self.next_id]['positions'].append((x, y))
                    self.next_id += 1
            return self.tracked_objects
            
        # Calculate cost matrix between existing tracks and new detections
        track_ids = list(self.tracked_objects.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            for j, (x, y, r) in enumerate(detections):
                # Simple distance-based cost
                last_pos = self.tracked_objects[track_id]['positions'][-1]
                cost_matrix[i, j] = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
                
        # Use Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Update matched tracks
        used_detections = set()
        for i, j in zip(track_indices, det_indices):
            if cost_matrix[i, j] < 50:  # Maximum allowed distance
                track_id = track_ids[i]
                x, y, r = detections[j]
                self.tracked_objects[track_id]['positions'].append((x, y))
                self.tracked_objects[track_id]['predicted'] = (x, y, r)
                self.tracked_objects[track_id]['disappeared'] = 0
                used_detections.add(j)
                
        # Handle unmatched detections (new tracks)
        for j in range(len(detections)):
            if j not in used_detections:
                x, y, r = detections[j]
                self.tracked_objects[self.next_id] = {
                    'positions': deque(maxlen=30),
                    'predicted': (x, y, r),
                    'disappeared': 0,
                    'color': tuple(np.random.randint(0, 255, 3).tolist())
                }
                self.tracked_objects[self.next_id]['positions'].append((x, y))
                self.next_id += 1
                
        # Handle disappeared tracks
        for track_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[track_id]['disappeared'] >= self.max_disappeared:
                del self.tracked_objects[track_id]
            else:
                self.tracked_objects[track_id]['disappeared'] += 1
                
        return self.tracked_objects
    
    def process_frame(self, frame):
        # Preprocess frame
        mask = self.preprocess_frame(frame)
        
        # Detect potential golf balls
        detections = self.detect_balls(frame, mask)
        
        # Update tracks
        tracks = self.update_tracks(detections)
        
        # Draw tracks and update track statistics
        output = frame.copy()
        debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw all detections in blue
        for (x, y, r) in detections:
            cv2.circle(debug_mask, (x, y), r, (255, 0, 0), 1)
        
        # Draw tracked objects
        for track_id, track in list(tracks.items()):
            # Update track age and visibility
            track['age'] += 1
            
            if len(track['positions']) > 1:
                # Only draw if we have enough history
                points = np.array(track['positions'], np.int32)
                
                # Calculate speed (pixels/frame)
                if len(track['positions']) > 2:
                    dx = points[-1][0] - points[-2][0]
                    dy = points[-1][1] - points[-2][1]
                    speed = np.sqrt(dx*dx + dy*dy)
                    
                    # Filter out slow-moving or stationary objects
                    if speed < 0.5:  # Minimum speed threshold
                        track['consecutive_invisible'] += 1
                    else:
                        track['consecutive_invisible'] = max(0, track['consecutive_invisible'] - 1)
                
                # Draw trajectory
                cv2.polylines(output, [points], False, track['color'], 2)
                
                # Draw current position
                x, y, r = track['predicted']
                cv2.circle(output, (x, y), r, track['color'], 2)
                
                # Draw ID and speed
                info = f"{track_id}"
                if 'speed' in locals():
                    info += f" {speed:.1f}px/f"
                cv2.putText(output, info, (x - 10, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, track['color'], 1)
        
        # Remove stale tracks
        max_invisible_frames = 10  # Maximum frames a track can be invisible before removal
        tracks = {k: v for k, v in tracks.items() 
                 if v['consecutive_invisible'] < max_invisible_frames}
        
        # Resize debug mask to fit in corner
        debug_mask = cv2.resize(debug_mask, (320, 180))
        output[10:190, 10:330] = debug_mask  # Top-left corner
        
        # Add FPS counter
        if hasattr(self, 'prev_time'):
            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time)
            cv2.putText(output, f"FPS: {fps:.1f}", (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.prev_time = time.time()
                          
        return output, tracks
