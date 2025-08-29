import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path
from .yolo_detector import YOLOBallDetector
from .roboflow_detector import RoboflowBallDetector
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)

class EnhancedGolfBallDetector:
    """Enhanced golf ball detector with advanced preprocessing and multiple detection strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced detector with configuration."""
        self.config = config
        self.detectors = self._initialize_detectors()
        
        # Detection parameters
        self.min_ball_radius = config.get('min_ball_radius', 2)
        self.max_ball_radius = config.get('max_ball_radius', 200)
        self.min_aspect_ratio = config.get('min_aspect_ratio', 0.5)
        self.max_aspect_ratio = config.get('max_aspect_ratio', 2.0)
        self.min_circularity = config.get('min_circularity', 0.6)
        
        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,  # Number of frames to consider for background
            varThreshold=16,  # Higher value = more sensitive to changes
            detectShadows=False
        )
        self.prev_gray = None
        
    def _initialize_detectors(self) -> list:
        """Initialize the base detectors based on config."""
        detectors = []
        
        # Initialize YOLO detector if enabled
        if self.config.get('yolo', {}).get('enabled', True):
            detectors.append(
                YOLOBallDetector(
                    model_path=self.config['yolo'].get('model_path', 'weights/yolov8n-golf.pt'),
                    conf_thresh=self.config['yolo'].get('conf_thresh', 0.15),  # Lower threshold for more detections
                    iou_thresh=self.config['yolo'].get('iou_thresh', 0.4)
                )
            )
            logger.info("YOLO detector initialized")
        
        # Initialize Roboflow detector if enabled
        if self.config.get('roboflow', {}).get('enabled', False):
            detectors.append(
                RoboflowBallDetector(
                    api_key=self.config['roboflow'].get('api_key', ''),
                    model_name=self.config['roboflow'].get('model_name', 'golf-ball-detection-hii2e'),
                    version=self.config['roboflow'].get('version', 2),
                    conf_thresh=self.config['roboflow'].get('conf_thresh', 0.15)  # Lower threshold
                )
            )
            logger.info("Roboflow detector initialized")
            
        if not detectors:
            raise ValueError("At least one detector must be enabled in the config")
            
        return detectors
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing to enhance ball visibility."""
        if frame is None or frame.size == 0:
            return frame
            
        # Convert to LAB color space for better lighting invariance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge enhanced L channel with original a and b channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR color space
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Increase contrast
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 0      # Brightness control (0-100)
        contrast_enhanced = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
        
        return contrast_enhanced
    
    def detect_motion(self, frame: np.ndarray) -> np.ndarray:
        """Detect motion in the frame using background subtraction."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Apply thresholding to get binary mask
        _, motion_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        return motion_mask
    
    def detect_contours(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect potential ball candidates using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 10 or area > 1000:  # Adjust based on expected ball size
                continue
                
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Filter by size
            if not (self.min_ball_radius <= radius <= self.max_ball_radius):
                continue
                
            # Calculate contour properties for filtering
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            
            # Filter by circularity
            if circularity < self.min_circularity:
                continue
                
            # Calculate aspect ratio
            _, (w, h), _ = cv2.minAreaRect(contour)
            if w == 0 or h == 0:
                continue
                
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > self.max_aspect_ratio or aspect_ratio < self.min_aspect_ratio:
                continue
                
            # If we get here, we have a valid ball candidate
            confidence = min(1.0, 0.5 + (circularity * 0.5))  # Higher circularity = higher confidence
            detections.append((x, y, radius, confidence))
            
        return detections
    
    def detect_using_watershed(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect balls using watershed algorithm for better segmentation."""
        if frame is None or frame.size == 0:
            return []
            
        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Threshold the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(frame, markers)
        
        # Process the markers
        detections = []
        for marker in np.unique(markers):
            if marker <= 1:  # Skip background
                continue
                
            # Create a mask for the current marker
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == marker] = 255
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get the bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Filter by size
                if not (self.min_ball_radius <= radius <= self.max_ball_radius):
                    continue
                    
                # Calculate circularity
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < self.min_circularity:
                    continue
                    
                # Add to detections
                confidence = min(1.0, 0.3 + (circularity * 0.7))  # Higher weight to circularity
                detections.append((x, y, radius, confidence))
                
        return detections
    
    def detect_using_hough_circles(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect circles using Hough Circle Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,  # Minimum distance between circles
            param1=50,   # Upper threshold for edge detection
            param2=30,   # Threshold for center detection (lower = more false positives)
            minRadius=self.min_ball_radius,
            maxRadius=self.max_ball_radius
        )
        
        if circles is None:
            return []
            
        circles = np.uint16(np.around(circles))
        return [(x, y, r, 0.7) for x, y, r in circles[0, :]]  # Fixed confidence of 0.7
    
    def combine_detections(self, detections_list: List[List[Tuple[float, float, float, float]]], 
                          iou_threshold: float = 0.5) -> List[Tuple[float, float, float, float]]:
        """Combine detections from multiple methods using weighted boxes fusion."""
        if not detections_list:
            return []
            
        # Flatten all detections
        all_detections = []
        for detections in detections_list:
            all_detections.extend(detections)
            
        if not all_detections:
            return []
            
        # Sort by confidence (highest first)
        all_detections.sort(key=lambda x: x[3], reverse=True)
        
        # Apply Non-Maximum Suppression
        keep = []
        
        while all_detections:
            # Take the detection with highest confidence
            best = all_detections.pop(0)
            keep.append(best)
            
            # Calculate IoU with remaining detections
            to_remove = []
            for i, det in enumerate(all_detections):
                iou = self._calculate_iou(best, det)
                if iou > iou_threshold:
                    to_remove.append(i)
            
            # Remove overlapping detections
            for i in reversed(to_remove):
                all_detections.pop(i)
                
        return keep
    
    def _calculate_iou(self, det1: Tuple[float, float, float, float], 
                      det2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union between two detections."""
        x1, y1, r1, _ = det1
        x2, y2, r2, _ = det2
        
        # Calculate distance between centers
        dx = x1 - x2
        dy = y1 - y2
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Calculate intersection and union areas
        if distance >= r1 + r2:
            return 0.0  # No intersection
            
        if distance <= abs(r1 - r2):
            # One circle is inside the other
            r_min = min(r1, r2)
            intersection = np.pi * r_min * r_min
        else:
            # Calculate intersection area
            r1_sq = r1 * r1
            r2_sq = r2 * r2
            
            d1 = (r1_sq - r2_sq + distance * distance) / (2 * distance)
            d2 = distance - d1
            
            intersection = (r1_sq * np.arccos(d1 / r1) - d1 * np.sqrt(r1_sq - d1 * d1) +
                           r2_sq * np.arccos(d2 / r2) - d2 * np.sqrt(r2_sq - d2 * d2))
        
        # Calculate union
        union = np.pi * (r1 * r1 + r2 * r2) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect golf balls in the frame using multiple strategies."""
        if frame is None or frame.size == 0:
            return []
            
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Get detections from base detectors
        base_detections = []
        for detector in self.detectors:
            try:
                detections = detector.detect(processed_frame)
                if detections:
                    base_detections.extend(detections)
            except Exception as e:
                logger.error(f"Error in base detector: {str(e)}")
        
        # Get detections from additional methods
        contour_detections = self.detect_contours(processed_frame)
        watershed_detections = self.detect_using_watershed(processed_frame)
        hough_detections = self.detect_using_hough_circles(processed_frame)
        
        # Combine all detections
        all_detections = self.combine_detections([
            base_detections,
            contour_detections,
            watershed_detections,
            hough_detections
        ])
        
        # Filter by size and aspect ratio
        filtered_detections = []
        for x, y, r, conf in all_detections:
            # Skip if radius is outside expected range
            if not (self.min_ball_radius <= r <= self.max_ball_radius):
                continue
                
            # Add to filtered detections
            filtered_detections.append((x, y, r, conf))
        
        return filtered_detections
