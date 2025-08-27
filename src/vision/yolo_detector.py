import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from ultralytics import YOLO
import torch
import logging

logger = logging.getLogger(__name__)

class YOLOBallDetector:
    """YOLOv8-based golf ball detector with enhanced error handling and logging.
    
    This class provides a simple interface for detecting golf balls in images using YOLOv8.
    It includes fallback mechanisms and detailed logging for better debugging.
    """
    
    def __init__(self, model_path: Union[str, Path], conf_thresh: float = 0.2, iou_thresh: float = 0.4):
        """Initialize the YOLO detector with enhanced detection parameters.
        
        Args:
            model_path: Path to the YOLOv8 model file (.pt or .onnx).
            conf_thresh: Confidence threshold for detections (0-1). Lower for more detections.
            iou_thresh: IOU threshold for non-maximum suppression (0-1). Lower for more aggressive NMS.
            
        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If model loading fails.
        """
        # Set device (GPU if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Detection parameters - optimized for golf ball detection
        self.conf_thresh = max(0.01, min(1.0, conf_thresh))
        self.iou_thresh = max(0.01, min(1.0, iou_thresh))
        
        # Ball detection constraints
        self.min_ball_radius = 2       # Lowered minimum radius for small/far balls
        self.max_ball_radius = 200     # Increased maximum radius for close-up shots
        self.max_aspect_ratio = 2.0    # Increased to handle motion blur
        self.min_confidence = 0.1      # Lowered minimum confidence for more detections
        
        # Motion detection and tracking
        self.prev_positions = []
        self.frame_count = 0
        self.model = None
        
        # Convert to Path object if string
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
        # Verify model exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Initialize model with optimizations
            logger.info(f"Loading YOLO model from {model_path} (device: {self.device})")
            self.model = YOLO(str(model_path))
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model parameters
            self.model.overrides['conf'] = self.conf_thresh
            self.model.overrides['iou'] = self.iou_thresh
            self.model.overrides['agnostic_nms'] = True
            self.model.overrides['verbose'] = False
            
            # Warmup run
            if torch.cuda.is_available():
                self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                
            logger.info(f"YOLO model loaded successfully. Using {self.device.upper()} for inference.")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise RuntimeError(f"YOLO model initialization failed: {e}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference."""
        # Convert to RGB (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def _is_valid_detection(self, x: float, y: float, width: float, height: float, conf: float) -> bool:
        """Check if a detection meets size and aspect ratio constraints."""
        radius = max(width, height) / 2
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        return (self.min_ball_radius <= radius <= self.max_ball_radius and
                aspect_ratio <= self.max_aspect_ratio and
                conf >= self.min_confidence)
    
    def _apply_nms(self, boxes, scores, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to filter overlapping boxes."""
        if len(boxes) == 0:
            return []
            
        # Convert boxes to x1,y1,x2,y2 format
        x1 = boxes[:, 0] - boxes[:, 2]/2
        y1 = boxes[:, 1] - boxes[:, 3]/2
        x2 = boxes[:, 0] + boxes[:, 2]/2
        y2 = boxes[:, 1] + boxes[:, 3]/2
        
        # Calculate areas
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score (highest first)
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            # Get the index with highest score
            current = indices[0]
            keep.append(current)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[current], x1[indices[1:]])
            yy1 = np.maximum(y1[current], y1[indices[1:]])
            xx2 = np.minimum(x2[current], x2[indices[1:]])
            yy2 = np.minimum(y2[current], y2[indices[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            intersection = w * h
            iou = intersection / (areas[current] + areas[indices[1:]] - intersection)
            
            # Keep boxes with IoU < threshold
            remaining_indices = np.where(iou <= iou_threshold)[0]
            indices = indices[remaining_indices + 1]
            
        return keep
    
    def _filter_static_detections(self, detections, max_movement=5.0):
        """Filter out detections that don't move between frames."""
        self.frame_count += 1
        
        # Skip filtering for first few frames
        if self.frame_count < 3:
            self.prev_positions = [d[:2] for d in detections]
            return detections
            
        # Filter detections that are close to previous positions
        filtered = []
        for det in detections:
            is_moving = True
            x, y, r, conf = det
            
            # Check distance to previous positions
            for prev_x, prev_y in self.prev_positions:
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if distance < max_movement:
                    is_moving = False
                    break
                    
            if is_moving:
                filtered.append(det)
                
        # Update previous positions
        self.prev_positions = [d[:2] for d in filtered]
        return filtered
    
    def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect golf balls in the input image with enhanced filtering.
        
        Args:
            image: Input image in BGR format (numpy array).
            
        Returns:
            List of detections as (x, y, radius, confidence) tuples.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized")
            
        if image is None or image.size == 0:
            logger.warning("Empty image provided for detection")
            return []
            
        try:
            # Preprocess - resize if needed for better performance
            h, w = image.shape[:2]
            if max(h, w) > 1280:  # Downscale large images for faster processing
                scale = 1280.0 / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to RGB and normalize
            image_rgb = self.preprocess(image)
            
            # Run inference with optimized parameters
            results = self.model(
                image_rgb, 
                verbose=False,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                agnostic_nms=True,  # Better for single class detection
                max_det=10,         # Limit max detections to prevent false positives
                classes=[0]          # Only detect golf balls (class 0)
            )
            
            # Process detections
            detections = []
            boxes = []
            scores = []
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    try:
                        # Get box coordinates (x1, y1, x2, y2)
                        box_data = box.xyxy[0].cpu().numpy()
                        if len(box_data) < 4:
                            continue
                            
                        x1, y1, x2, y2 = box_data
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Convert to center, width, height
                        x = (x1 + x2) / 2
                        y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Check if detection meets size/ratio constraints
                        if self._is_valid_detection(x, y, width, height, conf):
                            radius = max(width, height) / 2
                            detections.append((x, y, radius, conf))
                            boxes.append([x, y, width, height])
                            scores.append(conf)
                            
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue
            
            # Apply Non-Maximum Suppression
            if len(detections) > 0:
                boxes_np = np.array(boxes)
                scores_np = np.array(scores)
                keep_indices = self._apply_nms(boxes_np, scores_np, self.iou_thresh)
                detections = [detections[i] for i in keep_indices]
            
            # Filter static detections (optional, can be enabled/disabled)
            if len(detections) > 1:
                detections = self._filter_static_detections(detections)
            
            logger.debug(f"Detected {len(detections)} valid golf balls")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def __call__(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Alias for detect method."""
        return self.detect(frame)
