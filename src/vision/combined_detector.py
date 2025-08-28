from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from pathlib import Path
import logging
from .yolo_detector import YOLOBallDetector
from .roboflow_detector import RoboflowBallDetector

logger = logging.getLogger(__name__)

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect objects in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detections as (x, y, radius, confidence) tuples
        """
        pass

class CombinedDetector(BaseDetector):
    """Combines multiple detection models for improved detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the combined detector.
        
        Args:
            config: Configuration dictionary containing detector settings
        """
        self.detectors = []
        self.config = config
        
        # Initialize YOLO detector if enabled
        if config.get('yolo', {}).get('enabled', True):
            self.detectors.append(
                YOLOBallDetector(
                    model_path=config['yolo'].get('model_path', 'weights/yolov8n-golf.pt'),
                    conf_thresh=config['yolo'].get('conf_thresh', 0.2),
                    iou_thresh=config['yolo'].get('iou_thresh', 0.4)
                )
            )
            logger.info("YOLO detector initialized")
        
        # Initialize Roboflow detector if enabled
        if config.get('roboflow', {}).get('enabled', False):
            self.detectors.append(
                RoboflowBallDetector(
                    api_key=config['roboflow'].get('api_key', ''),
                    model_name=config['roboflow'].get('model_name', 'golf-ball-detection-hii2e'),
                    version=config['roboflow'].get('version', 2),
                    conf_thresh=config['roboflow'].get('conf_thresh', 0.2)
                )
            )
            logger.info("Roboflow detector initialized")
        
        if not self.detectors:
            raise ValueError("At least one detector must be enabled in the config")
        
        self.min_ball_radius = config.get('min_ball_radius', 2)
        self.max_ball_radius = config.get('max_ball_radius', 200)
    
    def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Run detection using all enabled detectors and combine results.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Combined list of detections as (x, y, radius, confidence) tuples
        """
        all_detections = []
        
        for detector in self.detectors:
            try:
                detections = detector.detect(image)
                if detections:
                    all_detections.extend(detections)
            except Exception as e:
                logger.error(f"Error in {detector.__class__.__name__}: {str(e)}")
        
        # Filter detections by size
        filtered_detections = [
            (x, y, r, conf) for x, y, r, conf in all_detections
            if self.min_ball_radius <= r <= self.max_ball_radius
        ]
        
        # Sort by confidence (highest first)
        filtered_detections.sort(key=lambda x: x[3], reverse=True)
        
        # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
        return self._nms(filtered_detections)
    
    def _nms(self, detections: List[Tuple[float, float, float, float]], 
            iou_threshold: float = 0.5) -> List[Tuple[float, float, float, float]]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not detections:
            return []
        
        # Convert to numpy array for easier manipulation
        boxes = np.array([(x - r, y - r, x + r, y + r, conf) 
                         for x, y, r, conf in detections])
        
        if len(boxes) == 0:
            return []
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(-boxes[:, 4])
        keep = []
        
        while sorted_indices.size > 0:
            # Pick the most confident detection
            best_idx = sorted_indices[0]
            keep.append(best_idx)
            
            if sorted_indices.size == 1:
                break
                
            # Get IoU of all other boxes with the best box
            best_box = boxes[best_idx, :4]
            other_boxes = boxes[sorted_indices[1:], :4]
            
            # Calculate IoU
            xx1 = np.maximum(best_box[0], other_boxes[:, 0])
            yy1 = np.maximum(best_box[1], other_boxes[:, 1])
            xx2 = np.minimum(best_box[2], other_boxes[:, 2])
            yy2 = np.minimum(best_box[3], other_boxes[:, 3])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            area1 = (best_box[2] - best_box[0] + 1) * (best_box[3] - best_box[1] + 1)
            area2 = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * \
                   (other_boxes[:, 3] - other_boxes[:, 1] + 1)
            
            union = area1 + area2 - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            keep_indices = np.where(iou <= iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        
        return [tuple(detections[i]) for i in keep]

    def __call__(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Alias for detect method."""
        return self.detect(image)
