import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from roboflow import Roboflow

logger = logging.getLogger(__name__)

class RoboflowBallDetector:
    """Roboflow-based golf ball detector."""
    
    def __init__(self, api_key: str, model_name: str = "golf-ball-detection-hii2e", version: int = 2, conf_thresh: float = 0.2):
        """Initialize the Roboflow detector.
        
        Args:
            api_key: Your Roboflow API key
            model_name: Name of the Roboflow model
            version: Model version number
            conf_thresh: Confidence threshold for detections
        """
        self.conf_thresh = conf_thresh
        
        # Initialize Roboflow
        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace().project(model_name)
            self.model = project.version(version).model
            logger.info(f"Successfully loaded Roboflow model {model_name} v{version}")
        except Exception as e:
            logger.error(f"Failed to initialize Roboflow model: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect golf balls in the input image.
        
        Args:
            image: Input image in BGR format (numpy array)
            
        Returns:
            List of detections as (x, y, radius, confidence) tuples
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Roboflow model is not initialized")
            
        if image is None or image.size == 0:
            logger.warning("Empty image provided for detection")
            return []
            
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            predictions = self.model.predict(image_rgb, confidence=self.conf_thresh).json()
            
            detections = []
            for pred in predictions.get('predictions', []):
                if pred['class'] == 'golf-ball' and pred['confidence'] >= self.conf_thresh:
                    x = pred['x']
                    y = pred['y']
                    width = pred['width']
                    height = pred['height']
                    
                    # Convert to (x, y, radius, confidence)
                    radius = (width + height) / 4  # Average of half width and half height
                    detections.append((x, y, radius, pred['confidence']))
            
            logger.debug(f"Detected {len(detections)} golf balls")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def __call__(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Alias for detect method."""
        return self.detect(image)
