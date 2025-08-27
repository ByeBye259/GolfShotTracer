import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SuperResolution:
    """Super-resolution model for enhancing image quality before detection.
    
    This class provides a unified interface for super-resolution models with fallback
    to basic upscaling if the model is not available or fails to load.
    """
    
    def __init__(self, model_path: Optional[str] = None, scale: int = 2, device: str = 'cpu'):
        """Initialize the super-resolution model.
        
        Args:
            model_path: Path to the super-resolution model file (optional).
            scale: Scale factor for super-resolution (2x, 3x, 4x, 8x).
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.scale = scale
        self.device = device
        self.model = None
        self._initialized = False
        
        # Check if OpenCV has dnn_superres module
        self.has_superres = hasattr(cv2, 'dnn_superres')
        
        if model_path and Path(model_path).exists() and self.has_superres:
            try:
                # Try to initialize the super-resolution model
                self.model = cv2.dnn_superres.DnnSuperResImpl_create()
                self.model.readModel(str(model_path))
                
                # Set backend and target
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                target = cv2.dnn.DNN_TARGET_CPU if device == 'cpu' else cv2.dnn.DNN_TARGET_CUDA
                self.model.setPreferableTarget(target)
                
                # Set model type based on file extension
                model_type = 'edsr'  # Default to EDSR
                if 'fsrcnn' in str(model_path).lower():
                    model_type = 'fsrcnn'
                elif 'espcn' in str(model_path).lower():
                    model_type = 'espcn'
                elif 'lapsrn' in str(model_path).lower():
                    model_type = 'lapsrn'
                
                self.model.setModel(model_type, scale)
                self._initialized = True
                logger.info(f"Initialized {model_type.upper()} super-resolution model (scale: {scale}x)")
                
            except Exception as e:
                logger.warning(f"Could not initialize super-resolution model: {e}")
                self.model = None
        else:
            if not self.has_superres:
                logger.warning("OpenCV dnn_superres module not available. Using basic upscaling.")
            elif not model_path:
                logger.warning("No model path provided. Using basic upscaling.")
            else:
                logger.warning(f"Model file not found: {model_path}. Using basic upscaling.")
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance the input image using super-resolution or basic upscaling.
        
        Args:
            image: Input image (BGR format).
            
        Returns:
            Enhanced image with higher resolution.
        """
        if self._initialized and self.model is not None:
            try:
                return self.model.upsample(image)
            except Exception as e:
                logger.warning(f"Super-resolution failed: {e}. Falling back to basic upscaling.")
                self._initialized = False  # Disable after failure
        
        # Fallback to basic upscaling
        height, width = image.shape[:2]
        return cv2.resize(
            image, 
            (width * self.scale, height * self.scale), 
            interpolation=cv2.INTER_CUBIC
        )
    
    def is_available(self) -> bool:
        """Check if super-resolution is available."""
        return self._initialized and self.model is not None
