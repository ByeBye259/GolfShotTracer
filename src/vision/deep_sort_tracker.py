import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import cv2
import logging
from pathlib import Path
from dataclasses import dataclass

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("deep_sort_realtime not available. Tracking will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """Represents a tracked object with its properties."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    track_id: int
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        x1, _, x2, _ = self.bbox
        return x2 - x1
    
    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        _, y1, _, y2 = self.bbox
        return y2 - y1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'track_id': self.track_id,
            'center': self.center,
            'width': self.width,
            'height': self.height
        }

class DummyTracker:
    """Dummy tracker for when DeepSORT is not available."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using DummyTracker as DeepSORT is not available")
        self.track_id_counter = 0
    
    def update_tracks(self, *args, **kwargs):
        return []
    
    def reset(self):
        self.track_id_counter = 0

class DeepSortTracker:
    """Wrapper around DeepSORT tracker for easier integration with enhanced error handling.
    
    This class provides a simplified interface to the DeepSORT tracker with:
    - Automatic fallback to a dummy tracker if DeepSORT is not available
    - Better error handling and logging
    - Support for both detection-based and feature-based tracking
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 max_iou_distance: float = 0.7,
                 max_cosine_distance: float = 0.2,
                 nms_max_overlap: float = 1.0,
                 embedder: str = "mobilenet",
                 embedder_model_path: Optional[Union[str, Path]] = None):
        """Initialize the DeepSORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track alive without updates.
            max_iou_distance: IoU threshold for association.
            max_cosine_distance: Cosine distance threshold for appearance matching.
            nms_max_overlap: Non-maximum suppression threshold.
            embedder: Feature extractor to use ('mobilenet', 'clip', 'torchreid', etc.).
            embedder_model_path: Path to custom embedder model weights.
            
        Raises:
            RuntimeError: If DeepSORT initialization fails and no fallback is available.
        """
        self.max_age = max(1, max_age)
        self.max_iou_distance = max(0.1, min(1.0, max_iou_distance))
        self.max_cosine_distance = max(0.1, min(1.0, max_cosine_distance))
        self.nms_max_overlap = max(0.1, min(1.0, nms_max_overlap))
        self.embedder = embedder
        self.embedder_model_path = str(embedder_model_path) if embedder_model_path else None
        
        # Initialize tracker (or dummy if not available)
        self.tracker = self._initialize_tracker()
        
        logger.info(f"Initialized {self.__class__.__name__} with max_age={self.max_age}, "
                   f"max_iou={self.max_iou_distance}, max_cosine={self.max_cosine_distance}")
    
    def _initialize_tracker(self):
        """Initialize the DeepSORT tracker with error handling."""
        if not DEEPSORT_AVAILABLE:
            logger.warning("DeepSORT not available. Using dummy tracker.")
            return DummyTracker()
            
        try:
            return DeepSort(
                max_age=self.max_age,
                n_init=3,
                nms_max_overlap=self.nms_max_overlap,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=None,
                override_track_class=None,
                embedder=self.embedder,
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=self.embedder_model_path,
                polygon=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}")
            logger.warning("Falling back to dummy tracker")
            return DummyTracker()
    
    def update(self, 
              detections: List[List[float]], 
              frame: np.ndarray,
              frame_id: Optional[int] = None) -> List[TrackedObject]:
        """Update the tracker with new detections.
        
        Args:
            detections: List of detections in format [x1, y1, x2, y2, confidence].
            frame: Current frame (BGR format) for feature extraction.
            frame_id: Optional frame ID for tracking across sequences.
            
        Returns:
            List of tracked objects with updated states.
        """
        if not detections or frame is None or frame.size == 0:
            return []
            
        try:
            # Ensure frame is in the correct format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            # Convert detections to numpy array [x1, y1, x2, y2, confidence, class_id?]
            detections_np = np.array(detections, dtype=np.float32)
            if detections_np.size == 0:
                return []
                
            # Ensure detections have the right shape (N, 5)
            if detections_np.ndim == 1:
                detections_np = detections_np.reshape(1, -1)
                
            if detections_np.shape[1] < 5:  # If no confidence, add default
                detections_np = np.column_stack([
                    detections_np,
                    np.ones(len(detections_np))  # Default confidence
                ])
            
            # Update tracker
            tracks = self.tracker.update_tracks(
                detections_np,
                frame=frame,
                yolo_preds=True
            )
            
            # Convert to TrackedObject format
            tracked_objects = []
            for track in tracks:
                if not hasattr(track, 'is_confirmed') or not track.is_confirmed():
                    continue
                    
                # Skip tracks that haven't been updated recently
                if hasattr(track, 'time_since_update') and track.time_since_update > 1:
                    continue
                
                # Get bounding box
                if hasattr(track, 'to_tlbr'):
                    bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                elif hasattr(track, 'to_tlwh'):
                    # Convert from [x,y,w,h] to [x1,y1,x2,y2]
                    x, y, w, h = track.to_tlwh()
                    bbox = np.array([x, y, x + w, y + h])
                else:
                    logger.warning("Track has no bounding box conversion method")
                    continue
                
                # Get class ID (default to 0 for golf ball)
                class_id = 0
                if hasattr(track, 'get_det_class'):
                    class_id = track.get_det_class()
                elif hasattr(track, 'det_class'):
                    class_id = track.det_class
                
                # Get confidence score
                confidence = 1.0
                if hasattr(track, 'get_det_conf'):
                    confidence = track.get_det_conf()
                elif hasattr(track, 'det_conf'):
                    confidence = track.det_conf
                elif len(detections_np) > 0 and detections_np.shape[1] > 4:
                    # Try to get confidence from original detections
                    idx = min(len(detections_np) - 1, int(getattr(track, 'detection_id', 0)))
                    confidence = float(detections_np[idx, 4])
                
                # Ensure valid bbox
                bbox = np.clip(bbox, 0, [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                
                tracked_objects.append(TrackedObject(
                    bbox=tuple(bbox.astype(float)),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    track_id=int(getattr(track, 'track_id', -1))
                ))
            
            logger.debug(f"Tracking {len(tracked_objects)} objects")
            return tracked_objects
            
        except Exception as e:
            logger.error(f"Error in tracker update: {e}", exc_info=True)
            return []
    
    def reset(self):
        """Reset the tracker state."""
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
    
    def __call__(self, detections: List[List[float]], frame: np.ndarray) -> List[TrackedObject]:
        """Alias for update method."""
        return self.update(detections, frame)
