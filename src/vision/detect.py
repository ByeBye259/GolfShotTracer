"""Ball detection module.

Two backends:
- ONNX: loads a small object detector if weights are present.
- Heuristic: motion/brightness-based blob detection, robust to blur.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore


@dataclass
class Detection:
    t_ms: float
    cx: float
    cy: float
    r: float
    conf: float


@dataclass
class DetectorConfig:
    backend: str = "auto"
    min_conf: float = 0.1
    min_radius_px: int = 1
    max_radius_px: int = 30


class BallDetector:
    def __init__(self, cfg: DetectorConfig, model_path: Optional[str] = None):
        self.cfg = cfg
        self.session = None
        if cfg.backend in ("onnx", "auto") and model_path and Path(model_path).exists() and ort is not None:
            try:
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                self.backend = "onnx"
            except Exception:
                self.backend = "heuristic"
        else:
            self.backend = "heuristic"
        self.prev_gray = None

    def detect_frame(self, frame_bgr: np.ndarray, t_ms: float) -> List[Detection]:
        if self.backend == "onnx" and self.session is not None:
            return self._detect_onnx(frame_bgr, t_ms)
        return self._detect_heuristic(frame_bgr, t_ms)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame to enhance motion and ball visibility."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame if needed
        if not hasattr(self, 'prev_frame'):
            self.prev_frame = gray
            return np.zeros_like(gray)
            
        # Compute absolute difference between current and previous frame
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray
        
        # Apply threshold to get significant changes
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh

    def _detect_heuristic(self, frame_bgr: np.ndarray, t_ms: float) -> List[Detection]:
        # Get preprocessed motion mask
        motion_mask = self._preprocess(frame_bgr)
        dets: List[Detection] = []
        
        # Find contours in the motion mask
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 10:  # Minimum area threshold
                continue
                
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)
            
            # Skip if radius is too small or too large
            if radius < 2 or radius > 100:
                continue
                
            # Calculate circularity
            area = cv2.contourArea(contour)
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area if circle_area > 0 else 0
            
            # Only keep reasonably circular detections with significant motion
            if 0.3 < circularity < 1.5:  # More lenient circularity for motion
                # Calculate confidence based on motion intensity and size
                motion_roi = motion_mask[y-radius:y+radius, x-radius:x+radius]
                motion_intensity = np.mean(motion_roi) / 255.0
                confidence = min(1.0, motion_intensity * (radius / 10.0))
                
                if confidence > 0.2:  # Minimum confidence threshold
                    # Create Detection with proper attributes (t_ms, x, y, r, conf)
                    dets.append(Detection(t_ms, x, y, radius, confidence))
        
        # Sort by confidence (highest first)
        dets.sort(key=lambda d: d.conf, reverse=True)
        
        # Keep only the top detection to avoid multiple detections
        return dets[:1] if dets else []
        h, w = gray.shape
        for c in contours:
            if len(c) < 3:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < self.cfg.min_radius_px or r > self.cfg.max_radius_px:
                continue
            area = cv2.contourArea(c)
            circ = 4 * np.pi * area / (np.pi * r * r + 1e-6)
            if circ < 0.2:
                continue
            conf = min(1.0, float(area / (np.pi * r * r)))
            dets.append(Detection(t_ms, float(x), float(y), float(r), conf))
        # Sort by conf descending
        dets.sort(key=lambda d: -d.conf)
        return dets[:5]

    def _detect_onnx(self, frame_bgr: np.ndarray, t_ms: float) -> List[Detection]:
        # Minimal placeholder: users can drop a model and adjust pre/post-processing.
        # For now, fallback to heuristic to keep pipeline functional.
        return self._detect_heuristic(frame_bgr, t_ms)
