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
        """Preprocess frame to enhance ball visibility."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhance brightness and contrast
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        
        # Recombine and convert back to BGR
        enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
        
        # Convert to grayscale with better contrast
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray

    def _detect_heuristic(self, frame_bgr: np.ndarray, t_ms: float) -> List[Detection]:
        gray = self._preprocess(frame_bgr)
        dets: List[Detection] = []
        if self.prev_gray is None:
            self.prev_gray = gray
            return dets
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)
        th = cv2.medianBlur(th, 5)
        th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
