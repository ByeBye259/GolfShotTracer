# Accuracy Notes

 This Lite build outputs an overlay-only trajectory (no numeric metrics). Accuracy here refers to how closely the drawn path follows the ball across frames.

 Tips for best alignment:
- Prefer high contrast and stable framing (down-the-line or face-on with minimal pan).
- Use ≥720p and ≥30 FPS; higher FPS reduces motion blur.
- Avoid heavy stabilization/filters that warp frames.

 Known limitations:
- Single-camera, 2D overlay; depth/azimuth are not estimated.
- Small, fast balls can exceed detector recall in extreme compression or low light.
