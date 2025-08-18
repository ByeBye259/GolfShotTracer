# ApexTracer-Lite: How it Works

- High-recall detection: heuristic motion/brightness detector with optional ONNX small-object model.
- Tracking: 2D Kalman (constant velocity) with physics-aware smoothing.
- Launch detection: speed jump heuristic at early frames.
- Auto-scale: fit vertical quadratic to estimate meters-per-pixel via gravity.
- Physics fit: 2D ballistic (no drag) least-squares for v0 and elevation.
- Rendering: smoothed tracer polyline + HUD; optional ffmpeg H.264 + audio remux.
- Outputs: MP4 with overlay, metrics.json, metrics.csv.

Limitations in Lite mode:
- Single-view with minimal assumptions; azimuth and 3D depth underconstrained. Reported with low confidence.
- Heuristic detector works best on sky/grass contrast and stable camera.

Upgrade path:
- Plug-in ONNX detector (YOLOv8n/RT-DETR tiny) via `assets/models/*.onnx`.
- Add drag/Magnus in `src/physics/model.py` + robust fit in `src/physics/fit.py`.
- Horizon/ground plane and focal refinement to move from 2D->3D.
