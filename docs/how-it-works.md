# ApexTracer-Lite: How it Works

- Detection: high-recall heuristic motion/brightness detector with optional ONNX small-object model.
- Tracking: 2D Kalman (constant velocity) to maintain a smooth, continuous ball path.
- Launch trim: simple speed-jump heuristic to trim pre-launch frames for a clean trace.
- Rendering: smoothed tracer polyline over the original frames; optional ffmpeg H.264 + audio remux.
- Output: a single MP4 with the overlaid trajectory.

Limitations in Lite mode:
- Single-view and 2D: overlay shows the apparent path in image space; no 3D reconstruction.
- Heuristic detector works best on high contrast (white ball vs sky/grass) and a stable camera.

Upgrade path:
- Plug-in ONNX detector (YOLOv8n/RT-DETR tiny) via `assets/models/*.onnx`.
- If you need metrics (speed/carry/etc.), re-introduce physics and scale estimation modules and HUD overlay.
