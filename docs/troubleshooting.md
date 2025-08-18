# Troubleshooting

- Ball not detected:
  - Ensure good contrast (white ball vs. sky/grass).
  - Use higher FPS or less compression.
  - Try the synthetic demo or drop a small ONNX model into `assets/models/`.

- Overlay not visible:
  - Check that the ball is actually tracked (console shows progress but output has no line if detection failed).
  - Try a shorter clip focusing on contact and early flight.
  - Increase resolution (≥720p) and avoid heavy stabilization or rolling shutter artifacts.

- Export errors / codec:
  - If ffmpeg is not installed or H.264 unavailable, the app falls back to mp4v. Install ffmpeg and ensure `ffmpeg` is on PATH.
