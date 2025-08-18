# Troubleshooting

- Ball not detected:
  - Ensure good contrast (white ball vs. sky/grass).
  - Use higher FPS or less compression.
  - Try the synthetic demo or drop a small ONNX model into `assets/models/`.

- Wrong speed/carry:
  - The Lite pipeline infers scale from vertical acceleration; large perspective/rolling shutter may bias it.
  - Try cropping around impact area or a more down-the-line view.

- Export errors / codec:
  - If ffmpeg is not installed or H.264 unavailable, the app falls back to mp4v. Install ffmpeg and ensure `ffmpeg` is on PATH.
