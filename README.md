# ApexTracer-Lite

 Single-camera golf shot tracer. Input a video (phone/action cam/DSLR), output the same video with an overlaid trajectory. No external sensors or props; everything inferred from video. Runs locally from the terminal and auto-opens the result.

## Repo layout
- `src/` — core pipeline (detection, tracking, lightweight launch trim, render)
- `configs/` — defaults
- `scripts/run_e2e.py` — end-to-end on a demo clip
- `docs/` — how-it-works, troubleshooting, accuracy notes

## Quickstart (CLI)
1) Create venv and install deps (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2) Run end-to-end (generates a synthetic demo if needed and auto-opens the output):
```powershell
python scripts/run_e2e.py
```
Output in `outputs/`: `tracer.mp4`.

## Process your own clip
```powershell
python scripts/run_e2e.py --input path\to\your\clip.mp4 --outdir outputs
```
Use `--no-open` to skip auto-opening the result.

## Notes
- Baseline detector is heuristic (motion/brightness) with optional ONNX tiny model fallback (drop weights in `assets/models/`).
- If `ffmpeg` is on PATH, we encode H.264 and remux original audio; otherwise fallback to mp4v.

See `docs/how-it-works.md` and `docs/troubleshooting.md` for details and tips.
