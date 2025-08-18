# ApexTracer-Lite

 Single-camera golf shot tracer. Input a video (phone/action cam/DSLR), output the same video with an overlaid trajectory and a metrics JSON/CSV. No external sensors or props; everything inferred from video. Runs locally.

 ## Repo layout
 - `apps/web/` — Next.js UI (upload, progress, preview, download)
 - `backend/` — FastAPI worker (upload → job → status → results)
 - `src/` — core pipeline (vision, calib-lite, physics-lite, render)
 - `configs/` — defaults
 - `scripts/run_e2e.py` — end-to-end on a demo clip
 - `docs/` — how-it-works, troubleshooting, accuracy notes

 ## Quickstart (CLI E2E)
 1) Create venv and install deps (Windows PowerShell):
 ```powershell
 python -m venv .venv
 .\.venv\Scripts\Activate.ps1
 pip install -r requirements.txt
 ```
 2) Run end-to-end (generates a synthetic demo if needed):
 ```powershell
 python scripts/run_e2e.py
 ```
 Outputs in `outputs/`: `tracer.mp4`, `metrics.json`, `metrics.csv`.

 ## Run backend API
 ```powershell
 uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
 ```
 Endpoints:
 - POST `/upload` — multipart video upload
 - GET `/status/{job_id}` — job progress
 - GET `/result/{job_id}/video` — traced MP4
 - GET `/result/{job_id}/metrics` — metrics JSON

 ## Run web UI
 ```powershell
 cd apps/web
 npm install
 $env:NEXT_PUBLIC_BACKEND_URL="http://localhost:8000"
 npm run dev
 ```
 Open http://localhost:3000, upload a video, watch progress, then preview or download results.

 ## Notes
 - Baseline detector is heuristic (motion/brightness) with optional ONNX tiny model fallback (drop weights in `assets/models/`).
 - Scale is estimated from vertical acceleration (gravity) — robust enough for plausible metrics without props.
 - If `ffmpeg` is on PATH, we encode H.264 and remux original audio; otherwise fallback to mp4v.
 - Confidence reflects residuals and track length; azimuth is low-confidence in Lite 2D mode.

 See `docs/how-it-works.md` and `docs/troubleshooting.md` for details and tips.
