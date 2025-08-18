from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import shutil
import yaml
from typing import Dict

from .job_manager import JobManager
from .worker import run_job

app = FastAPI(title="ApexTracer-Lite Backend")

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "backend" / "storage"
CONFIG = ROOT / "configs" / "defaults.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    CFG: Dict = yaml.safe_load(f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobManager(STORAGE)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    job = jobs.create_job()
    job_dir = STORAGE / f"job_{job.id}"
    input_path = job_dir / "input" / file.filename
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job.input_path = input_path
    job.status = "processing"

    # Run job in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_job, job, CFG)

    return {"job_id": job.id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    return {
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
    }


@app.get("/result/{job_id}/video")
async def result_video(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.out_video or not job.out_video.exists():
        raise HTTPException(404, detail="Result not found")
    return FileResponse(job.out_video)


# Metrics endpoint removed in overlay-only mode
