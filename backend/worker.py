import traceback
from pathlib import Path
from typing import Dict

from .job_manager import Job

# Ensure src in sys.path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pipeline import process_video  # noqa: E402


def run_job(job: Job, cfg: Dict):
    try:
        if not job.input_path:
            raise ValueError("Job has no input_path")
        job_dir = job.input_path.parent
        out_dir = job_dir / "out"
        out_dir.mkdir(exist_ok=True)
        process_video(
            input_video=str(job.input_path),
            output_dir=str(out_dir),
            config_path=str(ROOT / "configs" / "defaults.yaml"),
            progress_cb=lambda p, m: _update(job, p, m),
        )
        job.out_video = out_dir / "tracer.mp4"
        _update(job, 1.0, "done")
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.message = f"{e}\n{traceback.format_exc()}"


def _update(job: Job, progress: float, message: str):
    job.progress = progress
    job.message = message
