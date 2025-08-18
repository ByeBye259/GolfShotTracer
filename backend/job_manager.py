import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued, processing, done, error
    progress: float = 0.0
    message: str = ""
    input_path: Optional[Path] = None
    out_video: Optional[Path] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobManager:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self) -> Job:
        with self._lock:
            jid = uuid.uuid4().hex[:12]
            job_dir = self.storage_dir / f"job_{jid}"
            job_dir.mkdir(parents=True, exist_ok=True)
            job = Job(id=jid)
            self.jobs[jid] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def update(self, job_id: str, **kwargs):
        job = self.jobs.get(job_id)
        if not job:
            return
        with job.lock:
            for k, v in kwargs.items():
                setattr(job, k, v)
