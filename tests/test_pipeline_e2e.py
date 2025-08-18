from pathlib import Path

from src.data.synth import generate_synthetic_shot
from src.pipeline import process_video


def test_e2e(tmp_path: Path):
    input_video = tmp_path / "demo.mp4"
    outdir = tmp_path / "out"
    generate_synthetic_shot(str(input_video), duration_s=1.5, fps=30)
    process_video(str(input_video), str(outdir), "configs/defaults.yaml")
    assert (outdir / "tracer.mp4").exists()
    assert (outdir / "metrics.json").exists()
