import argparse
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synth import generate_synthetic_shot
from src.pipeline import process_video


def main():
    parser = argparse.ArgumentParser(description="Run ApexTracer-Lite E2E on a demo clip")
    parser.add_argument("--input", type=str, default="outputs/demo_input.mp4")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default="configs/defaults.yaml")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not Path(args.input).exists():
        print("Generating synthetic demo clip...")
        generate_synthetic_shot(args.input)

    def progress(p, m):
        print(f"[{p*100:5.1f}%] {m}")

    process_video(args.input, args.outdir, args.config, progress_cb=progress)
    print("Done. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
