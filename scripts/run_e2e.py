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
    parser.add_argument("--input", type=str, default="outputs/demo_input.mp4", help="Input video file path")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--config", type=str, default="configs/defaults.yaml", help="Path to config file")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the output video")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enable interactive mode to select ball position in first frame")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if not Path(args.input).exists():
        print("Generating synthetic demo clip...")
        generate_synthetic_shot(args.input)

    def progress(p, m):
        print(f"[{p*100:5.1f}%] {m}")

    process_video(args.input, args.outdir, args.config, progress_cb=progress, interactive=args.interactive)
    out_video = Path(args.outdir) / "tracer.mp4"
    print("Done. Output video:", out_video)

    if not args.no_open and out_video.exists():
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out_video))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f"open \"{out_video}\"")
            else:
                os.system(f"xdg-open \"{out_video}\"")
        except Exception as e:
            print("Could not open video automatically:", e)


if __name__ == "__main__":
    main()
