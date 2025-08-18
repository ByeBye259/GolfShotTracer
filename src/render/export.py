from typing import Dict, List, Tuple
from pathlib import Path
import json
import csv
import shutil
import subprocess
import cv2
import numpy as np

from .overlay import draw_tracer, draw_hud


def write_overlay_video(input_video: str, output_video: str, points_px: List[Tuple[float, float]], metrics: Dict, codec: str = "h264", ffmpeg_path: str = "ffmpeg"):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tmp_out = str(Path(output_video).with_suffix(".tmp.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))

    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(points_px)
    pts_for_frame = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use prefix of points up to current index for nice trail
        if idx < len(points_px):
            pts_for_frame.append(points_px[idx])
        frame = draw_tracer(frame, pts_for_frame)
        frame = draw_hud(frame, metrics)
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()

    # Try to remux audio and encode H.264 if available
    out = Path(output_video)
    if codec.lower() in ("h264", "libx264"):
        try:
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", tmp_out,
                "-i", input_video,
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-c:a", "aac",
                str(out)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            Path(tmp_out).unlink(missing_ok=True)
        except Exception:
            shutil.move(tmp_out, str(out))
    else:
        shutil.move(tmp_out, str(out))


def write_metrics(output_json: str, output_csv: str, times_ms: List[float], points_px: List[Tuple[float, float]], metrics: Dict):
    data = {
        "times_ms": times_ms,
        "points_px": points_px,
        "metrics": metrics,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    # CSV: t_ms, u_px, v_px, x_m, y_m if available
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ms", "u_px", "v_px"])
        for (t, (u, v)) in zip(times_ms, points_px):
            writer.writerow([f"{t:.3f}", f"{u:.3f}", f"{v:.3f}"])
