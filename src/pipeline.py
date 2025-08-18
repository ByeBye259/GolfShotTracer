from typing import Callable, Dict, List, Tuple
from pathlib import Path
import yaml
import cv2
import numpy as np

from src.utils.logging import get_logger
from src.vision.detect import BallDetector, DetectorConfig
from src.vision.track import SingleBallTracker
from src.vision.launch_find import find_launch_index
from src.calib.scale_fit import fit_scale_from_vertical_accel
from src.physics.fit import fit_ballistics_2d
from src.physics.conf import estimate_confidence
from src.render.export import write_overlay_video, write_metrics


logger = get_logger("pipeline")


def process_video(input_video: str, output_dir: str, config_path: str, progress_cb: Callable[[float, str], None] = lambda p, m: None):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    det_cfg = DetectorConfig(
        backend=cfg.get("detection", {}).get("backend", "auto"),
        min_conf=float(cfg.get("detection", {}).get("min_conf", 0.1)),
        min_radius_px=int(cfg.get("detection", {}).get("min_radius_px", 1)),
        max_radius_px=int(cfg.get("detection", {}).get("max_radius_px", 30)),
    )
    detector = BallDetector(det_cfg)
    tracker = SingleBallTracker(
        process_noise=float(cfg.get("tracking", {}).get("process_noise", 5.0)),
        measurement_noise=float(cfg.get("tracking", {}).get("measurement_noise", 2.0)),
    )

    points: List[Tuple[float, float]] = []
    times_ms: List[float] = []

    progress_cb(0.05, "detect+track")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        dets = detector.detect_frame(frame, t_ms)
        det_tuples = [(d.cx, d.cy, d.r, d.conf) for d in dets]
        tp = tracker.step(t_ms, det_tuples)
        points.append((tp.cx, tp.cy))
        times_ms.append(t_ms)
        frame_idx += 1
        if total > 0 and frame_idx % 30 == 0:
            progress_cb(0.05 + 0.4 * (frame_idx / total), "detect+track")

    cap.release()

    if len(points) < 5:
        raise RuntimeError("Ball not detected reliably. Try better lighting/contrast or different clip.")

    # Launch detection
    launch_idx = find_launch_index(tracker.get_track(), float(cfg.get("launch", {}).get("speed_jump_thresh", 2.0)))
    points = points[launch_idx:]
    times_ms = [t - times_ms[launch_idx] for t in times_ms[launch_idx:]]
    track = tracker.get_track()[launch_idx:]

    progress_cb(0.5, "scale fit")
    m_per_px = fit_scale_from_vertical_accel(track, float(cfg.get("physics", {}).get("g", 9.80665)))

    progress_cb(0.6, "physics fit")
    fit = fit_ballistics_2d(track, m_per_px, g=float(cfg.get("physics", {}).get("g", 9.80665)))

    # Derived metrics
    traj = np.array(fit["traj_m"])  # Nx2
    apex_m = float(np.max(traj[:, 1]))
    carry_m = float(np.max(traj[:, 0]) - np.min(traj[:, 0]))
    tof_s = float(times_ms[-1] / 1000.0)
    conf = estimate_confidence(fit["residual_rmse_m"], len(track))

    metrics = {
        "launch_speed_mps": fit["launch_speed_mps"],
        "elevation_deg": fit["elevation_deg"],
        "azimuth_deg": fit["azimuth_deg"],
        "apex_m": apex_m,
        "carry_m": carry_m,
        "time_of_flight_s": tof_s,
        "confidence": conf,
        "notes": "Single-view 2D fit; azimuth and 3D depth are underconstrained; report is approximate.",
    }

    progress_cb(0.8, "render")
    out_video = str(out_dir / "tracer.mp4")
    write_overlay_video(input_video, out_video, points, metrics, codec=cfg.get("export", {}).get("codec", "h264"), ffmpeg_path=cfg.get("export", {}).get("ffmpeg_path", "ffmpeg"))

    progress_cb(0.95, "export metrics")
    out_json = str(out_dir / "metrics.json")
    out_csv = str(out_dir / "metrics.csv")
    write_metrics(out_json, out_csv, times_ms, points, metrics)

    progress_cb(1.0, "done")
