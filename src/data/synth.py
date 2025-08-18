from pathlib import Path
import numpy as np
import cv2


def generate_synthetic_shot(out_path: str, duration_s: float = 3.0, fps: int = 60, resolution=(1280, 720)) -> str:
    w, h = resolution
    total = int(duration_s * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Simple green field and sky gradient
    for i in range(total):
        t = i / fps
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Sky
        img[:] = (180, 220, 255)
        # Grass
        cv2.rectangle(img, (0, int(h * 0.6)), (w, h), (60, 160, 60), -1)
        # Tee position
        x0 = int(w * 0.1)
        y0 = int(h * 0.6)
        # Ballistics: 2D
        v0 = np.array([40.0, 25.0])  # m/s
        g = 9.80665
        m_per_px = 0.02
        x = x0 + int(v0[0] * t / m_per_px)
        y = y0 - int((v0[1] * t - 0.5 * g * t * t) / m_per_px)
        # Draw ball
        cv2.circle(img, (x, y), 6, (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return out_path
