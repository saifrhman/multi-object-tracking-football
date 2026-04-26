"""
utils.py
────────
Shared helper functions used across the pipeline.

Sections:
  1. Video I/O — open/read/write video with OpenCV.
  2. Visualization — draw bounding boxes, IDs, trails.
  3. Configuration — load config.yaml into a dict.
  4. CSV logging — append detection rows for offline analysis.
"""

from __future__ import annotations

import csv
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

from tracker import Track


# ──────────────────────────────────────────────────────────────────────────────
# 1. Video I/O
# ──────────────────────────────────────────────────────────────────────────────

def open_video(path: str) -> tuple[cv2.VideoCapture, dict]:
    """
    Open a video file and return the capture object plus metadata.

    Returns
    -------
    cap : cv2.VideoCapture
    meta : dict with keys: width, height, fps, total_frames, duration_s
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    meta = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration_s"] = meta["total_frames"] / meta["fps"] if meta["fps"] > 0 else 0
    return cap, meta


def frame_generator(
    cap: cv2.VideoCapture,
    target_w: int = 0,
    target_h: int = 0,
    frame_skip: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_index, frame_bgr) from an open VideoCapture.

    Parameters
    ----------
    cap : cv2.VideoCapture
    target_w, target_h : int
        If both non-zero, resize each frame before yielding.
        Preserves aspect ratio via letterbox if needed — here we just resize.
    frame_skip : int
        Yield every Nth frame (1 = every frame, 2 = every other frame, …).
    """
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            if target_w > 0 and target_h > 0:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            yield idx, frame

        idx += 1


def open_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """Create and return a VideoWriter. Creates parent directories automatically."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    return writer


# ──────────────────────────────────────────────────────────────────────────────
# 2. Visualization
# ──────────────────────────────────────────────────────────────────────────────

# Generate a stable, visually distinct BGR color per track ID.
def _id_color(track_id: int) -> tuple[int, int, int]:
    random.seed(track_id * 137 + 31)   # deterministic but spread out
    return (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))


class TrailBuffer:
    """Stores recent centroid positions per track ID for trajectory drawing."""

    def __init__(self, max_len: int = 30) -> None:
        self._trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_len))

    def update(self, tracks: list[Track]) -> None:
        for t in tracks:
            cx = int((t.bbox_xyxy[0] + t.bbox_xyxy[2]) / 2)
            cy = int((t.bbox_xyxy[1] + t.bbox_xyxy[3]) / 2)
            self._trails[t.track_id].append((cx, cy))

    def draw(self, frame: np.ndarray) -> None:
        for track_id, pts in self._trails.items():
            color = _id_color(track_id)
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                # Fade older trail segments
                alpha = i / len(pts_list)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts_list[i - 1], pts_list[i], faded, thickness=2)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    trail_buffer: TrailBuffer | None = None,
    show_ids: bool = True,
    show_conf: bool = False,
    box_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes and labels onto a frame (in-place, also returns frame).

    Parameters
    ----------
    frame : np.ndarray  BGR image
    tracks : list[Track]
    trail_buffer : TrailBuffer | None  — pass to draw motion trails
    show_ids : bool  — draw player ID label
    show_conf : bool  — append detection confidence to label
    """
    if trail_buffer is not None:
        trail_buffer.draw(frame)

    for t in tracks:
        x1, y1, x2, y2 = t.bbox_xyxy.astype(int)
        color = _id_color(t.track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        if show_ids:
            label = f"ID:{t.track_id}"
            if show_conf:
                label += f" {t.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Filled background pill behind text
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay current FPS in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def draw_track_count(frame: np.ndarray, count: int) -> np.ndarray:
    """Overlay active player count."""
    cv2.putText(
        frame, f"Players: {count}",
        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# 3. Configuration
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 4. CSV logging
# ──────────────────────────────────────────────────────────────────────────────

class DetectionLogger:
    """Appends one row per track per frame to a CSV for offline analysis."""

    HEADER = ["frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence"]

    def __init__(self, csv_path: str) -> None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADER)
        self._writer.writeheader()

    def log(self, frame_id: int, tracks: list[Track]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t.bbox_xyxy.astype(int)
            self._writer.writerow({
                "frame_id": frame_id,
                "track_id": t.track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": f"{t.confidence:.4f}",
            })

    def close(self) -> None:
        self._file.close()


# ──────────────────────────────────────────────────────────────────────────────
# 5. FPS meter
# ──────────────────────────────────────────────────────────────────────────────

class FPSMeter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        return 1.0 / (sum(self._times) / len(self._times)) if self._times else 0.0
