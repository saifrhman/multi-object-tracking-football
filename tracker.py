"""
tracker.py
──────────
DeepSORT-based multi-object tracker.

Responsibilities:
  • Accept per-frame detections from detector.py.
  • Maintain consistent track IDs across frames using Kalman filtering +
    Hungarian algorithm + appearance-based Re-ID matching.
  • Return Track objects: (track_id, bbox_xyxy, is_confirmed).

DeepSORT internals (brief):
  1. Kalman filter predicts where each existing track will be next frame.
  2. Appearance extractor (MobileNet CNN) produces a 128-d embedding per crop.
  3. Hungarian algorithm matches predictions → detections using a combined
     Mahalanobis (motion) + cosine (appearance) cost matrix.
  4. Unmatched detections become tentative new tracks.
  5. Tracks with n_init consecutive hits are "confirmed" and shown to the user.
  6. Tracks unseen for max_age frames are deleted.

We use deep_sort_realtime which bundles the Re-ID model so no separate
download step is needed beyond `pip install deep-sort-realtime`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from detector import Detection


@dataclass
class Track:
    """A confirmed track for one frame."""
    track_id: int
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2]
    confidence: float
    is_confirmed: bool


class PlayerTracker:
    """
    Wraps DeepSORT for football player tracking.

    Parameters
    ----------
    max_age : int
        Frames to keep a track alive without a matching detection.
        Increase for handling longer occlusions (e.g. player behind a wall).
    n_init : int
        Consecutive hits required before a track is promoted to confirmed.
        Higher = fewer false positives, higher = slower ID assignment.
    max_cosine_distance : float
        Threshold for appearance embedding similarity.
        Lower = stricter matching (fewer ID switches, more track fragments).
    nms_max_overlap : float
        NMS overlap threshold for detections passed to DeepSORT.
    min_confidence : float
        Detections below this are discarded before tracking.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nms_max_overlap: float = 1.0,
        min_confidence: float = 0.35,
    ) -> None:
        self.min_confidence = min_confidence

        # DeepSort constructor — embedder="mobilenet" uses its built-in Re-ID CNN.
        # Pass embedder=None and embedder_model_name to use a custom Re-ID model.
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nms_max_overlap=nms_max_overlap,
            embedder="mobilenet",
            half=True,          # FP16 Re-ID inference — halves GPU memory usage
            bgr=True,           # OpenCV frames are BGR
        )
        print("[Tracker] DeepSORT initialised (embedder=mobilenet, half=True)")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """
        Feed one frame's detections into DeepSORT and return current tracks.

        Parameters
        ----------
        detections : list[Detection]
            Output of PlayerDetector.detect() for this frame.
        frame : np.ndarray
            The original BGR frame — used by the Re-ID CNN to extract crops.

        Returns
        -------
        list[Track]
            Only confirmed tracks are returned (tentative tracks are hidden).
        """
        # DeepSort expects: list of ([x1, y1, w, h], confidence, class_id)
        raw_detections = [
            (self._xyxy_to_xywh(d.bbox_xyxy), d.confidence, d.class_id)
            for d in detections
            if d.confidence >= self.min_confidence
        ]

        ds_tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        tracks: list[Track] = []
        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()   # [x1, y1, x2, y2]
            tracks.append(
                Track(
                    # deep_sort_realtime returns track_id as str in some versions;
                    # normalise to int here so downstream code can do arithmetic on it.
                    track_id=int(t.track_id),
                    bbox_xyxy=np.array(ltrb, dtype=np.float32),
                    confidence=t.det_conf if t.det_conf is not None else 0.0,
                    is_confirmed=True,
                )
            )

        return tracks

    def reset(self) -> None:
        """Clear all track state (useful for processing multiple clips)."""
        self._tracker.delete_all_tracks()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _xyxy_to_xywh(bbox: np.ndarray) -> list[float]:
        """Convert [x1, y1, x2, y2] → [x1, y1, width, height]."""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
