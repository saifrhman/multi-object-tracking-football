"""
detector.py
───────────
YOLOv8-based person detector.

Responsibilities:
  • Load a YOLOv8 model (downloads weights automatically on first run).
  • Run inference on a single frame or a batch of frames.
  • Return structured Detection objects: (bbox_xyxy, confidence, class_id).

Design notes:
  • The class is stateless between frames — all temporal logic lives in tracker.py.
  • `detect_batch` is provided for HPC/GPU throughput scenarios where you want to
    fill the GPU with multiple frames at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detected object in one frame."""
    bbox_xyxy: np.ndarray   # shape (4,): [x1, y1, x2, y2] in pixel coords
    confidence: float
    class_id: int


class PlayerDetector:
    """
    Wraps YOLOv8 for person detection.

    Parameters
    ----------
    weights : str
        Path to .pt weights file or an Ultralytics model name (e.g. 'yolov8m.pt').
        Ultralytics auto-downloads named models on first use.
    confidence : float
        Minimum detection confidence to return.
    iou : float
        NMS IoU threshold used inside YOLOv8.
    target_classes : list[int]
        COCO class IDs to keep. Default [0] = person only.
    device : str
        'cuda', 'cpu', or a specific GPU like 'cuda:0'.
    imgsz : int
        Inference image size (longest side). Must be a multiple of 32.
    """

    def __init__(
        self,
        weights: str = "yolov8m.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        target_classes: list[int] | None = None,
        device: str = "cuda",
        imgsz: int = 1280,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.target_classes = target_classes if target_classes is not None else [0]
        self.device = device
        self.imgsz = imgsz

        self.model = YOLO(weights)
        # Warm-up: pushes model to device and JIT-compiles the graph once.
        self.model.to(device)
        print(f"[Detector] Loaded {weights} on {device}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame (as returned by cv2.VideoCapture).

        Returns a list of Detection objects sorted by descending confidence.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.target_classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        return self._parse_results(results[0])

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        Run inference on a batch of frames in a single GPU forward pass.
        More efficient than calling detect() in a loop when you can buffer frames.

        Returns one list of Detection per input frame, in the same order.
        """
        results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou,
            classes=self.target_classes,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        return [self._parse_results(r) for r in results]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(result) -> list[Detection]:
        """Convert an Ultralytics Results object into Detection dataclasses."""
        detections: list[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # (N, 4)
        confidences = result.boxes.conf.cpu().numpy()  # (N,)
        class_ids   = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for bbox, conf, cls in zip(boxes_xyxy, confidences, class_ids):
            detections.append(Detection(bbox_xyxy=bbox, confidence=float(conf), class_id=cls))

        # Sort highest confidence first — helps tracker when there are many overlapping boxes
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
