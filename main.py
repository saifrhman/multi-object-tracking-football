"""
main.py
───────
Entry point for the football player detection and tracking pipeline.

Can be used two ways:

  1. CLI (local / HPC):
       python main.py --input data/input/match.mp4 --output data/output/tracked.mp4
       python main.py --preview          # live window (local only)
       python main.py --max-frames 300   # quick smoke test

  2. Imported from a Colab notebook (or any Python script):
       from main import run_pipeline
       run_pipeline(input_path='data/input/match.mp4', max_frames=300)

Pipeline flow:
    Video → frames → YOLOv8 detect → DeepSORT track → annotate → write output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from detector import PlayerDetector
from tracker import PlayerTracker
from utils import (
    DetectionLogger,
    FPSMeter,
    TrailBuffer,
    draw_fps,
    draw_track_count,
    draw_tracks,
    frame_generator,
    load_config,
    open_video,
    open_writer,
)


# ──────────────────────────────────────────────────────────────────────────────
# Environment detection
# ──────────────────────────────────────────────────────────────────────────────

def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline  (importable — used by both CLI and Colab notebook)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str = "config.yaml",
    max_frames: int = 0,
    preview: bool = False,
) -> None:
    """
    Run the full detection + tracking pipeline.

    Parameters
    ----------
    input_path : str | None
        Path to the input video. Falls back to config value if None.
    output_path : str | None
        Path for the annotated output video. Falls back to config value if None.
    config_path : str
        Path to config.yaml.
    max_frames : int
        Stop after this many frames (0 = process entire video).
    preview : bool
        Show a live OpenCV window. Automatically disabled in Colab.
    """
    cfg = load_config(config_path)

    input_path  = input_path  or cfg["video"]["input_path"]
    output_path = output_path or cfg["video"]["output_path"]

    # Colab has no display server — silently disable preview there
    can_preview = preview and not is_colab()
    if preview and not can_preview:
        print("[main] Note: live preview disabled in Colab (headless environment).")

    # ── Validate input ────────────────────────────────────────────────────────
    if not Path(input_path).exists():
        print(f"[main] ERROR: Input video not found: {input_path}")
        print(
            "  Download a football clip first:\n"
            "  python download_video.py --url <youtube_url> --out data/input/match.mp4\n"
            "  Or run the 'Download / Upload Video' cell in football_tracker.ipynb."
        )
        sys.exit(1)

    cap, meta = open_video(input_path)
    print(f"[main] Input : {input_path}")
    print(f"       {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps  "
          f"({meta['total_frames']} frames, {meta['duration_s']:.1f}s)")

    target_w   = cfg["video"]["frame_width"]
    target_h   = cfg["video"]["frame_height"]
    frame_skip = cfg["video"]["frame_skip"]
    out_fps    = cfg["output"]["fps"] or meta["fps"]

    # ── Build pipeline components ──────────────────────────────────────────────
    detector = PlayerDetector(
        weights=cfg["model"]["weights"],
        confidence=cfg["model"]["confidence"],
        iou=cfg["model"]["iou"],
        target_classes=cfg["model"]["target_classes"],
        device=cfg["model"]["device"],
        imgsz=cfg["model"]["imgsz"],
    )

    tracker = PlayerTracker(
        max_age=cfg["tracker"]["max_age"],
        n_init=cfg["tracker"]["n_init"],
        max_cosine_distance=cfg["tracker"]["max_cosine_distance"],
        nms_max_overlap=cfg["tracker"]["nms_max_overlap"],
        min_confidence=cfg["tracker"]["min_confidence"],
    )

    trail_buf = TrailBuffer(max_len=cfg["visualization"]["trail_length"])
    fps_meter = FPSMeter(window=30)

    # ── Open output ────────────────────────────────────────────────────────────
    writer = open_writer(output_path, out_fps, target_w, target_h, cfg["output"]["codec"])
    print(f"[main] Output: {output_path}  ({target_w}x{target_h} @ {out_fps:.1f} fps)")

    csv_logger: DetectionLogger | None = None
    if cfg["output"]["save_csv"]:
        csv_logger = DetectionLogger(cfg["output"]["csv_path"])
        print(f"[main] CSV   : {cfg['output']['csv_path']}")

    # ── Main loop ──────────────────────────────────────────────────────────────
    total = meta["total_frames"] // frame_skip
    if max_frames > 0:
        total = min(total, max_frames)

    pbar = tqdm(total=total, unit="frame", desc="Tracking")

    try:
        for frame_idx, frame in frame_generator(cap, target_w, target_h, frame_skip):
            if max_frames > 0 and frame_idx // frame_skip >= max_frames:
                break

            detections = detector.detect(frame)
            tracks     = tracker.update(detections, frame)

            trail_buf.update(tracks)

            vis_cfg   = cfg["visualization"]
            annotated = draw_tracks(
                frame.copy(),
                tracks,
                trail_buffer=trail_buf if vis_cfg["trail_length"] > 0 else None,
                show_ids=vis_cfg["show_ids"],
                box_thickness=vis_cfg["box_thickness"],
                font_scale=vis_cfg["font_scale"],
            )

            current_fps = fps_meter.tick()
            draw_fps(annotated, current_fps)
            draw_track_count(annotated, len(tracks))

            writer.write(annotated)

            if csv_logger:
                csv_logger.log(frame_idx, tracks)

            if can_preview:
                cv2.imshow("Football Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[main] Preview closed by user.")
                    break

            pbar.set_postfix({"tracks": len(tracks), "fps": f"{current_fps:.1f}"})
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()
        if csv_logger:
            csv_logger.close()
        if can_preview:
            cv2.destroyAllWindows()

    print(f"\n[main] Done. Output saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Football player tracking pipeline.")
    p.add_argument("--config",      default="config.yaml", help="Path to config.yaml")
    p.add_argument("--input",       default=None, help="Override video.input_path")
    p.add_argument("--output",      default=None, help="Override video.output_path")
    p.add_argument("--preview",     action="store_true", help="Show live preview (local only)")
    p.add_argument("--max-frames",  type=int, default=0,
                   help="Stop after N frames (0 = full video)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        max_frames=args.max_frames,
        preview=args.preview,
    )
