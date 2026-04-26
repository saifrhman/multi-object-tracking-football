"""
download_video.py
─────────────────
Downloads a football match clip from YouTube using yt-dlp and saves it
as an MP4 ready for the tracking pipeline.

Usage:
    python download_video.py --url "https://www.youtube.com/watch?v=XXXXX" \
                             --out data/input/match.mp4 \
                             --quality 1080

Recommended video characteristics for best tracking results:
  • Broadcast or wide-angle camera (not close-up replays)
  • Clear pitch-level view — avoid bird's-eye if players appear as dots
  • Steady camera (not heavy panning / zoom cuts)
  • 720p or 1080p resolution
  • At least 30 fps
"""

import argparse
import os
import subprocess
import sys


def download(url: str, output_path: str, max_height: int = 1080) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # yt-dlp format string: best video+audio up to max_height, remux to mp4
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+"
        f"bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best"
    )

    cmd = [
        "yt-dlp",
        "--format", fmt,
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
        url,
    ]

    print(f"[download_video] Downloading: {url}")
    print(f"[download_video] Target path: {output_path}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("[download_video] ERROR: yt-dlp failed. Check the URL or install yt-dlp.")
        sys.exit(1)

    print(f"[download_video] Saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a YouTube football clip.")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--out", default="data/input/match.mp4", help="Output file path"
    )
    parser.add_argument(
        "--quality", type=int, default=1080, help="Max video height (e.g. 720, 1080)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.url, args.out, args.quality)
