# Football Player Detection and Tracking

Real-time football player detection and tracking using **YOLOv8** (detection) and **DeepSORT** (tracking) on actual match video footage.

---

## Pipeline

```text
YouTube Video
     │
     ▼
yt-dlp download
     │
     ▼
OpenCV frame reader  →  resize / normalise
     │
     ▼
YOLOv8 detector      →  [bbox, confidence, class_id] per frame
     │
     ▼
DeepSORT tracker     →  [bbox, track_id] per frame  (persistent IDs)
     │
     ▼
draw_tracks()        →  annotated frame
     │
     ▼
OpenCV VideoWriter   →  output.mp4 + detections.csv
```

**YOLOv8** runs per-frame and outputs raw bounding boxes — it has no memory.
**DeepSORT** maintains Kalman-filter state per player and uses Hungarian matching + a MobileNet Re-ID appearance embedding to link detections across frames, surviving short occlusions.

---

## Project Structure

```text
├── main.py              Entry point — orchestrates the pipeline
├── detector.py          YOLOv8 wrapper — detection only
├── tracker.py           DeepSORT wrapper — tracking only
├── utils.py             Video I/O, drawing, CSV logging, FPS meter
├── download_video.py    yt-dlp helper — download YouTube footage
├── config.yaml          All tunable parameters in one place
├── requirements.txt     Python dependencies
├── data/
│   ├── input/           Place input videos here
│   └── output/          Annotated videos and CSVs land here
└── models/              Optional: custom Re-ID model weights
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download a football clip

```bash
python download_video.py \
  --url "https://www.youtube.com/watch?v=XXXXX" \
  --out data/input/match.mp4 \
  --quality 1080
```

Best video types for tracking:

- Broadcast wide-angle camera (full pitch view)
- Clear player visibility, stable camera
- 720p–1080p, 25–60 fps

### 3. Run the pipeline

```bash
# Default (reads paths from config.yaml)
python main.py

# With live preview + custom paths
python main.py --input data/input/match.mp4 --output data/output/tracked.mp4 --preview

# Quick test on first 300 frames
python main.py --max-frames 300 --preview
```

---

## Configuration

All parameters live in `config.yaml`. Key ones:

| Section   | Key                   | Effect                                                        |
| --------- | --------------------- | ------------------------------------------------------------- |
| `video`   | `frame_skip`          | Process every Nth frame (2 = 2× speed, half temporal resolution) |
| `model`   | `weights`             | `yolov8n.pt` (fast) → `yolov8x.pt` (accurate)               |
| `model`   | `confidence`          | Lower = more detections, more false positives                 |
| `model`   | `device`              | `cuda` or `cpu`                                               |
| `tracker` | `max_age`             | Frames to keep a track alive during occlusion                 |
| `tracker` | `n_init`              | Hits before a track is shown (higher = fewer ghost tracks)    |
| `tracker` | `max_cosine_distance` | Re-ID matching strictness (lower = stricter)                  |

---

## GPU / HPC Usage

YOLOv8 automatically uses the GPU set in `config.yaml → model.device`.

For **HPC batch jobs** (SLURM example):

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
python main.py --input data/input/match.mp4 --max-frames 0
```

For **higher throughput**, set `frame_skip: 2` (halves frames processed) or use `yolov8n.pt` for faster inference. The `detect_batch()` method in `detector.py` supports multi-frame batching for pure-inference benchmarking.

---

## Metrics

### Detection (requires labelled ground truth)

| Metric      | What it measures                                            |
| ----------- | ----------------------------------------------------------- |
| **mAP@0.5** | Mean Average Precision at 50% IoU — overall detection quality |
| **IoU**     | Overlap between predicted and ground-truth box              |

### Tracking (requires labelled ground truth)

| Metric          | What it measures                                       |
| --------------- | ------------------------------------------------------ |
| **MOTA**        | Multi-Object Tracking Accuracy — combines FP, FN, ID switches |
| **ID switches** | How often a player gets a new ID (lower = better Re-ID) |

### Without ground truth

- Watch the output video and count visible ID switches manually over a 30-second clip.
- Check the `detections.csv`: plot track lifetimes — long-lived tracks (many frames) indicate stable tracking.
- Count average active tracks per frame vs. visible players.

---

## Optional Extensions

| Feature                  | Approach                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| **Trajectory heatmap**   | Accumulate all centroids from `detections.csv` into a 2D histogram; render with `cv2.applyColorMap` |
| **Speed estimation**     | Homography-map pixel centroids to real-world pitch coordinates; differentiate positions over time |
| **Team classification**  | K-means cluster dominant jersey HSV colours within each bounding box crop; assign team label per track |
| **Formation analysis**   | Voronoi tesselation of player positions at key frames                                     |
