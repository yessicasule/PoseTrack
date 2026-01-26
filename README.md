# Upper-Body Keypoint Tracking Using MediaPipe

2-Week Assignment: Real-time upper-body pose tracking from monocular camera using MediaPipe Pose Landmarker.

## Overview

This project implements real-time upper-body pose tracking using MediaPipe's Pose Landmarker API. It tracks key landmarks (head, shoulders, elbows, wrists) from a live webcam feed, records video with skeleton overlay, saves landmark data, and provides trajectory visualization.

## Features

- ✅ Real-time upper-body pose tracking using MediaPipe Pose Landmarker
- ✅ Tracks head, shoulders, elbows, and wrists
- ✅ Video recording with keypoint and skeleton overlay
- ✅ Landmark data export (CSV and JSON formats)
- ✅ FPS tracking and system specifications reporting
- ✅ Trajectory visualization for elbow joints

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MediaPipe Model

Download the Pose Landmarker model from:
- **Full Model**: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

Place the downloaded `.task` file in the `models/` directory:
```
models/
  └── pose_landmarker_full.task
```

Alternatively, you can use:
- **Lite Model**: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
- **Heavy Model**: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

## Usage

### 1. Run Pose Tracking

```bash
python pose_tracking.py
```

This will:
- Start webcam feed
- Track upper-body landmarks in real-time
- Record video for 60 seconds (configurable)
- Save landmark data to CSV and JSON
- Display FPS and system information

**Controls:**
- Press `q` to quit early

**Output Files:**
- `output/pose_tracking_YYYYMMDD_HHMMSS.mp4` - Video with skeleton overlay
- `output/landmarks_YYYYMMDD_HHMMSS.csv` - Landmark data in CSV format
- `output/landmarks_YYYYMMDD_HHMMSS.json` - Landmark data in JSON format
- `output/system_info_YYYYMMDD_HHMMSS.txt` - System specifications

### 2. Visualize Trajectory

After recording, visualize the elbow trajectory:

```bash
python visualize_trajectory.py --input output/landmarks_YYYYMMDD_HHMMSS.json --elbow left
```

Or using CSV:
```bash
python visualize_trajectory.py --input output/landmarks_YYYYMMDD_HHMMSS.csv --elbow right
```

**Options:**
- `--input`: Path to JSON or CSV file with landmark data
- `--elbow`: Which elbow to visualize (`left` or `right`, default: `left`)
- `--output`: Output directory for plots (default: same as input file)

**Output:**
- `{elbow}_elbow_trajectory.png` - 2D and 3D trajectory plots
- `{elbow}_elbow_timeseries.png` - Coordinate values over time

## Configuration

### Adjust Recording Duration

Edit `pose_tracking.py`:
```python
tracker = PoseTracker(
    model_path=str(model_path),
    record_duration=60  # Change this value (in seconds)
)
```

### Change Model

Update the model path in `pose_tracking.py`:
```python
model_path = Path('models/pose_landmarker_lite.task')  # or 'pose_landmarker_heavy.task'
```

## Data Format

### CSV Format

Each row contains:
- `frame_index`: Frame number
- `timestamp`: Time in seconds
- For each landmark (nose, left_shoulder, right_elbow, etc.):
  - `{landmark}_x`, `{landmark}_y`, `{landmark}_z`: Normalized coordinates (0-1)
  - `{landmark}_visibility`: Visibility score (0-1)
  - `{landmark}_world_x`, `{landmark}_world_y`, `{landmark}_world_z`: 3D world coordinates

### JSON Format

Structured format with metadata and frame-by-frame landmark data:
```json
{
  "metadata": {
    "total_frames": 1800,
    "duration_seconds": 60,
    "average_fps": 30.0,
    "upper_body_landmarks": [...],
    "timestamp": "..."
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "landmarks": {
        "left_elbow": {
          "x": 0.5,
          "y": 0.6,
          "z": -0.1,
          "visibility": 0.9,
          "world_x": 0.2,
          "world_y": 0.3,
          "world_z": -0.5
        },
        ...
      }
    },
    ...
  ]
}
```

## Tracked Landmarks

The following upper-body landmarks are tracked:

- **Head**: nose, eyes, ears, mouth
- **Shoulders**: left_shoulder, right_shoulder
- **Elbows**: left_elbow, right_elbow
- **Wrists**: left_wrist, right_wrist

See MediaPipe documentation for complete landmark list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

## System Requirements

- Python 3.8+
- Webcam
- Windows/Linux/macOS

## Troubleshooting

### Camera Not Accessing
- Ensure no other application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

### Model Not Found
- Download the model file and place it in `models/` directory
- Check the file path in `pose_tracking.py`

### Low FPS
- Use the lite model instead of full model
- Reduce video resolution
- Close other applications

## References

- MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- MediaPipe Python API: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

## License

This project is for educational purposes as part of a 2-week assignment.

