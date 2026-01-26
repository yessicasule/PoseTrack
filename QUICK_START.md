# Quick Start Guide

## Complete Workflow in 4 Steps

### Step 1: Test Camera
```bash
python test_camera.py
```
- Press 'q' to quit when done

### Step 2: Record Session
```bash
python main.py
```
**Controls:**
- Press **SPACE** to start recording
- Move your arms around (30-60 seconds)
- Press **SPACE** to stop recording
- Press 'q' to quit

**Output:**
- Video saved to `outputs/videos/`
- Data saved to `outputs/data/`

### Step 3: Analyze Session
```bash
python analyze_session.py
```
- Generates trajectory plots
- Saves to `outputs/plots/`

### Step 4: Generate Report
```bash
python generate_report.py
```
- Creates text report
- Saves to `outputs/data/REPORT_*.txt`

## Automated Workflow

Run all steps automatically:
```bash
python run_complete_workflow.py
```

## What Gets Created

After running the complete workflow:

1. **Video File**: `outputs/videos/pose_tracking_*.mp4`
   - 30-60 second video with skeleton overlay

2. **Data Files**: `outputs/data/`
   - `landmarks_*.csv` - Coordinate data
   - `landmarks_*.json` - Structured data
   - `metadata_*.json` - Session metadata
   - `REPORT_*.txt` - Complete report

3. **Plots**: `outputs/plots/`
   - Left elbow trajectory
   - Right elbow trajectory
   - Upper-body summary

## Troubleshooting

**Camera not working?**
```bash
python test_camera.py
```

**No pose detected?**
- Stand in front of camera
- Ensure good lighting
- Face the camera directly

**Low FPS?**
- Edit `config/config.py`
- Set `MODEL_COMPLEXITY = 0` (lite model)
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
