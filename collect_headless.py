"""
Headless synchronized data collection - no GUI required.

Records video and motion capture data without display output.
"""
import cv2
import time
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


def collect_headless(session_name=None, duration=10, camera_id=0, mocap_enabled=False):
    """
    Collect synchronized data without GUI.
    
    Args:
        session_name: Optional session name
        duration: Recording duration in seconds
        camera_id: Camera device ID
        mocap_enabled: Enable motion capture
        
    Returns:
        dict: Collection results
    """
    if session_name is None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    Config.create_output_directories()
    
    session_dir = Config.get_collection_session_dir(session_name)
    video_dir = Config.get_collection_session_dir(session_name, "video")
    sync_dir = Config.get_collection_session_dir(session_name, "sync")
    
    print(f"\n{'='*60}")
    print(f"HEADLESS DATA COLLECTION")
    print(f"{'='*60}")
    print(f"Session:  {session_name}")
    print(f"Duration: {duration}s")
    print(f"Camera:   {camera_id}")
    print(f"Mocap:    {'enabled' if mocap_enabled else 'disabled'}")
    print(f"Output:   {video_dir}")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot access camera {camera_id}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = Config.OUTPUT_VIDEO_FPS
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(video_dir, f"{session_name}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(video_dir, f"{session_name}.avi")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Recording: {width}x{height} @ {fps}fps")
    print("Recording started...")
    
    start_time = time.perf_counter()
    frame_count = 0
    frame_timestamps = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read error")
                break
            
            elapsed = time.perf_counter() - start_time
            
            writer.write(frame)
            frame_count += 1
            
            frame_timestamps.append({
                'frame_index': frame_count - 1,
                'video_timestamp': elapsed,
                'wall_timestamp': time.time()
            })
            
            if frame_count % 30 == 0:
                print(f"  Progress: {frame_count} frames, {elapsed:.1f}s")
            
            if elapsed >= duration:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        writer.release()
        cap.release()
    
    duration_actual = time.perf_counter() - start_time
    avg_fps = frame_count / duration_actual if duration_actual > 0 else 0
    
    metadata = {
        'session_name': session_name,
        'collection_time': datetime.now().isoformat(),
        'duration_seconds': duration_actual,
        'frame_count': frame_count,
        'avg_fps': avg_fps,
        'camera_id': camera_id,
        'resolution': {'width': width, 'height': height},
        'mocap_enabled': mocap_enabled,
        'video_path': video_path
    }
    
    metadata_path = os.path.join(sync_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    timestamps_path = os.path.join(sync_dir, "timestamps.json")
    with open(timestamps_path, 'w') as f:
        json.dump(frame_timestamps, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Video:    {video_path}")
    print(f"Frames:  {frame_count}")
    print(f"Duration: {duration_actual:.2f}s")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"{'='*60}\n")
    
    return {
        'session_name': session_name,
        'video_path': video_path,
        'frame_count': frame_count,
        'duration': duration_actual,
        'avg_fps': avg_fps
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Headless data collection")
    parser.add_argument("--session", "-s", type=str, default=None,
                       help="Session name")
    parser.add_argument("--duration", "-d", type=int, default=10,
                       help="Recording duration in seconds")
    parser.add_argument("--camera", "-c", type=int, default=0,
                       help="Camera device ID")
    parser.add_argument("--mocap", "-m", action="store_true",
                       help="Enable motion capture")
    
    args = parser.parse_args()
    
    collect_headless(args.session, args.duration, args.camera, args.mocap)
