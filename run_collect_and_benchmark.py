"""
Complete workflow: Synchronized data collection + Benchmark processing

This script performs two main steps:
1. Collect synchronized video and motion capture data
2. Post-process the video with MediaPipe Pose, PoseNet, and MoveNet for benchmarking

Usage:
    python run_collect_and_benchmark.py
    python run_collect_and_benchmark.py --session my_session --duration 30
    python run_collect_and_benchmark.py --benchmark-only --session existing_session
"""
import sys
import os
import cv2
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sync_data_collector import SynchronizedRecorder
from src.benchmark_processor import (
    BenchmarkProcessor, MediaPipePoseProcessor, 
    PoseNetProcessor, MoveNetProcessor
)
from config.config import Config


def collect_synchronized_data(session_name=None, duration=None, mocap_enabled=False):
    """
    Step 1: Collect synchronized video and motion capture data.
    
    Args:
        session_name: Optional session name (auto-generated if None)
        duration: Maximum recording duration in seconds
        mocap_enabled: Enable motion capture
        
    Returns:
        dict: Session info including video path and metadata
    """
    print("\n" + "="*70)
    print("STEP 1: SYNCHRONIZED DATA COLLECTION")
    print("="*70)
    
    if session_name is None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    Config.create_output_directories()
    
    recorder = SynchronizedRecorder(session_name=session_name, mocap_enabled=mocap_enabled)
    
    try:
        width, height, fps = recorder.start_camera()
        
        print(f"\nRecording session: {session_name}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print("\nControls:")
        print("  SPACE - Start/Stop recording")
        print("  q     - Quit")
        print("="*70)
        
        recording = False
        
        while True:
            ret, frame = recorder.cap.read()
            if not ret:
                print("Camera error")
                break
            
            display = frame.copy()
            
            if recording:
                cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(display, "REC", (50, 38), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                elapsed = time.perf_counter() - recorder.start_time
                cv2.putText(display, f"Time: {elapsed:.1f}s", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Frames: {recorder.frame_count}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if duration and elapsed >= duration:
                    print(f"\nDuration limit ({duration}s) reached")
                    break
            
            cv2.putText(display, "Synchronized Data Collection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if mocap_enabled:
                status = "MOCAP CONNECTED" if recorder.mocap_receiver and recorder.mocap_receiver.running else "MOCAP OFFLINE"
                color = (0, 255, 0) if "CONNECTED" in status else (0, 0, 255)
                cv2.putText(display, status, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Data Collection", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and not recording:
                recorder.start_recording(width, height, fps)
                recording = True
            elif key == ord(' ') and recording:
                break
        
        result = None
        if recording:
            result = recorder.stop_recording()
        
    finally:
        recorder.cleanup()
    
    session_info = {
        'session_name': session_name,
        'video_path': recorder.video_path,
        'sync_dir': recorder.sync_dir,
        'frame_count': recorder.frame_count,
        'mocap_enabled': mocap_enabled
    }
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE")
    print("="*70)
    print(f"Session: {session_name}")
    print(f"Video:   {session_info['video_path']}")
    print(f"Frames:  {session_info['frame_count']}")
    print("="*70)
    
    return session_info


def run_benchmark(session_info, frameworks=None):
    """
    Step 2: Post-process video with multiple pose estimation frameworks.
    
    Args:
        session_info: Session info from data collection
        frameworks: List of frameworks to run (default: all three)
        
    Returns:
        dict: Benchmark results
    """
    print("\n" + "="*70)
    print("STEP 2: POST-PROCESSING BENCHMARK")
    print("="*70)
    
    session_name = session_info['session_name']
    video_path = session_info['video_path']
    
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    if frameworks is None:
        frameworks = ['mediapipe', 'posenet', 'movenet']
    
    print(f"\nSession:    {session_name}")
    print(f"Video:      {video_path}")
    print(f"Frameworks: {frameworks}")
    print("="*70)
    
    processor = BenchmarkProcessor(session_name, video_path)
    
    for fw in frameworks:
        if fw == 'mediapipe':
            processor.add_processor(MediaPipePoseProcessor())
            print("  Added: MediaPipe Pose")
        elif fw == 'posenet':
            processor.add_processor(PoseNetProcessor())
            print("  Added: PoseNet")
        elif fw == 'movenet':
            processor.add_processor(MoveNetProcessor())
            print("  Added: MoveNet")
    
    results, timings = processor.run()
    
    benchmark_info = {
        'session_name': session_name,
        'video_path': video_path,
        'frameworks': frameworks,
        'results_dir': processor.output_dir
    }
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"Results saved to: {processor.output_dir}")
    print("="*70)
    
    return benchmark_info


def list_available_sessions():
    """List available collection sessions."""
    if not os.path.exists(Config.COLLECTION_DIR):
        print("No collection sessions found.")
        return []
    
    sessions = []
    for item in os.listdir(Config.COLLECTION_DIR):
        item_path = os.path.join(Config.COLLECTION_DIR, item)
        if os.path.isdir(item_path):
            video_path = os.path.join(item_path, "video")
            sync_path = os.path.join(item_path, "sync")
            
            if os.path.exists(video_path):
                videos = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
                if videos:
                    metadata = {}
                    meta_path = os.path.join(sync_path, "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            metadata = json.load(f)
                    
                    sessions.append({
                        'name': item,
                        'video': os.path.join(video_path, videos[0]),
                        'path': item_path,
                        'metadata': metadata
                    })
    
    return sessions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect synchronized data and run pose estimation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full workflow (collection + benchmark)
    python run_collect_and_benchmark.py
    
    # Run with specific session name and duration
    python run_collect_and_benchmark.py --session test_session --duration 30
    
    # Run benchmark only on existing session
    python run_collect_and_benchmark.py --benchmark-only --session test_session
    
    # List available sessions
    python run_collect_and_benchmark.py --list
    
    # Use specific frameworks
    python run_collect_and_benchmark.py --frameworks mediapipe movenet
        """
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available collection sessions')
    parser.add_argument('--session', '-s', type=str, default=None,
                       help='Session name')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Recording duration in seconds')
    parser.add_argument('--mocap', '-m', action='store_true',
                       help='Enable motion capture')
    parser.add_argument('--benchmark-only', '-b', action='store_true',
                       help='Skip data collection, run benchmark only')
    parser.add_argument('--frameworks', '-f', nargs='+',
                       choices=['mediapipe', 'posenet', 'movenet'],
                       default=['mediapipe', 'posenet', 'movenet'],
                       help='Frameworks to benchmark')
    
    args = parser.parse_args()
    
    if args.list:
        sessions = list_available_sessions()
        print("\nAvailable Sessions:")
        print("-" * 70)
        if not sessions:
            print("No sessions found.")
        else:
            for s in sessions:
                print(f"  {s['name']}")
                print(f"    Video: {s['video']}")
                if s['metadata']:
                    print(f"    Date: {s['metadata'].get('collection_time', 'N/A')}")
                print()
        return
    
    session_info = None
    
    if args.benchmark_only:
        if not args.session:
            print("Error: --session required for benchmark-only mode")
            return
        
        session_path = os.path.join(Config.COLLECTION_DIR, args.session)
        if not os.path.exists(session_path):
            print(f"Error: Session '{args.session}' not found")
            print("Use --list to see available sessions")
            return
        
        video_dir = os.path.join(session_path, "video")
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        if not videos:
            print(f"Error: No video found for session '{args.session}'")
            return
        
        session_info = {
            'session_name': args.session,
            'video_path': os.path.join(video_dir, videos[0]),
            'sync_dir': os.path.join(session_path, "sync"),
            'frame_count': 0,
            'mocap_enabled': False
        }
        
        meta_path = os.path.join(session_info['sync_dir'], "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                session_info['metadata'] = json.load(f)
    else:
        session_info = collect_synchronized_data(
            session_name=args.session,
            duration=args.duration,
            mocap_enabled=args.mocap
        )
    
    benchmark_info = run_benchmark(session_info, args.frameworks)
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nCollection: {session_info['session_name']}")
    print(f"Video:      {session_info['video_path']}")
    print(f"Frameworks: {args.frameworks}")
    print(f"Results:    {benchmark_info['results_dir']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
