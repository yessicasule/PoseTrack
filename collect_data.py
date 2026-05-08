"""
Data collection script for synchronized video and motion capture.

This script provides a simple interface to collect synchronized data
for pose estimation benchmarking.

Usage:
    python collect_data.py --session my_session --duration 30 --mocap
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sync_data_collector import collect_session
from config.config import Config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Synchronized data collection for pose estimation benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect 60 seconds of video
    python collect_data.py --duration 60
    
    # Collect with custom session name
    python collect_data.py --session test_001 --duration 30
    
    # Collect with motion capture enabled
    python collect_data.py --session mocap_test --mocap
    
    # Manual stop (no duration limit)
    python collect_data.py --session manual_test
        """
    )
    
    parser.add_argument('--session', '-s', type=str, default=None,
                       help='Session name (auto-generated if not provided)')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Maximum recording duration in seconds')
    parser.add_argument('--mocap', '-m', action='store_true',
                       help='Enable motion capture synchronization')
    parser.add_argument('--camera', '-c', type=int, default=None,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=Config.FRAME_WIDTH,
                       help=f'Frame width (default: {Config.FRAME_WIDTH})')
    parser.add_argument('--height', type=int, default=Config.FRAME_HEIGHT,
                       help=f'Frame height (default: {Config.FRAME_HEIGHT})')
    
    args = parser.parse_args()
    
    Config.FRAME_WIDTH = args.width
    Config.FRAME_HEIGHT = args.height
    
    if args.camera is not None:
        Config.CAMERA_ID = args.camera
    
    print("\n" + "="*70)
    print("SYNCHRONIZED DATA COLLECTION FOR POSE ESTIMATION BENCHMARKING")
    print("="*70)
    print(f"\nSession:     {args.session or 'auto-generated'}")
    print(f"Duration:    {args.duration if args.duration else 'manual stop'}")
    print(f"Mocap:      {'enabled' if args.mocap else 'disabled'}")
    print(f"Camera:     {Config.CAMERA_ID}")
    print(f"Resolution: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
    print(f"\nOutput:     {Config.COLLECTION_DIR}")
    print("="*70 + "\n")
    
    result = collect_session(
        session_name=args.session,
        duration=args.duration,
        mocap_enabled=args.mocap
    )
    
    if result:
        print("\n" + "="*70)
        print("COLLECTION COMPLETE")
        print("="*70)
        print(f"Video file: {result['video_path']}")
        print(f"Frames:    {result['frame_count']}")
        print(f"Duration:  {result['duration']:.2f}s")
        print(f"Avg FPS:   {result['avg_fps']:.2f}")
        print("="*70 + "\n")
        
        print("Next steps:")
        print(f"  1. Run benchmark: python run_benchmark.py --session {args.session or 'SESSION_NAME'}")
        print(f"  2. Compare results: python compare_results.py --session {args.session or 'SESSION_NAME'}")


if __name__ == "__main__":
    main()
