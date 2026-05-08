"""
Post-processing benchmark script for pose estimation frameworks.

This script processes recorded video with multiple pose estimation
frameworks for consistent benchmarking.

Usage:
    python run_benchmark.py --session my_session
    python run_benchmark.py --session my_session --processors mediapipe_c0 mediapipe_c1
    python run_benchmark.py --session my_session --video custom_video.mp4
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_processor import run_benchmark_on_session
from src.benchmark_comparator import generate_comparison_report
from config.config import Config


def list_sessions():
    """List available collection sessions."""
    if not os.path.exists(Config.COLLECTION_DIR):
        print("No collection sessions found.")
        return []
    
    sessions = []
    for item in os.listdir(Config.COLLECTION_DIR):
        item_path = os.path.join(Config.COLLECTION_DIR, item)
        if os.path.isdir(item_path):
            video_path = os.path.join(item_path, "video")
            if os.path.exists(video_path):
                videos = [f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi'))]
                if videos:
                    sessions.append({
                        'name': item,
                        'video': os.path.join(video_path, videos[0]),
                        'path': item_path
                    })
    
    return sessions


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Post-processing benchmark for pose estimation frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available sessions
    python run_benchmark.py --list
    
    # Run benchmark on a session (uses default processors)
    python run_benchmark.py --session 20240101_120000
    
    # Run with specific processors
    python run_benchmark.py --session test --processors mediapipe_c1 mediapipe_c2
    
    # Generate comparison report after benchmark
    python run_benchmark.py --session test --report
        """
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available collection sessions')
    parser.add_argument('--session', '-s', type=str, default=None,
                       help='Collection session name to benchmark')
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Custom video path (overrides session video)')
    parser.add_argument('--processors', '-p', nargs='+',
                       choices=['mediapipe_c0', 'mediapipe_c1', 'mediapipe_c2', 
                                'mediapipe_task', 'posenet_mobilenet_v1', 
                                'movenet_lightning', 'movenet_thunder',
                                'opencv_dnn'],
                       help='Processors to benchmark')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate comparison report after benchmark')
    
    args = parser.parse_args()
    
    if args.list:
        sessions = list_sessions()
        print("\nAvailable Collection Sessions:")
        print("-" * 70)
        if not sessions:
            print("No sessions found. Run collect_data.py first.")
        else:
            for s in sessions:
                print(f"  {s['name']}")
                print(f"    Video: {s['video']}")
                print()
        return
    
    if not args.session:
        print("Error: --session is required (or use --list to see available sessions)")
        parser.print_help()
        return
    
    session_path = os.path.join(Config.COLLECTION_DIR, args.session)
    if not os.path.exists(session_path):
        print(f"Error: Session '{args.session}' not found.")
        print("Use --list to see available sessions.")
        return
    
    video_path = args.video
    if not video_path:
        video_dir = os.path.join(session_path, "video")
        if os.path.exists(video_dir):
            videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
            if videos:
                video_path = os.path.join(video_dir, videos[0])
    
    if not video_path or not os.path.exists(video_path):
        print(f"Error: No video found for session '{args.session}'.")
        return
    
    print("\n" + "="*70)
    print("POSE ESTIMATION FRAMEWORK BENCHMARK")
    print("="*70)
    print(f"\nSession:     {args.session}")
    print(f"Video:      {video_path}")
    print(f"Processors: {args.processors or 'all enabled'}")
    print(f"Output:     {Config.get_benchmark_session_dir(args.session)}")
    print("="*70 + "\n")
    
    try:
        results, timings = run_benchmark_on_session(
            args.session,
            video_path,
            args.processors
        )
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        
        if args.report:
            print("\nGenerating comparison report...")
            report_path = generate_comparison_report(args.session)
            print(f"Report saved to: {report_path}")
        
        print("\n" + "="*70)
        print("BENCHMARKING COMPLETE")
        print("="*70)
        print(f"\nResults: {Config.get_benchmark_session_dir(args.session)}")
        print(f"\nView visualizations in: {Config.PLOT_OUTPUT_DIR}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
