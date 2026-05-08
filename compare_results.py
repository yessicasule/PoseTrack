"""
Comparison results script for benchmark analysis.

This script generates comparison reports and visualizations
for benchmark results across multiple frameworks.

Usage:
    python compare_results.py --session my_session
    python compare_results.py --session my_session --format markdown
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_comparator import generate_comparison_report, BenchmarkComparator
from config.config import Config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare and visualize benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate markdown report with visualizations
    python compare_results.py --session 20240101_120000
    
    # Generate JSON report
    python compare_results.py --session test --format json
    
    # List available benchmark sessions
    python compare_results.py --list
        """
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available benchmark sessions')
    parser.add_argument('--session', '-s', type=str, default=None,
                       help='Benchmark session name')
    parser.add_argument('--format', '-f', choices=['json', 'markdown'],
                       default='markdown', help='Report format')
    parser.add_argument('--keypoint', '-k', type=int, default=0,
                       help='Keypoint index for detailed analysis (default: 0=nose)')
    
    args = parser.parse_args()
    
    if args.list:
        list_benchmark_sessions()
        return
    
    if not args.session:
        print("Error: --session is required (or use --list to see available sessions)")
        parser.parse_args(['--help'])
        return
    
    session_path = Config.get_benchmark_session_dir(args.session)
    if not os.path.exists(session_path):
        print(f"Error: Benchmark session '{args.session}' not found.")
        print("Run run_benchmark.py first.")
        return
    
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON REPORT")
    print("="*70)
    print(f"\nSession: {args.session}")
    print(f"Format:  {args.format}")
    print(f"Output: {os.path.join(Config.PLOT_OUTPUT_DIR, args.session)}")
    print("="*70 + "\n")
    
    comparator = BenchmarkComparator(args.session)
    
    print("Computing statistics...")
    stats = comparator.compute_processing_statistics()
    
    if not stats:
        print("No benchmark results found.")
        return
    
    print("\n" + "-"*70)
    print("PROCESSING TIME COMPARISON")
    print("-"*70)
    print(f"{'Framework':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Est FPS':<10}")
    print("-"*70)
    
    for name, stat in sorted(stats.items(), key=lambda x: x[1]['mean_time']):
        print(f"{name:<25} {stat['mean_time']:<12.2f} {stat['std_time']:<12.2f} {stat['estimated_fps']:<10.1f}")
    
    print("\n" + "-"*70)
    print("DETECTION RATE COMPARISON")
    print("-"*70)
    print(f"{'Framework':<25} {'Detections':<12} {'Total':<10} {'Rate':<10}")
    print("-"*70)
    
    for name, stat in sorted(stats.items(), key=lambda x: x[1]['detection_rate'], reverse=True):
        rate = stat['detection_rate'] * 100
        print(f"{name:<25} {stat['detections']:<12} {stat['total_frames']:<10} {rate:<10.1f}%")
    
    if args.keypoint is not None:
        keypoint_names = Config.LANDMARK_NAMES
        keypoint_name = keypoint_names.get(args.keypoint, f"keypoint_{args.keypoint}")
        
        print(f"\n" + "-"*70)
        print(f"KEYPOINT COMPARISON: {keypoint_name.upper()} (idx={args.keypoint})")
        print("-"*70)
        
        pck = comparator.compute_pck(args.keypoint)
        if pck:
            print(f"{'Framework':<25} {'PCK':<10}")
            print("-"*70)
            for name, rate in sorted(pck.items(), key=lambda x: x[1], reverse=True):
                print(f"{name:<25} {rate*100:<10.1f}%")
    
    print("\n" + "-"*70)
    print("Generating visualizations...")
    print("-"*70)
    
    report_path = generate_comparison_report(args.session, args.format)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nReport: {report_path}")
    print(f"Plots:  {os.path.join(Config.PLOT_OUTPUT_DIR, args.session)}")
    print("="*70 + "\n")


def list_benchmark_sessions():
    """List available benchmark sessions."""
    if not os.path.exists(Config.BENCHMARK_DIR):
        print("No benchmark sessions found.")
        return []
    
    sessions = []
    for item in os.listdir(Config.BENCHMARK_DIR):
        item_path = os.path.join(Config.BENCHMARK_DIR, item)
        if os.path.isdir(item_path):
            summary_path = os.path.join(item_path, "benchmark_summary.json")
            if os.path.exists(summary_path):
                sessions.append({
                    'name': item,
                    'summary': summary_path,
                    'path': item_path
                })
    
    print("\nAvailable Benchmark Sessions:")
    print("-" * 70)
    if not sessions:
        print("No sessions found. Run run_benchmark.py first.")
    else:
        for s in sessions:
            print(f"  {s['name']}")
            print(f"    Path: {s['path']}")
            print()
    
    return sessions


if __name__ == "__main__":
    main()
