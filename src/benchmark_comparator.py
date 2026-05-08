"""
Benchmark results comparison and visualization module.

Provides comprehensive analysis and visualization of pose estimation
benchmark results across multiple frameworks.
"""
import json
import os
import csv
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config.config import Config


class BenchmarkComparator:
    """
    Compare and analyze results from multiple pose estimation frameworks.
    """
    
    def __init__(self, session_name):
        """
        Initialize comparator for a benchmark session.
        
        Args:
            session_name: Name of the benchmark session
        """
        self.session_name = session_name
        self.benchmark_dir = Config.get_benchmark_session_dir(session_name)
        self.results = {}
        self.summary = None
        
        self._load_results()
    
    def _load_results(self):
        """Load all benchmark results from the session directory."""
        summary_path = os.path.join(self.benchmark_dir, "benchmark_summary.json")
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        
        for processor_dir in os.listdir(self.benchmark_dir):
            processor_path = os.path.join(self.benchmark_dir, processor_dir)
            
            if not os.path.isdir(processor_path):
                continue
            
            json_path = os.path.join(processor_path, f"{processor_dir}_full_results.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.results[processor_dir] = json.load(f)
    
    def compute_pck(self, keypoint_idx=0, threshold=0.5):
        """
        Compute Percentage of Correct Keypoints (PCK).
        
        Args:
            keypoint_idx: Index of keypoint to analyze
            threshold: Distance threshold (as fraction of torso length)
            
        Returns:
            dict: PCK values for each processor
        """
        pck_results = {}
        
        for name, data in self.results.items():
            frames = data.get('frames', [])
            correct = 0
            total = 0
            
            for frame in frames:
                result = frame.get('result')
                if result and 'landmarks' in result:
                    landmarks = result['landmarks']
                    if keypoint_idx in landmarks:
                        if landmarks[keypoint_idx].get('visibility', 0) > 0.5:
                            correct += 1
                    total += 1
            
            pck_results[name] = correct / total if total > 0 else 0
        
        return pck_results
    
    def compute_timeline_comparison(self, keypoint_idx=0):
        """
        Compare keypoint positions over time across processors.
        
        Args:
            keypoint_idx: Index of keypoint to track
            
        Returns:
            dict: Timeline data for each processor
        """
        timelines = {}
        
        for name, data in self.results.items():
            frames = data.get('frames', [])
            x_vals = []
            y_vals = []
            timestamps = []
            
            for frame in frames:
                result = frame.get('result')
                ts = frame.get('timestamp_ms', 0)
                
                if result and 'landmarks' in result:
                    landmarks = result['landmarks']
                    if keypoint_idx in landmarks:
                        lm = landmarks[keypoint_idx]
                        x_vals.append(lm.get('x', 0))
                        y_vals.append(lm.get('y', 0))
                        timestamps.append(ts)
                    else:
                        x_vals.append(np.nan)
                        y_vals.append(np.nan)
                        timestamps.append(ts)
                else:
                    x_vals.append(np.nan)
                    y_vals.append(np.nan)
                    timestamps.append(ts)
            
            timelines[name] = {
                'timestamps': timestamps,
                'x': x_vals,
                'y': y_vals
            }
        
        return timelines
    
    def compute_processing_statistics(self):
        """
        Compute detailed processing statistics for each processor.
        
        Returns:
            dict: Statistics for each processor
        """
        stats = defaultdict(dict)
        
        for name, data in self.results.items():
            frames = data.get('frames', [])
            times = [f.get('processing_time_ms', 0) for f in frames]
            
            detection_times = []
            no_detection_times = []
            
            for f in frames:
                t = f.get('processing_time_ms', 0)
                if f.get('result'):
                    detection_times.append(t)
                else:
                    no_detection_times.append(t)
            
            stats[name] = {
                'total_frames': len(frames),
                'detections': sum(1 for f in frames if f.get('result')),
                'detection_rate': sum(1 for f in frames if f.get('result')) / len(frames) if frames else 0,
                'times': times,
                'mean_time': np.mean(times) if times else 0,
                'std_time': np.std(times) if times else 0,
                'min_time': np.min(times) if times else 0,
                'max_time': np.max(times) if times else 0,
                'median_time': np.median(times) if times else 0,
                'p95_time': np.percentile(times, 95) if times else 0,
                'p99_time': np.percentile(times, 99) if times else 0,
                'mean_time_with_detection': np.mean(detection_times) if detection_times else 0,
                'mean_time_no_detection': np.mean(no_detection_times) if no_detection_times else 0,
                'estimated_fps': 1000 / np.mean(times) if times and np.mean(times) > 0 else 0
            }
        
        return stats
    
    def compute_inter_framework_difference(self, reference_processor, target_processor, keypoint_idx=0):
        """
        Compute difference between two processors for a specific keypoint.
        
        Args:
            reference_processor: Name of reference processor
            target_processor: Name of processor to compare
            keypoint_idx: Keypoint to analyze
            
        Returns:
            dict: Difference metrics
        """
        ref_data = self.results.get(reference_processor, {}).get('frames', [])
        target_data = self.results.get(target_processor, {}).get('frames', [])
        
        if not ref_data or not target_data:
            return None
        
        differences = []
        
        for ref_frame, target_frame in zip(ref_data, target_data):
            ref_result = ref_frame.get('result', {})
            target_result = target_frame.get('result', {})
            
            if ref_result and target_result:
                ref_landmarks = ref_result.get('landmarks', {})
                target_landmarks = target_result.get('landmarks', {})
                
                if keypoint_idx in ref_landmarks and keypoint_idx in target_landmarks:
                    ref_lm = ref_landmarks[keypoint_idx]
                    target_lm = target_landmarks[keypoint_idx]
                    
                    dx = target_lm.get('x', 0) - ref_lm.get('x', 0)
                    dy = target_lm.get('y', 0) - ref_lm.get('y', 0)
                    dz = target_lm.get('z', 0) - ref_lm.get('z', 0)
                    
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    differences.append({
                        'frame': ref_frame.get('frame_index'),
                        'dx': dx,
                        'dy': dy,
                        'dz': dz,
                        'distance': dist
                    })
        
        if not differences:
            return None
        
        distances = [d['distance'] for d in differences]
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'median_distance': np.median(distances),
            'differences': differences
        }


class BenchmarkVisualizer:
    """
    Create visualizations for benchmark comparison results.
    """
    
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def __init__(self, comparator, output_dir=None):
        """
        Initialize visualizer.
        
        Args:
            comparator: BenchmarkComparator instance
            output_dir: Directory for saving plots
        """
        self.comparator = comparator
        self.output_dir = output_dir or os.path.join(Config.PLOT_OUTPUT_DIR, comparator.session_name)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_processing_times(self, figsize=(12, 6)):
        """Plot processing time comparison across processors."""
        stats = self.comparator.compute_processing_statistics()
        
        if not stats:
            print("No data to plot")
            return
        
        processors = list(stats.keys())
        mean_times = [stats[p]['mean_time'] for p in processors]
        std_times = [stats[p]['std_time'] for p in processors]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(processors))
        bars = ax.bar(x, mean_times, yerr=std_times, capsize=5,
                     color=self.COLORS[:len(processors)], alpha=0.7)
        
        ax.set_xlabel('Framework')
        ax.set_ylabel('Processing Time (ms)')
        ax.set_title(f'Processing Time Comparison - {self.comparator.session_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(processors, rotation=45, ha='right')
        
        for i, (bar, mean) in enumerate(zip(bars, mean_times)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_times[i] + 0.5,
                   f'{mean:.1f}ms', ha='center', va='bottom', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'processing_times.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_detection_rates(self, figsize=(10, 6)):
        """Plot detection rate comparison."""
        stats = self.comparator.compute_processing_statistics()
        
        if not stats:
            return
        
        processors = list(stats.keys())
        rates = [stats[p]['detection_rate'] * 100 for p in processors]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(processors))
        bars = ax.bar(x, rates, color=self.COLORS[:len(processors)], alpha=0.7)
        
        ax.set_xlabel('Framework')
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title(f'Detection Rate Comparison - {self.comparator.session_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(processors, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'detection_rates.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_keypoint_timeline(self, keypoint_idx=0, keypoint_name='nose', figsize=(14, 6)):
        """Plot keypoint position over time for all processors."""
        timelines = self.comparator.compute_timeline_comparison(keypoint_idx)
        
        if not timelines:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        for i, (name, data) in enumerate(timelines.items()):
            color = self.COLORS[i % len(self.COLORS)]
            ts = [t/1000 for t in data['timestamps']]
            
            ax1.plot(ts, data['x'], label=name, color=color, alpha=0.8, linewidth=1.5)
            ax2.plot(ts, data['y'], label=name, color=color, alpha=0.8, linewidth=1.5)
        
        ax1.set_ylabel(f'{keypoint_name.capitalize()} X Position')
        ax2.set_ylabel(f'{keypoint_name.capitalize()} Y Position')
        ax2.set_xlabel('Time (seconds)')
        
        ax1.set_title(f'{keypoint_name.capitalize()} Position Over Time - {self.comparator.session_name}')
        
        ax1.legend(loc='upper right', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        
        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'timeline_{keypoint_name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_box_whisker_times(self, figsize=(12, 6)):
        """Create box-whisker plot for processing time distribution."""
        stats = self.comparator.compute_processing_statistics()
        
        if not stats:
            return
        
        data = [stats[p]['times'] for p in stats.keys()]
        labels = list(stats.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], self.COLORS[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Processing Time (ms)')
        ax.set_title(f'Processing Time Distribution - {self.comparator.session_name}')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'processing_times_boxplot.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_pck_comparison(self, keypoint_idx=0, keypoint_name='nose', threshold=0.5, figsize=(10, 6)):
        """Plot PCK comparison for a specific keypoint."""
        pck = self.comparator.compute_pck(keypoint_idx, threshold)
        
        if not pck:
            return
        
        processors = list(pck.keys())
        values = [pck[p] * 100 for p in processors]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(processors))
        bars = ax.bar(x, values, color=self.COLORS[:len(processors)], alpha=0.7)
        
        ax.set_xlabel('Framework')
        ax.set_ylabel(f'PCK@{threshold*100:.0f}% (%)')
        ax.set_title(f'{keypoint_name.capitalize()} PCK@{threshold*100:.0f}% - {self.comparator.session_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(processors, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'pck_{keypoint_name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_summary_dashboard(self, figsize=(16, 10)):
        """Create a comprehensive summary dashboard."""
        stats = self.comparator.compute_processing_statistics()
        
        if not stats:
            return
        
        fig = plt.figure(figsize=figsize)
        
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        ax4 = fig.add_subplot(gs[2, :])
        
        processors = list(stats.keys())
        mean_times = [stats[p]['mean_time'] for p in processors]
        est_fps = [stats[p]['estimated_fps'] for p in processors]
        detection_rates = [stats[p]['detection_rate'] * 100 for p in processors]
        
        x = np.arange(len(processors))
        
        ax1.bar(x, mean_times, color=self.COLORS[:len(processors)], alpha=0.7)
        ax1.set_title('Mean Processing Time (ms)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(processors, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(x, est_fps, color=self.COLORS[:len(processors)], alpha=0.7)
        ax2.set_title('Estimated FPS')
        ax2.set_xticks(x)
        ax2.set_xticklabels(processors, rotation=45, ha='right', fontsize=8)
        ax2.grid(axis='y', alpha=0.3)
        
        bp = ax3.boxplot([stats[p]['times'] for p in processors], 
                        labels=processors, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.COLORS[:len(processors)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_title('Processing Time Distribution')
        ax3.set_xticklabels(processors, rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)
        
        ax4.bar(x, detection_rates, color=self.COLORS[:len(processors)], alpha=0.7)
        ax4.set_title('Detection Rate (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(processors, rotation=45, ha='right', fontsize=8)
        ax4.set_ylim(0, 105)
        ax4.grid(axis='y', alpha=0.3)
        
        for i, (rate, proc) in enumerate(zip(detection_rates, processors)):
            ax4.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        fig.suptitle(f'Benchmark Summary Dashboard - {self.comparator.session_name}', 
                    fontsize=14, fontweight='bold')
        
        save_path = os.path.join(self.output_dir, 'dashboard.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    
    def generate_all(self):
        """Generate all visualizations."""
        print(f"\nGenerating visualizations for: {self.comparator.session_name}")
        self.plot_processing_times()
        self.plot_detection_rates()
        self.plot_keypoint_timeline(keypoint_idx=0, keypoint_name='nose')
        self.plot_keypoint_timeline(keypoint_idx=11, keypoint_name='left_shoulder')
        self.plot_box_whisker_times()
        self.plot_pck_comparison(keypoint_idx=0, keypoint_name='nose')
        self.plot_summary_dashboard()
        print("Visualization complete!")


def generate_comparison_report(session_name, output_format='html'):
    """
    Generate a comprehensive comparison report.
    
    Args:
        session_name: Name of the benchmark session
        output_format: Report format ('html', 'markdown', 'json')
        
    Returns:
        str: Path to generated report
    """
    comparator = BenchmarkComparator(session_name)
    stats = comparator.compute_processing_statistics()
    
    report_data = {
        'session_name': session_name,
        'generated_at': datetime.now().isoformat(),
        'frameworks': {}
    }
    
    for name, stat in stats.items():
        report_data['frameworks'][name] = {
            'detection_rate': f"{stat['detection_rate']*100:.2f}%",
            'mean_processing_time_ms': f"{stat['mean_time']:.2f}",
            'std_processing_time_ms': f"{stat['std_time']:.2f}",
            'min_processing_time_ms': f"{stat['min_time']:.2f}",
            'max_processing_time_ms': f"{stat['max_time']:.2f}",
            'median_processing_time_ms': f"{stat['median_time']:.2f}",
            'p95_processing_time_ms': f"{stat['p95_time']:.2f}",
            'estimated_fps': f"{stat['estimated_fps']:.2f}"
        }
    
    output_dir = os.path.join(Config.PLOT_OUTPUT_DIR, session_name)
    
    if output_format == 'json':
        output_path = os.path.join(output_dir, 'comparison_report.json')
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    elif output_format == 'markdown':
        output_path = os.path.join(output_dir, 'comparison_report.md')
        
        with open(output_path, 'w') as f:
            f.write(f"# Benchmark Comparison Report: {session_name}\n\n")
            f.write(f"Generated: {report_data['generated_at']}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Framework | Detection Rate | Mean Time (ms) | Est. FPS |\n")
            f.write("|-----------|----------------|----------------|----------|\n")
            
            for name, data in report_data['frameworks'].items():
                f.write(f"| {name} | {data['detection_rate']} | {data['mean_processing_time_ms']} | {data['estimated_fps']} |\n")
            
            f.write("\n## Detailed Statistics\n\n")
            
            for name, data in report_data['frameworks'].items():
                f.write(f"### {name}\n\n")
                f.write(f"- Detection Rate: {data['detection_rate']}\n")
                f.write(f"- Mean Processing Time: {data['mean_processing_time_ms']} ms\n")
                f.write(f"- Std Dev: {data['std_processing_time_ms']} ms\n")
                f.write(f"- Min/Max: {data['min_processing_time_ms']} / {data['max_processing_time_ms']} ms\n")
                f.write(f"- 95th Percentile: {data['p95_processing_time_ms']} ms\n")
                f.write(f"- Estimated FPS: {data['estimated_fps']}\n\n")
    
    visualizer = BenchmarkVisualizer(comparator)
    visualizer.generate_all()
    
    return output_path if output_format in ['json', 'markdown'] else output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument("--session", "-s", type=str, required=True,
                       help="Benchmark session name")
    parser.add_argument("--format", "-f", choices=['json', 'markdown', 'html'], 
                       default='markdown', help="Report format")
    
    args = parser.parse_args()
    
    report_path = generate_comparison_report(args.session, args.format)
    print(f"Report saved to: {report_path}")
