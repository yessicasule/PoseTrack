"""Source package for pose tracking and benchmarking project"""

from src.pose_tracker import PoseTracker
from src.data_recorder import DataRecorder
from src.video_recorder import VideoRecorder
from src.visualizer import PoseVisualizer
from src.sync_data_collector import SynchronizedRecorder, MotionCaptureReceiver, collect_session
from src.benchmark_processor import (
    BenchmarkProcessor,
    MediaPipePoseProcessor,
    MediaPipeTaskAPIProcessor,
    OpenCVDnnPoseProcessor,
    run_benchmark_on_session
)
from src.benchmark_comparator import (
    BenchmarkComparator,
    BenchmarkVisualizer,
    generate_comparison_report
)
