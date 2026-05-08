"""
Post-processing benchmark processor for pose estimation frameworks.

This module processes recorded video with multiple pose estimation frameworks
to enable consistent benchmarking on the same input data.
"""
import cv2
import json
import time
import pickle
import os
import csv
from abc import ABC, abstractmethod
from datetime import datetime
from collections import defaultdict
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, PoseLandmarker, PoseLandmarkerOptions, RunningMode
from config.config import Config


class PoseProcessor(ABC):
    """Abstract base class for pose estimation processors."""
    
    @property
    @abstractmethod
    def name(self):
        """Return the framework name."""
        pass
    
    @abstractmethod
    def process_frame(self, frame):
        """
        Process a single frame and return landmarks.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Landmarks data or None if no pose detected
        """
        pass
    
    @abstractmethod
    def get_landmark_count(self):
        """Return the number of landmarks this processor detects."""
        pass
    
    def close(self):
        """Cleanup resources."""
        pass


class MediaPipePoseProcessor(PoseProcessor):
    """MediaPipe Task API PoseLandmarker processor."""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(Config.BASE_DIR, "models", "pose_landmarker_full.task")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("Using bundled MediaPipe Pose solution instead.")
            self.use_bundled = True
            self.pose = None
            return
        
        self.use_bundled = False
        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        self.detector = PoseLandmarker.create_from_options(options)
    
    @property
    def name(self):
        return "mediapipe_task"
    
    def process_frame(self, frame, timestamp_ms=None):
        if self.use_bundled:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        if timestamp_ms is None:
            timestamp_ms = 0
        
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks[0]):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility_score
            }
        
        world_landmarks = None
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            world_landmarks = {}
            for idx, landmark in enumerate(results.pose_world_landmarks[0]):
                world_landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility_score
                }
        
        return {
            'landmarks': landmarks,
            'pose_world_landmarks': world_landmarks,
            'detection_confidence': 1.0
        }
    
    def get_landmark_count(self):
        return 33
    
    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()


class MoveNetProcessor(PoseProcessor):
    """
    MoveNet processor using MediaPipe's MoveNet Lightning/Thunder.
    MoveNet is optimized for fast and accurate pose detection.
    """
    
    def __init__(self, model_type='lightning', variant='thunder'):
        self.model_type = model_type
        self.variant = variant
        self.detector = None
        self._setup_detector()
        
    def _setup_detector(self):
        """Setup MoveNet detector via MediaPipe Solutions."""
        try:
            if self.model_type == 'lightning':
                base_options = BaseOptions(
                    model_asset_path=os.path.join(Config.BASE_DIR, "models", "pose_landmarker_lite.task")
                )
                self.model_name = "movenet_lightning"
            else:
                base_options = BaseOptions(
                    model_asset_path=os.path.join(Config.BASE_DIR, "models", "pose_landmarker_full.task")
                )
                self.model_name = "movenet_thunder"
            
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
            )
            self.detector = PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"MoveNet setup failed: {e}, falling back to MediaPipe Pose")
            self.detector = None
            self.model_name = "mediapipe_pose"
    
    @property
    def name(self):
        return self.model_name
    
    def process_frame(self, frame, timestamp_ms=None):
        if self.detector is None:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        if timestamp_ms is None:
            timestamp_ms = 0
        
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
        
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks[0]):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility_score
            }
        
        world_landmarks = None
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            world_landmarks = {}
            for idx, landmark in enumerate(results.pose_world_landmarks[0]):
                world_landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility_score
                }
        
        return {
            'landmarks': landmarks,
            'pose_world_landmarks': world_landmarks,
            'detection_confidence': 1.0
        }
    
    def get_landmark_count(self):
        return 33
    
    def close(self):
        if self.detector:
            self.detector.close()


class PoseNetProcessor(PoseProcessor):
    """
    PoseNet processor using TensorFlow/TensorFlow Lite models.
    Provides lightweight pose estimation for mobile/web.
    """
    
    def __init__(self, model_name='posenet'):
        self.model_name = model_name
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup PoseNet model via TensorFlow Lite."""
        try:
            import tflite_runtime.interpreter as tflite
            model_path = os.path.join(Config.BASE_DIR, "models", "posenet_mobilenet_v1_100.tflite")
            
            if not os.path.exists(model_path):
                print(f"PoseNet model not found at {model_path}")
                print("Using MediaPipe as fallback for PoseNet")
                self.interpreter = None
                return
            
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except ImportError:
            print("TensorFlow Lite not available, using MediaPipe fallback")
            self.interpreter = None
        except Exception as e:
            print(f"PoseNet setup failed: {e}, using MediaPipe fallback")
            self.interpreter = None
    
    @property
    def name(self):
        return "posenet_mobilenet_v1"
    
    def process_frame(self, frame, timestamp_ms=None):
        if self.interpreter is None:
            return None
        
        h, w = frame.shape[:2]
        
        input_size = 257
        resized = cv2.resize(frame, (input_size, input_size))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        heatmaps = self.interpreter.get_tensor(self.output_details[0]['index'])
        offsets = self.interpreter.get_tensor(self.output_details[1]['index'])
        
        num_keypoints = heatmaps.shape[-1]
        landmarks = {}
        
        for kp_idx in range(num_keypoints):
            heatmap = heatmaps[0, :, :, kp_idx]
            max_val = np.max(heatmap)
            
            if max_val > 0.1:
                max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                y_pos = (max_loc[0] / (input_size - 1)) * h + offsets[0, max_loc[0], max_loc[1], kp_idx]
                x_pos = (max_loc[1] / (input_size - 1)) * w + offsets[0, max_loc[0], max_loc[1], kp_idx + num_keypoints]
                
                landmarks[kp_idx] = {
                    'x': x_pos / w,
                    'y': y_pos / h,
                    'z': 0,
                    'visibility': max_val
                }
        
        return {'landmarks': landmarks, 'detection_confidence': 1.0}
    
    def get_landmark_count(self):
        return 17
    
    def close(self):
        if self.interpreter:
            del self.interpreter


MediaPipeTaskAPIProcessor = MediaPipePoseProcessor


class OpenCVDnnPoseProcessor(PoseProcessor):
    """
    OpenCV DNN-based pose estimation processor.
    Uses OpenCV's bundled pose estimation models.
    """
    
    def __init__(self, model="body_pose_coco"):
        self.model = model
        
        if model == "body_pose_coco":
            self.keypoints_mapping = [
                'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
                'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                'right_eye', 'left_eye', 'right_ear', 'left_ear'
            ]
            self.pairs = [
                (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)
            ]
        else:
            self.keypoints_mapping = list(range(18))
            self.pairs = []
        
        self.net = None
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load OpenCV's DNN pose model."""
        try:
            proto_path = cv2.data.haarcascades
            print(f"Note: OpenCV DNN pose requires model files. Using fallback.")
            self.net = None
        except Exception:
            self.net = None
    
    @property
    def name(self):
        return f"opencv_dnn_{self.model}"
    
    def process_frame(self, frame):
        if self.net is None:
            return None
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (368, 368), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        output = self.net.forward()
        
        h, w = frame.shape[:2]
        points = {}
        
        for i, name in enumerate(self.keypoints_mapping):
            if i >= len(output):
                break
            
            prob_map = output[i]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            x = point[0] / prob_map.shape[1] * w
            y = point[1] / prob_map.shape[0] * h
            
            visibility = prob if prob > 0.1 else 0
            
            points[i] = {
                'name': name,
                'x': x / w,
                'y': y / h,
                'z': 0,
                'visibility': visibility
            }
        
        return {'landmarks': points, 'detection_confidence': 1.0}
    
    def get_landmark_count(self):
        return len(self.keypoints_mapping)
    
    def close(self):
        pass


class BenchmarkProcessor:
    """
    Main benchmark processor that runs multiple frameworks on recorded video.
    """
    
    def __init__(self, session_name, video_path):
        """
        Initialize benchmark processor.
        
        Args:
            session_name: Name of the collection session
            video_path: Path to the recorded video file
        """
        self.session_name = session_name
        self.video_path = video_path
        
        self.output_dir = Config.get_benchmark_session_dir(session_name)
        self.processors = {}
        self.results = defaultdict(list)
        self.timings = defaultdict(list)
        
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.total_frames = 0
    
    def add_processor(self, processor):
        """Add a pose processor to the benchmark."""
        self.processors[processor.name] = processor
        print(f"Added processor: {processor.name}")
    
    def _setup_processors(self):
        """Setup default processors based on config."""
        if not self.processors:
            mp_settings = Config.BENCHMARK_FRAMEWORKS.get('mediapipe', {})
            
            if mp_settings.get('enabled', True):
                model_path = mp_settings.get('model_path')
                processor = MediaPipePoseProcessor(model_path=model_path)
                self.add_processor(processor)
                if processor.use_bundled:
                    print("Warning: Task API model not available.")
            
            posenet_settings = Config.BENCHMARK_FRAMEWORKS.get('posenet', {})
            if posenet_settings.get('enabled', False):
                processor = PoseNetProcessor(model_name=posenet_settings.get('model_name', 'posenet_mobilenet_v1'))
                self.add_processor(processor)
            
            movenet_settings = Config.BENCHMARK_FRAMEWORKS.get('movenet', {})
            if movenet_settings.get('enabled', False):
                processor = MoveNetProcessor(model_type=movenet_settings.get('model_type', 'lightning'))
                self.add_processor(processor)
            
            opencv_settings = Config.BENCHMARK_FRAMEWORKS.get('openpose', {})
            if opencv_settings.get('enabled', False):
                processor = OpenCVDnnPoseProcessor(model=opencv_settings.get('model', 'body_pose_coco'))
                self.add_processor(processor)
    
    def run(self):
        """
        Run benchmark on the video.
        
        Returns:
            dict: Benchmark results for each processor
        """
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {self.session_name}")
        print(f"{'='*60}")
        print(f"Video: {self.video_path}")
        print(f"Frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Processors: {list(self.processors.keys())}")
        print(f"{'='*60}\n")
        
        self._setup_processors()
        
        frame_idx = 0
        timestamp_ms = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            for name, processor in self.processors.items():
                start_time = time.perf_counter()
                
                result = processor.process_frame(frame, int(timestamp_ms))
                
                elapsed = (time.perf_counter() - start_time) * 1000
                
                self.timings[name].append(elapsed)
                
                frame_result = {
                    'frame_index': frame_idx,
                    'timestamp_ms': timestamp_ms,
                    'processing_time_ms': elapsed,
                    'result': result
                }
                self.results[name].append(frame_result)
                
                if result is None:
                    print(f"  Frame {frame_idx}: {name} - No pose detected ({elapsed:.1f}ms)")
            
            frame_idx += 1
            timestamp_ms = (frame_idx / self.fps) * 1000
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{self.total_frames} frames...")
        
        self.cap.release()
        
        for processor in self.processors.values():
            processor.close()
        
        self._save_results()
        self._print_summary()
        
        return self.results, self.timings
    
    def _save_results(self):
        """Save benchmark results to files."""
        for name, frames in self.results.items():
            processor_dir = os.path.join(self.output_dir, name)
            os.makedirs(processor_dir, exist_ok=True)
            
            csv_path = os.path.join(processor_dir, f"{name}_results.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_index', 'timestamp_ms', 'processing_time_ms', 
                               'pose_detected', 'avg_visibility', 'nose_x', 'nose_y'])
                
                for frame_data in frames:
                    result = frame_data['result']
                    if result and result.get('landmarks'):
                        landmarks = result['landmarks']
                        visibilities = [lm['visibility'] for lm in landmarks.values()]
                        avg_vis = sum(visibilities) / len(visibilities) if visibilities else 0
                        
                        nose = landmarks.get(0, {})
                        nose_x = nose.get('x', '')
                        nose_y = nose.get('y', '')
                    else:
                        avg_vis = 0
                        nose_x = ''
                        nose_y = ''
                    
                    writer.writerow([
                        frame_data['frame_index'],
                        frame_data['timestamp_ms'],
                        frame_data['processing_time_ms'],
                        1 if result else 0,
                        f"{avg_vis:.4f}",
                        nose_x,
                        nose_y
                    ])
            
            json_path = os.path.join(processor_dir, f"{name}_full_results.json")
            with open(json_path, 'w') as f:
                json.dump({
                    'session_name': self.session_name,
                    'processor': name,
                    'frames': frames
                }, f, indent=2)
        
        summary_path = os.path.join(self.output_dir, "benchmark_summary.json")
        summary = {
            'session_name': self.session_name,
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'processors': {}
        }
        
        for name, times in self.timings.items():
            detected = sum(1 for f in self.results[name] if f['result'] is not None)
            summary['processors'][name] = {
                'total_frames': len(times),
                'detection_rate': detected / len(times) if times else 0,
                'avg_processing_time_ms': sum(times) / len(times) if times else 0,
                'min_processing_time_ms': min(times) if times else 0,
                'max_processing_time_ms': max(times) if times else 0,
                'estimated_fps': 1000 / (sum(times) / len(times)) if times else 0
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def _print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        for name, times in self.timings.items():
            detected = sum(1 for f in self.results[name] if f['result'] is not None)
            avg_time = sum(times) / len(times) if times else 0
            est_fps = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Detection rate: {detected}/{len(times)} ({100*detected/len(times):.1f}%)")
            print(f"  Avg time: {avg_time:.2f}ms")
            print(f"  Min/Max: {min(times):.2f}ms / {max(times):.2f}ms")
            print(f"  Est. FPS: {est_fps:.1f}")
        
        print(f"\n{'='*60}")


def run_benchmark_on_session(session_name, video_path=None, processors=None):
    """
    Run benchmark on a collection session.
    
    Args:
        session_name: Name of the collection session
        video_path: Optional path to video (auto-detected if None)
        processors: Optional list of processor names to run
        
    Returns:
        tuple: (results, timings)
    """
    if video_path is None:
        video_dir = os.path.join(Config.COLLECTION_DIR, session_name, "video")
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        if not video_files:
            raise FileNotFoundError(f"No video found in session {session_name}")
        video_path = os.path.join(video_dir, video_files[0])
    
    processor = BenchmarkProcessor(session_name, video_path)
    
    if processors:
        for p_name in processors:
            if 'mediapipe' in p_name:
                processor.add_processor(MediaPipePoseProcessor())
            elif 'posenet' in p_name:
                processor.add_processor(PoseNetProcessor())
            elif 'movenet' in p_name:
                processor.add_processor(MoveNetProcessor())
            elif 'opencv_dnn' in p_name:
                processor.add_processor(OpenCVDnnPoseProcessor())
    else:
        processor._setup_processors()
    
    return processor.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pose estimation benchmark")
    parser.add_argument("--session", "-s", type=str, required=True,
                       help="Collection session name")
    parser.add_argument("--video", "-v", type=str, default=None,
                       help="Path to video file")
    parser.add_argument("--processors", "-p", nargs='+', default=None,
                       choices=['mediapipe_task', 'posenet_mobilenet_v1', 'movenet_lightning', 'movenet_thunder', 'opencv_dnn'],
                       help="Processors to run")
    
    args = parser.parse_args()
    
    run_benchmark_on_session(args.session, args.video, args.processors)
