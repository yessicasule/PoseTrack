"""
Synchronized data collection for video and motion capture benchmarking.

This module provides functionality to:
- Record synchronized video frames with precise timestamps
- Collect motion capture data (or simulate if unavailable)
- Align video and mocap data for post-processing
"""
import cv2
import time
import json
import socket
import struct
import threading
import queue
from datetime import datetime
from collections import defaultdict
import os
from collections import defaultdict
from config.config import Config


class MotionCaptureReceiver:
    """
    UDP-based motion capture data receiver.
    Supports OptiTrack, Vicon, and other NatNet-compatible systems.
    """
    
    def __init__(self, host=None, port=None, rate=None):
        self.host = host or Config.MOCAP_HOST
        self.port = port or Config.MOCAP_PORT
        self.rate = rate or Config.MOCAP_RATE
        
        self.socket = None
        self.running = False
        self.thread = None
        self.data_queue = queue.Queue(maxsize=100)
        self.latest_data = {}
        
        self._body_definitions = {}
        self._frame_count = 0
        self._dropped_frames = 0
    
    def start(self):
        """Start the mocap receiver in a background thread."""
        if self.running:
            return
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(1.0)
            self.socket.bind(('', self.port))
            self.running = True
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            print(f"Motion capture receiver started on port {self.port}")
        except Exception as e:
            print(f"Failed to start mocap receiver: {e}")
            self.running = False
    
    def stop(self):
        """Stop the mocap receiver."""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _receive_loop(self):
        """Background loop to receive mocap packets."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                self._process_packet(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Mocap receive error: {e}")
    
    def _process_packet(self, data):
        """Process received mocap packet."""
        try:
            mocap_frame = self._parse_natnet_packet(data)
            if mocap_frame:
                self.latest_data = mocap_frame
                self._frame_count += 1
                if not self.data_queue.empty():
                    self._dropped_frames += 1
                try:
                    self.data_queue.put_nowait(mocap_frame)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"Packet parse error: {e}")
    
    def _parse_natnet_packet(self, data):
        """
        Parse NatNet packet format.
        Returns dict with rigid body poses and marker positions.
        """
        mocap_frame = {
            'timestamp': time.time(),
            'frame_number': None,
            'rigid_bodies': [],
            'markers': []
        }
        
        try:
            if len(data) < 4:
                return None
            
            msg_id = int.from_bytes(data[0:2], byteorder='little')
            
            if msg_id == 7:
                offset = 4
                if len(data) < offset + 4:
                    return None
                    
                mocap_frame['frame_number'] = int.from_bytes(
                    data[offset:offset+4], byteorder='little')
                offset += 4
                
                marker_set_size = int.from_bytes(
                    data[offset:offset+4], byteorder='little')
                offset += 4
                
                for _ in range(marker_set_size):
                    if offset + 130 > len(data):
                        break
                    name_len = int.from_bytes(
                        data[offset:offset+4], byteorder='little')
                    offset += 4
                    
                    name = data[offset:offset+name_len].decode('utf-8', errors='ignore')
                    offset += name_len
                    
                    x, y, z = struct.unpack('<fff', data[offset:offset+12])
                    offset += 12
                    
                    qx, qy, qz, qw = struct.unpack('<ffff', data[offset:offset+16])
                    offset += 16
                    
                    mean_error = struct.unpack('<f', data[offset:offset+4])[0]
                    offset += 4
                    
                    mocap_frame['rigid_bodies'].append({
                        'name': name,
                        'position': [x, y, z],
                        'rotation': [qx, qy, qz, qw],
                        'mean_error': mean_error
                    })
                    
        except Exception:
            pass
        
        return mocap_frame if mocap_frame['rigid_bodies'] else None
    
    def get_latest(self):
        """Get the most recent mocap frame."""
        return self.latest_data.copy() if self.latest_data else None
    
    def get_stats(self):
        """Get receiver statistics."""
        return {
            'frame_count': self._frame_count,
            'dropped_frames': self._dropped_frames,
            'drop_rate': self._dropped_frames / max(1, self._frame_count),
            'running': self.running
        }


class SynchronizedRecorder:
    """
    Records synchronized video and motion capture data.
    Provides frame-accurate synchronization for benchmarking.
    """
    
    def __init__(self, session_name=None, camera_id=None, mocap_enabled=None):
        """
        Initialize synchronized recorder.
        
        Args:
            session_name: Optional session name (auto-generated if None)
            camera_id: Camera device ID (default from config)
            mocap_enabled: Enable mocap recording (default from config)
        """
        Config.create_output_directories()
        
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        
        self.session_dir = Config.get_collection_session_dir(session_name)
        self.video_dir = Config.get_collection_session_dir(session_name, "video")
        self.mocap_dir = Config.get_collection_session_dir(session_name, "mocap")
        self.sync_dir = Config.get_collection_session_dir(session_name, "sync")
        
        self.camera_id = camera_id if camera_id is not None else Config.CAMERA_ID
        self.mocap_enabled = mocap_enabled if mocap_enabled is not None else Config.MOCAP_ENABLED
        
        self.cap = None
        self.mocap_receiver = None
        
        self.video_writer = None
        self.video_path = None
        
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        
        self.sync_log = []
        self.frame_timestamps = []
        
        self._setup_mocap()
    
    def _setup_mocap(self):
        """Initialize motion capture receiver if enabled."""
        if self.mocap_enabled:
            try:
                import struct
                self.mocap_receiver = MotionCaptureReceiver()
                self.mocap_receiver.start()
                print(f"Motion capture enabled - listening on port {Config.MOCAP_PORT}")
            except Exception as e:
                print(f"Failed to setup mocap: {e}")
                self.mocap_enabled = False
                self.mocap_receiver = None
        else:
            print("Motion capture disabled - using simulated timestamps")
            self.mocap_receiver = None
    
    def start_camera(self):
        """Initialize and start camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot access camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.OUTPUT_VIDEO_FPS)
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
        return actual_width, actual_height, actual_fps
    
    def _init_video_writer(self, width, height, fps):
        """Initialize video writer for recording."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = os.path.join(self.video_dir, f"{self.session_name}.mp4")
        self.video_writer = cv2.VideoWriter(
            self.video_path, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_path = os.path.join(self.video_dir, f"{self.session_name}.avi")
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, fps, (width, height))
        
        print(f"Video writer initialized: {self.video_path}")
    
    def start_recording(self, width, height, fps):
        """Start synchronized recording session."""
        if self.is_recording:
            return
        
        self._init_video_writer(width, height, fps)
        
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.sync_log = []
        self.frame_timestamps = []
        self.is_recording = True
        
        print("Recording started")
    
    def record_frame(self, frame):
        """
        Record a single frame with synchronized timestamp.
        
        Args:
            frame: Video frame from camera
            
        Returns:
            dict: Frame metadata including sync info
        """
        if not self.is_recording:
            return None
        
        video_timestamp = time.perf_counter() - self.start_time
        wall_timestamp = time.time()
        
        mocap_data = None
        if self.mocap_receiver:
            mocap_data = self.mocap_receiver.get_latest()
        
        frame_info = {
            'frame_index': self.frame_count,
            'video_timestamp': video_timestamp,
            'wall_timestamp': wall_timestamp,
            'mocap_data': mocap_data
        }
        
        self.frame_timestamps.append({
            'frame_index': self.frame_count,
            'video_timestamp': video_timestamp,
            'wall_timestamp': wall_timestamp
        })
        
        if mocap_data:
            self.sync_log.append({
                'frame_index': self.frame_count,
                'video_ts': video_timestamp,
                'mocap_ts': mocap_data.get('timestamp', 0),
                'sync_diff': abs(video_timestamp - (mocap_data.get('timestamp', 0) - self.start_time))
            })
        
        self.video_writer.write(frame)
        self.frame_count += 1
        
        return frame_info
    
    def stop_recording(self):
        """Stop recording and save synchronization data."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.video_writer:
            self.video_writer.release()
        
        self._save_sync_data()
        
        duration = time.perf_counter() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / duration if duration > 0 else 0
        
        print(f"\nRecording stopped:")
        print(f"  Frames: {self.frame_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Avg FPS: {avg_fps:.2f}")
        print(f"  Video: {self.video_path}")
        
        return {
            'frame_count': self.frame_count,
            'duration': duration,
            'avg_fps': avg_fps,
            'video_path': self.video_path
        }
    
    def _save_sync_data(self):
        """Save synchronization metadata and timestamps."""
        metadata = {
            'session_name': self.session_name,
            'collection_time': datetime.now().isoformat(),
            'start_time': self.start_time,
            'camera_id': self.camera_id,
            'mocap_enabled': self.mocap_enabled,
            'config': {
                'frame_width': Config.FRAME_WIDTH,
                'frame_height': Config.FRAME_HEIGHT,
                'output_fps': Config.OUTPUT_VIDEO_FPS,
                'mocap_host': Config.MOCAP_HOST,
                'mocap_port': Config.MOCAP_PORT
            }
        }
        
        metadata_path = os.path.join(self.sync_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        timestamps_path = os.path.join(self.sync_dir, "timestamps.json")
        with open(timestamps_path, 'w') as f:
            json.dump(self.frame_timestamps, f, indent=2)
        
        if self.sync_log:
            sync_log_path = os.path.join(self.sync_dir, "sync_log.json")
            with open(sync_log_path, 'w') as f:
                json.dump(self.sync_log, f, indent=2)
            
            sync_errors = [s['sync_diff'] for s in self.sync_log]
            print(f"  Sync errors: min={min(sync_errors):.4f}s, max={max(sync_errors):.4f}s, avg={sum(sync_errors)/len(sync_errors):.4f}s")
        
        if self.mocap_receiver:
            stats = self.mocap_receiver.get_stats()
            stats_path = os.path.join(self.sync_dir, "mocap_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.mocap_receiver:
            self.mocap_receiver.stop()
        cv2.destroyAllWindows()


def collect_session(session_name=None, duration=None, mocap_enabled=None):
    """
    Collect a synchronized video/mocap session.
    
    Args:
        session_name: Optional session name
        duration: Maximum recording duration in seconds (None = manual stop)
        mocap_enabled: Enable motion capture
        
    Returns:
        dict: Session results
    """
    recorder = SynchronizedRecorder(session_name, mocap_enabled=mocap_enabled)
    
    try:
        width, height, fps = recorder.start_camera()
        
        print("\n" + "="*60)
        print("SYNCHRONIZED DATA COLLECTION")
        print("="*60)
        print("\nControls:")
        print("   Press SPACE to start/stop recording")
        print("   Press 'q' to quit")
        print("="*60 + "\n")
        
        recording = False
        
        while True:
            ret, frame = recorder.cap.read()
            if not ret:
                print("Camera read error")
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
                    print(f"Duration limit ({duration}s) reached")
                    break
            
            cv2.putText(display, "Synchronized Data Collection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if recorder.mocap_enabled:
                status = "MOCAP CONNECTED" if recorder.mocap_receiver and recorder.mocap_receiver.running else "MOCAP DISCONNECTED"
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
        
        if recording:
            result = recorder.stop_recording()
        else:
            result = None
        
    finally:
        recorder.cleanup()
    
    return result


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Synchronized data collection")
    parser.add_argument("--name", "-n", type=str, default=None,
                       help="Session name")
    parser.add_argument("--duration", "-d", type=int, default=None,
                       help="Maximum duration in seconds")
    parser.add_argument("--mocap", "-m", action="store_true",
                       help="Enable motion capture")
    
    args = parser.parse_args()
    
    collect_session(args.name, args.duration, args.mocap)
