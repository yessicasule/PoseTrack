"""
Configuration settings for pose tracking and benchmarking
"""
import os

class Config:
    # Camera settings
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # MediaPipe settings (default values)
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 1
    SMOOTH_LANDMARKS = True
    
    # Recording settings
    RECORD_DURATION = 60  # Maximum recording duration in seconds
    OUTPUT_VIDEO_FPS = 30
    VIDEO_CODEC = 'mp4v'  # or 'XVID', 'MJPG'
    
    # Output paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "videos")
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "data")
    PLOT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
    COLLECTION_DIR = os.path.join(BASE_DIR, "outputs", "collection")
    BENCHMARK_DIR = os.path.join(BASE_DIR, "outputs", "benchmark")
    
    # Motion capture settings (for synchronized recording)
    MOCAP_ENABLED = False
    MOCAP_HOST = '127.0.0.1'
    MOCAP_PORT = 22345
    MOCAP_RATE = 100  # Hz
    SYNC_TOLERANCE = 0.01  # seconds tolerance for synchronization
    
    # Benchmarking settings
    BENCHMARK_FRAMEWORKS = {
        'mediapipe': {
            'enabled': True,
            'model_complexity': [0, 1, 2],
        },
        'posenet': {
            'enabled': True,
            'model_name': 'posenet_mobilenet_v1',
        },
        'movenet': {
            'enabled': True,
            'model_type': 'lightning',
        },
        'openpose': {
            'enabled': False,
            'model': 'body25',
        },
        'blazepose': {
            'enabled': True,
            'model_path': 'models/pose_landmarker_full.task',
        },
    }
    
    # Benchmark output formats
    BENCHMARK_OUTPUT_FORMATS = ['csv', 'json', 'pkl']
    COMPARE_VISUALIZATIONS = True
    
    # Upper-body landmark indices
    UPPER_BODY_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]
    
    # Landmark names mapping
    LANDMARK_NAMES = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist"
    }
    
    @classmethod
    def create_output_directories(cls):
        """Create output directories if they don't exist"""
        os.makedirs(cls.VIDEO_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.PLOT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.COLLECTION_DIR, exist_ok=True)
        os.makedirs(cls.BENCHMARK_DIR, exist_ok=True)
        print("Output directories created/verified")
    
    @classmethod
    def get_collection_session_dir(cls, session_name, subdir=None):
        """Get path for a specific collection session"""
        session_dir = os.path.join(cls.COLLECTION_DIR, session_name)
        os.makedirs(session_dir, exist_ok=True)
        if subdir:
            subdir_path = os.path.join(session_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            return subdir_path
        os.makedirs(os.path.join(session_dir, "video"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "mocap"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "sync"), exist_ok=True)
        return session_dir
    
    @classmethod
    def get_benchmark_session_dir(cls, session_name, framework=None):
        """Get path for benchmark results"""
        if framework:
            session_dir = os.path.join(cls.BENCHMARK_DIR, session_name, framework)
        else:
            session_dir = os.path.join(cls.BENCHMARK_DIR, session_name)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
