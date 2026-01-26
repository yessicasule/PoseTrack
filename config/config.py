"""
Configuration settings for pose tracking
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
        print("Output directories created/verified")
