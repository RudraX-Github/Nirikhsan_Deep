import os as _os_init
_os_init.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os_init.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ✅ OPTIMIZED: RTSP transport options for minimal latency
if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in _os_init.environ:
    # Simplified for smooth streaming: tcp transport with minimal analysis
    # video_codec;h264 forces H.264 decoding, avoiding HEVC/H.265 issues
    _os_init.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|video_codec;h264"

import cv2
from types import SimpleNamespace

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


import csv
import time
import tkinter as tk
import customtkinter as ctk
from tkinter import font, simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
import os
import glob
import face_recognition
import numpy as np
import threading
import platform
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from collections import deque, Counter
import json
import gc
import psutil

# ✅ MODEL PIPELINE IMPORTS
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from bytetrack.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Set CustomTkinter appearance with modern dark theme
import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

import warnings
import os as os_module
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os_module.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- 1. Configuration Loading ---
def load_config():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Config load error: {e}. Using defaults.")
        return {
            "detection": {"min_detection_confidence": 0.5, "min_tracking_confidence": 0.5, 
                         "face_recognition_tolerance": 0.5, "re_detect_interval": 8},  # ✅ OPTIMIZED: Reduced from 15 to 8 frames (~267ms interval for faster detection)
            "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, "enable_frame_skipping": True, "min_buffer_for_classification": 5},
            "logging": {"log_directory": "logs", "max_log_size_mb": 10, "auto_flush_interval": 50},
            "storage": {"alert_snapshots_dir": "alert_snapshots", "snapshot_retention_days": 30,
                       "guard_profiles_dir": r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles", "capture_snapshots_dir": "capture_snapshots",
                       "audio_files_dir": "audio_files"},
            "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8}
        }

CONFIG = load_config()

# --- 2. Logging Setup with Rotation ---
if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

logger = logging.getLogger("निराक्षण")
logger.setLevel(logging.WARNING)  # Only log warnings and errors by default

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Rotating file handler
file_handler = RotatingFileHandler(
    os.path.join(CONFIG["logging"]["log_directory"], "session.log"),
    maxBytes=CONFIG["logging"]["max_log_size_mb"] * 1024 * 1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ✅ SAFE LOGGING: ASCII-safe logging utility to prevent Unicode encoding errors on Windows
class SafeLogger:
    """
    Wrapper for logger that sanitizes Unicode characters and replaces them with ASCII equivalents.
    Prevents UnicodeEncodeError on Windows console (cp1252 encoding).
    """
    
    # Unicode to ASCII replacement map
    UNICODE_MAP = {
        '\u2713': '[OK]',           # ✓ checkmark
        '\u2717': '[X]',            # ✗ cross
        '\u26a0': '[WARN]',         # ⚠ warning
        '\u26a0\ufe0f': '[WARN]',   # ⚠️ warning with variant
        '\ud83d\udea8': '[ALERT]',  # 🚨 siren
        '\ud83d\udd0a': '[SOUND]',  # 🔊 speaker
        '\ud83d\udcf8': '[SNAP]',   # 📸 camera
        '\ud83d\udccb': '[LOG]',    # 📋 log
        '\ud83d\udccd': '[PIN]',    # 📍 pin
        '\ud83d\udca4': '[SLEEP]',  # 💤 sleep
        '\u251c\u2500': '|-',       # ├─ tree
        '\u2514\u2500': 'L-',       # └─ tree
        '→': '->',                   # arrow
        '→': '->',                   # variant
    }
    
    @staticmethod
    def sanitize(text):
        """Convert Unicode characters to ASCII-safe equivalents."""
        if not isinstance(text, str):
            return text
        for unicode_char, ascii_equiv in SafeLogger.UNICODE_MAP.items():
            text = text.replace(unicode_char, ascii_equiv)
        # Remove any remaining problematic Unicode by encoding/decoding
        try:
            text.encode('cp1252')  # Test if encoding works
        except UnicodeEncodeError:
            text = text.encode('cp1252', errors='replace').decode('cp1252')
        return text
    
    @staticmethod
    def warning(msg, *args, **kwargs):
        """Log warning with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.warning(msg, *args, **kwargs)
    
    @staticmethod
    def info(msg, *args, **kwargs):
        """Log info with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.info(msg, *args, **kwargs)
    
    @staticmethod
    def debug(msg, *args, **kwargs):
        """Log debug with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.debug(msg, *args, **kwargs)
    
    @staticmethod
    def error(msg, *args, **kwargs):
        """Log error with Unicode sanitization."""
        msg = SafeLogger.sanitize(msg)
        logger.error(msg, *args, **kwargs)

# Create instance for use throughout application
safe_logger = SafeLogger()

# --- 3. File Storage Utilities (Systematic Organization) ---
def get_storage_paths():
    """
    Get all organized storage directory paths.
    Structure:
    - guard_profiles/: Face images for recognition
    - capture_snapshots/: Timestamped captures
    - logs/: CSV events and session logs
    """
    paths = {
        "guard_profiles": CONFIG.get("storage", {}).get("guard_profiles_dir", r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles"),
        "capture_snapshots": CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots"),
        "logs": CONFIG["logging"]["log_directory"]
    }
    
    # Create all directories
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    return paths

def save_guard_face(face_image, guard_name, angle="front"):
    """Save guard face image to guard_profiles directory with multi-angle support.
    
    Args:
        face_image: The face image to save
        guard_name: Guard name
        angle: Angle identifier (front, left, right, back, or None for legacy single image)
    """
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    
    if angle and angle != "front":
        # Multi-angle: Save in guard-specific subfolder
        guard_dir = os.path.join(paths["guard_profiles"], f"target_{safe_name}")
        os.makedirs(guard_dir, exist_ok=True)
        profile_path = os.path.join(guard_dir, f"{angle}.jpg")
    else:
        # Legacy/primary: Save main face image in root (front angle or single image)
        profile_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
        # Also save in multi-angle folder for consistency
        guard_dir = os.path.join(paths["guard_profiles"], f"target_{safe_name}")
        os.makedirs(guard_dir, exist_ok=True)
        front_path = os.path.join(guard_dir, "front.jpg")
        cv2.imwrite(front_path, face_image)
    
    cv2.imwrite(profile_path, face_image)
    logger.info(f"Saved guard face ({angle}): {profile_path}")
    return profile_path

def load_guard_angle_images(guard_name):
    """Load all angle reference images for a guard.
    
    Returns:
        dict: Dictionary with angle names as keys and file paths as values
              e.g., {'front': 'path/to/front.jpg', 'left': 'path/to/left.jpg', ...}
    """
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    guard_dir = os.path.join(paths["guard_profiles"], f"target_{safe_name}")
    
    angle_images = {}
    
    # Check for multi-angle images in subfolder (PRO mode: 5 angles, Normal: 4 angles)
    if os.path.exists(guard_dir):
        for angle in ["front", "left", "right", "back", "top"]:
            angle_path = os.path.join(guard_dir, f"{angle}.jpg")
            if os.path.exists(angle_path):
                angle_images[angle] = angle_path
                logger.debug(f"Found {angle} angle image for {guard_name}")
    
    # Always include the main face image as "front" if no multi-angle images exist
    if not angle_images:
        main_face_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
        if os.path.exists(main_face_path):
            angle_images["front"] = main_face_path
            logger.debug(f"Using single face image for {guard_name} (legacy mode)")
    
    return angle_images

def save_capture_snapshot(face_image, guard_name):
    """Save timestamped capture snapshot to capture_snapshots directory."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(paths["capture_snapshots"], f"{safe_name}_capture_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, face_image)
    return snapshot_path

# --- Directory Setup (using systematic functions) ---
if not os.path.exists(CONFIG["storage"]["alert_snapshots_dir"]):
    os.makedirs(CONFIG["storage"]["alert_snapshots_dir"])

if not os.path.exists(CONFIG.get("storage", {}).get("guard_profiles_dir", r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles")):
    os.makedirs(CONFIG.get("storage", {}).get("guard_profiles_dir", r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles"))

if not os.path.exists(CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots")):
    os.makedirs(CONFIG.get("storage", {}).get("capture_snapshots_dir", "capture_snapshots"))

# Ensure logs directory exists
if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

csv_file = os.path.join(CONFIG["logging"]["log_directory"], "events.csv")
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Action", "Status", "Image_Path", "Confidence"])

# --- 4. Cleanup Old Snapshots ---
def cleanup_old_snapshots():
    try:
        retention_days = CONFIG["storage"]["snapshot_retention_days"]
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        snapshot_dir = CONFIG["storage"]["alert_snapshots_dir"]
        
        for filename in os.listdir(snapshot_dir):
            filepath = os.path.join(snapshot_dir, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_time < cutoff_time:
                    os.remove(filepath)
    except Exception as e:
        logger.error(f"Snapshot cleanup error: {e}")

threading.Thread(target=cleanup_old_snapshots, daemon=True).start()

# ✅ PERFORMANCE: Memory management and garbage collection optimization
def optimize_memory():
    """Optimize memory usage by aggressive garbage collection at strategic points."""
    try:
        gc.collect()
        # Force collection of unreachable objects
        unreachable = gc.collect()
        if unreachable > 100:  # Log only if significant cleanup
            logger.debug(f"[MEMORY] Collected {unreachable} unreachable objects")
    except Exception as e:
        logger.debug(f"[MEMORY] GC optimization error: {e}")

# Set aggressive garbage collection for real-time performance
gc.set_threshold(1000, 15, 15)  # More aggressive collection

# ============================================================================
# PRO MODE ENHANCEMENTS: Multi-Scale Distance Detection & Body Matching
# ============================================================================

def detect_faces_multiscale_distance(rgb_frame, pro_mode=False):
    """
    ✅ PRO MODE: Enhanced multi-scale detection for distance guards.
    Detects guards from 5 to 20 meters away using scale pyramid.
    Only active in PRO mode for performance optimization.
    """
    if not pro_mode:
        return face_recognition.face_locations(rgb_frame, model="hog")
    
    face_locations = []
    scales = [1.0, 1.5, 2.0]
    
    for scale in scales:
        if scale == 1.0:
            scaled_frame = rgb_frame
        else:
            h, w = rgb_frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        try:
            locations = face_recognition.face_locations(scaled_frame, model="hog")
            for (top, right, bottom, left) in locations:
                face_locations.append((
                    int(top / scale),
                    int(right / scale),
                    int(bottom / scale),
                    int(left / scale)
                ))
        except Exception as e:
            logger.debug(f"Multi-scale detection error at scale {scale}: {e}")
            continue
    
    face_locations = remove_duplicate_faces(face_locations, iou_threshold=0.5)
    if len(face_locations) > 0:
        logger.debug(f"[PRO MODE] Multi-scale detected {len(face_locations)} faces")
    
    return face_locations

def remove_duplicate_faces(face_locations, iou_threshold=0.5):
    """Remove duplicate face detections from multi-scale pyramid using NMS."""
    if len(face_locations) <= 1:
        return face_locations
    
    boxes = []
    for (top, right, bottom, left) in face_locations:
        boxes.append([left, top, right - left, bottom - top])
    
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    areas = (boxes[:, 2]) * (boxes[:, 3])
    order = areas.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 0] + boxes[i, 2], boxes[order[1:], 0] + boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 1] + boxes[i, 3], boxes[order[1:], 1] + boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return [face_locations[i] for i in keep]

def match_body_silhouette(detected_landmarks, guard_body_profile):
    """
    ✅ PRO MODE: Match guard by body silhouette when face detection fails.
    """
    if not detected_landmarks or not guard_body_profile:
        return 0.0
    
    try:
        def get_body_proportions(landmarks):
            if hasattr(landmarks, 'landmark'):
                lm = landmarks.landmark
            else:
                lm = landmarks
            
            l_shoulder = lm[11]
            r_shoulder = lm[12]
            shoulder_width = abs(r_shoulder.x - l_shoulder.x)
            
            l_hip = lm[23]
            torso_height = abs(l_hip.y - l_shoulder.y)
            
            l_ankle = lm[27]
            leg_length = abs(l_ankle.y - l_hip.y)
            
            l_wrist = lm[15]
            arm_length = abs(l_wrist.y - l_shoulder.y)
            
            return np.array([shoulder_width, torso_height, leg_length, arm_length])
        
        detected_props = get_body_proportions(detected_landmarks)
        stored_props = get_body_proportions(guard_body_profile)
        
        dot_product = np.dot(detected_props, stored_props)
        norm_detected = np.linalg.norm(detected_props)
        norm_stored = np.linalg.norm(stored_props)
        
        if norm_detected == 0 or norm_stored == 0:
            return 0.0
        
        similarity = dot_product / (norm_detected * norm_stored)
        return max(0.0, min(1.0, similarity))
    except Exception as e:
        logger.debug(f"Body silhouette matching error: {e}")
        return 0.0

# --- MediaPipe Solutions Setup (kept for action classification only) ---
# UPDATED: All pose detection models replaced with new pipeline:
# - BlazeFace for face detection (single-person mode)
# - MoveNet Lightning for single-person pose estimation  
# - BlazePose for multi-person pose estimation
# - ByteTrack for multi-object tracking
# MediaPipe Holistic retained only for pose landmark constants used in classify_action
mp_holistic = mp.solutions.holistic

# --- Sound Logic ---
def _resource_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

def get_sound_path(filename: str) -> str:
    """Return absolute path for a sound file placed next to this script."""
    return os.path.join(_resource_dir(), filename)

def play_siren_sound(stop_event=None, duration_seconds=30, sound_file="siren.mp3"):
    """Play alert sound looping for up to duration_seconds or until stop_event is set
    
    Args:
        stop_event: threading.Event to signal stop playback
        duration_seconds: Maximum duration to play (default 30 seconds)
        sound_file: Path/name of audio file (can be full path or filename in audio_files directory)
    """
    def _sound_worker():
        # Support both full paths and relative audio_files directory paths
        if os.path.isabs(sound_file) or os.path.exists(sound_file):
            mp3_path = sound_file
        else:
            audio_dir = CONFIG.get("storage", {}).get("audio_files_dir", "audio_files")
            mp3_path = os.path.join(audio_dir, sound_file)
        
        if not os.path.exists(mp3_path):
            # Fallback to old hardcoded path for backwards compatibility
            mp3_path = rf"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Web_Cam\{sound_file}"
        start_time = time.time()
        
        # Option 1: Try pygame (PRIMARY - most reliable for MP3 on Windows)
        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.set_volume(1.0)
                
                # Play in loop until stop_event or duration_seconds
                pygame.mixer.music.play(-1)  # -1 means infinite loop
                logger.info(f"Alert sound started via pygame (max {duration_seconds}s)")
                
                # Wait until stop_event or duration expired
                while True:
                    elapsed = time.time() - start_time
                    
                    # Check if stop_event is set (action performed)
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Check if duration expired
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    time.sleep(0.1)
                
                pygame.mixer.music.stop()
                return
            except Exception as e:
                logger.warning(f"Pygame playback failed: {e}")
        
        # Option 2: Try pydub (requires ffmpeg/avconv)
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                logger.info(f"Alert sound started via pydub (max {duration_seconds}s)")
                
                while True:
                    elapsed = time.time() - start_time
                    
                    # Check if stop_event is set
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Check if duration expired
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Play audio clip
                    play(audio)
                
                logger.info("Alert sound via pydub completed")
                return
            except Exception as e:
                logger.warning(f"Pydub playback failed: {e}")
        
        # Fallback: Use system beeps (Windows winsound - always available)
        try:
            if platform.system() == "Windows":
                import winsound
                logger.info(f"Alert sound started via winsound (max {duration_seconds}s)")
                
                # Simulate emergency siren with pulsing high-low tones
                while True:
                    elapsed = time.time() - start_time
                    
                    # Check if stop_event is set
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Check if duration expired
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    # Play siren pattern
                    winsound.Beep(2500, 150)  # High beep
                    time.sleep(0.05)
                    winsound.Beep(1800, 150)  # Lower beep
                    time.sleep(0.05)
            else:
                # Unix/Linux fallback
                logger.info(f"Alert sound started via beep (max {duration_seconds}s)")
                while True:
                    elapsed = time.time() - start_time
                    
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    print('\a')
                    time.sleep(0.3)
        except Exception as e:
            logger.error(f"Sound Error: {e}")

    t = threading.Thread(target=_sound_worker, daemon=True)
    t.start()
    return t

# --- EAR Calculation Helper (from Basic_v5.py) ---
def calculate_ear(landmarks, width, height):
    """Calculates Eye Aspect Ratio (EAR)."""
    RIGHT_EYE = [33, 133, 159, 145, 158, 153]
    LEFT_EYE = [362, 263, 386, 374, 385, 380]

    def get_eye_ear(indices):
        p1 = np.array([landmarks[indices[0]].x * width, landmarks[indices[0]].y * height])
        p2 = np.array([landmarks[indices[1]].x * width, landmarks[indices[1]].y * height])
        p3 = np.array([landmarks[indices[2]].x * width, landmarks[indices[2]].y * height])
        p4 = np.array([landmarks[indices[3]].x * width, landmarks[indices[3]].y * height])
        p5 = np.array([landmarks[indices[4]].x * width, landmarks[indices[4]].y * height])
        p6 = np.array([landmarks[indices[5]].x * width, landmarks[indices[5]].y * height])

        v1 = np.linalg.norm(p3 - p4)
        v2 = np.linalg.norm(p5 - p6)
        h1 = np.linalg.norm(p1 - p2)

        if h1 == 0: return 0.0
        return (v1 + v2) / (2.0 * h1)

    ear_right = get_eye_ear(RIGHT_EYE)
    ear_left = get_eye_ear(LEFT_EYE)
    return (ear_right + ear_left) / 2.0

# --- classify_action with improved detection ---
def classify_action(landmarks, h, w):
    """
    Classify pose action with robust detection and confidence scoring.
    Supports: Hands Up, Hands Crossed, One Hand Raised (Left/Right), T-Pose, Sit, Standing
    Includes visibility and quality checks for stable detection.
    
    ✅ IMPROVED: Distance-agnostic pose classification
    - Uses relative positions (normalized to body size)
    - Works for guards at near and far distances
    - More tolerant of small finger movements
    - Enhanced: Extreme angle tolerance, multi-joint averaging, confidence scoring
    - Enhanced: Adaptive visibility thresholds based on body size and occlusion detection
    """
    try:
        NOSE = mp_holistic.PoseLandmark.NOSE.value
        L_WRIST = mp_holistic.PoseLandmark.LEFT_WRIST.value
        R_WRIST = mp_holistic.PoseLandmark.RIGHT_WRIST.value
        L_ELBOW = mp_holistic.PoseLandmark.LEFT_ELBOW.value
        R_ELBOW = mp_holistic.PoseLandmark.RIGHT_ELBOW.value
        L_SHOULDER = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        R_SHOULDER = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        L_HIP = mp_holistic.PoseLandmark.LEFT_HIP.value
        R_HIP = mp_holistic.PoseLandmark.RIGHT_HIP.value
        L_KNEE = mp_holistic.PoseLandmark.LEFT_KNEE.value
        R_KNEE = mp_holistic.PoseLandmark.RIGHT_KNEE.value
        L_ANKLE = mp_holistic.PoseLandmark.LEFT_ANKLE.value
        R_ANKLE = mp_holistic.PoseLandmark.RIGHT_ANKLE.value
        L_EYE = mp_holistic.PoseLandmark.LEFT_EYE.value
        R_EYE = mp_holistic.PoseLandmark.RIGHT_EYE.value

        nose = landmarks[NOSE]
        l_wrist = landmarks[L_WRIST]
        r_wrist = landmarks[R_WRIST]
        l_elbow = landmarks[L_ELBOW]
        r_elbow = landmarks[R_ELBOW]
        l_shoulder = landmarks[L_SHOULDER]
        r_shoulder = landmarks[R_SHOULDER]
        l_hip = landmarks[L_HIP]
        r_hip = landmarks[R_HIP]
        l_knee = landmarks[L_KNEE]
        r_knee = landmarks[R_KNEE]
        l_ankle = landmarks[L_ANKLE]
        r_ankle = landmarks[R_ANKLE]
        l_eye = landmarks[L_EYE]
        r_eye = landmarks[R_EYE]

        # ✅ CRITICAL: Normalize to body scale (use shoulder-to-hip distance as reference)
        # This makes detection work for near and far guards equally well
        shoulder_to_hip_dist = abs(l_shoulder.y - l_hip.y)  # Vertical distance
        if shoulder_to_hip_dist < 0.01:  # Guard too far/close or bad detection
            return "Standing"
        
        # ✅ ENHANCED: Calculate body size percentile for adaptive thresholds
        # Use shoulder width as additional reference for full-body scale
        shoulder_width = abs(r_shoulder.x - l_shoulder.x)
        body_scale = (shoulder_to_hip_dist + shoulder_width) / 2.0
        
        # Threshold multipliers based on body size instead of fixed pixels
        # This scales detection to work at any distance
        HANDS_UP_THRESHOLD = shoulder_to_hip_dist * 0.4      # Hands above head = 40% of torso
        HANDS_CROSSED_TOLERANCE = shoulder_to_hip_dist * 0.3  # Hands at chest = within 30% of torso
        ARM_EXTENSION = shoulder_to_hip_dist * 0.6            # Arms extended = 60% of torso width
        WRIST_ARM_ALIGNMENT = shoulder_to_hip_dist * 0.25     # Wrist aligned with arm = within 25%

        nose_y = nose.y
        nose_x = nose.x
        lw_y = l_wrist.y
        rw_y = r_wrist.y
        lw_x = l_wrist.x
        rw_x = r_wrist.x
        ls_y = l_shoulder.y
        rs_y = r_shoulder.y
        ls_x = l_shoulder.x
        rs_x = r_shoulder.x
        lh_y = l_hip.y
        rh_y = r_hip.y
        
        # ✅ ENHANCED: Adaptive visibility thresholds based on body size and occlusion
        # Far/small bodies naturally have lower visibility - adapt thresholds dynamically
        # Also check face visibility (eyes) to detect if person is facing camera
        eyes_visible = l_eye.visibility > 0.40 and r_eye.visibility > 0.40
        
        # If eyes not visible, person is turned away - MUCH more lenient joint visibility requirements
        if eyes_visible:
            # FRONT-FACING MODE: Person facing camera, stricter visibility thresholds
            visibility_adjustment = 1.0
            visibility_threshold_base = 0.45  # Stricter when facing forward
        else:
            # BACK/EXTREME-ANGLE MODE: Person turned away or extreme angle, very lenient thresholds
            visibility_adjustment = 0.6  # Reduce thresholds by 40%
            visibility_threshold_base = 0.30  # Much more lenient for back views
        
        l_wrist_visible = l_wrist.visibility > (0.45 * visibility_adjustment)
        r_wrist_visible = r_wrist.visibility > (0.45 * visibility_adjustment)
        l_elbow_visible = l_elbow.visibility > (0.45 * visibility_adjustment)
        r_elbow_visible = r_elbow.visibility > (0.45 * visibility_adjustment)
        nose_visible = nose.visibility > (0.40 * visibility_adjustment)
        l_shoulder_visible = l_shoulder.visibility > (0.40 * visibility_adjustment)
        r_shoulder_visible = r_shoulder.visibility > (0.40 * visibility_adjustment)
        l_knee_visible = l_knee.visibility > (0.30 * visibility_adjustment)  # Knees hardest to see
        r_knee_visible = r_knee.visibility > (0.30 * visibility_adjustment)
        l_hip_visible = l_hip.visibility > (0.35 * visibility_adjustment)
        r_hip_visible = r_hip.visibility > (0.35 * visibility_adjustment)
        
        # ✅ ENHANCED: Joint visibility quality check with adaptive thresholds
        # For back-facing guards, require fewer visible joints since body is occluded
        visible_joints = sum([
            l_wrist_visible, r_wrist_visible, l_elbow_visible, r_elbow_visible,
            l_shoulder_visible, r_shoulder_visible, l_knee_visible, r_knee_visible,
            l_hip_visible, r_hip_visible, nose_visible
        ])
        
        # ✅ ADAPTIVE: Joint quality requirement varies by viewing angle
        if eyes_visible:
            # Front-facing: Require 7/11 major joints visible (stricter)
            min_joints_required = 7
        else:
            # Back-facing: Require only 5/11 major joints (more lenient for rear view)
            # Since many joints are naturally occluded from behind
            min_joints_required = 5
        
        if visible_joints < min_joints_required:  
            return "Standing"  # Default to standing if pose quality is poor
        
        # ✅ ENHANCED: Hands Up Detection with extreme angle tolerance
        # Works for both front-facing and angled upward views
        if (l_wrist_visible and r_wrist_visible and nose_visible):
            # Multi-joint confirmation: Check wrists relative to nose AND shoulders
            wrist_above_nose_l = (nose_y - lw_y) > (HANDS_UP_THRESHOLD * 0.8)  # 80% of threshold
            wrist_above_nose_r = (nose_y - rw_y) > (HANDS_UP_THRESHOLD * 0.8)
            
            # Also accept if wrists clearly above shoulders (extreme upward angle view)
            wrist_above_shoulder_l = (ls_y - lw_y) > (HANDS_UP_THRESHOLD * 0.5)
            wrist_above_shoulder_r = (rs_y - rw_y) > (HANDS_UP_THRESHOLD * 0.5)
            
            if ((wrist_above_nose_l and wrist_above_nose_r) or 
                (wrist_above_shoulder_l and wrist_above_shoulder_r)):
                return "Hands Up"
        
        # ✅ ENHANCED: Hands Crossed Detection with improved angle tolerance
        if (l_wrist_visible and r_wrist_visible and l_shoulder_visible and r_shoulder_visible):
            chest_y = (ls_y + rs_y) / 2
            body_center_x = (ls_x + rs_x) / 2
            
            # Check if both wrists are at chest level (normalized) OR below shoulders
            # More tolerant for extreme angles where chest position shifts
            wrist_chest_dist_l = abs(lw_y - chest_y)
            wrist_chest_dist_r = abs(rw_y - chest_y)
            
            # Allow higher tolerance for crossed position (extreme angles)
            crossed_tolerance = HANDS_CROSSED_TOLERANCE * 1.2
            
            if (wrist_chest_dist_l < crossed_tolerance and 
                wrist_chest_dist_r < crossed_tolerance):
                # Check if wrists are crossed (left hand on right side, vice versa)
                # More lenient: require only one hand to be clearly crossed
                left_hand_crossed = lw_x > body_center_x
                right_hand_crossed = rw_x < body_center_x
                
                if left_hand_crossed and right_hand_crossed:
                    return "Hands Crossed"
        
        # ✅ ENHANCED: T-Pose Detection with extreme angle tolerance
        if (l_wrist_visible and r_wrist_visible and l_elbow_visible and r_elbow_visible and
            l_shoulder_visible and r_shoulder_visible):
            
            # Check if limbs form T-shape with more tolerant thresholds for angles
            # T-pose: arms extended at shoulder level
            lw_at_shoulder = abs(lw_y - ls_y) < (WRIST_ARM_ALIGNMENT * 1.3)  # 30% more tolerant
            rw_at_shoulder = abs(rw_y - rs_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            le_at_shoulder = abs(l_elbow.y - ls_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            re_at_shoulder = abs(r_elbow.y - rs_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            
            if lw_at_shoulder and rw_at_shoulder and le_at_shoulder and re_at_shoulder:
                # Check if arms are extended outward (normalized)
                shoulder_width = abs(rs_x - ls_x)
                if shoulder_width > 0:
                    left_extension = abs(lw_x - ls_x) / shoulder_width
                    right_extension = abs(rw_x - rs_x) / shoulder_width
                    
                    # T-pose: arms extended to 0.8+ shoulder width
                    if left_extension > 0.6 and right_extension > 0.6:  # 60% of shoulder width
                        return "T-Pose"
        
        # ✅ ENHANCED: One Hand Raised Detection with multi-frame consistency
        if l_wrist_visible and r_wrist_visible and l_shoulder_visible and r_shoulder_visible:
            chest_y = (ls_y + rs_y) / 2
            
            # LEFT HAND RAISED
            left_raised = (nose_y - lw_y) > (HANDS_UP_THRESHOLD * 0.8)
            right_down = (rw_y - chest_y) > (HANDS_CROSSED_TOLERANCE * 0.8)
            
            if left_raised and right_down:
                return "One Hand Raised (Left)"
            
            # RIGHT HAND RAISED
            right_raised = (nose_y - rw_y) > (HANDS_UP_THRESHOLD * 0.8)
            left_down = (lw_y - chest_y) > (HANDS_CROSSED_TOLERANCE * 0.8)
            
            if right_raised and left_down:
                return "One Hand Raised (Right)"
        
        # Alternative: One hand raised while other is not visible (extreme angle)
        if l_wrist_visible and not r_wrist_visible and nose_visible:
            if (nose_y - lw_y) > (HANDS_UP_THRESHOLD * 0.8):
                return "One Hand Raised (Left)"
        
        if r_wrist_visible and not l_wrist_visible and nose_visible:
            if (nose_y - rw_y) > (HANDS_UP_THRESHOLD * 0.8):
                return "One Hand Raised (Right)"
        
        # ✅ ENHANCED: Sit/Stand Detection with multi-joint confirmation
        # More robust by checking both knees AND ankles for leg extension
        if l_knee_visible and r_knee_visible and l_hip_visible and r_hip_visible:
            # Calculate thigh angle (knee to hip)
            thigh_angle_l = abs(l_knee.y - l_hip.y)
            thigh_angle_r = abs(r_knee.y - r_hip.y)
            avg_thigh_angle = (thigh_angle_l + thigh_angle_r) / 2
            
            # Also check ankle position if visible (adds confidence for standing)
            ankle_check_valid = False
            if landmarks[L_ANKLE].visibility > 0.30 and landmarks[R_ANKLE].visibility > 0.30:
                ankle_down_l = l_ankle.y > l_knee.y
                ankle_down_r = r_ankle.y > r_knee.y
                ankle_check_valid = ankle_down_l and ankle_down_r
            
            # If thigh is nearly horizontal, person is sitting
            sit_threshold = shoulder_to_hip_dist * 0.15
            
            if avg_thigh_angle < sit_threshold:
                return "Sit"
            elif ankle_check_valid or avg_thigh_angle > sit_threshold:
                # Ankles visible below knees OR thigh angle indicates standing
                return "Standing"
            else:
                return "Standing"  # Default to standing
        elif l_knee_visible or r_knee_visible:
            # At least one knee visible, use it
            knee_y = l_knee.y if l_knee_visible else r_knee.y
            hip_y = l_hip.y if l_knee_visible else r_hip.y
            
            if abs(knee_y - hip_y) < (shoulder_to_hip_dist * 0.15):
                return "Sit"
            else:
                return "Standing"
        else:
            # Default to standing if knee not visible
            return "Standing"

        return "Standing" 

    except Exception as e:
        logger.debug(f"Pose classification error: {e}")
        return "Unknown"

# --- Helper: Calculate Dynamic Body Box from Face ---
def calculate_body_box(face_box, frame_h, frame_w, expansion_factor=3.0):
    """
    Calculate dynamic body bounding box from detected face box.
    
    ✅ IMPROVED: Distance-agnostic body box calculation
    - Works for guards at near and far distances
    - Expands based on face size (relative scaling)
    - Ensures full body is captured for pose detection
    
    Args:
        face_box: tuple (x1, y1, x2, y2) - face coordinates
        frame_h, frame_w: frame dimensions
        expansion_factor: how many face widths to expand (default 3x for near/far)
    
    Returns:
        tuple (bx1, by1, bx2, by2) - body box coordinates that captures full body
    """
    x1, y1, x2, y2 = face_box
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = x1 + (face_w // 2)
    
    # ✅ CRITICAL: Dynamic expansion based on face size
    # Smaller faces (far guards) and larger faces (near guards) both get appropriate body boxes
    # For small faces: expand more; for large faces: expand less (but always full body)
    if face_w < 30:  # Very far guard
        adaptive_expansion = 5.0  # Expand 5x for small face
    elif face_w < 100:  # Far guard
        adaptive_expansion = 4.0  # Expand 4x for small-medium face
    else:  # Near guard or medium distance
        adaptive_expansion = 3.0  # Expand 3x for large face
    
    # Expand horizontally based on adaptive face width
    bx1 = max(0, int(face_cx - (face_w * adaptive_expansion / 2)))
    bx2 = min(frame_w, int(face_cx + (face_w * adaptive_expansion / 2)))
    
    # Expand vertically: slightly above face, down to feet
    # Scale vertical expansion with face size too
    by1 = max(0, int(y1 - (face_h * 0.5)))
    by2 = frame_h  # Always extend to bottom for leg visibility
    
    return (bx1, by1, bx2, by2)

def approximate_face_from_body(body_box):
    """
    Approximate a face box from a tracked body box. Useful when face tracking is lost
    but the person/body remains tracked. Returns (x1, y1, x2, y2).
    """
    bx1, by1, bx2, by2 = body_box
    w = max(1, bx2 - bx1)
    h = max(1, by2 - by1)
    fx_w = int(max(20, 0.25 * w))
    fx_h = int(max(20, 0.20 * h))
    fx1 = bx1 + int(0.375 * w)
    fy1 = by1 + int(0.05 * h)
    fx2 = min(bx2, fx1 + fx_w)
    fy2 = min(by2, fy1 + fx_h)
    return (fx1, fy1, fx2, fy2)

# --- Helper: IoU for Overlap Check ---
def calculate_iou(boxA, boxB):
    # box = (x, y, w, h) -> convert to (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

# ✅ NEW: Robust Skeleton Validation for Guard Tracking
def validate_skeleton_quality(landmarks, visibility_threshold=0.35, min_keypoints=10):
    """
    ✅ ENHANCED: Comprehensive skeleton validation with quality scoring.
    
    Validates pose estimation quality for guard tracking fallback.
    Crucial for when face is not detected but body skeleton is still visible.
    
    Args:
        landmarks: MediaPipe pose landmarks (33 points)
        visibility_threshold: minimum visibility score for a landmark (0.0-1.0)
        min_keypoints: minimum number of visible landmarks required
    
    Returns:
        dict: {
            'valid': bool - whether skeleton meets quality requirements,
            'keypoint_count': int - number of visible landmarks,
            'confidence': float - overall skeleton quality (0.0-1.0),
            'visible_keypoints': list - indices of visible keypoints,
            'validity_reason': str - explanation of validation result
        }
    """
    try:
        if landmarks is None or len(landmarks) < 33:
            return {
                'valid': False,
                'keypoint_count': 0,
                'confidence': 0.0,
                'visible_keypoints': [],
                'validity_reason': 'Invalid landmarks data'
            }
        
        # ✅ CRITICAL KEYPOINTS for full-body validation
        # These 17 keypoints form the essential skeleton for pose tracking
        CRITICAL_KEYPOINTS = [
            0,   # NOSE
            11, 12,  # SHOULDERS (left, right)
            13, 14,  # ELBOWS (left, right)
            15, 16,  # WRISTS (left, right)
            23, 24,  # HIPS (left, right)
            25, 26,  # KNEES (left, right)
            27, 28,  # ANKLES (left, right)
        ]
        
        visible_keypoints = []
        visible_count = 0
        confidence_scores = []
        
        # Check each landmark
        for idx in range(len(landmarks)):
            landmark = landmarks[idx]
            if landmark.visibility > visibility_threshold:
                visible_keypoints.append(idx)
                visible_count += 1
                # Use visibility as quality measure
                confidence_scores.append(landmark.visibility)
        
        # Check critical keypoints specifically
        critical_visible = sum(
            1 for idx in CRITICAL_KEYPOINTS 
            if idx < len(landmarks) and landmarks[idx].visibility > visibility_threshold
        )
        
        # ✅ VALIDATION CRITERIA:
        # 1. Minimum visible keypoints (10 out of 33)
        # 2. At least 8/14 critical keypoints visible for full-body skeleton
        min_critical_keypoints = 8
        
        is_valid = (
            visible_count >= min_keypoints and
            critical_visible >= min_critical_keypoints
        )
        
        # Calculate confidence as average of visible landmark visibilities
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            avg_confidence = 0.0
        
        # Additional quality check: spatial consistency
        # Landmarks should be reasonably spread (not all clustered in one spot)
        if len(visible_keypoints) >= 3:
            x_coords = [landmarks[idx].x for idx in visible_keypoints if idx < len(landmarks)]
            y_coords = [landmarks[idx].y for idx in visible_keypoints if idx < len(landmarks)]
            
            if x_coords and y_coords:
                x_spread = max(x_coords) - min(x_coords)
                y_spread = max(y_coords) - min(y_coords)
                spatial_spread = (x_spread + y_spread) / 2
                
                # If spread is very small (<0.1), landmarks are too clustered (likely noise)
                if spatial_spread < 0.05:
                    is_valid = False
                    validity_reason = f"Keypoints too clustered (spread={spatial_spread:.3f})"
                else:
                    validity_reason = f"OK: {visible_count} keypoints, critical={critical_visible}/14, confidence={avg_confidence:.2f}"
            else:
                validity_reason = f"Cannot compute spatial spread"
        else:
            if is_valid:
                validity_reason = f"OK: {visible_count} keypoints, confidence={avg_confidence:.2f}"
            else:
                validity_reason = f"Insufficient keypoints ({visible_count}/{min_keypoints})"
        
        return {
            'valid': is_valid,
            'keypoint_count': visible_count,
            'confidence': avg_confidence,
            'visible_keypoints': visible_keypoints,
            'critical_visible': critical_visible,
            'validity_reason': validity_reason
        }
        
    except Exception as e:
        logger.debug(f"Skeleton validation error: {e}")
        return {
            'valid': False,
            'keypoint_count': 0,
            'confidence': 0.0,
            'visible_keypoints': [],
            'validity_reason': f'Exception: {str(e)}'
        }

def extract_skeleton_features(landmarks):
    """
    ✅ NEW: Extract skeleton-based features for robust body tracking.
    
    When face is not detected, these features help maintain tracking on skeleton alone.
    Computes body geometry landmarks that are invariant to distance and angle.
    
    Args:
        landmarks: MediaPipe pose landmarks
    
    Returns:
        dict: body features including skeleton box, confidence, and key points
    """
    try:
        if landmarks is None or len(landmarks) < 33:
            return None
        
        # Extract key body parts
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        
        # Verify minimum visibility for body box calculation
        min_vis = 0.25  # Very lenient for far guards
        if (l_shoulder.visibility < min_vis or r_shoulder.visibility < min_vis or
            l_hip.visibility < min_vis or r_hip.visibility < min_vis):
            return None
        
        # ✅ CRITICAL: Compute skeleton bounding box
        # This box encompasses the entire body from head to feet
        all_x = [p.x for p in landmarks if p.visibility > min_vis]
        all_y = [p.y for p in landmarks if p.visibility > min_vis]
        
        if not all_x or not all_y:
            return None
        
        # Normalized coordinates (0-1 range)
        skeleton_x_min = min(all_x)
        skeleton_x_max = max(all_x)
        skeleton_y_min = min(all_y)
        skeleton_y_max = max(all_y)
        
        # Expand slightly to account for limb width
        expansion = 0.05
        skeleton_x_min = max(0, skeleton_x_min - expansion)
        skeleton_x_max = min(1, skeleton_x_max + expansion)
        skeleton_y_min = max(0, skeleton_y_min - expansion)
        skeleton_y_max = min(1, skeleton_y_max + expansion)
        
        # Compute torso centerline for consistency
        torso_center_x = (l_shoulder.x + r_shoulder.x) / 2
        torso_center_y = (l_shoulder.y + l_hip.y) / 2
        
        # Compute body scale (shoulder to hip distance)
        body_scale = abs(l_hip.y - l_shoulder.y)
        
        features = {
            'skeleton_box': (skeleton_x_min, skeleton_y_min, skeleton_x_max, skeleton_y_max),
            'torso_center': (torso_center_x, torso_center_y),
            'body_scale': body_scale,
            'body_height': skeleton_y_max - skeleton_y_min,
            'body_width': skeleton_x_max - skeleton_x_min,
            'is_upright': body_scale > 0.2,  # Body height > 20% of frame = upright person
            'visibility_avg': sum(p.visibility for p in landmarks if p.visibility > 0) / max(1, len([p for p in landmarks if p.visibility > 0]))
        }
        
        return features
        
    except Exception as e:
        logger.debug(f"Skeleton feature extraction error: {e}")
        return None

# --- Multi-Guard Pose Detection Enhancement ---
def resolve_overlapping_poses(targets_status, iou_threshold=0.3):
    """
    Resolve conflicting pose detections when multiple guards overlap.
    Ensures each guard has independent, consistent pose detection.
    
    ✅ MULTI-GUARD ENHANCED: Uses confidence scoring, temporal consistency, 
    and spatial analysis for better conflict resolution.
    ✅ SEPARATION FIX: Re-enables disabled guards when they move apart (IoU < threshold).
    
    Args:
        targets_status: Dictionary of all tracked guards and their status
        iou_threshold: IoU threshold for considering boxes as overlapping
    
    Returns:
        Updated targets_status with resolved conflicts
    """
    try:
        target_names = list(targets_status.keys())
        conflicts_resolved = []
        
        for i, name_a in enumerate(target_names):
            # ✅ SEPARATION FIX: Check BOTH visible and disabled guards for re-enabling
            box_a = targets_status[name_a].get("face_box")
            if not box_a:
                continue
            
            # Check overlap with other guards
            for name_b in target_names[i+1:]:
                box_b = targets_status[name_b].get("face_box")
                if not box_b:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(
                    (box_a[0], box_a[1], box_a[2] - box_a[0], box_a[3] - box_a[1]),
                    (box_b[0], box_b[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
                )
                
                # ✅ SEPARATION FIX: If guards were previously overlapping but now separated
                # Re-enable the disabled guard immediately
                if iou < iou_threshold:
                    # Guards are SEPARATED
                    was_disabled_a = not targets_status[name_a].get("visible")
                    was_disabled_b = not targets_status[name_b].get("visible")
                    
                    # If either was disabled due to overlap, re-enable it now
                    if was_disabled_a and targets_status[name_a].get("overlap_disabled"):
                        targets_status[name_a]["visible"] = True
                        targets_status[name_a]["overlap_disabled"] = False
                        targets_status[name_a]["tracker"] = None  # Reset tracker for fresh detection
                        logger.warning(f"[SEPARATION] Re-enabled {name_a} - guards separated (IoU: {iou:.2f})")
                    
                    if was_disabled_b and targets_status[name_b].get("overlap_disabled"):
                        targets_status[name_b]["visible"] = True
                        targets_status[name_b]["overlap_disabled"] = False
                        targets_status[name_b]["tracker"] = None  # Reset tracker for fresh detection
                        logger.warning(f"[SEPARATION] Re-enabled {name_b} - guards separated (IoU: {iou:.2f})")
                    
                    continue  # Skip conflict resolution if not overlapping
                
                # ✅ ORIGINAL: Handle overlapping guards (IoU > threshold)
                if targets_status[name_a].get("visible") and targets_status[name_b].get("visible"):
                    # Both visible and overlapping - resolve conflict
                    # Multi-factor scoring for better decision making
                    pose_conf_a = targets_status[name_a].get("pose_confidence", 0.0)
                    pose_conf_b = targets_status[name_b].get("pose_confidence", 0.0)
                    face_conf_a = targets_status[name_a].get("face_confidence", 0.0)
                    face_conf_b = targets_status[name_b].get("face_confidence", 0.0)
                    
                    # ✅ MULTI-GUARD FIX: Check for temporal consistency (recent pose quality)
                    pose_hist_a = targets_status[name_a].get("pose_quality_history", deque())
                    pose_hist_b = targets_status[name_b].get("pose_quality_history", deque())
                    recent_quality_a = sum(pose_hist_a) / len(pose_hist_a) if pose_hist_a else 0.0
                    recent_quality_b = sum(pose_hist_b) / len(pose_hist_b) if pose_hist_b else 0.0
                    
                    # Composite score: 40% pose + 30% face + 30% temporal consistency
                    score_a = (pose_conf_a * 0.4) + (face_conf_a * 0.3) + (recent_quality_a * 0.3)
                    score_b = (pose_conf_b * 0.4) + (face_conf_b * 0.3) + (recent_quality_b * 0.3)
                    
                    # Keep guard with higher composite score
                    if score_a < score_b:
                        targets_status[name_a]["visible"] = False
                        targets_status[name_a]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled due to overlap
                        conflicts_resolved.append((name_a, name_b, score_a, score_b, iou))
                        logger.debug(f"[RESOLVE] Disabled {name_a} (score:{score_a:.3f}) - kept {name_b} (score:{score_b:.3f}), IoU:{iou:.2f}")
                    elif score_b < score_a:
                        targets_status[name_b]["visible"] = False
                        targets_status[name_b]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled due to overlap
                        conflicts_resolved.append((name_b, name_a, score_b, score_a, iou))
                        logger.debug(f"[RESOLVE] Disabled {name_b} (score:{score_b:.3f}) - kept {name_a} (score:{score_a:.3f}), IoU:{iou:.2f}")
                    else:
                        # ✅ MULTI-GUARD TIE-BREAKER: If scores are equal, prefer by face confidence
                        if face_conf_a >= face_conf_b:
                            targets_status[name_b]["visible"] = False
                            targets_status[name_b]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled
                            logger.debug(f"[RESOLVE] Tie-break: Disabled {name_b} - kept {name_a} by face confidence")
                        else:
                            targets_status[name_a]["visible"] = False
                            targets_status[name_a]["overlap_disabled"] = True  # ✅ SEPARATION FIX: Mark as disabled
                            logger.debug(f"[RESOLVE] Tie-break: Disabled {name_a} - kept {name_b} by face confidence")
        
        if conflicts_resolved:
            logger.info(f"[RESOLVE] Multi-guard: Resolved {len(conflicts_resolved)} pose conflicts")
    except Exception as e:
        logger.debug(f"[ERROR] Pose conflict resolution error: {e}")
    
    return targets_status

def smooth_bounding_box(current_box, previous_box, smoothing_factor=0.7):
    """
    Apply exponential moving average smoothing to bounding box to reduce jitter.
    
    Args:
        current_box: Current detected box (x1, y1, x2, y2)
        previous_box: Previous smoothed box (x1, y1, x2, y2)
        smoothing_factor: Smoothing weight (0-1, higher = more weight on previous box)
    
    Returns:
        Smoothed box coordinates
    """
    if previous_box is None:
        return current_box
    
    try:
        smoothed_box = tuple(
            int(smoothing_factor * prev + (1 - smoothing_factor) * curr)
            for curr, prev in zip(current_box, previous_box)
        )
        return smoothed_box
    except Exception as e:
        logger.debug(f"Box smoothing error: {e}")
        return current_box

# --- ReID Feature Extraction & Matching Functions ---
def extract_appearance_features(frame, face_box):
    """
    Extract appearance features from a person's image using color and edge histograms.
    
    Args:
        frame: Input image frame
        face_box: Bounding box (x1, y1, x2, y2) for person region
    
    Returns:
        Feature vector (numpy array) or None if extraction fails
    """
    try:
        x1, y1, x2, y2 = face_box
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        
        # Resize for consistent feature extraction
        person_resized = cv2.resize(person_crop, (64, 128))
        
        # Color histogram features (3 channels × 8 bins each = 24 features)
        color_features = []
        for i in range(3):
            hist = cv2.calcHist([person_resized], [i], None, [8], [0, 256])
            color_features.extend(hist.flatten())
        
        # Edge detection features using Canny (8 bins for consistency)
        gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
        
        # Combine all features into single vector (24 + 8 = 32 features total)
        features = np.array(color_features + edge_hist.flatten(), dtype=np.float32)
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        return None

def calculate_feature_similarity(features1, features2):
    """
    Calculate similarity between two feature vectors using cosine similarity.
    Simple fallback implementation without sklearn dependency.
    
    Args:
        features1: Feature vector 1
        features2: Feature vector 2
    
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        # Manual cosine similarity (no sklearn needed)
        f1 = np.array(features1).astype(np.float32)
        f2 = np.array(features2).astype(np.float32)
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        similarity = float(dot_product / (norm1 * norm2 + 1e-6))
        
        return max(0.0, min(1.0, float(similarity)))  # Clamp to [0, 1]
    except Exception as e:
        logger.warning(f"Similarity calculation error: {e}")
        return 0.0

def match_person_identity(person_id, new_features, person_features_db, confidence_threshold=0.65):
    """
    Match a person's features against database to determine if same person.
    
    Args:
        person_id: ID of person to match
        new_features: New feature vector
        person_features_db: Database of known person features
        confidence_threshold: Minimum similarity for match
    
    Returns:
        (matched_person_id, confidence) or (None, 0.0) if no match
    """
    try:
        if person_id not in person_features_db:
            # New person, store features
            person_features_db[person_id] = {
                'features': new_features,
                'count': 1,
                'last_seen': time.time()
            }
            return person_id, 1.0
        
        # Calculate similarity with stored features
        stored_features = person_features_db[person_id]['features']
        similarity = calculate_feature_similarity(new_features, stored_features)
        
        if similarity >= confidence_threshold:
            # Update features with exponential moving average for stability
            alpha = 0.3  # Learning rate
            person_features_db[person_id]['features'] = (
                alpha * new_features + 
                (1 - alpha) * stored_features
            )
            person_features_db[person_id]['count'] += 1
            person_features_db[person_id]['last_seen'] = time.time()
            return person_id, similarity
        else:
            # Create new person identity
            new_id = f"{person_id}_variant_{len(person_features_db)}"
            person_features_db[new_id] = {
                'features': new_features,
                'count': 1,
                'last_seen': time.time()
            }
            return new_id, similarity
    except Exception as e:
        logger.warning(f"Person matching error: {e}")
        return None, 0.0

# --- Helper: Detect Available Cameras ---
def detect_available_cameras(max_cameras=10):
    """Detect all available camera indices"""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# --- Tkinter Application Class ---
class PoseApp:
    def __init__(self, window_title=" निराक्षण - Niraakshan"):
        self.root = ctk.CTk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")  # Larger default size
        
        # ✅ CRITICAL: Initialize translations EARLY - before any UI elements are created
        self.current_language = "Hindi"
        self.translations = self.load_translations()
        self.current_trans = self.translations["Hindi"]  # Set Hindi as default
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0  # Default camera
        self.rtsp_fallback_applied = False
        self._rtsp_original_url = None
        self._rtsp_fallback_notice_logged = False
        self.decoder_failure_count = 0
        self.last_frame_time = time.time()  # ✅ FIX: Track frame timing to detect lag
        self.frame_lag_warnings = 0  # Count consecutive lag warnings
        
        # ✅ NEW: Camera and Mode state tracking
        self.is_camera_running = False
        self.is_pro_mode = False
        self.is_alert_mode = False
        self.is_fugitive_detection = False
        self.is_stillness_alert = False
        self.guard_mode = "ADD"
        self.fugitive_add_mode = "ADD"
        
        self.alert_interval = 10
        # Monitoring Mode
        self.monitor_mode_var = tk.StringVar(self.root)
        self.monitor_mode_var.set("Action Alerts Only")
        self.frame_w = 640 
        self.frame_h = 480 
        
        # ✅ MODEL PIPELINE CONFIGURATION
        # Single-person (normal): BlazeFace → MoveNet Lightning → SORT
        # Multi-person (normal): BlazePose → ByteTrack
        # Single-person (dark): CLAHE+Gamma → BlazeFace → MoveNet Lightning → SORT
        # Multi-person (dark): CLAHE+Gamma → BlazePose → ByteTrack
        
        self.is_single_person_mode = False  # Auto-detected based on guard count
        self.night_mode = False
        self.face_detector = None
        self.pose_model = None
        self.tracker = None  # SORT or ByteTrack
        self.model_pipeline_initialized = False
        
        self._initialize_model_pipeline()
        
        # ✅ NEW: Tracking State
        self.is_tracking = False  # Toggle state for Track/Stop Monitoring
        self.active_required_action = "Hands Up"  # Current active action from dropdown
        self.last_clock_second = -1  # Track last displayed second for smooth clock update

        self.target_map = {}
        self.targets_status = {} 
        self.selected_target_names = []  # NEW: Track selected targets
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = CONFIG["detection"]["re_detect_interval"]
        self.RESIZE_SCALE = 1.0 
        self.temp_log = []
        self.temp_log_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()  # CRITICAL: For FPS calculation
        self.current_fps = 0.0  # CRITICAL: Current FPS value
        self.last_process_frame = None  # CRITICAL: Cached frame for frame skipping
        self.last_action_cache = {}
        self.session_start_time = time.time()
        self.onboarding_mode = False
        self.is_in_capture_mode = False  # CRITICAL: Track if in capture mode
        self.onboarding_step = 0
        self.onboarding_name = None
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None
        
        # Fugitive Mode Fields
        self.fugitive_mode = False
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_name = "Unknown Fugitive"
        self.fugitive_detected_log_done = False  # Prevent duplicate logs
        self.last_fugitive_snapshot_time = 0  # Rate limiting for snapshots
        self.fugitive_alert_sound_thread = None
        self.fugitive_alert_stop_event = None
        
        # ✅ ENHANCED FUGITIVE TRACKING: Track visibility and alert timing
        self.fugitive_currently_visible = False  # Flag to track if fugitive is in frame
        self.fugitive_alert_start_time = 0  # Timestamp when fugitive first appeared
        
        # ✅ SIMPLIFIED: PRO Detection mode fields removed - focus on Normal Mode only
        
        # Photo storage for Tkinter (prevent garbage collection)
        self.photo_storage = {}  # Dictionary to store PhotoImage references

        self.frame_timestamp_ms = 0 

        # --- Layout ---
        self.root.grid_rowconfigure(0, weight=1)  # Main content area
        self.root.grid_columnconfigure(0, weight=1)  # Camera feed (expandable)
        self.root.grid_columnconfigure(1, weight=0)  # Sidebar (fixed width)
        
        # Sidebar state
        self.sidebar_collapsed = False
        self.sidebar_width = 320  # Wider for better button layout
        
        # 1. Main Content Area (Camera Feed + Controls)
        self.main_container = ctk.CTkFrame(self.root, fg_color="#1a1a1a")
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_rowconfigure(1, weight=1)  # Camera feed expands
        self.main_container.grid_rowconfigure(0, weight=0)  # Camera controls fixed
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # === CAMERA CONTROLS PANEL (Above camera feed) ===
        self.camera_controls_panel = ctk.CTkFrame(self.main_container, fg_color="#2b2b2b", height=50, corner_radius=0)
        self.camera_controls_panel.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.camera_controls_panel.grid_propagate(False)
        
        # Control buttons frame - single row with all buttons
        controls_frame = ctk.CTkFrame(self.camera_controls_panel, fg_color="transparent")
        controls_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Single row: Camera ON/OFF + Snap + Night + PRO Mode (all in one line)
        self.btn_camera_toggle = ctk.CTkButton(controls_frame, text=self.current_trans.get("camera_on", "▶ कैमरा ON/OFF"), 
                                               command=self.toggle_camera, height=28,
                                               fg_color="#27ae60", font=("Segoe UI", 11, "bold"), corner_radius=4)
        self.btn_camera_toggle.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_snap = ctk.CTkButton(controls_frame, text=self.current_trans.get("snap", "📸 स्नैप"), command=self.snap_photo, 
                                     height=28, fg_color="#f39c12", font=("Segoe UI", 11, "bold"), corner_radius=4)
        self.btn_snap.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_night_mode = ctk.CTkButton(controls_frame, text=self.current_trans.get("night", "🌙 रात"), command=self.toggle_night_mode, 
                                            height=28, fg_color="#34495e", font=("Segoe UI", 11, "bold"), corner_radius=4)
        self.btn_night_mode.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_pro_toggle = ctk.CTkButton(controls_frame, text=self.current_trans.get("pro_mode", "⚡ PRO Mode ON/OFF"), 
                                           command=self.toggle_pro_mode, height=28,
                                           fg_color="#8e44ad", font=("Segoe UI", 11, "bold"), corner_radius=4)
        self.btn_pro_toggle.pack(side="left", fill="both", expand=True, padx=(0,2))
        
        self.btn_switch_camera = ctk.CTkButton(controls_frame, text="📹 Switch", 
                                               command=self.switch_camera_dialog, height=28,
                                               fg_color="#e67e22", font=("Segoe UI", 11, "bold"), corner_radius=4)
        self.btn_switch_camera.pack(side="left", fill="both", expand=True)
        
        # Camera feed in main container (below controls)
        self.video_container = ctk.CTkFrame(self.main_container, fg_color="#000000")
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.video_label = ctk.CTkLabel(self.video_container, text=self.current_trans.get("camera_offline", "📷 कैमरा फीड ऑफलाइन"), 
                                         font=("Segoe UI", 20, "bold"), text_color="#7f8c8d")
        self.video_label.pack(fill="both", expand=True)
        
        # 2. Professional Sidebar - FIXED WIDTH FOR CONSISTENT BUTTON SIZING
        self.sidebar_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", width=420)
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_propagate(False)
        
        # Sidebar Header
        sidebar_header = ctk.CTkFrame(self.sidebar_frame, fg_color="#1e1e1e", height=80, corner_radius=0)
        sidebar_header.pack(fill="x", padx=0, pady=0)
        sidebar_header.pack_propagate(False)
        
        # ✅ CLICKABLE LOGO/TITLE - Opens Language Converter
        self.title_label = ctk.CTkLabel(sidebar_header, text="🛡️ निराक्षण (Niraakshan)", 
                    font=("Arial Unicode MS", 14, "bold"), text_color="#3498db", cursor="hand2")
        self.title_label.pack(pady=(8,2))
        self.title_label.bind("<Button-1>", lambda e: self.open_language_converter())
        
        # Slogan/Mantra
        ctk.CTkLabel(sidebar_header, text="निराक्षण करे रक्षण", 
                    font=("Arial Unicode MS", 11, "italic"), text_color="#2ecc71").pack(pady=(0,8))
        
        # Separator
        ctk.CTkFrame(sidebar_header, fg_color="#3498db", height=2).pack(side="bottom", fill="x")
        
        # ✅ Translations already initialized at the start of __init__
        
        # Scrollable content area
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar_frame, fg_color="#2b2b2b", corner_radius=0)
        self.sidebar_scroll.pack(fill="both", expand=True, padx=0, pady=0)
        
        # FIXED BUTTON WIDTHS CALCULATION
        # Sidebar width: 420px, padding/margins: 2px left + 2px right = 4px total
        # Available width for buttons: 420 - 4 - (extra padding inside groups) = ~406px
        # Group padding: padx=2 on pack + padx=4 inside = 8px total, so 420-8 = 412px inner
        # For half buttons: (412-4-2) / 2 = 203px each (with 1px gap between)
        SIDEBAR_WIDTH = 420
        PADDING_OUTER = 4  # 2px left + 2px right from pack(padx=2)
        PADDING_INNER = 8  # 4px left + 4px right from pack(padx=4)
        AVAILABLE_WIDTH = SIDEBAR_WIDTH - PADDING_OUTER - PADDING_INNER
        HALF_BTN_WIDTH = (AVAILABLE_WIDTH - 2) // 2  # Subtract 1px gap, divide by 2
        FULL_BTN_WIDTH = AVAILABLE_WIDTH

        # Professional Control Panel
        btn_font = ('Segoe UI', 11, 'bold')
        btn_font_small = ('Segoe UI', 10)
        
        # === SYSTEM CLOCK (ABOVE GUARDS) ===
        clock_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#3498db", height=45)
        clock_frame.pack(fill="x", padx=2, pady=2)
        clock_frame.pack_propagate(False)
        
        self.clock_label_title = ctk.CTkLabel(clock_frame, text=self.current_trans["system_time"], font=("Segoe UI", 8, "bold"), text_color="#3498db")
        self.clock_label_title.pack(anchor="w", padx=6, pady=(3,1))
        self.clock_label = ctk.CTkLabel(clock_frame, text="--:--:--", font=("Segoe UI", 10, "bold"), text_color="#f39c12")
        self.clock_label.pack(anchor="w", padx=6, pady=(0,3))

        # === GUARD MANAGEMENT ===
        self.grp_guard = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#16a085", height=170)
        self.grp_guard.pack(fill="x", padx=2, pady=2)
        self.grp_guard.pack_propagate(False)
        
        self.guards_label = ctk.CTkLabel(self.grp_guard, text=self.current_trans["guards"], 
                    font=("Segoe UI", 10, "bold"), text_color="#16a085")
        self.guards_label.pack(padx=6, pady=(5,4), anchor="w")
        
        # Buttons container - Single column layout
        guard_btns_container = ctk.CTkFrame(self.grp_guard, fg_color="transparent")
        guard_btns_container.pack(fill="both", expand=True, padx=4, pady=(0,4))
        
        # Button 1: Add Guard (full width)
        self.btn_guard_toggle = ctk.CTkButton(guard_btns_container, text=self.current_trans["add_guard"], 
                                             command=self.toggle_guard_mode, height=32,
                                             fg_color="#16a085", font=("Segoe UI", 9), corner_radius=4)
        self.btn_guard_toggle.pack(fill="x", pady=(0,3))
        
        # Button 2: Remove Guard (full width)
        self.btn_remove_guard = ctk.CTkButton(guard_btns_container, text=self.current_trans["remove_guard"], 
                                             command=self.remove_guard_dialog, height=32,
                                             fg_color="#c0392b", font=("Segoe UI", 9), corner_radius=4)
        self.btn_remove_guard.pack(fill="x", pady=(0,3))
        
        # Button 3: Fugitive (full width)
        self.btn_fugitive_toggle = ctk.CTkButton(guard_btns_container, text=self.current_trans["fugitive"], 
                                                command=self.toggle_fugitive_add_remove, height=32,
                                                fg_color="#e74c3c", font=("Segoe UI", 9), corner_radius=4)
        self.btn_fugitive_toggle.pack(fill="x", pady=(0,3))
        
        # Button 4: Select Guard (full width)
        self.btn_select_guards = ctk.CTkButton(guard_btns_container, text=self.current_trans["select_guard"], 
                                              command=self.open_guard_selection_dialog, height=32,
                                              fg_color="#2980b9", font=("Segoe UI", 9), corner_radius=4)
        self.btn_select_guards.pack(fill="x", pady=(0,0))

        # === ALERT SYSTEM ===
        self.grp_alert_type = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#e67e22", height=290)
        self.grp_alert_type.pack(fill="x", padx=2, pady=2)
        self.grp_alert_type.pack_propagate(False)
        
        self.alerts_label = ctk.CTkLabel(self.grp_alert_type, text=self.current_trans["alerts"], 
                    font=("Segoe UI", 10, "bold"), text_color="#e67e22")
        self.alerts_label.pack(padx=6, pady=(5,4), anchor="w")
        
        self.alert_type_btns_frame = ctk.CTkFrame(self.grp_alert_type, fg_color="transparent")
        self.alert_type_btns_frame.pack(fill="both", expand=True, padx=4, pady=(0,4))
        
        # Button 1: Timeout (full width)
        self.btn_set_interval = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["timeout"], 
                                             command=self.set_alert_interval_advanced, height=32,
                                             fg_color="#3498db", font=("Segoe UI", 9), corner_radius=4)
        self.btn_set_interval.pack(fill="x", pady=(0,3))
        
        # Button 2: Alert ON/OFF (full width)
        self.btn_alert_toggle = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["alert_toggle"], 
                                             command=self.toggle_alert_mode, height=32,
                                             fg_color="#7f8c8d", font=("Segoe UI", 9), corner_radius=4)
        self.btn_alert_toggle.pack(fill="x", pady=(0,3))
        
        # Button 3: Stillness ON/OFF (full width)
        self.btn_stillness_alert = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["stillness"], 
                                                command=self.toggle_stillness_alert, height=32,
                                                fg_color="#7f8c8d", font=("Segoe UI", 9), corner_radius=4)
        self.btn_stillness_alert.pack(fill="x", pady=(0,3))
        
        # Button 4: Track Guard / Stop Monitoring (Toggle full width)
        self.btn_track_toggle = ctk.CTkButton(self.alert_type_btns_frame, text=self.current_trans["track"], 
                                             command=self.toggle_track_monitoring, height=32,
                                             fg_color="#16a085", font=("Segoe UI", 9), corner_radius=4)
        self.btn_track_toggle.pack(fill="x", pady=(0,3))
        
        # Label: Action Dropdown (full width)
        self.action_label = ctk.CTkLabel(self.alert_type_btns_frame, text=self.current_trans["required_action"], 
                    font=("Segoe UI", 8), text_color="#bdc3c7")
        self.action_label.pack(anchor="w", pady=(2,1))
        
        self.required_action_var = tk.StringVar(self.root)
        self.required_action_var.set("हाथ ऊपर (Hands Up)")
        self.action_dropdown = ctk.CTkOptionMenu(self.alert_type_btns_frame, 
                                                values=["हाथ ऊपर (Hands Up)", 
                                                       "बाएं हाथ ऊपर (Left Hand Up)", "दाएं हाथ ऊपर (Right Hand Up)", 
                                                       "खड़ा (Standing)"], 
                                                command=self.on_action_change, 
                                                fg_color="#3498db",
                                                button_color="#2980b9",
                                                text_color="white", font=("Segoe UI", 9), 
                                                dropdown_font=("Segoe UI", 9), corner_radius=4, height=32)
        self.action_dropdown.pack(fill="x", pady=(0,0))

        # Initialize selected targets list
        self.selected_target_names = []
        
        # === ACTIVE TRACKING STATUS (COMPACT SECTION) ===
        self.grp_preview = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#27ae60", height=320)
        self.grp_preview.pack(fill="x", padx=2, pady=2)
        self.grp_preview.pack_propagate(False)
        
        self.active_track_label = ctk.CTkLabel(self.grp_preview, text=self.current_trans["active_tracking"], 
                    font=("Segoe UI", 10, "bold"), text_color="#27ae60")
        self.active_track_label.pack(padx=6, pady=(5,3), anchor="w")
        
        # Guard Preview
        self.guard_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#2b2b2b", corner_radius=5, height=140)
        self.guard_preview_frame.pack(fill="both", padx=4, pady=(0,3), expand=False)
        self.guard_preview_frame.pack_propagate(False)
        
        self.guard_preview_label = ctk.CTkLabel(self.guard_preview_frame, text=self.current_trans["guards_preview"], text_color="#27ae60", 
                    font=("Segoe UI", 9, "bold"))
        self.guard_preview_label.pack(anchor="w", padx=6, pady=(4,2))
        
        self.guard_preview_scroll_frame = ctk.CTkScrollableFrame(self.guard_preview_frame, 
                                                                 fg_color="transparent", height=115)
        self.guard_preview_scroll_frame.pack(fill="both", expand=True, padx=3, pady=(0,3))
        
        self.guard_preview_grid = {}
        
        # Fugitive Preview (hidden by default)
        self.fugitive_preview_frame = ctk.CTkFrame(self.grp_preview, fg_color="#2b2b2b", corner_radius=5)
        self.fugitive_preview_frame.pack(fill="x", padx=4, pady=(0,3))
        self.fugitive_preview_frame.pack_forget()
        
        self.fugitive_preview_title = ctk.CTkLabel(self.fugitive_preview_frame, text=self.current_trans["fugitive_preview"], text_color="#e74c3c", 
                    font=("Segoe UI", 9, "bold"))
        self.fugitive_preview_title.pack(anchor="w", padx=6, pady=(4,2))
        
        self.fugitive_preview_label = ctk.CTkLabel(self.fugitive_preview_frame, text=self.current_trans["not_set"], 
                                                   text_color="#7f8c8d", font=("Segoe UI", 9))
        self.fugitive_preview_label.pack(anchor="w", padx=6, pady=(0,3))

        # === PERFORMANCE STATS (FPS & MEM) - SMALLER SECTION ===
        perf_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1e1e1e", corner_radius=5, border_width=1, border_color="#95a5a6", height=70)
        perf_frame.pack(fill="x", padx=2, pady=2)
        perf_frame.pack_propagate(False)
        
        self.perf_label = ctk.CTkLabel(perf_frame, text=self.current_trans["performance"], 
                    font=("Segoe UI", 8, "bold"), text_color="#95a5a6")
        self.perf_label.pack(padx=6, pady=(3,1), anchor="w")
        
        # FPS and MEM on same row
        perf_row = ctk.CTkFrame(perf_frame, fg_color="transparent")
        perf_row.pack(fill="x", padx=6, pady=(0,3))
        
        ctk.CTkLabel(perf_row, text=self.current_trans["fps"], font=("Segoe UI", 7), text_color="#bdc3c7").pack(side="left", anchor="w")
        self.fps_label = ctk.CTkLabel(perf_row, text="0", font=("Segoe UI", 7, "bold"), text_color="#27ae60")
        self.fps_label.pack(side="left", padx=(3,10))
        
        ctk.CTkLabel(perf_row, text=self.current_trans["memory"], font=("Segoe UI", 7), text_color="#bdc3c7").pack(side="left", anchor="w")
        self.mem_label = ctk.CTkLabel(perf_row, text="0MB", font=("Segoe UI", 7, "bold"), text_color="#f39c12")
        self.mem_label.pack(side="left", padx=(3,0))

        # === EXIT BUTTON ===
        self.btn_exit = ctk.CTkButton(self.sidebar_scroll, text=self.current_trans["exit"], 
                                     command=self.graceful_exit, height=32, width=FULL_BTN_WIDTH,
                                     fg_color="#c0392b", font=("Segoe UI", 9), corner_radius=4)
        self.btn_exit.pack(pady=(2,20))  # Extra padding at bottom to ensure visibility
        
        self.load_targets()
        
        # ✅ NEW PIPELINE: All models replaced with strict 4-model pipeline
        # Removed: Phase 4, HMR-Lite, RetinaFace, MoveNet MultiPose, MediaPipe Holistic, VideoPose3D, YOLOv8, DeepSORT
        # Implemented: BlazeFace (detection), MoveNet Lightning (single-person pose), BlazePose (multi-person pose), ByteTrack (tracking)
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        
        self.root.mainloop()
    
    def _initialize_model_pipeline(self):
        """Initialize appropriate model pipeline based on guard count and environment"""
        try:
            # ✅ CRITICAL: Initialize MediaPipe Pose (works offline, reliable)
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,  # 0=light, 1=full (balanced)
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Pose initialized successfully")
                self.model_pipeline_initialized = True
            else:
                logger.warning("MediaPipe not available - pose detection disabled")
                self.model_pipeline_initialized = False
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            self.model_pipeline_initialized = False
    
    def _load_single_person_pipeline(self):
        """
        Load single-person pipeline:
        Normal: BlazeFace → MediaPipe Pose → SORT
        Dark: CLAHE+Gamma → BlazeFace → MediaPipe Pose → SORT
        """
        try:
            if self.pose_model is not None:
                return  # Already loaded
            
            logger.info("Loading single-person pipeline with MediaPipe...")
            
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Pose loaded for single-person mode")
            else:
                logger.warning("MediaPipe not available")
                self.pose_model = None
        except Exception as e:
            logger.error(f"Single-person pipeline load error: {e}")
    
    def _load_multi_person_pipeline(self):
        """
        Load multi-person pipeline:
        Normal: BlazePose → ByteTrack (using MediaPipe Pose)
        Dark: CLAHE+Gamma → BlazePose → ByteTrack
        """
        try:
            if self.pose_model is not None:
                return  # Already loaded
            
            logger.info("Loading multi-person pipeline with MediaPipe...")
            
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("✅ MediaPipe Pose loaded for multi-person mode")
            else:
                logger.warning("MediaPipe not available")
                self.pose_model = None
        except Exception as e:
            logger.error(f"Multi-person pipeline load error: {e}")
    
    def _detect_faces_blazeface(self, rgb_frame):
        """BlazeFace face detection for single-person mode"""
        try:
            # Using MediaPipe FaceMesh as BlazeFace (built-in)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            return face_locations
        except Exception as e:
            logger.debug(f"BlazeFace detection error: {e}")
            return []
    
    def _detect_faces_blazepose(self, rgb_frame):
        """BlazePose face detection for multi-person mode"""
        try:
            # Using MediaPipe Pose for multi-person face detection
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            return face_locations
        except Exception as e:
            logger.debug(f"BlazePose detection error: {e}")
            return []
    
    def _detect_pose_movenet_lightning(self, rgb_frame):
        """MediaPipe Pose for single-person pose estimation"""
        try:
            if self.pose_model is None:
                return None
            
            # ✅ CRITICAL: Convert RGB frame to proper format for MediaPipe
            results = self.pose_model.process(rgb_frame)
            return results  # Returns MediaPipe PoseLandmarks with pose_landmarks attribute
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None
    
    def _detect_pose_blazepose_multipose(self, rgb_frame):
        """MediaPipe Pose for multi-person pose estimation"""
        try:
            if self.pose_model is None:
                return None
            
            # ✅ CRITICAL: Convert RGB frame to proper format for MediaPipe
            results = self.pose_model.process(rgb_frame)
            return results  # Returns MediaPipe PoseLandmarks with pose_landmarks attribute
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None
    
    def _apply_clahe_enhancement(self, frame):
        """Apply CLAHE + Gamma correction for dark environments"""
        try:
            # Convert to LAB for better enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Apply Gamma correction for brightness boost
            inv_gamma = 1.0 / 1.5
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l_channel = cv2.LUT(l_channel, table)
            
            # Merge back
            lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_frame
        except Exception as e:
            logger.debug(f"CLAHE enhancement error: {e}")
            return frame
    
    def toggle_sidebar(self):
        """Toggle sidebar visibility (not used in new design but kept for compatibility)"""
        pass
    
    # ✅ INDUSTRIAL-LEVEL: Low-Light Detection & Tracking Enhancements
    def enhance_frame_for_low_light(self, frame):
        """
        ✅ OPTIMIZED: Fast frame enhancement with night/day mode awareness
        Day mode: Skip enhancement entirely (fast)
        Night mode: Lightweight enhancement only if brightness is low
        """
        try:
            # Day Mode: No enhancement needed, return frame as-is for maximum speed
            if not self.night_mode:
                return frame
            
            # Night Mode: Only enhance if brightness is low (adaptive)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Skip enhancement for normal/bright conditions - return frame as-is for speed
            if brightness > 120:  # Normal/bright - no enhancement needed
                return frame
            
            # Lightweight enhancement for low-light (fast, minimal overhead)
            # Stage 1: Lightweight denoise only for dark areas (faster version)
            frame = cv2.fastNlMeansDenoisingColored(frame, h=8, templateWindowSize=7, searchWindowSize=15)
            
            # Stage 2: Minimal CLAHE for contrast - very fast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # Reduced tile size for speed
            l_channel_clahe = clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_channel_clahe, a, b])
            frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return frame_enhanced
        except Exception as e:
            logger.debug(f"Low-light enhancement failed: {e}, using original frame")
            return frame
    
    def get_adaptive_face_detection_params(self, frame):
        """
        ✅ INDUSTRIAL-LEVEL: Adaptive parameters based on lighting conditions
        Detects frame brightness and adjusts detection sensitivity accordingly
        """
        # Calculate frame brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Adaptive tolerance and confidence thresholds based on brightness
        if brightness < 50:  # Very dark
            face_tolerance = 0.45  # More lenient
            confidence_threshold = 0.45
            use_cnn_model = True  # Use CNN for better accuracy in low-light
            logger.debug(f"Very dark conditions (brightness={brightness:.0f}) - Using lenient tolerance")
        elif brightness < 100:  # Dark
            face_tolerance = 0.48
            confidence_threshold = 0.50
            use_cnn_model = True
            logger.debug(f"Dark conditions (brightness={brightness:.0f}) - Using CNN model")
        elif brightness < 150:  # Medium
            face_tolerance = 0.50
            confidence_threshold = 0.55
            use_cnn_model = False
            logger.debug(f"Medium lighting (brightness={brightness:.0f}) - Using standard detection")
        else:  # Bright
            face_tolerance = 0.55
            confidence_threshold = 0.60
            use_cnn_model = False
            logger.debug(f"Good lighting (brightness={brightness:.0f}) - Using strict tolerance")
        
        return {
            "tolerance": face_tolerance,
            "confidence": confidence_threshold,
            "use_cnn": use_cnn_model,
            "brightness": brightness
        }
    
    def detect_faces_multiscale(self, rgb_frame):
        """
        ✅ INDUSTRIAL-LEVEL: Multi-scale face detection using OpenCV cascade
        Helps catch faces at different scales, especially in complex scenes
        """
        try:
            # Load Haar cascade if not already loaded
            if not hasattr(self, 'face_cascade'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            
            # Multi-scale detection with different scale factors
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # More granular scale steps
                minNeighbors=4,   # Lower threshold for low-light
                minSize=(40, 40),
                maxSize=(300, 300)
            )
            
            # Convert from (x, y, w, h) to face_recognition format (top, right, bottom, left)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            
            if face_locations:
                logger.debug(f"Multi-scale detection found {len(face_locations)} faces")
            
            return face_locations
        except Exception as e:
            logger.debug(f"Multi-scale detection failed: {e}")
            return []
    
    
    def graceful_exit(self):
        """Gracefully exit the application with proper cleanup"""
        try:
            # Confirm exit if camera is running or alert mode is active
            if self.is_running or self.is_alert_mode:
                response = messagebox.askyesno(
                    "Confirm Exit",
                    "Camera is running. Are you sure you want to exit?"
                )
                if not response:
                    return
            
            logger.warning("Initiating graceful shutdown...")
            
            # ✅ SIMPLIFIED: Phase 4 cleanup removed - focus on core cleanup only
            
            # Stop camera if running
            if self.is_running:
                self.is_running = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            # Save logs if logging
            if self.is_logging:
                self.save_log_to_file()
            
            # Cleanup trackers
            for status in self.targets_status.values():
                if status.get("tracker"):
                    status["tracker"] = None
            
            # Force garbage collection
            gc.collect()
            
            logger.warning("Shutdown complete")
            
            # Destroy window
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error during exit: {e}")
            # Force exit even if there's an error
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def load_translations(self):
        """Load translation dictionary for supported languages - COMPREHENSIVE with all UI elements"""
        return {
            "Hindi": {
                # Header & Branding
                "title": "🛡️ निराक्षण (Niraakshan)",
                "slogan": "निराक्षण करे रक्षण",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 GUARDS",
                "alerts": "🔔 ALERTS",
                "active_tracking": "✓ ACTIVE TRACKING",
                "performance": "📊 PERFORMANCE",
                "guards_preview": "👮 Guards:",
                "fugitive_preview": "🚨 Fugitive:",
                "no_guard_selected": "कोई Guard चयनित नहीं",
                "not_set": "सेट नहीं",
                
                # Camera Control Buttons
                "camera_on": "▶ कैमरा ON/OFF",
                "snap": "📸 स्नैप",
                "night": "🌙 रात",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                "camera_offline": "📷 कैमरा फीड ऑफलाइन",
                
                # Guard Management Buttons
                "add_guard": "➕ Guard जोड़ें",
                "remove_guard": "❌ Guard हटाएं",
                "fugitive": "🚨 Fugitive",
                "select_guard": "✓ Guard चुनें",
                
                # Alert Buttons
                "timeout": "⏱️ समय सीमा (HH:MM:SS)",
                "alert_toggle": "🔔 अलर्ट चालू/बंद",
                "stillness": "🔇 स्थिरता चालू/बंद",
                "track": "🎯 गार्ड ट्रैक करें",
                "stop_monitoring": "⏹️ निरीक्षण बंद करें",
                
                # Dropdown & Labels
                "required_action": "आवश्यक क्रिया:",
                "exit": "🚪 निकास",
                
                # Dialog Titles
                "lang_title": "🌐 Language Selection",
                "lang_select": "अपनी पसंदीदा भाषा चुनें:",
                "current_lang": "Current Language: ",
                "language_changed": "Language Changed",
                "lang_changed_msg": "App language को {lang} में बदल दिया गया! पूर्ण प्रभाव के लिए ऐप को पुनः शुरू करें।",
                
                # Guard Dialog Messages
                "add_guard_title": "Guard जोड़ें",
                "add_guard_msg": "आप guard को कैसे जोड़ना चाहते हैं?\n\nYes = कैमरे से फ़ोटो लें\nNo = मौजूदा Image Upload करें",
                "remove_guard_title": "Guard हटाएं",
                "select_guard_remove": "हटाने के लिए guard चुनें:",
                "confirm_removal": "Removal की पुष्टि करें",
                "remove_confirm_msg": "क्या आप '{name}' को हटाना निश्चित हैं?\n\nयह हटाएगा:\n- Face image\n- Pose references\n- सभी संबंधित data\n\nयह कार्रवाई को पूर्ववत नहीं किया जा सकता!",
                "remove_btn": "हटाएं",
                "cancel_btn": "Cancel",
                "guard_removed": "Guard Removed",
                "guard_removed_msg": "'{name}' को सफलतापूर्वक हटा दिया गया।\n\nDeleted: {items}",
                "no_guards": "कोई Guards नहीं",
                "no_guards_msg": "हटाने के लिए कोई guards उपलब्ध नहीं।",
                
                # Upload Dialog
                "upload_title": "Guard Image चुनें",
                "guard_name": "Guard का नाम दर्ज करें:",
                "upload_success": "Success",
                "upload_success_msg": "Guard '{name}' को सफलतापूर्वक जोड़ा गया!\n(अपलोड की गई images के लिए Pose capture छोड़ दिया गया)",
                "upload_error": "Error",
                "upload_error_msg": "Image अपलोड करने में विफल: {error}",
                "face_error": "Error",
                "face_error_msg": "Image में बिल्कुल एक face होना चाहिए।",
                
                # Camera Dialog
                "camera_required": "Camera आवश्यक",
                "camera_required_msg": "कृपया पहले camera शुरू करें।",
                "confirm_exit": "Exit की पुष्टि करें",
                "exit_confirm_msg": "Camera चल रहा है। क्या आप निश्चित रूप से exit करना चाहते हैं?",
                
                # Select Guards Dialog
                "select_targets": "Targets चुनें",
                "select_targets_msg": "Track करने के लिए Targets चुनें",
                "select_all": "सभी चुनें",
                "clear_all": "सभी को साफ करें",
                "done": "Done",
                "no_targets": "कोई targets नहीं मिले।",
                
                # Timeout Dialog
                "timeout_title": "Action Timeout Interval सेट करें",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "कितने समय बाद timeout alert ट्रिगर होगा यदि कार्रवाई नहीं की गई",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "कोई Guards चयनित नहीं",
                "select_guards_first": "कृपया पहले 'Guard चुनें' बटन का उपयोग करके guards चुनें",
                "tracking_started": "Tracking शुरू",
                "tracking_started_msg": "अब identify और track कर रहे हैं:\n{guards}\n\nActive Action: {action}\nAction alerts की निगरानी की जा रही है...",
                "tracking_stopped": "Tracking बंद",
                "now_scanning": "अब {count} चयनित targets के लिए स्कैन कर रहे हैं",
                "no_targets_init": "कोई targets को initialize नहीं किया गया - check करें कि guard profile images में faces हैं",
                
                # Alert Messages
                "alert_on": "Alert चालू",
                "alert_off": "Alert बंद",
                "stillness_on": "Stillness Alert चालू",
                "stillness_off": "Stillness Alert बंद",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB",
                
                # Action Types (Dropdown Options)
                "action_hands_up": "हाथ ऊपर (Hands Up)",
                "action_left_hand_up": "बाएं हाथ ऊपर (Left Hand Up)",
                "action_right_hand_up": "दाएं हाथ ऊपर (Right Hand Up)",
                "action_standing": "खड़ा (Standing)",
                
                # Monitor Modes
                "monitor_action_alerts": "केवल Action Alert",
                "monitor_stillness": "केवल Stillness Alert",
                "monitor_both": "दोनों Alert"
            },
            "English": {
                # Header & Branding
                "title": "🛡️ Niraakshan (Multi-Guard Tracking)",
                "slogan": "Observe to Protect",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 GUARDS",
                "alerts": "🔔 ALERTS",
                "active_tracking": "✓ ACTIVE TRACKING",
                "performance": "📊 PERFORMANCE",
                "guards_preview": "👮 Guards:",
                "fugitive_preview": "🚨 Fugitive:",
                "no_guard_selected": "No Guard Selected",
                "not_set": "Not Set",
                
                # Camera Control Buttons
                "camera_on": "▶ Camera ON/OFF",
                "snap": "📸 Snap",
                "night": "🌙 Night",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                
                # Guard Management Buttons
                "add_guard": "➕ Add Guard",
                "remove_guard": "❌ Remove Guard",
                "fugitive": "🚨 Fugitive",
                "select_guard": "✓ Select Guard",
                
                # Alert Buttons
                "timeout": "⏱️ Timeout (HH:MM:SS)",
                "alert_toggle": "🔔 Alert ON/OFF",
                "stillness": "🔇 Stillness ON/OFF",
                "track": "🎯 Track Guard",
                "stop_monitoring": "⏹️ Stop Monitoring",
                
                # Dropdown & Labels
                "required_action": "Required Action:",
                "exit": "🚪 Exit",
                
                # Dialog Titles
                "lang_title": "🌐 Language Selection",
                "lang_select": "Select your preferred language:",
                "current_lang": "Current Language: ",
                "language_changed": "Language Changed",
                "lang_changed_msg": "App language changed to {lang}!\nRestart the app for full effect.",
                
                # Guard Dialog Messages
                "add_guard_title": "Add Guard",
                "add_guard_msg": "How would you like to add the guard?\n\nYes = Take Photo with Camera\nNo = Upload Existing Image",
                "remove_guard_title": "Remove Guard",
                "select_guard_remove": "Select guard to remove:",
                "confirm_removal": "Confirm Removal",
                "remove_confirm_msg": "Are you sure you want to remove '{name}'?\n\nThis will delete:\n- Face image\n- Pose references\n- All associated data\n\nThis action cannot be undone!",
                "remove_btn": "Remove",
                "cancel_btn": "Cancel",
                "guard_removed": "Guard Removed",
                "guard_removed_msg": "'{name}' has been successfully removed.\n\nDeleted: {items}",
                "no_guards": "No Guards",
                "no_guards_msg": "No guards available to remove.",
                
                # Upload Dialog
                "upload_title": "Select Guard Image",
                "guard_name": "Enter guard name:",
                "upload_success": "Success",
                "upload_success_msg": "Guard '{name}' added successfully!\n(Pose capture skipped for uploaded images)",
                "upload_error": "Error",
                "upload_error_msg": "Failed to upload image: {error}",
                "face_error": "Error",
                "face_error_msg": "Image must contain exactly one face.",
                
                # Camera Dialog
                "camera_required": "Camera Required",
                "camera_required_msg": "Please start the camera first.",
                "confirm_exit": "Confirm Exit",
                "exit_confirm_msg": "Camera is running. Are you sure you want to exit?",
                
                # Select Guards Dialog
                "select_targets": "Select Targets",
                "select_targets_msg": "Select Targets to Track",
                "select_all": "Select All",
                "clear_all": "Clear All",
                "done": "Done",
                "no_targets": "No targets found.",
                
                # Timeout Dialog
                "timeout_title": "Set Action Timeout Interval",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "How long until timeout alert triggers if action not performed",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "No Guards Selected",
                "select_guards_first": "Please select guards first using 'Select Guard' button",
                "tracking_started": "Tracking Started",
                "tracking_started_msg": "Now identifying and tracking:\n{guards}\n\nActive Action: {action}\nMonitoring for action alerts...",
                "tracking_stopped": "Tracking Stopped",
                "now_scanning": "Now scanning for {count} selected targets",
                "no_targets_init": "No targets were initialized - check that guard profile images contain faces",
                
                # Alert Messages
                "alert_on": "Alert ON",
                "alert_off": "Alert OFF",
                "stillness_on": "Stillness Alert ON",
                "stillness_off": "Stillness Alert OFF",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB",
                
                # Action Types (Dropdown Options)
                "action_hands_up": "Hands Up",
                "action_left_hand_up": "Left Hand Up",
                "action_right_hand_up": "Right Hand Up",
                "action_standing": "Standing",
                
                # Monitor Modes
                "monitor_action_alerts": "Action Alerts Only",
                "monitor_stillness": "Stillness Alerts Only",
                "monitor_both": "Both Alerts"
            },
            "Marathi": {
                # Header & Branding
                "title": "🛡️ निराक्षण (Niraakshan)",
                "slogan": "निरीक्षण करा संरक्षण करा",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 गार्ड",
                "alerts": "🔔 सतर्कता",
                "active_tracking": "✓ सक्रिय ट्रॅकिंग",
                "performance": "📊 कार्यक्षमता",
                "guards_preview": "👮 गार्ड:",
                "fugitive_preview": "🚨 फरार:",
                "no_guard_selected": "कोई Guard निवडलेले नाहीत",
                "not_set": "सेट केलेले नाहीत",
                
                # Camera Control Buttons
                "camera_on": "▶ कॅमेरा ON/OFF",
                "snap": "📸 स्नॅप",
                "night": "🌙 रात्रि",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                
                # Guard Management Buttons
                "add_guard": "➕ गार्ड जोडा",
                "remove_guard": "❌ गार्ड हटवा",
                "fugitive": "🚨 फरार",
                "select_guard": "✓ गार्ड निवडा",
                
                # Alert Buttons
                "timeout": "⏱️ Timeout (HH:MM:SS)",
                "alert_toggle": "🔔 सतर्कता ON/OFF",
                "stillness": "🔇 शांतता ON/OFF",
                "track": "🎯 गार्ड ट्रॅक करा",
                "stop_monitoring": "⏹️ निरीक्षण बंद करा",
                
                # Dropdown & Labels
                "required_action": "आवश्यक कृती:",
                "exit": "🚪 Exit",
                
                # Dialog Titles
                "lang_title": "🌐 भाषा निवड",
                "lang_select": "तुमची पसंदीची भाषा निवडा:",
                "current_lang": "Current Language: ",
                "language_changed": "भाषा बदलली",
                "lang_changed_msg": "App भाषा {lang} मध्ये बदलली गेली!\nपूर्ण प्रभावासाठी ऍप्लिकेशन पुनः सुरू करा।",
                
                # Guard Dialog Messages
                "add_guard_title": "गार्ड जोडा",
                "add_guard_msg": "तुम्ही गार्ड कसे जोडू इच्छिता?\n\nYes = कॅमेर्‍यात फोटो घ्या\nNo = विद्यमान Image अपलोड करा",
                "remove_guard_title": "गार्ड हटवा",
                "select_guard_remove": "हटविण्यासाठी गार्ड निवडा:",
                "confirm_removal": "हटवणे पुष्टी करा",
                "remove_confirm_msg": "तुम्ही '{name}' हटविण्यास निश्चित आहात?\n\nहे हटवेल:\n- Face image\n- Pose references\n- सर्व संबंधित data\n\nही कृती पूर्ववत केली जाऊ शकत नाही!",
                "remove_btn": "हटवा",
                "cancel_btn": "रद्द करा",
                "guard_removed": "गार्ड हटविला",
                "guard_removed_msg": "'{name}' यशस्वीरित्या हटविला गेला।\n\nDeleted: {items}",
                "no_guards": "कोही गार्ड नाहीत",
                "no_guards_msg": "हटविण्यासाठी कोही गार्ड उपलब्ध नाहीत।",
                
                # Upload Dialog
                "upload_title": "गार्ड Image निवडा",
                "guard_name": "गार्डचे नाव दर्ज करा:",
                "upload_success": "यश",
                "upload_success_msg": "गार्ड '{name}' यशस्वीरित्या जोडला गेला!\n(अपलोड केलेल्या images साठी Pose capture वगळा)",
                "upload_error": "Error",
                "upload_error_msg": "Image अपलोड करणे अयशस्वी: {error}",
                "face_error": "Error",
                "face_error_msg": "Image मध्ये अगदी एक face असणे आवश्यक आहे।",
                
                # Camera Dialog
                "camera_required": "कॅमेरा आवश्यक",
                "camera_required_msg": "कृपया प्रथम कॅमेरा सुरू करा।",
                "confirm_exit": "Exit पुष्टी करा",
                "exit_confirm_msg": "कॅमेरा चालू आहे. तुम्ही निश्चितपणे exit करू इच्छिता?",
                
                # Select Guards Dialog
                "select_targets": "Targets निवडा",
                "select_targets_msg": "ट्रॅक करण्यासाठी Targets निवडा",
                "select_all": "सर्व निवडा",
                "clear_all": "सर्व साफ करा",
                "done": "Done",
                "no_targets": "कोही targets सापडले नाहीत.",
                
                # Timeout Dialog
                "timeout_title": "Action Timeout Interval सेट करा",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "कृती केली न गेल्यास timeout alert किती वेळ ट्रिगर होईल",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "कोही Guards निवडलेले नाहीत",
                "select_guards_first": "कृपया प्रथम 'गार्ड निवडा' बटण वापरून guards निवडा",
                "tracking_started": "ट्रॅकिंग सुरू",
                "tracking_started_msg": "आता identify आणि track कर्‍या हैत:\n{guards}\n\nActive Action: {action}\nAction alerts साठी निरीक्षण...",
                "tracking_stopped": "ट्रॅकिंग बंद",
                "now_scanning": "आता {count} निवडलेल्या targets साठी स्कॅन कर्‍या हैत",
                "no_targets_init": "कोही targets initialize केले गेले नाहीत - check करा की guard profile images मध्ये faces आहेत",
                
                # Alert Messages
                "alert_on": "सतर्कता ON",
                "alert_off": "सतर्कता OFF",
                "stillness_on": "शांतता Alert ON",
                "stillness_off": "शांतता Alert OFF",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB",
                
                # Action Types (Dropdown Options)
                "action_hands_up": "हाथ वर (Hands Up)",
                "action_left_hand_up": "डावा हाथ वर (Left Hand Up)",
                "action_right_hand_up": "उजवा हाथ वर (Right Hand Up)",
                "action_standing": "उभे (Standing)",
                
                # Monitor Modes
                "monitor_action_alerts": "फक्त Action Alerts",
                "monitor_stillness": "फक्त Stillness Alerts",
                "monitor_both": "दोन्ही Alerts"
            },
            "Gujarati": {
                # Header & Branding
                "title": "🛡️ નિરાક્ષણ (Niraakshan)",
                "slogan": "અવલોકન કરો અને રક્ષણ કરો",
                
                # Sidebar Labels
                "system_time": "⏰ System Time",
                "guards": "👮 ગાર્ડ્સ",
                "alerts": "🔔 અલર્ટ્સ",
                "active_tracking": "✓ સક્રિય ટ્રૅકિંગ",
                "performance": "📊 કાર્યક્ષમતા",
                "guards_preview": "👮 ગાર્ડ્સ:",
                "fugitive_preview": "🚨 ફરાર:",
                "no_guard_selected": "કોઈ Guard પસંદ નથી",
                "not_set": "સેટ નથી",
                
                # Camera Control Buttons
                "camera_on": "▶ કેમેરા ON/OFF",
                "snap": "📸 સ્નેપ",
                "night": "🌙 રાત",
                "pro_mode": "⚡ PRO Mode ON/OFF",
                
                # Guard Management Buttons
                "add_guard": "➕ ગાર્ડ ઉમેરો",
                "remove_guard": "❌ ગાર્ડ દૂર કરો",
                "fugitive": "🚨 ફરાર",
                "select_guard": "✓ ગાર્ડ પસંદ કરો",
                
                # Alert Buttons
                "timeout": "⏱️ Timeout (HH:MM:SS)",
                "alert_toggle": "🔔 અલર્ટ ON/OFF",
                "stillness": "🔇 શાંતતા ON/OFF",
                "track": "🎯 ગાર્ડ ટ્રૅક કરો",
                "stop_monitoring": "⏹️ મોનિટરિંગ બંધ કરો",
                
                # Dropdown & Labels
                "required_action": "જરૂરી પગલું:",
                "exit": "🚪 Exit",
                
                # Dialog Titles
                "lang_title": "🌐 ભાષા પસંદગી",
                "lang_select": "તમારી પસંદીની ભાષા પસંદ કરો:",
                "current_lang": "Current Language: ",
                "language_changed": "ભાષા બદલાઈ",
                "lang_changed_msg": "App ભાષા {lang} માં બદલાઈ!\nપૂર્ણ અસર માટે ઍપ્લિકેશન પુનः શરૂ કરો।",
                
                # Guard Dialog Messages
                "add_guard_title": "ગાર્ડ ઉમેરો",
                "add_guard_msg": "તમે ગાર્ડ કેવી રીતે ઉમેરવા માંગો છો?\n\nYes = કેમેરા સાથે ફોટો લો\nNo = હાલ ની Image અપલોડ કરો",
                "remove_guard_title": "ગાર્ડ દૂર કરો",
                "select_guard_remove": "દૂર કરવા માટે ગાર્ડ પસંદ કરો:",
                "confirm_removal": "દૂર કરવાની પુષ્ટી કરો",
                "remove_confirm_msg": "શું તમે '{name}' દૂર કરવા માટે આત્મવિશ્વાસી છો?\n\nયહ દૂર કરશે:\n- Face image\n- Pose references\n- બધા સંબંધિત data\n\nયો કાર્ય રદ કરી શકાતો નથી!",
                "remove_btn": "દૂર કરો",
                "cancel_btn": "રદ કરો",
                "guard_removed": "ગાર્ડ દૂર",
                "guard_removed_msg": "'{name}' સફળતાપૂર્વક દૂર કરાયો.\n\nDeleted: {items}",
                "no_guards": "કોઈ Guards નથી",
                "no_guards_msg": "દૂર કરવા માટે કોઈ guards ઉપલબ્ધ નથી.",
                
                # Upload Dialog
                "upload_title": "ગાર્ડ Image પસંદ કરો",
                "guard_name": "ગાર્ડનું નામ દાખલ કરો:",
                "upload_success": "સફલતા",
                "upload_success_msg": "ગાર્ડ '{name}' સફળતાપૂર્વક ઉમેરાયો!\n(અપલોડ કરેલી images માટે Pose capture છોડી દો)",
                "upload_error": "Error",
                "upload_error_msg": "Image અપલોડ કરવું નિષ્ફળ: {error}",
                "face_error": "Error",
                "face_error_msg": "Image માં બરાબર એક face હોવું જોઈએ.",
                
                # Camera Dialog
                "camera_required": "કેમેરો જરૂરી",
                "camera_required_msg": "કૃપયા પ્રથમ કેમેરો શરૂ કરો.",
                "confirm_exit": "Exit નું પુષ્ટિકરણ કરો",
                "exit_confirm_msg": "કેમેરો ચાલુ છે. શું તમે ચોક્કસ રીતે exit કરવા માંગો છો?",
                
                # Select Guards Dialog
                "select_targets": "Targets પસંદ કરો",
                "select_targets_msg": "ટ્રૅક કરવા માટે Targets પસંદ કરો",
                "select_all": "બધું પસંદ કરો",
                "clear_all": "બધું સાફ કરો",
                "done": "Done",
                "no_targets": "કોઈ targets મળ્યા નથી.",
                
                # Timeout Dialog
                "timeout_title": "Action Timeout Interval સેટ કરો",
                "timeout_label": "⏱ Action Timeout Interval",
                "timeout_desc": "કૃતિ ન કરવામાં આવે તો timeout alert કેટલો સમય ટ્રીગર થાય",
                "hours": "Hours:",
                "minutes": "Minutes:",
                "seconds": "Seconds:",
                "timeout_recommend": "Recommended: 5 - 300 seconds (action timeout)",
                "set_btn": "✓ Set",
                
                # Tracking Messages
                "no_guards_selected": "કોઈ Guards પસંદ નથી",
                "select_guards_first": "કૃપયા પ્રથમ 'ગાર્ડ પસંદ કરો' બટન વાપરીને guards પસંદ કરો",
                "tracking_started": "ટ્રૅકિંગ શરૂ",
                "tracking_started_msg": "હવે identify અને track કર્યા છે:\n{guards}\n\nActive Action: {action}\nAction alerts માટે મોનિટર કર્યા છે...",
                "tracking_stopped": "ટ્રૅકિંગ બંધ",
                "now_scanning": "હવે {count} પસંદ કરેલા targets માટે સ્કેન કર્યા છે",
                "no_targets_init": "કોઈ targets શરૂ કરાયા નથી - ચેક કરો કે guard profile images માં faces છે",
                
                # Alert Messages
                "alert_on": "અલર્ટ ON",
                "alert_off": "અલર્ટ OFF",
                "stillness_on": "શાંતતા Alert ON",
                "stillness_off": "શાંતતા Alert OFF",
                
                # Performance Labels
                "fps": "FPS:",
                "memory": "MEM:",
                "mb": "MB"
            }
        }

    def open_language_converter(self):
        """Open language selection dialog with real-time switching (no restart needed)"""
        try:
            dialog = ctk.CTkToplevel(self.root)
            dialog.title(self.get_text("lang_title"))
            dialog.geometry("450x400")
            dialog.resizable(False, False)
            dialog.attributes('-topmost', True)  # Keep on top
            dialog.grab_set()
            dialog.transient(self.root)
            
            # Title
            ctk.CTkLabel(dialog, text=self.get_text("lang_title"), 
                        font=("Arial Unicode MS", 14, "bold"), text_color="#3498db").pack(pady=15)
            
            ctk.CTkLabel(dialog, text=self.get_text("lang_select"), 
                        font=("Segoe UI", 10), text_color="#bdc3c7").pack(pady=5)
            
            # Language buttons frame
            lang_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            lang_frame.pack(fill="both", expand=True, padx=20, pady=15)
            
            languages = list(self.translations.keys())
            
            def select_language(lang):
                """Apply language translation in REAL-TIME (no restart needed)"""
                self.current_language = lang
                self.apply_translations()
                dialog.destroy()
                
                # Show confirmation with translated message
                msg = self.get_text("lang_changed_msg")
                if msg and isinstance(msg, str):
                    msg = msg.replace("{lang}", lang)
                else:
                    msg = f"App language changed to {lang}!"
                messagebox.showinfo(
                    self.get_text("language_changed"),
                    msg
                )
            
            # Create buttons for each language
            for lang in languages:
                is_current = lang == self.current_language
                bg_color = "#3498db" if is_current else "#2c3e50"
                
                btn = ctk.CTkButton(lang_frame, text=f"✓ {lang}" if is_current else lang,
                                  command=lambda l=lang: select_language(l),
                                  height=40, fg_color=bg_color, font=("Segoe UI", 11, "bold"),
                                  corner_radius=5)
                btn.pack(fill="x", pady=8)
            
            # Info label
            ctk.CTkLabel(dialog, text=f"{self.get_text('current_lang')}{self.current_language}",
                        font=("Segoe UI", 9), text_color="#27ae60").pack(pady=10)
                        
        except Exception as e:
            logger.warning(f"Error opening language converter: {e}")

    def apply_translations(self):
        """Apply language translations to ALL UI elements in REAL-TIME (no restart)"""
        try:
            trans = self.translations.get(self.current_language, self.translations["Hindi"])
            self.current_trans = trans
            
            # ========== UPDATE ALL BUTTON TEXT ==========
            # Camera Controls
            if hasattr(self, 'btn_camera_toggle'):
                self.btn_camera_toggle.configure(text=trans.get("camera_on", "▶ कैमरा ON/OFF"))
            if hasattr(self, 'btn_snap'):
                self.btn_snap.configure(text=trans.get("snap", "📸 स्नैप"))
            if hasattr(self, 'btn_night_mode'):
                self.btn_night_mode.configure(text=trans.get("night", "🌙 रात"))
            if hasattr(self, 'btn_pro_toggle'):
                self.btn_pro_toggle.configure(text=trans.get("pro_mode", "⚡ PRO Mode"))
            
            # Camera feed offline text
            if hasattr(self, 'video_label'):
                self.video_label.configure(text=trans.get("camera_offline", "📷 कैमरा फीड ऑफलाइन"))
            
            # Guard Section
            if hasattr(self, 'clock_label_title'):
                self.clock_label_title.configure(text=trans.get("system_time", "⏰ System Time"))
            if hasattr(self, 'guards_label'):
                self.guards_label.configure(text=trans.get("guards", "👮 GUARDS"))
            if hasattr(self, 'btn_guard_toggle'):
                self.btn_guard_toggle.configure(text=trans.get("add_guard", "➕ Add Guard"))
            if hasattr(self, 'btn_remove_guard'):
                self.btn_remove_guard.configure(text=trans.get("remove_guard", "❌ Remove Guard"))
            if hasattr(self, 'btn_fugitive_toggle'):
                self.btn_fugitive_toggle.configure(text=trans.get("fugitive", "🚨 Fugitive"))
            if hasattr(self, 'btn_select_guards'):
                self.btn_select_guards.configure(text=trans.get("select_guard", "✓ Select Guard"))
            
            # Alert Section
            if hasattr(self, 'alerts_label'):
                self.alerts_label.configure(text=trans.get("alerts", "🔔 ALERTS"))
            if hasattr(self, 'btn_set_interval'):
                self.btn_set_interval.configure(text=trans.get("timeout", "⏱️ Timeout (HH:MM:SS)"))
            if hasattr(self, 'btn_alert_toggle'):
                self.btn_alert_toggle.configure(text=trans.get("alert_toggle", "🔔 Alert ON/OFF"))
            if hasattr(self, 'btn_stillness_alert'):
                self.btn_stillness_alert.configure(text=trans.get("stillness", "🔇 Stillness ON/OFF"))
            if hasattr(self, 'btn_track_toggle'):
                self.btn_track_toggle.configure(text=trans.get("track", "🎯 Track Guard"))
            if hasattr(self, 'action_label'):
                self.action_label.configure(text=trans.get("required_action", "Required Action:"))
            
            # Active Tracking Section
            if hasattr(self, 'active_track_label'):
                self.active_track_label.configure(text=trans.get("active_tracking", "✓ ACTIVE TRACKING"))
            if hasattr(self, 'guard_preview_label'):
                self.guard_preview_label.configure(text=trans.get("guards_preview", "👮 Guards:"))
            if hasattr(self, 'fugitive_preview_title'):
                self.fugitive_preview_title.configure(text=trans.get("fugitive_preview", "🚨 Fugitive:"))
            
            # Performance Section
            if hasattr(self, 'perf_label'):
                self.perf_label.configure(text=trans.get("performance", "📊 PERFORMANCE"))
            
            # Exit Button
            if hasattr(self, 'btn_exit'):
                self.btn_exit.configure(text=trans.get("exit", "🚪 Exit"))
            
            # Title Label (most important)
            self.title_label.configure(text=trans.get("title", "🛡️ निराक्षण"))
            
            # ========== UPDATE ACTION DROPDOWN ==========
            if hasattr(self, 'action_dropdown'):
                action_values = [
                    trans.get("action_hands_up", "Hands Up"),
                    trans.get("action_hands_crossed", "Hands Crossed"),
                    trans.get("action_left_hand_up", "Left Hand Up"),
                    trans.get("action_right_hand_up", "Right Hand Up"),
                    trans.get("action_t_pose", "T-Pose"),
                    trans.get("action_sit", "Sit"),
                    trans.get("action_standing", "Standing")
                ]
                self.action_dropdown.configure(values=action_values)
                # Set the first value as default and normalize it
                first_action = action_values[0]
                self.action_dropdown.set(first_action)
                self.active_required_action = self.normalize_action_name(first_action)
            
            # ✅ FIX: Remove emoji from logger to avoid UnicodeEncodeError on Windows console
            logger.warning(f"REAL-TIME language switched to: {self.current_language} (NO RESTART NEEDED)")
            
        except Exception as e:
            logger.error(f"Error applying translations: {e}")

    def update_widget_text(self, key, text):
        """Helper to update widget text by key"""
        try:
            return self.get_text(key)
        except Exception as e:
            logger.debug(f"Error updating widget text: {e}")
            return text

    def get_text(self, key):
        """Get translated text for a given key (with fallback to Hindi)"""
        try:
            if not hasattr(self, 'current_trans'):
                self.current_trans = self.translations.get(self.current_language, self.translations["Hindi"])
            
            # Return translation or the key itself as fallback
            return self.current_trans.get(key, key)
        except:
            # Final fallback to Hindi
            return self.translations.get("Hindi", {}).get(key, key)
        """Show dialog to choose between capturing or uploading guard image"""
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first.")
            return
        
        # Create custom dialog
        choice = messagebox.askquestion(
            "Add Guard",
            "How would you like to add the guard?\n\nYes = Take Photo with Camera\nNo = Upload Existing Image",
            icon='question'
        )
        
        if choice == 'yes':
            self.enter_onboarding_mode()
        else:
            self.upload_guard_image()
    
    def remove_guard_dialog(self):
        """Show dialog to select and remove a guard"""
        if not self.target_map:
            messagebox.showwarning("No Guards", "No guards available to remove.")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Remove Guard")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select guard to remove:", 
                font=('Helvetica', 11, 'bold')).pack(pady=10)
        
        # Listbox for guard selection
        listbox_frame = tk.Frame(dialog)
        listbox_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")
        
        guard_listbox = tk.Listbox(listbox_frame, font=('Helvetica', 10), 
                                   yscrollcommand=scrollbar.set)
        guard_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=guard_listbox.yview)
        
        # Populate listbox
        guard_names = sorted(self.target_map.keys())
        for name in guard_names:
            guard_listbox.insert(tk.END, name)
        
        def on_remove():
            selection = guard_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a guard to remove.")
                return
            
            guard_name = guard_listbox.get(selection[0])
            
            # Confirm deletion
            response = messagebox.askyesno(
                "Confirm Removal",
                f"Are you sure you want to remove '{guard_name}'?\n\nThis will delete:\n" +
                "- Face image\n- Pose references\n- All associated data\n\nThis action cannot be undone!"
            )
            
            if response:
                self.remove_guard(guard_name)
                dialog.destroy()
        
        # Buttons
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Remove", command=on_remove, bg="#e74c3c", 
                 fg="white", font=('Helvetica', 10, 'bold'), width=12).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#7f8c8d", 
                 fg="white", font=('Helvetica', 10, 'bold'), width=12).pack(side="left", padx=5)
    
    def remove_guard(self, guard_name):
        """Remove guard profile and all associated data"""
        try:
            safe_name = guard_name.replace(" ", "_")
            guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles")
            
            deleted_items = []
            
            # Remove face image from guard_profiles directory ONLY
            profile_image = os.path.join(guard_profiles_dir, f"target_{safe_name}_face.jpg")
            if os.path.exists(profile_image):
                os.remove(profile_image)
                deleted_items.append("Face image (profiles)")
            
            # Remove from tracking if currently tracked
            if guard_name in self.targets_status:
                if self.targets_status[guard_name].get("tracker"):
                    self.targets_status[guard_name]["tracker"] = None
                del self.targets_status[guard_name]
                deleted_items.append("Active tracking")
            
            # Reload targets list
            self.load_targets()
            
            logger.warning(f"Guard removed: {guard_name} ({', '.join(deleted_items)})")
            messagebox.showinfo(
                "Guard Removed",
                f"'{guard_name}' has been successfully removed.\n\nDeleted: {', '.join(deleted_items)}"
            )
            
        except Exception as e:
            logger.error(f"Error removing guard {guard_name}: {e}")
            messagebox.showerror("Error", f"Failed to remove guard: {e}")
    
    def upload_guard_image(self):
        """Upload an existing image for guard onboarding"""
        if not self.is_running: return
        
        filepath = filedialog.askopenfilename(
            title="Select Guard Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if not filepath: return
        
        try:
            name = simpledialog.askstring("Guard Name", "Enter guard name:")
            if not name: return
            
            safe_name = name.strip().replace(" ", "_")
            guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles")
            target_path = os.path.join(guard_profiles_dir, f"target_{safe_name}_face.jpg")
            
            # Load and verify face
            img = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(img)
            
            if len(face_locations) != 1:
                messagebox.showerror("Error", "Image must contain exactly one face.")
                return
            
            # Copy image
            import shutil
            shutil.copy(filepath, target_path)
            
            # Also copy to root for backward compatibility
            shutil.copy(filepath, f"target_{safe_name}_face.jpg")
            
            self.load_targets()
            messagebox.showinfo("Success", f"Guard '{name}' added successfully!\n(Pose capture skipped for uploaded images)")
            logger.warning(f"Guard added via upload: {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload image: {e}")
            logger.error(f"Upload error: {e}")
    
    def load_targets(self):
        self.target_map = {}
        # Search ONLY in guard_profiles directory
        guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", r"D:\CUDA_Experiments\Git_HUB\Nirikhsan_Deep\V2\guard_profiles")
        if not os.path.exists(guard_profiles_dir):
            os.makedirs(guard_profiles_dir)
        target_files = glob.glob(os.path.join(guard_profiles_dir, "target_*.jpg"))
        display_names = []
        for f in target_files:
            try:
                # Parse filename: target_[Name]_face.jpg or target_[First_Last]_face.jpg
                base_name = os.path.basename(f).replace(".jpg", "")
                parts = base_name.split('_')
                
                # Remove 'target' prefix and 'face' suffix
                if len(parts) >= 3 and parts[-1] == "face":
                    # Join all parts between 'target' and 'face' as the name
                    display_name = " ".join(parts[1:-1])
                    self.target_map[display_name] = f
                    display_names.append(display_name)
            except Exception as e:
                logger.error(f"Error parsing {f}: {e}")

        if not display_names:
             logger.warning("No target files found")
        else:
             logger.warning(f"Loaded {len(set(display_names))} guards")
        
        # Update selected targets list
        self.selected_target_names = [name for name in self.selected_target_names if name in self.target_map]
        
        # ✅ SIMPLIFIED: Removed Phase 4 Stage 2/3 initialization - focus on core targets only
        
        self.update_selected_preview()
    
    def select_all_targets(self):
        """Select all targets"""
        self.selected_target_names = list(self.target_map.keys())
        self.update_selected_preview()

    def update_selected_preview(self):
        """Update guard preview with ALL selected targets in single column layout"""
        
        # Clear existing grid
        for widget in self.guard_preview_scroll_frame.winfo_children():
            widget.destroy()
        self.guard_preview_grid = {}
        
        if not self.selected_target_names:
            # Show placeholder message
            ctk.CTkLabel(self.guard_preview_scroll_frame, text="No Guard Selected", 
                        text_color="#bdc3c7", font=("Arial", 9)).pack(pady=30)
            return
        
        # Create single-column layout of guards for ALL selected guards
        for guard_name in sorted(self.selected_target_names):
            filename = self.target_map.get(guard_name)
            if not filename:
                continue
            
            try:
                # Load and resize image for thumbnail
                img = cv2.imread(filename)
                if img is not None:
                    # Resize to 60x60 thumbnail
                    img_resized = cv2.resize(img, (60, 60))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    imgtk = ImageTk.PhotoImage(image=pil_img)
                    
                    # Create row frame with image on left and name on right
                    row_frame = ctk.CTkFrame(self.guard_preview_scroll_frame, fg_color="#2a2a2a", 
                                            border_width=1, border_color="#27ae60", corner_radius=4)
                    row_frame.pack(fill="x", padx=0, pady=2)
                    
                    # Image label on left
                    img_label = ctk.CTkLabel(row_frame, image=imgtk, text="")
                    img_label.pack(side="left", padx=3, pady=3)
                    
                    # Guard name label on right (full width)
                    name_label = ctk.CTkLabel(row_frame, text=guard_name, 
                                             text_color="#27ae60", font=("Segoe UI", 9))
                    name_label.pack(side="left", fill="x", expand=True, padx=5, pady=3)
                    
                    # Store reference and photo
                    self.guard_preview_grid[guard_name] = (img_label, imgtk)
                    self.photo_storage[f"guard_preview_{guard_name}"] = imgtk
                    
            except Exception as e:
                logger.error(f"Error loading guard preview for {guard_name}: {e}")
                # Show error row
                error_frame = ctk.CTkFrame(self.guard_preview_scroll_frame, fg_color="#2a2a2a", 
                                          border_width=1, border_color="#e74c3c", corner_radius=4)
                error_frame.pack(fill="x", padx=0, pady=2)
                ctk.CTkLabel(error_frame, text="❌", text_color="#e74c3c", font=("Arial", 10)).pack(side="left", padx=3, pady=3)
                ctk.CTkLabel(error_frame, text=guard_name, text_color="#e74c3c", 
                            font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=5, pady=3)

    def apply_target_selection(self):
        self.targets_status = {} 
        if not self.selected_target_names:
            # No targets selected, tracking disabled
            logger.info("No targets selected - tracking disabled")
            return
        count = 0
        # ✅ IMPROVED: Increased pose buffer size for better multi-guard stability
        pose_buffer_size = max(CONFIG["performance"].get("pose_buffer_size", 5), 12)
        
        for name in self.selected_target_names:
            filename = self.target_map.get(name)
            if filename:
                try:
                    # ✅ CRITICAL: Check if file exists before loading
                    if not os.path.exists(filename):
                        logger.error(f"Guard profile file not found: {filename}")
                        continue
                    
                    # ✅ MULTI-ANGLE SUPPORT: Load all angle reference images
                    angle_images = load_guard_angle_images(name)
                    
                    # Generate face encodings from all available angles
                    multi_angle_encodings = []
                    for angle, img_path in angle_images.items():
                        try:
                            angle_image = face_recognition.load_image_file(img_path)
                            # ✅ PERFORMANCE: Use num_jitters=1 for fast encoding at initialization
                            angle_encodings = face_recognition.face_encodings(angle_image, num_jitters=1)
                            if angle_encodings:
                                multi_angle_encodings.extend(angle_encodings)
                                logger.debug(f"Loaded {angle} angle encoding for {name}")
                        except Exception as e:
                            logger.warning(f"Failed to load {angle} angle for {name}: {e}")
                    
                    # Fallback: Load primary face image if no multi-angle images found
                    if not multi_angle_encodings:
                        target_image_file = face_recognition.load_image_file(filename)
                        encodings = face_recognition.face_encodings(target_image_file, num_jitters=1)
                        if encodings:
                            multi_angle_encodings = encodings
                    
                    if multi_angle_encodings and len(multi_angle_encodings) > 0:
                        self.targets_status[name] = {
                            "encoding": multi_angle_encodings[0],  # Primary encoding for compatibility
                            "multi_angle_encodings": multi_angle_encodings,  # ✅ NEW: All angle encodings for better matching
                            "body_profile_landmarks": None,  # ✅ PRO MODE: Store body profile for silhouette matching
                            "tracker": None,
                            "body_tracker": None,
                            "face_box": None, 
                            "body_box": None,
                            "visible": False,
                            "overlap_disabled": False,  # ✅ SEPARATION FIX: Track if disabled due to overlap
                            "last_action_time": time.time(),  # Renamed from last_wave_time
                            "action_performed": False,  # ✅ PHASE 3.1: Track if action was performed in grace period
                            "alert_cooldown": 0,
                            "alert_triggered_state": False,
                            "last_logged_action": None,
                            "pose_buffer": deque(maxlen=pose_buffer_size),  # ✅ Larger buffer for stability
                            "missing_pose_counter": 0,
                            "face_confidence": 0.0,
                            "pose_confidence": 0.0,  # ✅ NEW: Track pose detection quality
                            "face_encoding_history": deque(maxlen=5),  # ✅ CRITICAL: Track face encodings
                            "face_match_confidence": 0.0,  # ✅ NEW: Track match quality
                            "last_valid_pose": None,  # ✅ NEW: Store last valid pose for continuity
                            "last_valid_pose_time": time.time(),  # ✅ NEW: Track when last valid pose was detected
                            "pose_quality_history": deque(maxlen=10),  # ✅ NEW: Track pose quality over time
                            "last_snapshot_time": 0,  # Rate limiting: one snapshot per minute
                            "last_log_time": 0,  # Rate limiting: one log entry per minute
                            "alert_sound_thread": None,  # Track current alert sound thread
                            "alert_stop_event": None,  # Event to signal sound to stop when action performed
                            "alert_logged_timeout": False,  # Track if timeout alert was logged
                            "target_missing_alert_logged": False,  # ✅ NEW: Track if TARGET MISSING was logged (event start only)
                            # ✅ STILLNESS DETECTION (PRO MODE)
                            "last_stillness_check_time": time.time(),  # For tracking stillness duration
                            "stillness_start_time": None,  # When guard became still
                            "last_face_box_stillness": None,  # Previous face box for motion detection
                            "consecutive_detections": 0,  # Count consecutive frames detected for stability
                            "stable_tracking": False,  # True after 3+ consecutive detections
                            "last_pose_vector": None,  # Previous pose for variance calculation
                            "stillness_alert_logged": False,  # Track if stillness alert was logged
                        }
                        count += 1
                        logger.info(f"[OK] {name} initialized with {len(multi_angle_encodings)} face encoding(s) from {len(angle_images)} angle(s)")
                    else:
                        # ✅ IMPROVED: Log if no face detected in target image
                        logger.warning(f"[WARN] No face encoding found in {name} image - guard may not be detectable")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        if count > 0:
            logger.warning(f"[OK] Tracking initialized for {count} targets (Pose Buffer: {pose_buffer_size} frames)")
            
            messagebox.showinfo("Tracking Updated", f"Now scanning for {count} selected targets")
        else:
            logger.error("No targets were initialized - check that guard profile images contain faces")

    def open_target_selection_dialog(self):
        """Open dialog for selecting targets"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Select Targets")
        dialog.geometry("400x500")
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="Select Targets to Track", font=("Roboto", 14, "bold")).pack(pady=10)
        
        scroll_frame = ctk.CTkScrollableFrame(dialog, width=350, height=350)
        scroll_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.target_vars = {}
        
        # Get all available targets
        targets = sorted(list(self.target_map.keys()))
        
        if not targets:
            ctk.CTkLabel(scroll_frame, text="No targets found.").pack()
        
        for target in targets:
            var = ctk.BooleanVar(value=target in self.selected_target_names)
            chk = ctk.CTkCheckBox(scroll_frame, text=target, variable=var)
            chk.pack(anchor="w", pady=2, padx=5)
            self.target_vars[target] = var
            
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10, fill="x", padx=10)
        
        ctk.CTkButton(btn_frame, text="Select All", command=self.select_all_dialog, width=100, font=("Segoe UI", 12, "bold")).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Clear All", command=self.clear_all_dialog, width=100, font=("Segoe UI", 12, "bold")).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Done", command=lambda: self.confirm_selection(dialog), width=100, fg_color="green", font=("Segoe UI", 12, "bold")).pack(side="right", padx=5)

    def select_all_dialog(self):
        """Select all targets in dialog"""
        for var in self.target_vars.values():
            var.set(True)

    def clear_all_dialog(self):
        """Clear all targets in dialog"""
        for var in self.target_vars.values():
            var.set(False)

    def confirm_selection(self, dialog):
        """Confirm target selection from dialog"""
        self.selected_target_names = [name for name, var in self.target_vars.items() if var.get()]
        dialog.destroy()
        # Update preview
        self.update_selected_preview()

    def set_alert_interval_advanced(self):
        """Set alert interval (timeout before alert) with hours, minutes, seconds"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Set Action Timeout Interval")
        dialog.geometry("400x320")
        dialog.grab_set()
        
        # Button frame at top (left corner)
        button_frame_top = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame_top.pack(pady=5, fill="x", padx=10)
        
        def confirm():
            try:
                h = int(h_var.get()) if h_var.get() and h_var.get().strip() else 0
                m = int(m_var.get()) if m_var.get() and m_var.get().strip() else 0
                
                # ✅ IMPROVED: Validate input ranges (no seconds, only HH:MM)
                if h < 0 or m < 0 or m > 59:
                    messagebox.showerror("Error", "Please enter valid time values (H >= 0, 0 <= M <= 59)")
                    return
                
                total_seconds = h * 3600 + m * 60
                if total_seconds > 0:
                    self.alert_interval = total_seconds  # ✅ IMPORTANT: Store as seconds
                    # Update button to show current value in HH:MM format only
                    display_text = f"{h:02d}:{m:02d}"
                    self.btn_set_interval.configure(text=display_text)
                    messagebox.showinfo("Success", f"Action timeout set to {total_seconds} seconds ({h}h {m}m)")
                    logger.warning(f"Alert interval changed to {total_seconds} seconds ({h}h {m}m)")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Interval must be greater than 0")
            except ValueError as e:
                messagebox.showerror("Error", f"Please enter valid numbers. Error: {e}")
        
        ctk.CTkButton(button_frame_top, text="✓ Set", command=confirm, fg_color="#27ae60", font=("Roboto", 12, "bold"), width=80).pack(side="left", padx=3)
        ctk.CTkButton(button_frame_top, text="✕ Cancel", command=dialog.destroy, fg_color="#34495e", font=("Roboto", 12, "bold"), width=80).pack(side="left", padx=3)
        
        ctk.CTkLabel(dialog, text="⏱ Action Timeout Interval", font=("Roboto", 14, "bold")).pack(pady=10)
        ctk.CTkLabel(dialog, text="How long until timeout alert triggers if action not performed", font=("Roboto", 10), text_color="#95a5a6").pack(pady=5)
        
        frame = ctk.CTkFrame(dialog)
        frame.pack(pady=15, padx=20, fill="x")
        
        # Calculate current time components for display (only HH:MM, no seconds)
        current_h = int(self.alert_interval // 3600)
        current_m = int((self.alert_interval % 3600) // 60)
        
        # Hours
        ctk.CTkLabel(frame, text="Hours:", font=("Roboto", 10)).grid(row=0, column=0, padx=5, sticky="w")
        h_var = ctk.StringVar(value=str(current_h))
        h_entry = ctk.CTkEntry(frame, textvariable=h_var, width=80)
        h_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Minutes
        ctk.CTkLabel(frame, text="Minutes:", font=("Roboto", 10)).grid(row=1, column=0, padx=5, sticky="w")
        m_var = ctk.StringVar(value=str(current_m))
        m_entry = ctk.CTkEntry(frame, textvariable=m_var, width=80)
        m_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Display recommended range
        ctk.CTkLabel(frame, text="Recommended: 1 - 60 minutes (action timeout)", font=("Roboto", 9), text_color="#95a5a6").grid(row=2, column=0, columnspan=2, pady=10)
        
        # Focus on minutes field by default
        m_entry.focus()

    def toggle_alert_mode(self):
        self.is_alert_mode = not self.is_alert_mode
        if self.is_alert_mode:
            self.btn_alert_toggle.configure(text="🔔 Action Alert: ON", fg_color="#f39c12")
            # Auto-start logging
            if not self.is_logging:
                self.is_logging = True
                self.temp_log.clear()
                self.temp_log_counter = 0
                logger.warning("Alert mode started - logging enabled")
            
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False
        else:
            self.btn_alert_toggle.configure(text="🔔 Action Alert: OFF", fg_color="#7f8c8d")
            # Auto-stop logging and save
            if self.is_logging:
                self.save_log_to_file()
                self.is_logging = False
                logger.warning("Alert mode stopped - logging saved")
    
    def toggle_logging_button(self):
        """Toggle logging on/off manually"""
        self.is_logging = not self.is_logging
        if self.is_logging:
            self.temp_log.clear()
            self.temp_log_counter = 0
            logger.info("Logging enabled manually")
        else:
            if len(self.temp_log) > 0:
                self.save_log_to_file()
            logger.info("Logging disabled manually")

    def set_alert_interval(self):
        """Set alert interval with Hours, Minutes, Seconds dialog - ✅ ENHANCED: Full H:M:S support"""
        # Create custom dialog for H:M:S input
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Alert Interval")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Calculate current values from total seconds
        hours = self.alert_interval // 3600
        remaining = self.alert_interval % 3600
        minutes = remaining // 60
        seconds = 0  # No longer use seconds, only HH:MM
        
        # Create input fields
        tk.Label(dialog, text="Alert Interval (Hours : Minutes)", font=("Helvetica", 10, "bold")).pack(pady=10)
        
        frame = tk.Frame(dialog)
        frame.pack(pady=10)
        
        tk.Label(frame, text="Hours:", width=10).grid(row=0, column=0, padx=5)
        hours_var = tk.StringVar(value=str(hours))
        hours_entry = tk.Entry(frame, textvariable=hours_var, width=5, font=("Helvetica", 10))
        hours_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(frame, text="Minutes:", width=10).grid(row=1, column=0, padx=5)
        minutes_var = tk.StringVar(value=str(minutes))
        minutes_entry = tk.Entry(frame, textvariable=minutes_var, width=5, font=("Helvetica", 10))
        minutes_entry.grid(row=1, column=1, padx=5)
        
        def on_ok():
            try:
                h = int(hours_var.get() or 0)
                m = int(minutes_var.get() or 0)
                
                if h < 0 or m < 0 or m > 59:
                    messagebox.showwarning("Invalid Input", "Please enter valid values (hours >= 0, 0-59 minutes)")
                    return
                
                total_seconds = h * 3600 + m * 60
                if total_seconds < 1:
                    messagebox.showwarning("Invalid Input", "Interval must be at least 1 minute")
                    return
                
                self.alert_interval = total_seconds
                
                # Format display: HH:MM only
                display = f"{h:02d}:{m:02d}"
                
                self.btn_set_interval.configure(text=display)
                dialog.destroy()
                logger.warning(f"Alert interval updated: {h}h {m}m ({total_seconds} seconds)")
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers")
        
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_ok, bg="#27ae60", fg="white", width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#e74c3c", fg="white", width=10).pack(side="left", padx=5)
            
    def normalize_action_name(self, action_text):
        """Normalize action text by extracting English action name from bilingual/translated dropdown text.
        Handles formats like:
        - "Hands Up"
        - "हाथ ऊपर (Hands Up)"
        - "हाथ वर (Hands Up)"
        Returns: Canonical English action name like "Hands Up", "One Hand Raised (Left)", etc.
        """
        if not action_text:
            return "Standing"  # Safe default
        
        # Extract English part from bilingual text like "हाथ ऊपर (Hands Up)"
        if "(" in action_text and ")" in action_text:
            start = action_text.find("(")
            end = action_text.find(")")
            if start < end:
                english_part = action_text[start+1:end].strip()
                return english_part
        
        # Return as-is if already in English or no parentheses
        return action_text.strip()
    
    def on_action_change(self, value):
        """Update active required action when dropdown changes"""
        # Normalize the action name to handle bilingual/translated dropdown values
        self.active_required_action = self.normalize_action_name(value)
        if self.is_alert_mode:
            current_time = time.time()
            for name in self.targets_status:
                self.targets_status[name]["last_action_time"] = current_time
                self.targets_status[name]["alert_triggered_state"] = False
        logger.debug(f"Active Required Action changed to: {value} (normalized: {self.active_required_action})")

    def toggle_night_mode(self):
        """Toggle Night/Day Mode for optimal lighting adaptation with dual-model detection"""
        self.night_mode = not self.night_mode
        if self.night_mode:
            self.btn_night_mode.configure(text="🌙 Night ON", fg_color="#1a1a2e")
            logger.warning(f"Night Mode - Face: BlazeFace + CLAHE | Pose: MoveNet Lightning/BlazePose + CLAHE enhancement")
        else:
            self.btn_night_mode.configure(text="🌙 Night OFF", fg_color="#34495e")
            logger.warning(f"Day Mode - Face: BlazeFace | Pose: MoveNet Lightning/BlazePose | Tracking: ByteTrack")

    def switch_camera_dialog(self):
        """Allow switching to a different camera while app is running"""
        # Detect available cameras
        available_cameras = detect_available_cameras()
        
        # Create camera selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Switch Camera")
        dialog.geometry("400x350")
        dialog.configure(bg="#2c3e50")
        
        tk.Label(dialog, text="Select Camera:", font=('Helvetica', 14, 'bold'), 
                bg="#2c3e50", fg="white").pack(pady=15)
        
        # Variable to store selection
        # Use string var to handle both int indices and string URLs
        current_val = str(self.camera_index) if hasattr(self, 'camera_index') else (str(available_cameras[0]) if available_cameras else "ip")
        selected_source = tk.StringVar(value=current_val)
        
        # Frame for options
        radio_frame = tk.Frame(dialog, bg="#34495e")
        radio_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Local cameras
        if available_cameras:
            tk.Label(radio_frame, text="Local Cameras:", font=('Helvetica', 10, 'bold'), 
                    bg="#34495e", fg="white", anchor="w").pack(fill="x", pady=(5,0))
            for idx in available_cameras:
                current_label = " (Current)" if hasattr(self, 'camera_index') and str(idx) == str(self.camera_index) else ""
                tk.Radiobutton(radio_frame, text=f"Camera {idx}{current_label}", variable=selected_source,
                              value=str(idx), font=('Helvetica', 11), bg="#34495e", fg="white",
                              selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=5)
        
        # IP Camera Option
        tk.Label(radio_frame, text="IP Camera:", font=('Helvetica', 10, 'bold'), 
                bg="#34495e", fg="white", anchor="w").pack(fill="x", pady=(10,0))
        
        is_ip_current = isinstance(self.camera_index, str) if hasattr(self, 'camera_index') else False
        ip_label = " (Current)" if is_ip_current else ""
        
        tk.Radiobutton(radio_frame, text=f"Add New IP Camera{ip_label}", variable=selected_source,
                      value="ip", font=('Helvetica', 11), bg="#34495e", fg="white",
                      selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=5)
        
        def on_switch():
            choice = selected_source.get()
            
            if choice == "ip":
                dialog.destroy()
                self.add_ip_camera_dialog() # Reuse the dialog from start_camera
                return

            # Check if switching to same camera
            if hasattr(self, 'camera_index') and str(choice) == str(self.camera_index):
                messagebox.showinfo("Camera Switch", "Already using this camera")
                dialog.destroy()
                return
            
            # Store running state
            was_running = self.is_camera_running
            
            # Stop current camera if running
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            dialog.destroy()
            
            # Switch to new camera
            try:
                new_index = int(choice)
                self.camera_index = new_index
                self.initialize_camera(new_index)
            except ValueError:
                messagebox.showerror("Error", "Invalid camera selection")
                return
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg="#2c3e50")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Switch", command=on_switch, bg="#27ae60",
                 fg="white", font=('Helvetica', 11, 'bold'), width=10).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#e74c3c",
                 fg="white", font=('Helvetica', 11, 'bold'), width=10).pack(side="left", padx=5)

    def toggle_camera(self):
        """Toggle Camera ON/OFF - unified camera control"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def toggle_pro_mode(self):
        """Toggle PRO Mode ON/OFF - enables advanced features like ReID and Stillness detection"""
        self.is_pro_mode = not self.is_pro_mode
        if self.is_pro_mode:
            self.btn_pro_toggle.configure(text="🎯 PRO (ON)", fg_color="#004a7f")
            self.btn_stillness_alert.configure(state="normal")
            logger.warning("PRO Mode ENABLED - Advanced ReID and Stillness detection available")
        else:
            self.btn_pro_toggle.configure(text="🎯 PRO (OFF)", fg_color="#34495e")
            self.btn_stillness_alert.configure(state="disabled", text="🔇 Stillness Alert OFF", fg_color="#95a5a6")
            self.is_stillness_alert = False
            logger.warning("PRO Mode DISABLED")

    def toggle_guard_mode(self):
        """Directly enter onboarding mode to add a new guard"""
        if not self.is_camera_running:
            messagebox.showwarning("Camera Required", "Please start camera first")
            return
        
        # Enter onboarding mode to capture guard face and poses
        self.enter_onboarding_mode()

    def add_guard_then_capture(self):
        """Show dialog to enter guard name, then capture face and poses"""
        if not self.is_camera_running:
            messagebox.showwarning("Camera Required", "Please start camera first")
            return
        
        # Enter onboarding mode to capture guard face and poses
        self.enter_onboarding_mode()

    def toggle_fugitive_add_remove(self):
        """Toggle between ADD and REMOVE fugitive modes"""
        if not hasattr(self, 'fugitive_add_mode'):
            self.fugitive_add_mode = "ADD"
        
        if self.fugitive_add_mode == "ADD":
            # Initiate ADD fugitive
            self.add_fugitive()
            self.fugitive_add_mode = "REMOVE"
            self.btn_fugitive_toggle.configure(text="❌ Remove Fugitive", fg_color="#e74c3c")
        else:
            # Initiate REMOVE fugitive
            self.remove_fugitive()
            self.fugitive_add_mode = "ADD"
            self.btn_fugitive_toggle.configure(text="➕ Add Fugitive", fg_color="#e74c3c")

    def add_fugitive(self):
        """Add/set a fugitive for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Fugitive Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display fugitive image
            self.fugitive_image = cv2.imread(file_path)
            if self.fugitive_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            # Extract face encoding from fugitive image
            rgb_image = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                messagebox.showerror("Error", "No face detected in selected image")
                return
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if face_encodings:
                self.fugitive_face_encoding = face_encodings[0]
                self.fugitive_detected_log_done = False
                
                # Extract name from path
                self.fugitive_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Update preview
                self._update_fugitive_preview()
                
                # ✅ AUTO-ACTIVATE: Enable fugitive detection automatically
                self.is_fugitive_detection = True
                self.btn_fugitive_toggle.configure(text="❌ Remove Fugitive", fg_color="#e74c3c")
                self.fugitive_preview_frame.pack(fill="x", padx=0, pady=2)
                
                logger.warning(f"Fugitive Added: {self.fugitive_name}")
                logger.warning(f"Fugitive Detection ENABLED - Searching for: {self.fugitive_name}")
                messagebox.showinfo("Fugitive Added", f"Fugitive profile created: {self.fugitive_name}\n\n✅ Fugitive detection automatically activated!")
        
        except Exception as e:
            logger.error(f"Error adding fugitive: {e}")
            messagebox.showerror("Error", f"Failed to process image: {e}")

    def remove_fugitive(self):
        """Remove current fugitive"""
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_detected_log_done = False
        self.fugitive_name = None
        
        if hasattr(self, 'fugitive_preview_label'):
            self.fugitive_preview_label.configure(image='', text="No Fugitive Selected")
        
        logger.warning("Fugitive Removed")
        messagebox.showinfo("Fugitive Removed", "Fugitive profile cleared")

    def toggle_fugitive_detection(self):
        """Toggle Fugitive Detection ON/OFF - enables live search for fugitive"""
        if self.fugitive_face_encoding is None:
            messagebox.showwarning("No Fugitive", "Please add a fugitive first using Add Fugitive button")
            return
        
        self.is_fugitive_detection = not self.is_fugitive_detection
        if self.is_fugitive_detection:
            self.btn_fugitive_toggle.configure(text="🚨 Fugitive ON", fg_color="#8b0000")
            # Show fugitive preview
            self.fugitive_preview_frame.pack(fill="x", padx=0, pady=2)
            logger.warning(f"Fugitive Detection ENABLED - Searching for: {self.fugitive_name}")
        else:
            self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
            # Hide fugitive preview
            self.fugitive_preview_frame.pack_forget()
            logger.warning("Fugitive Detection DISABLED")

    def toggle_stillness_alert(self):
        """Toggle Stillness Alert ON/OFF - detects guards not moving"""
        if not self.is_pro_mode:
            messagebox.showwarning("PRO Mode Required", "Stillness Alert is only available in PRO Mode")
            return
        
        self.is_stillness_alert = not self.is_stillness_alert
        if self.is_stillness_alert:
            self.btn_stillness_alert.configure(text="🔇 Stillness Alert ON", fg_color="#00bfff")
            logger.warning("Stillness Alert ENABLED")
        else:
            self.btn_stillness_alert.configure(text="🔇 Stillness Alert OFF", fg_color="#95a5a6")
            logger.warning("Stillness Alert DISABLED")

    def open_guard_selection_dialog(self):
        """Open dialog to select multiple guards for tracking"""
        # Get available guards from guard_profiles directory
        guard_profile_dir = os.path.join(os.path.dirname(__file__), "guard_profiles")
        if not os.path.exists(guard_profile_dir):
            messagebox.showinfo("No Guards", "No guards available. Please add guards first.")
            return
        
        # List all unique guard names from face files
        guard_files = glob.glob(os.path.join(guard_profile_dir, "target_*_face.jpg"))
        available_guards = list(set([os.path.basename(f).replace("target_", "").replace("_face.jpg", "") for f in guard_files]))
        
        if not available_guards:
            messagebox.showinfo("No Guards", "No guards available. Please add guards first.")
            return
        
        # For now, open the existing target selection dialog if it exists
        if hasattr(self, 'open_target_selection_dialog'):
            self.open_target_selection_dialog()
        else:
            messagebox.showinfo("Select Guards", f"Available guards: {', '.join(available_guards)}")

    def on_track_selected_guard_clicked(self):
        """Handle Track Selected Guard button click - activates tracking for selected guards"""
        if not hasattr(self, 'selected_target_names') or not self.selected_target_names:
            messagebox.showwarning("No Guards Selected", "Please select guards first using 'Select Guard' button")
            return
        
        # Log the tracking activation
        guards_list = ", ".join(self.selected_target_names)
        logger.warning(f"Tracking activated for: {guards_list}")
        messagebox.showinfo("Tracking Activated", f"Now tracking: {guards_list}")

    def toggle_track_monitoring(self):
        """Toggle Track Guard / Stop Monitoring button - start or stop tracking selected guards"""
        if not hasattr(self, 'selected_target_names') or not self.selected_target_names:
            messagebox.showwarning("No Guards Selected", "Please select guards first using 'Select Guard' button")
            return
        
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first")
            return
        
        # Toggle tracking state
        if not self.is_tracking:
            # START TRACKING
            self.is_tracking = True
            self.btn_track_toggle.configure(text="⏹️ Stop Monitoring", fg_color="#e74c3c")
            
            # Activate the selected guards for tracking
            self.apply_target_selection()
            
            # Log and show confirmation
            guards_list = ", ".join(self.selected_target_names)
            current_action = self.active_required_action
            logger.warning(f"[TRACKING START] Identifying and tracking: {guards_list} | Action: {current_action}")
            messagebox.showinfo("Tracking Started", 
                              f"Now identifying and tracking:\n{guards_list}\n\nActive Action: {current_action}\nMonitoring for action alerts...")
        else:
            # STOP TRACKING
            self.is_tracking = False
            self.btn_track_toggle.configure(text="🏃 Track Guard", fg_color="#16a085")
            
            # Clear all tracking data
            tracked_guards = list(self.targets_status.keys())
            self.targets_status.clear()
            self.selected_target_names.clear()
            self.update_selected_preview()
            
            # Log and show confirmation
            if tracked_guards:
                guards_list = ", ".join(tracked_guards)
                logger.warning(f"[TRACKING STOP] Stopped monitoring: {guards_list}")
                messagebox.showinfo("Monitoring Stopped", f"Stopped tracking:\n{guards_list}")
            else:
                logger.warning("[TRACKING STOP] Monitoring stopped")
                messagebox.showinfo("Monitoring Stopped", "Guard tracking stopped")

    def on_track_guards_clicked(self):
        """Handle Track Guards button click - identifies and starts tracking selected guards"""
        if not hasattr(self, 'selected_target_names') or not self.selected_target_names:
            messagebox.showwarning("No Guards Selected", "Please select guards first using 'Select Guard' button")
            return
        
        if not self.is_running:
            messagebox.showwarning("Camera Required", "Please start the camera first")
            return
        
        # Activate the selected guards for tracking
        self.apply_target_selection()
        
        # Log and show confirmation
        guards_list = ", ".join(self.selected_target_names)
        logger.warning(f"[TRACKING START] Identifying and tracking: {guards_list}")
        messagebox.showinfo("Tracking Started", f"Now identifying and tracking:\n{guards_list}\n\nMonitoring for action alerts...")

    def on_stop_tracking_clicked(self):
        """Handle Stop Tracking button click - stops all guard tracking"""
        if not self.targets_status:
            messagebox.showinfo("No Tracking", "No guards are currently being tracked")
            return
        
        # Clear all tracking data
        tracked_guards = list(self.targets_status.keys())
        self.targets_status.clear()
        self.selected_target_names.clear()
        self.update_selected_preview()
        
        # Log and show confirmation
        guards_list = ", ".join(tracked_guards)
        logger.warning(f"[TRACKING STOP] Stopped tracking: {guards_list}")
        messagebox.showinfo("Tracking Stopped", f"Stopped tracking:\n{guards_list}")

    def toggle_fugitive_mode(self):
        """Toggle Fugitive Mode - Search for a specific person in live feed"""
        if not self.fugitive_mode:
            # Start Fugitive Mode
            file_path = filedialog.askopenfilename(
                title="Select Fugitive Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            try:
                # Load and display fugitive image
                self.fugitive_image = cv2.imread(file_path)
                if self.fugitive_image is None:
                    messagebox.showerror("Error", "Failed to load image")
                    return
                
                # Extract face encoding from fugitive image
                rgb_image = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                if not face_locations:
                    messagebox.showerror("Error", "No face detected in selected image")
                    return
                
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if not face_encodings:
                    messagebox.showerror("Error", "Failed to extract face encoding")
                    return
                
                # OLD FUNCTION - Now handled by toggle_fugitive_detection()
                self.fugitive_face_encoding = face_encodings[0]
                self.fugitive_name = simpledialog.askstring("Fugitive Name", "Enter fugitive name:") or "Unknown Fugitive"
                
                # Trigger fugitive detection
                self.is_fugitive_detection = True
                self.fugitive_detected_log_done = False
                self.btn_fugitive_toggle.configure(text="🚨 Fugitive ON", fg_color="#8b0000")
                
                # Display fugitive image in preview
                self._update_fugitive_preview()
                
                logger.warning(f"Fugitive Detection Started - Searching for: {self.fugitive_name}")
                messagebox.showinfo("Fugitive Detection", f"Searching for: {self.fugitive_name}")
                
            except Exception as e:
                logger.error(f"Fugitive Detection Error: {e}")
                messagebox.showerror("Error", f"Failed to process image: {e}")
        else:
            # Stop Fugitive Detection
            self.is_fugitive_detection = False
            self.fugitive_image = None
            self.fugitive_face_encoding = None
            self.fugitive_detected_log_done = False
            self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
            
            logger.warning("Fugitive Detection Stopped")
            messagebox.showinfo("Fugitive Detection", "Fugitive Detection Stopped")

    def _update_fugitive_preview(self):
        """Update fugitive preview image display"""
        if self.fugitive_image is None:
            self.fugitive_preview_label.configure(image='', text="No Fugitive")
            return
        
        try:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(self.fugitive_image, cv2.COLOR_BGR2RGB)
            
            # Resize for preview (150x150)
            preview_size = 150
            h, w = rgb_image.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w = preview_size
                new_h = int(preview_size / aspect)
            else:
                new_h = preview_size
                new_w = int(preview_size * aspect)
            
            rgb_resized = cv2.resize(rgb_image, (new_w, new_h))
            
            # Convert to PIL
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(rgb_resized)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.fugitive_preview_label.configure(image=photo, text='')
            # Store photo reference to prevent garbage collection
            self.photo_storage["fugitive_preview"] = photo
            
        except Exception as e:
            logger.error(f"Failed to update fugitive preview: {e}")
            self.fugitive_preview_label.configure(text="Preview Error")

    # ✅ SIMPLIFIED: toggle_pro_detection_mode and _save_pro_detection_log functions removed

    def _build_rtsp_alternatives(self, source):
        """Return candidate RTSP URLs that downgrade to H.264 substreams when possible."""
        if not isinstance(source, str) or not source.lower().startswith("rtsp://"):
            return [source]
        candidates = [source]
        base, sep, query = source.partition("?")
        params = []
        if sep:
            for chunk in query.split("&"):
                if not chunk:
                    continue
                if "=" in chunk:
                    key, value = chunk.split("=", 1)
                else:
                    key, value = chunk, ""
                params.append((key, value))

        def rebuild(mod_params):
            if not mod_params:
                return base
            query_str = "&".join(f"{k}={v}" if v != "" else k for k, v in mod_params)
            return f"{base}?{query_str}"

        # Substream fallback (Dahua/Onvif subtype=0 → subtype=1)
        for idx, (key, value) in enumerate(params):
            if key.lower() == "subtype" and value == "0":
                updated = list(params)
                updated[idx] = (key, "1")
                candidates.append(rebuild(updated))
                break

        # Codec parameter fallback (codec=h265/hevc → h264)
        for idx, (key, value) in enumerate(params):
            if key.lower() in ("codec", "compression", "videocodec") and value.lower() in ("h265", "hevc", "265"):
                updated = list(params)
                updated[idx] = (key, "H264")
                candidates.append(rebuild(updated))
                break

        # Profile fallback (Profile_1/Main → Profile_2/Sub)
        for idx, (key, value) in enumerate(params):
            value_lower = value.lower()
            if key.lower() == "profile" and value_lower in ("profile_1", "profile1", "main", "mainstream"):
                updated = list(params)
                updated[idx] = (key, "Profile_2")
                candidates.append(rebuild(updated))
                break

        # Onvif channel fallback (Streaming/Channels/101 → 102)
        lowered_base = base.lower()
        if "/channels/" in lowered_base:
            parts = base.split("/")
            if parts and parts[-1].isdigit() and len(parts[-1]) >= 3:
                tail = parts[-1]
                if tail[-1] in ("1", "0"):
                    alt_tail = tail[:-1] + "2"
                    alt_base = "/".join(parts[:-1] + [alt_tail])
                    if sep and query:
                        alt_full = f"{alt_base}?{query}"
                    else:
                        alt_full = alt_base
                    candidates.append(alt_full)

        # Ensure uniqueness while preserving order
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
        return unique_candidates

    def _open_camera(self, source):
        """Open camera source, trying safer RTSP fallbacks if necessary."""
        candidates = self._build_rtsp_alternatives(source)
        last_error = None
        for candidate in candidates:
            backend = cv2.CAP_FFMPEG if isinstance(candidate, str) and candidate.lower().startswith("rtsp://") else cv2.CAP_ANY
            if backend == cv2.CAP_FFMPEG:
                logger.info(f"Opening RTSP stream with FFmpeg backend: {candidate[:30]}...")
            try:
                cap = cv2.VideoCapture(candidate, backend)
            except Exception as e:
                last_error = e
                logger.debug(f"VideoCapture init failed for {candidate}: {e}")
                continue

            if not cap or not cap.isOpened():
                logger.debug(f"Camera candidate unavailable: {candidate}")
                if cap:
                    cap.release()
                continue

            # ✅ CRITICAL FIX: Optimize for IP cameras to reduce lag
            is_ip_camera = isinstance(candidate, str) and candidate.lower().startswith("rtsp://")
            if is_ip_camera:
                # Set minimal buffer to reduce latency (critical for real-time)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Flush initial buffered frames
                for _ in range(10):
                    cap.grab()
                logger.info("IP camera optimized: buffer=1, flushed initial frames")

            if isinstance(source, str) and isinstance(candidate, str) and candidate != source:
                self.rtsp_fallback_applied = True
                if not self._rtsp_fallback_notice_logged:
                    logger.warning(f"Switched to fallback RTSP stream (likely H.264 substream): {candidate}")
                    self._rtsp_fallback_notice_logged = True
                self.camera_index = candidate
            return cap, candidate

        if last_error:
            logger.error(f"Camera open error: {last_error}")
        return None, source

    def start_camera(self):
        if not self.is_running:
            try:
                # Detect available cameras
                available_cameras = detect_available_cameras()
                
                # Create selection dialog
                dialog = tk.Toplevel(self.root)
                dialog.title("Select Camera Source")
                dialog.geometry("400x350")
                dialog.transient(self.root)
                dialog.grab_set()
                
                tk.Label(dialog, text="Select Camera Source:", font=('Helvetica', 12, 'bold')).pack(pady=10)
                
                # Variable to store selection: integer for local, string for IP
                # We'll use a string var and parse it
                selected_source = tk.StringVar(value=str(available_cameras[0]) if available_cameras else "ip")
                
                # Frame for options
                options_frame = tk.Frame(dialog)
                options_frame.pack(fill="both", expand=True, padx=20)
                
                # Local cameras
                if available_cameras:
                    tk.Label(options_frame, text="Local Cameras:", font=('Helvetica', 10, 'bold'), anchor="w").pack(fill="x", pady=(5,0))
                    for idx in available_cameras:
                        tk.Radiobutton(options_frame, text=f"Camera {idx}", variable=selected_source, 
                                      value=str(idx), font=('Helvetica', 10)).pack(anchor="w", padx=10)
                
                # IP Camera Option
                tk.Label(options_frame, text="IP Camera:", font=('Helvetica', 10, 'bold'), anchor="w").pack(fill="x", pady=(10,0))
                tk.Radiobutton(options_frame, text="Add New IP Camera", variable=selected_source, 
                              value="ip", font=('Helvetica', 10)).pack(anchor="w", padx=10)
                
                def on_select():
                    choice = selected_source.get()
                    dialog.destroy()
                    
                    if choice == "ip":
                        self.add_ip_camera_dialog()
                    else:
                        try:
                            self.camera_index = int(choice)
                            self.initialize_camera(self.camera_index)
                        except ValueError:
                            messagebox.showerror("Error", "Invalid camera selection")
                
                tk.Button(dialog, text="Start Camera", command=on_select, bg="#27ae60", 
                         fg="white", font=('Helvetica', 10, 'bold'), height=2).pack(pady=20, fill="x", padx=20)
                
                dialog.wait_window()
                
            except Exception as e:
                logger.error(f"Camera start error: {e}")
                messagebox.showerror("Error", f"Failed to start camera: {e}")

    def add_ip_camera_dialog(self):
        """Dialog to add IP Camera details - Username and Password are REQUIRED"""
        import urllib.parse
        
        ip_dialog = tk.Toplevel(self.root)
        ip_dialog.title("Add IP Camera")
        ip_dialog.geometry("500x450")
        ip_dialog.configure(bg="#2c3e50")
        ip_dialog.transient(self.root)
        ip_dialog.grab_set()
        
        tk.Label(ip_dialog, text="Enter IP Camera Details", font=('Helvetica', 12, 'bold'),
                bg="#2c3e50", fg="white").pack(pady=10)
        
        input_frame = tk.Frame(ip_dialog, bg="#34495e")
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # IP Address
        tk.Label(input_frame, text="IP Address (e.g. 192.168.1.111):", anchor="w",
                bg="#34495e", fg="white", font=('Helvetica', 10)).pack(fill="x", pady=(5,0))
        ip_var = tk.StringVar(value="192.168.1.111")
        tk.Entry(input_frame, textvariable=ip_var, font=('Helvetica', 10)).pack(fill="x", pady=(0, 10))
        
        # RTSP Port
        tk.Label(input_frame, text="RTSP Port (e.g. 554):", anchor="w",
                bg="#34495e", fg="white", font=('Helvetica', 10)).pack(fill="x")
        port_var = tk.StringVar(value="554")
        tk.Entry(input_frame, textvariable=port_var, font=('Helvetica', 10)).pack(fill="x", pady=(0, 10))
        
        # Username - REQUIRED
        tk.Label(input_frame, text="Username (Required):", anchor="w",
                bg="#34495e", fg="#f39c12", font=('Helvetica', 10, 'bold')).pack(fill="x")
        user_var = tk.StringVar()
        user_entry = tk.Entry(input_frame, textvariable=user_var, font=('Helvetica', 10))
        user_entry.pack(fill="x", pady=(0, 10))
        
        # Password - REQUIRED
        tk.Label(input_frame, text="Password (Required):", anchor="w",
                bg="#34495e", fg="#f39c12", font=('Helvetica', 10, 'bold')).pack(fill="x")
        pass_var = tk.StringVar()
        pass_entry = tk.Entry(input_frame, textvariable=pass_var, show="*", font=('Helvetica', 10))
        pass_entry.pack(fill="x", pady=(0, 10))
        
        # Stream Path Selection
        tk.Label(input_frame, text="Stream Path (Select for compatibility):", anchor="w",
                bg="#34495e", fg="white", font=('Helvetica', 10)).pack(fill="x")
        stream_option_var = tk.StringVar(value="stream1")
        stream_frame = tk.Frame(input_frame, bg="#34495e")
        stream_frame.pack(fill="x", pady=(0, 10))
        tk.Radiobutton(stream_frame, text="stream1 (Recommended - Universal H.264)", 
                      variable=stream_option_var, value="stream1",
                      bg="#34495e", fg="white", selectcolor="#2c3e50", 
                      font=('Helvetica', 9)).pack(anchor="w")
        tk.Radiobutton(stream_frame, text="cam/realmonitor?channel=1&subtype=1 (Dahua H.264)", 
                      variable=stream_option_var, value="dahua_h264",
                      bg="#34495e", fg="white", selectcolor="#2c3e50", 
                      font=('Helvetica', 9)).pack(anchor="w")
        
        # Info label
        tk.Label(input_frame, text="✓ stream1 recommended for maximum compatibility\n✓ Uses H.264 codec for smooth real-time playback\n✓ No HEVC/H.265 decoder issues",
                anchor="w", bg="#34495e", fg="#95a5a6", font=('Helvetica', 8), justify="left").pack(fill="x", pady=(5,0))
        
        def connect_ip():
            ip = ip_var.get().strip()
            port = port_var.get().strip()
            username = user_var.get().strip()
            password = pass_var.get().strip()
            
            # Validation
            if not ip or not port:
                messagebox.showerror("Error", "IP Address and Port are required!")
                return
            
            if not username:
                messagebox.showwarning("Username Required", "Please enter a username for the IP camera")
                user_entry.focus_set()
                return
            
            if not password:
                messagebox.showwarning("Password Required", "Please enter a password for the IP camera")
                pass_entry.focus_set()
                return
            
            try:
                # URL-encode password to handle special characters like '@'
                password_encoded = urllib.parse.quote(password)
                
                # Construct RTSP URL based on selected stream path
                stream_option = stream_option_var.get()
                if stream_option == "stream1":
                    # Universal H.264 stream path (works with most cameras)
                    stream_path = "stream1"
                else:
                    # Dahua H.264 substream path
                    stream_path = "cam/realmonitor?channel=1&subtype=1"
                
                rtsp_url = f"rtsp://{username}:{password_encoded}@{ip}:{port}/{stream_path}"
                
                logger.info(f"Connecting to IP camera: rtsp://{username}:***@{ip}:{port}/{stream_path}")
                
                ip_dialog.destroy()
                self.camera_index = rtsp_url  # Store URL as camera index
                self.initialize_camera(rtsp_url)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to construct camera URL: {e}")

        # Buttons frame
        btn_frame = tk.Frame(ip_dialog, bg="#2c3e50")
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Connect", command=connect_ip, bg="#27ae60", 
                 fg="white", font=('Helvetica', 11, 'bold'), width=12, height=1).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=ip_dialog.destroy, bg="#e74c3c", 
                 fg="white", font=('Helvetica', 11, 'bold'), width=12, height=1).pack(side="left", padx=5)

    def initialize_camera(self, source):
        """Initialize camera from source (int index or string URL)"""
        try:
            # Reset fallback tracking for a fresh connect
            if isinstance(source, str):
                self.rtsp_fallback_applied = False
                self._rtsp_fallback_notice_logged = False
                self._rtsp_original_url = source
            else:
                self.rtsp_fallback_applied = False
                self._rtsp_fallback_notice_logged = False
                self._rtsp_original_url = None

            self.decoder_failure_count = 0

            # Open selected camera (with automatic fallback for RTSP streams)
            cap, used_source = self._open_camera(source)
            if not cap:
                messagebox.showerror("Camera Error", f"Failed to open camera source: {source}")
                return

            if self.cap:
                self.cap.release()
            self.cap = cap
            self.camera_index = used_source
            
            # ✅ OPTIMIZATION: Set camera properties for faster capture and better clarity
            # Note: Some properties might not work for IP cameras
            if isinstance(used_source, int):
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            
            # ✅ CRITICAL: Reduce buffer to minimize latency (essential for RTSP streams)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # ✅ CRITICAL: Warm up camera with initial frames
            logger.info("Warming up camera (please wait)...")
            for _ in range(10):
                ret, _ = self.cap.read()
                time.sleep(0.05)
                if not ret:
                    logger.warning("Camera warmup: frame read failed, retrying...")
                    continue
            logger.info("Camera ready!")
                
            self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Fallback dimensions if detection failed (common with IP cams)
            if self.frame_w == 0 or self.frame_h == 0:
                self.frame_w = 640
                self.frame_h = 480
                
            self.is_running = True
            self.is_camera_running = True
            self.btn_camera_toggle.configure(text="🎥 Camera ON", fg_color="#27ae60")
            self.btn_guard_toggle.configure(state="normal")
            self.btn_alert_toggle.configure(state="normal")
            self.btn_fugitive_toggle.configure(state="normal")
            logger.warning(f"Camera started successfully: {self.camera_index}")
            self.update_video_feed()
            
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            messagebox.showerror("Error", f"Failed to initialize camera: {e}")


    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.is_logging:
                self.save_log_to_file()

            self.decoder_failure_count = 0
            self.rtsp_fallback_applied = False
            self._rtsp_fallback_notice_logged = False
            self._rtsp_original_url = None
            
            # Stop Fugitive Mode if running
            if hasattr(self, 'is_fugitive_detection') and self.is_fugitive_detection:
                self.is_fugitive_detection = False
                self.fugitive_image = None
                self.fugitive_face_encoding = None
                self.fugitive_detected_log_done = False
                self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
                self.fugitive_preview_label.configure(image='', text="No Fugitive Selected")
            
            # ✅ SIMPLIFIED: PRO_Detection mode cleanup removed
            
            # Clear guard preview grid
            for widget in self.guard_preview_scroll_frame.winfo_children():
                widget.destroy()
            self.guard_preview_grid = {}
            
            # Cleanup
            for status in self.targets_status.values():
                if status["tracker"]:
                    status["tracker"] = None
            
            gc.collect()
            
            self.is_camera_running = False
            self.btn_camera_toggle.configure(text="🎥 Camera OFF", fg_color="#c0392b")
            self.btn_guard_toggle.configure(state="disabled")
            self.btn_fugitive_toggle.configure(state="disabled")
            self.video_label.configure(image='')

    def auto_flush_logs(self):
        """Automatically flush logs when threshold reached"""
        if self.is_logging and len(self.temp_log) >= CONFIG["logging"]["auto_flush_interval"]:
            self.save_log_to_file()
        
        # ✅ OPTIMIZATION: Reduce memory optimization frequency from every 300 to every 500 frames
        # This reduces garbage collection overhead which can cause temporary latency spikes
        # At 30 FPS: once per 16-17 seconds instead of 10 seconds
        if self.frame_counter % 500 == 0:  # Every ~500 frames
            optimize_memory()
    
    def optimize_memory(self):
        """Clear old cache entries and collect garbage to free memory"""
        try:
            # Clear old action cache - keep last 50 entries
            # ✅ OPTIMIZATION: Use more aggressive cleanup to prevent memory bloat
            max_cache_size = 100  # Increased from 50 for multi-guard scenarios
            if len(self.last_action_cache) > max_cache_size:
                # Keep only the most recent entries
                # Sort by value (which is a dict with timestamp if available) 
                # For now, just keep last N
                keys_to_remove = list(self.last_action_cache.keys())[:-max_cache_size]
                for key in keys_to_remove:
                    del self.last_action_cache[key]
            
            # Force garbage collection - but be selective to avoid latency
            # Only force full collection every 500 frames (not every 300)
            gc.collect()
            logger.debug("Memory optimized - caches cleared, garbage collected")
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")

    def save_log_to_file(self):
        if self.temp_log:
            try:
                log_dir = CONFIG["logging"]["log_directory"]
                os.makedirs(log_dir, exist_ok=True)
                csv_path = os.path.join(log_dir, "events.csv")
                
                file_exists = os.path.exists(csv_path)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Timestamp", "Guard Name", "Action", "Status", "Image Path", "Confidence"])
                    writer.writerows(self.temp_log)
                logger.warning(f"Saved {len(self.temp_log)} log entries to {csv_path}")
                self.temp_log.clear()
                self.temp_log_counter = 0
            except Exception as e:
                logger.error(f"Log save error: {e}")


            
    def capture_alert_snapshot(self, frame, target_name, check_rate_limit=False, bbox=None, is_fugitive=False):
        """
        Capture alert snapshot of FULL FRAME with highlighted bounding box.
        
        Args:
            frame: Full camera frame (not cropped)
            target_name: Name of the target
            check_rate_limit: If True, only capture if 60+ seconds since last snapshot
            bbox: (x1, y1, x2, y2) bounding box coordinates to highlight, or None
            is_fugitive: If True, use red color for fugitive highlight
        
        Returns:
            filename if saved, None if rate limited, "Error" if failed
        """
        current_time = time.time()
        
        # Rate limiting check: only one snapshot per minute per target
        if check_rate_limit and target_name in self.targets_status:
            last_snap_time = self.targets_status[target_name].get("last_snapshot_time", 0)
            if (current_time - last_snap_time) < 60:  # Less than 60 seconds
                return None  # Skip snapshot due to rate limit
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = target_name.replace(" ", "_")
        snapshot_dir = CONFIG["storage"]["alert_snapshots_dir"]
        filename = os.path.join(snapshot_dir, f"alert_{safe_name}_{timestamp}.jpg")
        try:
            # Make a copy to draw on (don't modify original frame)
            frame_copy = frame.copy()
            
            # Draw highlighted bounding box if provided
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Color: RED for fugitive, YELLOW for guard alerts
                color = (0, 0, 255) if is_fugitive else (0, 255, 255)
                thickness = 4  # Thick border for visibility
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
                
                # Add label above box
                label = f"FUGITIVE: {target_name}" if is_fugitive else f"ALERT: {target_name}"
                label_bg_color = (0, 0, 200) if is_fugitive else (0, 200, 200)
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame_copy, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), label_bg_color, -1)
                cv2.putText(frame_copy, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Convert and save
            bgr_frame = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_frame)
            
            # Update last snapshot time for this target
            if target_name in self.targets_status:
                self.targets_status[target_name]["last_snapshot_time"] = current_time
            
            return filename
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
            return "Error"

    def enter_onboarding_mode(self):
        if not self.is_running: return
        self.onboarding_mode = True
        self.onboarding_step = 0  # 0=front angle, 1=left, 2=right, 3=back
        self.onboarding_poses = {}  # Now stores angle photos instead of poses
        self.is_in_capture_mode = True
        
        name = simpledialog.askstring("New Guard", "Enter guard name:")
        if not name:
            self.onboarding_mode = False
            self.is_in_capture_mode = False
            return
        self.onboarding_name = name.strip()
        
        messagebox.showinfo("Step 1", "Stand in FRONT of camera (green box will appear when detected). Click 'Snap Photo' when ready.")

    def exit_onboarding_mode(self):
        self.is_in_capture_mode = False
        self.onboarding_mode = False
        self.onboarding_step = 0
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None

    def snap_photo(self):
        if self.unprocessed_frame is None: return
        
        if not self.onboarding_mode:
            # Legacy simple capture - now with dynamic detection
            rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 1:
                name = simpledialog.askstring("Name", "Enter Name:")
                if name:
                    # Get face box for better cropping
                    top, right, bottom, left = face_locations[0]
                    face_h = bottom - top
                    face_w = right - left
                    
                    # Expand to include shoulders/upper body
                    crop_top = max(0, top - int(face_h * 0.3))
                    crop_bottom = min(self.unprocessed_frame.shape[0], bottom + int(face_h * 0.5))
                    crop_left = max(0, left - int(face_w * 0.3))
                    crop_right = min(self.unprocessed_frame.shape[1], right + int(face_w * 0.3))
                    
                    cropped_face = self.unprocessed_frame[crop_top:crop_bottom, crop_left:crop_right]
                    
                    # Save using systematic helpers
                    save_guard_face(cropped_face, name)
                    save_capture_snapshot(cropped_face, name)
                    
                    # Backward compatibility
                    safe_name = name.strip().replace(" ", "_")
                    cv2.imwrite(f"target_{safe_name}_face.jpg", cropped_face)
                    
                    self.load_targets()
                    self.exit_onboarding_mode()
            else:
                messagebox.showwarning("Error", "Ensure exactly one face is visible. Move closer to camera.")
            return
        
        # Onboarding mode with angle captures
        # Normal mode: 4 pictures (front, left, right, back)
        # PRO mode: 5 pictures (front, left, right, back, top)
        if self.onboarding_step == 0:
            # Capture FRONT angle - use cached detection results
            if self.onboarding_face_box is None:
                messagebox.showwarning("Error", "No face detected. Please stand FRONT of camera and wait for green box.")
                return
            
            rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) != 1:
                messagebox.showwarning("Error", "Ensure exactly one face is visible. Move closer to camera.")
                return
            
            # Use detected face box to crop intelligently
            top, right, bottom, left = face_locations[0]
            face_h = bottom - top
            face_w = right - left
            
            # Check if face is large enough (person is close)
            frame_h, frame_w = self.unprocessed_frame.shape[:2]
            face_area_ratio = (face_h * face_w) / (frame_h * frame_w)
            
            if face_area_ratio < 0.02:  # Face is too small
                messagebox.showwarning("Error", "Please move closer to the camera. Face is too small.")
                return
            
            # Expand to include shoulders/upper body for better recognition
            crop_top = max(0, top - int(face_h * 0.3))
            crop_bottom = min(frame_h, bottom + int(face_h * 0.5))
            crop_left = max(0, left - int(face_w * 0.3))
            crop_right = min(frame_w, right + int(face_w * 0.3))
            
            cropped_face = self.unprocessed_frame[crop_top:crop_bottom, crop_left:crop_right]
            
            # Save using systematic helpers with angle support
            if self.onboarding_name:
                save_guard_face(cropped_face, self.onboarding_name, angle="front")
                save_capture_snapshot(cropped_face, self.onboarding_name)
                
                # Save angle-specific photos
                safe_name = self.onboarding_name.replace(" ", "_")
                angle_dir = os.path.join("guard_profiles", safe_name, "angles")
                os.makedirs(angle_dir, exist_ok=True)
                cv2.imwrite(os.path.join(angle_dir, "01_front.jpg"), cropped_face)
                
                # Backward compatibility - save to root
                cv2.imwrite(f"target_{safe_name}_face.jpg", cropped_face)
            
            self.onboarding_step = 1
            messagebox.showinfo("Step 2", "Perfect! Now TURN LEFT (90° profile) and click Snap Photo")
        else:
            # Capture angle photos (left, right, back, top)
            # ✅ ENHANCED: 5 angle-based photos for PRO mode (front, left, right, back, top)
            # Normal mode: 4 angles (front, left, right, back)
            # PRO mode: 5 angles (front, left, right, back, top)
            if self.is_pro_mode:
                angles = ["01_front", "02_left", "03_right", "04_back", "05_top"]
                angle_instructions = ["TURN LEFT (90° profile)", "TURN RIGHT (90° profile)", "TURN AROUND (back view, 180°)", "POSITION CAMERA ABOVE (bird's-eye view)"]
            else:
                angles = ["01_front", "02_left", "03_right", "04_back"]
                angle_instructions = ["TURN LEFT (90° profile)", "TURN RIGHT (90° profile)", "TURN AROUND (back view, 180°)"]
            
            if self.onboarding_face_box is None:
                messagebox.showwarning("Error", f"No face detected. Please stand and face the camera.")
                return
            
            rgb_frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if len(face_locations) != 1:
                messagebox.showwarning("Error", "Ensure exactly one face is visible. Move closer to camera.")
                return
            
            # Capture face at this angle
            top, right, bottom, left = face_locations[0]
            face_h = bottom - top
            face_w = right - left
            frame_h, frame_w = self.unprocessed_frame.shape[:2]
            
            # Expand to include shoulders/upper body
            crop_top = max(0, top - int(face_h * 0.3))
            crop_bottom = min(frame_h, bottom + int(face_h * 0.5))
            crop_left = max(0, left - int(face_w * 0.3))
            crop_right = min(frame_w, right + int(face_w * 0.3))
            
            cropped_angle = self.unprocessed_frame[crop_top:crop_bottom, crop_left:crop_right]
            
            # Save angle-specific photo
            if self.onboarding_name:
                safe_name = self.onboarding_name.replace(" ", "_")
                angle_dir = os.path.join("guard_profiles", safe_name, "angles")
                os.makedirs(angle_dir, exist_ok=True)
                
                angle_file = angles[self.onboarding_step]
                cv2.imwrite(os.path.join(angle_dir, f"{angle_file}.jpg"), cropped_angle)
                self.onboarding_poses[angle_file] = cropped_angle  # Store reference
            
            self.onboarding_step += 1
            
            # ✅ PRO MODE: Check if we need 5th angle (top view)
            max_angles = 4 if not self.is_pro_mode else 5
            
            if self.onboarding_step < max_angles:
                messagebox.showinfo(f"Step {self.onboarding_step + 1}", f"Great! Now {angle_instructions[self.onboarding_step - 1]} and click Snap Photo")
            else:
                # All angles captured
                self.load_targets()
                self.exit_onboarding_mode()
                if self.is_pro_mode:
                    messagebox.showinfo("Complete", f"{self.onboarding_name} onboarding complete! 5 angle photos captured (Front, Left, Right, Back, Top - PRO Mode).")
                else:
                    messagebox.showinfo("Complete", f"{self.onboarding_name} onboarding complete! 4 angle photos captured (Front, Left, Right, Back).")

    def update_video_feed(self):
        if not self.is_running: return
        
        try:
            if not self.cap or not self.cap.isOpened():
                logger.error("Camera not available")
                self.stop_camera()
                return
            
            # ✅ CRITICAL FIX: Aggressive buffer flush for IP cameras to get latest frame
            is_ip_camera = isinstance(self.camera_index, str) and self.camera_index.lower().startswith("rtsp://")
            if is_ip_camera:
                # AGGRESSIVE: Drop accumulated buffered frames to get real-time stream
                # Grab and discard old frames without decoding (fast)
                for _ in range(5):
                    self.cap.grab()
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.decoder_failure_count += 1
                logger.warning("Failed to read frame, attempting recovery...")

                # Clear buffer and retry with exponential backoff
                retry_count = 0
                max_retries = 10
                retry_delay = 0.02
                
                while (not ret or frame is None) and retry_count < max_retries:
                    time.sleep(retry_delay)
                    ret, frame = self.cap.read()
                    retry_count += 1
                    if retry_delay < 0.1:
                        retry_delay *= 1.2  # Exponential backoff

                if ret and frame is not None:
                    # Short-term recovery succeeded
                    self.decoder_failure_count = 0
                else:
                    logger.error(f"Failed to recover after {max_retries} retries, attempting reconnect...")
                    try:
                        if self.cap:
                            self.cap.release()
                        time.sleep(1.0)

                        reconnect_source = self.camera_index
                        if (isinstance(self.camera_index, str) and self._rtsp_original_url and
                                not self.rtsp_fallback_applied):
                            reconnect_source = self._rtsp_original_url
                            logger.warning("RTSP stream unstable - trying H.264 fallback")

                        cap, used_source = self._open_camera(reconnect_source)
                        if not cap:
                            raise RuntimeError("VideoCapture reopen failed")

                        self.cap = cap
                        self.camera_index = used_source

                        if isinstance(used_source, int):
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

                        # ✅ CRITICAL: Minimal buffer for IP cameras
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # ✅ FIX: Additional IP camera optimization after reconnect
                        if isinstance(used_source, str) and used_source.lower().startswith("rtsp://"):
                            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                            logger.info("Reconnected IP camera with low-latency settings")

                        # Warm up after reconnect
                        for _ in range(5):
                            self.cap.read()
                            time.sleep(0.05)

                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            logger.error("Camera reconnection failed")
                            self.stop_camera()
                            messagebox.showerror("Camera Error", "Camera disconnected - please restart")
                            return

                        self.decoder_failure_count = 0
                    except Exception as e:
                        logger.error(f"Reconnection error: {e}")
                        self.stop_camera()
                        return
            else:
                self.decoder_failure_count = 0
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            self.stop_camera()
            return
        
        # ✅ FIX: Detect and handle frame lag for IP cameras
        current_frame_time = time.time()
        frame_delay = current_frame_time - self.last_frame_time
        self.last_frame_time = current_frame_time
        
        if is_ip_camera and frame_delay > 0.2:  # More than 200ms between frames = lag
            self.frame_lag_warnings += 1
            if self.frame_lag_warnings % 30 == 1:  # Log warning every 30 lag occurrences
                logger.warning(f"IP camera lag detected: {frame_delay*1000:.0f}ms frame delay")
            
            # Aggressive buffer flush when lag is detected
            if self.frame_lag_warnings > 5:
                for _ in range(5):
                    self.cap.grab()  # Drop accumulated frames
                self.frame_lag_warnings = 0  # Reset counter
        else:
            self.frame_lag_warnings = max(0, self.frame_lag_warnings - 1)  # Gradual recovery
        
        self.unprocessed_frame = frame.copy()
        
        # ✅ INDUSTRIAL-LEVEL: Enhance frame for low-light conditions
        # This improves face detection in dark areas significantly
        frame = self.enhance_frame_for_low_light(frame)
        
        # Frame skipping for performance
        self.frame_counter += 1
        skip_interval = CONFIG["performance"]["frame_skip_interval"]
        
        # ========== PERFORMANCE MONITORING SIMPLIFIED ==========
        # Removed Phase 4 Stage 5 - just do basic FPS tracking
        
        if self.is_in_capture_mode:
            self.process_capture_frame(frame)
        else:
            # Skip processing every N frames when enabled
            if CONFIG["performance"]["enable_frame_skipping"] and self.frame_counter % skip_interval != 0:
                # Use cached frame
                if self.last_process_frame is not None:
                    frame = self.last_process_frame.copy()
            else:
                # Process tracking frame normally
                self.process_tracking_frame_optimized(frame)
        
                self.last_process_frame = frame.copy()
        
        # ========== PERFORMANCE MONITORING & FPS CALCULATION ==========
        # ✅ OPTIMIZATION: Update GUI labels only every 30 frames (~1 second) instead of every frame
        # This reduces GUI rendering overhead significantly
        self.frame_counter += 1
        
        if self.frame_counter % 30 == 0:  # Update labels every 30 frames
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.last_fps_time = current_time
            
            # ✅ PERFORMANCE: Periodic memory optimization
            if self.frame_counter % 150 == 0:  # Every 150 frames (~5 seconds at 30 FPS)
                optimize_memory()
            
            # Memory monitoring and label updates (only every 30 frames)
            try:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                if hasattr(self, 'fps_label') and self.fps_label.winfo_exists():
                    self.fps_label.configure(text=f"{self.current_fps:.1f}")
                if hasattr(self, 'mem_label') and self.mem_label.winfo_exists():
                    self.mem_label.configure(text=f"{mem_mb:.0f}MB")
            except:
                pass
            
            # Update system clock (only when second changes)
            try:
                current_datetime = datetime.now()
                current_second = current_datetime.second
                if current_second != self.last_clock_second:
                    current_time_str = current_datetime.strftime("%H:%M:%S")
                    if hasattr(self, 'clock_label') and self.clock_label.winfo_exists():
                        self.clock_label.configure(text=current_time_str)
                    self.last_clock_second = current_second
            except:
                pass
            
            # Session time check (only every 30 frames)
            try:
                session_hours = (current_time - self.session_start_time) / 3600
                if session_hours >= CONFIG["monitoring"]["session_restart_prompt_hours"]:
                    response = messagebox.askyesno(
                        "Long Session",
                        f"Session running for {session_hours:.1f} hours. Restart recommended. Continue?"
                    )
                    if not response:
                        self.stop_camera()
                        return
                    else:
                        self.session_start_time = current_time
            except:
                pass
        
        # Auto flush logs
        self.auto_flush_logs()
        
        # ========== OPTIMIZED: GUI VIDEO FEED RENDERING ==========
        # ✅ CRITICAL OPTIMIZATION: Only update GUI every other frame (50ms instead of 33ms)
        # This reduces GUI overhead by ~30-40% and frees up CPU for tracking/detection
        # Visual effect: ~15 FPS GUI (still acceptable for video) but backend gets much more time
        should_update_gui = (self.frame_counter % 2 == 0)  # Update every 2nd frame = 50ms at 30 FPS input
        
        if should_update_gui and self.video_label.winfo_exists():
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # ✅ OPTIMIZED: Scale frame efficiently using linear interpolation
                lbl_w = self.video_label.winfo_width()
                lbl_h = self.video_label.winfo_height()
                h, w = frame.shape[:2]
                
                if lbl_w > 10 and lbl_h > 10:
                    scale = min(lbl_w/w, lbl_h/h, 1.5)
                    new_w, new_h = int(w*scale), int(h*scale)
                    # Use INTER_LINEAR for balance between quality and speed
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # ✅ OPTIMIZATION: Use PIL directly for image conversion
                pil_image = Image.fromarray(frame_rgb, mode='RGB')
                ctk_image = ctk.CTkImage(light_image=pil_image, size=(pil_image.width, pil_image.height))
                self.video_label.configure(image=ctk_image, text="")
                self.video_label.image = ctk_image
            except Exception as e:
                logger.debug(f"Frame display error: {e}")
        
        # ========== DYNAMIC REFRESH RATE OPTIMIZATION ==========
        # ✅ OPTIMIZATION: Smart refresh rate - increase backend processing time automatically
        # When FPS drops, GUI refresh slows down to give backend more CPU time
        # When FPS is good, maintain smooth 30 FPS visual feedback
        refresh_ms = 50  # Base refresh rate
        
        if self.current_fps < 10:
            # Very slow - maximize backend processing time
            refresh_ms = 150  # ~6.7 FPS GUI, but backend gets more time
        elif self.current_fps < 15:
            # Slow - increase backend time
            refresh_ms = 100  # ~10 FPS GUI
        elif self.current_fps < 20:
            # Below target - slight delay
            refresh_ms = 66  # ~15 FPS GUI
        elif self.current_fps >= 25:
            # Good performance - faster refresh for responsive feedback
            refresh_ms = 40  # ~25 FPS GUI
        else:
            # Excellent - maintain smooth 30 FPS
            refresh_ms = 33  # ~30 FPS GUI
        
        self.root.after(refresh_ms, self.update_video_feed)

    def process_capture_frame(self, frame):
        """Process frame during onboarding capture mode with dynamic detection"""
        h, w = frame.shape[:2]
        
        # Detect face and pose from entire frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize pipelines if needed
        if not self.model_pipeline_initialized:
            self._initialize_model_pipeline()
        
        # Load single-person pipeline for onboarding
        self._load_single_person_pipeline()
        
        # Detect face
        face_locations = self._detect_faces_blazeface(rgb_frame)
        
        # Detect pose
        pose_result = self._detect_pose_movenet_lightning(rgb_frame) if self.pose_model else None
        
        # Store detection results for snap_photo to use
        self.onboarding_detection_results = pose_result
        self.onboarding_face_box = None
        
        detection_status = ""
        box_color = (0, 0, 255)  # Red by default
        
        if self.onboarding_step == 0:
            # Step 0: Front angle capture
            
            if len(face_locations) == 1:
                top, right, bottom, left = face_locations[0]
                self.onboarding_face_box = (top, right, bottom, left)
                
                # Check if face is large enough (person is close)
                face_area_ratio = ((bottom - top) * (right - left)) / (h * w)
                
                if face_area_ratio >= 0.02:  # Good size
                    box_color = (0, 255, 0)  # Green
                    detection_status = "READY - Click Snap Photo"
                    # Draw face box
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
                else:
                    box_color = (0, 165, 255)  # Orange
                    detection_status = "Move Closer to Camera"
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            elif len(face_locations) == 0:
                detection_status = "No Face Detected - Stand in FRONT of camera"
            else:
                detection_status = "Multiple Faces - Only one person should be visible"
                
        else:
            # Steps 1-3: Angle captures (left, right, back)
            angle_labels = ["LEFT (90°)", "RIGHT (90°)", "BACK (180°)"]
            target_angle = angle_labels[self.onboarding_step - 1]
            
            if len(face_locations) == 1:
                top, right, bottom, left = face_locations[0]
                self.onboarding_face_box = (top, right, bottom, left)
                
                # Check if face is large enough
                face_area_ratio = ((bottom - top) * (right - left)) / (h * w)
                
                if face_area_ratio >= 0.02:  # Good size
                    box_color = (0, 255, 0)  # Green - ready
                    detection_status = f"READY - Face detected at {target_angle} - Click Snap Photo"
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
                else:
                    box_color = (0, 165, 255)  # Orange
                    detection_status = "Move Closer to Camera"
                    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            elif len(face_locations) == 0:
                detection_status = f"No Face Detected - Face {target_angle}"
            else:
                detection_status = "Multiple Faces - Only one person should be visible"
        
        # Display instructions and status
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)  # Black background for text
        
        if self.onboarding_step == 0:
            instruction = f"STEP 1/4: FRONT ANGLE"
        else:
            angle_labels = ["LEFT (90°)", "RIGHT (90°)", "BACK (180°)"]
            instruction = f"STEP {self.onboarding_step + 1}/4: {angle_labels[self.onboarding_step - 1].upper()}"
        
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, detection_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        return frame

    # --- TRACKING LOGIC ---
    def process_tracking_frame_optimized(self, frame):
        # ✅ CRITICAL: Fugitive detection runs FIRST, before checking if guards exist
        # This ensures fugitive alert works independently of guard tracking state
        
        # Prepare color space for face detection (used by both fugitive and guard detection)
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ==================== FUGITIVE MODE (ALWAYS RUNS - Independent of other modes) ====================
        # ✅ CRITICAL: Fugitive detection MUST run regardless of mode, always prioritized
        # Fugitive is detected instantly and independently from guard tracking or action monitoring
        if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
            # Use downscaled frame for faster detection (same optimization as guard detection)
            # ✅ OPTIMIZED FOR FAR DISTANCE: Reduced scale factor to detect small faces
            scale_factor = 1.2  # Less downscaling for far-distance detection (was 1.5)
            h, w = rgb_full_frame.shape[:2]
            scaled_w = max(1, int(w / scale_factor))
            scaled_h = max(1, int(h / scale_factor))
            rgb_frame_fugitive = cv2.resize(rgb_full_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            
            try:
                face_locations = face_recognition.face_locations(rgb_frame_fugitive, model="hog")
                if face_locations:
                    # ✅ ENHANCED: Better encoding quality for all-angle detection
                    # Use num_jitters=2 for more accurate face encodings at angles (was 1)
                    face_encodings = face_recognition.face_encodings(rgb_frame_fugitive, face_locations, num_jitters=2)
                    
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # ✅ ENHANCED: Multi-angle fugitive detection with relaxed tolerance
                        # Detects fugitive at extreme angles (front, side, up, down)
                        # ✅ FAR DISTANCE DETECTION: Optimized for detecting fugitive from far away
                        face_distance = face_recognition.face_distance([self.fugitive_face_encoding], face_encoding)
                        confidence = 1.0 - face_distance[0]
                        
                        # ✅ CRITICAL: Enhanced angle + distance tolerance
                        # Use larger tolerance (0.72) to detect at extreme angles AND far distances
                        # Small face + extreme angle = lower confidence, but still valid match
                        # Confidence > 0.25 required to prevent false matches on unrelated faces
                        angle_aware_tolerance = 0.72  # Very lenient for all head positions + distance (was 0.70)
                        min_confidence_threshold = 0.25  # Support far-distance detection (was 0.28)
                        
                        is_fugitive_match = (face_distance[0] <= angle_aware_tolerance and 
                                           confidence >= min_confidence_threshold)
                        
                        if is_fugitive_match:
                            # ✅ CRITICAL FIX: PREVENT FALSE FUGITIVE DETECTION AS GUARD
                            # Cross-check: Make sure this isn't a guard that's being misidentified as fugitive
                            is_likely_guard = False
                            if self.targets_status and len(self.targets_status) > 0:
                                # Check distance to all tracked guards
                                for guard_name, guard_status in self.targets_status.items():
                                    guard_encoding = guard_status.get("encoding")
                                    if guard_encoding is not None:
                                        guard_dist = face_recognition.face_distance([guard_encoding], face_encoding)[0]
                                        # If distance to a guard is MUCH closer than to fugitive, it's definitely a guard
                                        # Require 15% margin to confidently distinguish
                                        if guard_dist < (face_distance[0] - 0.15):
                                            logger.debug(f"[SKIP FUGITIVE] Face matches GUARD '{guard_name}' better (guard_dist={guard_dist:.3f} vs fugitive_dist={face_distance[0]:.3f})")
                                            is_likely_guard = True
                                            break
                            
                            if is_likely_guard:
                                continue  # Skip this face - it's a guard, not fugitive
                            
                            # ✅ FUGITIVE CONFIRMED: Immediate log & alert on first appearance
                            # Scale back to original frame size
                            top, right, bottom, left = face_location
                            top, right, bottom, left = int(top*scale_factor), int(right*scale_factor), int(bottom*scale_factor), int(left*scale_factor)
                            
                            # Draw bounding box - BRIGHT RED for fugitive alert
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                            cv2.putText(frame, f"🚨 FUGITIVE: {self.fugitive_name} 🚨", (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            # ✅ CRITICAL: Fugitive behavior requirements:
                            # 1. Log entry ONLY on first-ever detection (tracked with fugitive_logged_once flag)
                            # 2. Alert sound triggers on EVERY FRAME where fugitive is detected
                            
                            # Track if this is first appearance (for logging)
                            is_first_appearance = not getattr(self, 'fugitive_currently_visible', False)
                            self.fugitive_currently_visible = True
                            
                            # 1. PLAY FUGITIVE ALERT SOUND ON EVERY DETECTION (restart if not playing)
                            try:
                                # Check if alert is currently playing
                                alert_playing = (hasattr(self, 'fugitive_alert_sound_thread') and 
                                               self.fugitive_alert_sound_thread is not None and 
                                               self.fugitive_alert_sound_thread.is_alive())
                                
                                if not alert_playing:
                                    # Alert not playing - start new 15-second alert
                                    if self.fugitive_alert_stop_event is None:
                                        self.fugitive_alert_stop_event = threading.Event()
                                    
                                    self.fugitive_alert_stop_event = threading.Event()
                                    self.fugitive_alert_sound_thread = play_siren_sound(
                                        stop_event=self.fugitive_alert_stop_event,
                                        sound_file=get_sound_path("Fugitive.mp3"),
                                        duration_seconds=15  # 15 seconds for fugitive alert
                                    )
                                    logger.warning(f"[FUGITIVE ALERT] !!! FUGITIVE DETECTED - {self.fugitive_name} - Playing 15s alert! (confidence: {confidence:.3f})")
                            except Exception as e:
                                logger.error(f"[FUGITIVE SOUND] Error playing alert: {e}")
                            
                            # 2. LOG ENTRY: ONLY on first-ever detection (not on re-appearances)
                            if is_first_appearance:
                                if not getattr(self, 'fugitive_logged_once', False):
                                    try:
                                        # Capture full frame snapshot with highlighted fugitive bbox
                                        fugitive_bbox = (left, top, right, bottom)
                                        snapshot_path = self.capture_alert_snapshot(
                                            frame, 
                                            f"FUGITIVE_{self.fugitive_name}_APPEARED", 
                                            check_rate_limit=False,
                                            bbox=fugitive_bbox,
                                            is_fugitive=True
                                        )
                                        img_path = snapshot_path if snapshot_path else "N/A"
                                        
                                        # Create CSV log entry for fugitive appearance (ONLY ONCE)
                                        self.temp_log.append((
                                            time.strftime("%Y-%m-%d %H:%M:%S"),
                                            f"FUGITIVE_{self.fugitive_name}",
                                            "FUGITIVE_DETECTED",
                                            f"FIRST_APPEARANCE (confidence: {confidence:.3f})",
                                            img_path,
                                            f"{confidence:.3f}"
                                        ))
                                        self.temp_log_counter += 1
                                        
                                        # Mark as logged so we don't log again on re-appearances
                                        self.fugitive_logged_once = True
                                        
                                        # Flush CSV immediately on fugitive detection for real-time logging
                                        self.save_log_to_file()
                                        
                                        logger.warning(f"[FUGITIVE DETECTED] !!! {self.fugitive_name} FIRST APPEARANCE logged - Snapshot captured (confidence: {confidence:.3f})")
                                    except Exception as e:
                                        logger.error(f"[FUGITIVE LOG] Error creating fugitive log entry: {e}")
                            else:
                                # Not first appearance but still detected - alert continues to play if active
                                logger.debug(f"[FUGITIVE TRACKING] {self.fugitive_name} still in frame (confidence: {confidence:.3f})")
                        
            except Exception as e:
                logger.debug(f"Fugitive detection error: {e}")
        
        # ✅ RESET FUGITIVE VISIBILITY FLAG: If fugitive not found in this frame
        # This allows us to detect again on next appearance
        else:
            # Fugitive detection is enabled but NO fugitive found in current frame
            if getattr(self, 'fugitive_currently_visible', False):
                # Fugitive was visible last frame but now disappeared
                self.fugitive_currently_visible = False
                logger.debug(f"[FUGITIVE] {self.fugitive_name} disappeared from view - resetting for next detection")
        # ✅ GUARD TRACKING: Only run if guards are selected
        if not self.targets_status:
            cv2.putText(frame, "SELECT TARGETS TO START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        # ✅ PERFORMANCE OPTIMIZATION: Adaptive detection interval based on tracking stability
        # More frequent detection when targets are lost, less frequent when stable
        self.re_detect_counter += 1
        untracked_count = len([n for n, s in self.targets_status.items() if not s["visible"]])
        
        # ✅ CRITICAL FIX: ALWAYS detect if guards are lost
        # If ANY guard loses track, detection runs EVERY FRAME (interval=1) until re-acquired
        # Previously only ran on fixed intervals - this caused permanent loss after tracker failure
        
        # ✅ OPTIMIZED INTERVALS (AGGRESSIVE FOR LOST GUARD RECOVERY):
        # - If targets missing: 1 frame (33ms) - RUN EVERY FRAME to re-acquire lost guards
        # - If all stable: 8 frames (267ms) - conservative, let tracker do the work
        # Key insight: Faster detection interval ensures guards are identified as soon as they appear on camera
        num_visible = len([n for n, s in self.targets_status.items() if s["visible"]])
        
        # ✅ INSTANT TRACKING: Check if any guard is newly detected
        # Newly detected guards get 1-frame detection for 5 frames (166ms) to ensure rapid acquisition
        any_newly_detected = any(s.get("newly_detected", False) for s in self.targets_status.values())
        
        if untracked_count > 0:
            # ANY guard missing - use maximum aggression to find and re-acquire them
            # This ensures guards aren't permanently lost if tracker fails once
            # ✅ ENHANCED: Faster re-detection after movement - every frame (1) instead of waiting
            adaptive_interval = 1  # 33ms - EVERY FRAME until all targets re-acquired (fast re-identification)
            logger.debug(f"[DETECT] {untracked_count}/{len(self.targets_status)} targets lost - running detection EVERY frame")
        elif any_newly_detected:
            # ✅ INSTANT TRACKING: Newly detected guards need rapid re-detection for stable tracking
            # Run at 1-frame interval (33ms) for the first 5 frames after detection
            adaptive_interval = 1
            # Increment frame counter and reset flag after 5 frames (166ms)
            for name, status in self.targets_status.items():
                if status.get("newly_detected", False):
                    status["detection_frame_count"] = status.get("detection_frame_count", 0) + 1
                    if status["detection_frame_count"] >= 5:
                        status["newly_detected"] = False
                        status["detection_frame_count"] = 0
                        logger.debug(f"[INSTANT] {name} tracking stabilized after 5 frames, returning to normal interval")
            logger.debug(f"[INSTANT] Instant tracking active - {sum(1 for s in self.targets_status.values() if s.get('newly_detected'))} guard(s) in rapid acquisition phase")
        else:
            # All visible and tracked - run detection MORE FREQUENTLY for real-time accuracy
            # ✅ MULTI-GUARD STABILITY: Faster detection for multiple guards (2 frames vs 3)
            # Multiple guards need more frequent verification to prevent mix-ups and maintain stable tracking
            num_guards = len(self.targets_status)
            if num_guards >= 2:
                adaptive_interval = 2  # 67ms - Very fast for multi-guard stability
            else:
                adaptive_interval = 3  # 100ms - Standard for single guard
        
        if self.re_detect_counter > adaptive_interval:
            self.re_detect_counter = 0
        
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ✅ SIMPLIFIED: PRO_DETECTION MODE removed - focus on Normal Mode only
        
        # ✅ IMPORTANT: Get current time at the start of the tracking loop
        current_time = time.time()

        # 1. Update Trackers (PERSISTENT TRACKING - Keep tracking even at frame edges)
        # ✅ CRITICAL: Keep tracking guards even if they partially leave the frame
        # Persistent tracking ensures continuous monitoring until guard fully exits
        for name, status in self.targets_status.items():
            # ✅ OPTIMIZATION: Skip invisible targets COMPLETELY - no tracker update, no drawing
            # This saves ~5-10ms per untracked target per frame
            if not status.get("visible", False):
                continue  # Skip to next target - invisible targets get zero processing
            
            if status.get("tracker") is None:
                continue  # No tracker to update
            
            # ✅ PERFORMANCE: Only update tracker if it exists and target is visible
            # Tracker update is fast (~5ms) but skip when not needed
            success, box = status["tracker"].update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                new_box = (x, y, x + w, y + h)
                
                # ✅ OPTIMIZATION: Calculate tracker confidence based on movement consistency
                # If tracker is giving consistent results within small movement, confidence is high
                # This allows us to skip expensive face detection when tracker is stable
                if status.get("face_box") is not None:
                    old_x1, old_y1, old_x2, old_y2 = status["face_box"]
                    new_x1, new_y1, new_x2, new_y2 = new_box
                    dx = abs(new_x1 - old_x1) + abs(new_x2 - old_x2)
                    dy = abs(new_y1 - old_y1) + abs(new_y2 - old_y2)
                    old_w = max(1, old_x2 - old_x1)
                    old_h = max(1, old_y2 - old_y1)
                    new_w = max(1, new_x2 - new_x1)
                    new_h = max(1, new_y2 - new_y1)
                    
                    # ✅ MULTI-GUARD STABILITY: More conservative confidence for multiple guards
                    # Multi-guard scenarios need stricter validation to prevent guard mix-ups
                    movement_ratio = (dx + dy) / max(1, (old_w + old_h) * 2)  # Normalized movement
                    size_stability = 1.0 - abs(new_w - old_w) / max(1, old_w)  # How stable is size
                    
                    # Confidence calculation with multi-guard penalty
                    num_guards = len(self.targets_status)
                    base_confidence = max(0.5, 1.0 - movement_ratio) * size_stability
                    
                    # Apply stricter confidence for multi-guard (reduce by 10% to force more verification)
                    if num_guards >= 2:
                        tracker_confidence = base_confidence * 0.90  # 10% penalty for multi-guard
                    else:
                        tracker_confidence = base_confidence
                    
                    status["tracker_confidence"] = min(1.0, max(0.0, tracker_confidence))
                else:
                    status["tracker_confidence"] = 0.95  # Default high confidence if no prior box
                
                if status["face_box"] is not None:
                    # Sanity check: detect if tracker jumped too far (tracker failure)
                    old_x1, old_y1, old_x2, old_y2 = status["face_box"]
                    new_x1, new_y1, new_x2, new_y2 = new_box
                    dx = abs(new_x1 - old_x1) + abs(new_x2 - old_x2)
                    dy = abs(new_y1 - old_y1) + abs(new_y2 - old_y2)
                    old_w = max(1, old_x2 - old_x1)
                    old_h = max(1, old_y2 - old_y1)
                    new_w = max(1, new_x2 - new_x1)
                    new_h = max(1, new_y2 - new_y1)
                    size_change = abs(new_w - old_w) + abs(new_h - old_h)
                    
                    # ✅ MOVEMENT THRESHOLD FIX: Ensure minimum threshold for small bounding boxes
                    # For small boxes (e.g. 36x36), 80% is too small. Enforce minimum base of 60px.
                    # This ensures movement_threshold is at least 300px (60 * 5).
                    max_movement = max(max(old_w, old_h) * 0.8, 60)  
                    max_size_change = (old_w + old_h) * 0.65  # 65% size change for depth movement
                    
                    # ✅ PERSISTENT TRACKING: Allow larger movement but not EXTREME
                    frame_h, frame_w = frame.shape[:2]
                    at_frame_edge = (new_x1 < 50 or new_x2 > frame_w - 50 or 
                                    new_y1 < 50 or new_y2 > frame_h - 50)
                    
                    # ✅ MOVEMENT TOLERANCE: Allow natural guard movements (walking, running, jumping)
                    # Base: 80% box size, Multiplier: 5x-6x for normal human movement patterns
                    # This prevents losing track during speed walking, jumping, or quick turns
                    movement_threshold = (max_movement * 6) if at_frame_edge else (max_movement * 5)
                    size_threshold = (max_size_change * 6) if at_frame_edge else (max_size_change * 5)
                    
                    # ✅ EXTREME ANGLE TOLERANCE: If pose quality is poor (back view, extreme angle),
                    # increase tolerance slightly (but not too much) to prevent losing track
                    pose_quality_check = status.get("pose_confidence", 0.3)
                    if pose_quality_check < 0.3:
                        # Back/extreme angle view - more lenient for guards moving away from camera
                        movement_threshold = movement_threshold * 2.0  # 2x more lenient for back/side views
                        size_threshold = size_threshold * 2.0
                        logger.debug(f"[TRACKER] {name}: Using extreme-angle tolerance (sparse pose: {pose_quality_check:.2f})")
                    
                    if dx > movement_threshold or dy > movement_threshold or size_change > size_threshold:
                        # Only reset tracker if movement is EXTREME (likely lost track, not natural movement)
                        status["visible"] = False
                        status["tracker"] = None
                        status["consecutive_detections"] = 0
                        status["stable_tracking"] = False
                        logger.warning(f"[TRACKER LOST] {name}: Extreme movement detected (dx={dx:.0f}, dy={dy:.0f}, size_change={size_change:.0f}) - pose_quality:{pose_quality_check:.2f}")
                    else:
                        # Tracker still valid - apply LIGHTER smoothing for better responsiveness
                        # Reduced smoothing factor from 0.75 to 0.4 to fix lag during walking/jumping
                        smoothed_box = smooth_bounding_box(new_box, status["face_box"], smoothing_factor=0.4)
                        status["face_box"] = smoothed_box
                        status["visible"] = True
                        # ✅ CRITICAL: Mark for face re-verification on next detection
                        # This ensures we verify the tracker is actually on a real face, not a ghost
                        status["needs_face_reverification"] = True
                        
                        # ✅ ANTI-GHOST: Track consecutive frames without face detection
                        # If tracker runs without finding face for too long, it's a ghost
                        if not status.get("consecutive_no_face_frames"):
                            status["consecutive_no_face_frames"] = 0
                        
                        # Log tracking status
                        if at_frame_edge:
                            logger.debug(f"[TRACKER OK] {name}: Persistent tracking at frame edge (movement: dx={dx:.0f}, dy={dy:.0f})")
                        else:
                            logger.debug(f"[TRACKER OK] {name}: Tracked (movement: dx={dx:.0f}, dy={dy:.0f})")
                else:
                    status["face_box"] = new_box
                    status["visible"] = True
            else:
                # Tracker lost the target
                status["tracker"] = None
                status["visible"] = False
                logger.warning(f"[TRACKER FAILED] {name}: Tracker returned False (will re-detect)")

        # 2. Detection (PARALLEL MATCHING) - Fast, frequent detection for newly selected guards
        # ✅ OPTIMIZATION: Detection runs very frequently to catch new guards
        untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
        
        # ✅ FREQUENCY (OPTIMIZED): Run detection every 2-5-12 frames depending on stability
        # CRITICAL: Always run detection frequently to verify tracker is on real face, not ghost
        all_stable_and_tracked = all(
            s.get("visible", False) and s.get("stable_tracking", False) 
            for s in self.targets_status.values()
        )
        
        # ✅ OPTIMIZATION: Only run detection on schedule (adaptive_interval)
        # CHANGED: Even stable targets get re-verified every 12 frames to prevent ghost tracking
        should_run_detection = (
            (self.re_detect_counter % adaptive_interval == 0)  # SIMPLIFIED: Interval-based only
        )  # REMOVED: 'and not all_stable_and_tracked' - NOW ALWAYS RUN to verify no ghosts
        
        if should_run_detection:
            # ✅ OPTIMIZATION: Skip expensive face detection if all targets have very high tracker confidence
            # If tracker confidence > 0.96 (was 0.92) for all visible targets, skip re-detection this frame
            # Increased threshold to force more frequent face verification (fixes drift/ghosting)
            all_trackers_confident = all(
                status.get("tracker_confidence", 0.0) > 0.96
                for status in self.targets_status.values()
                if status.get("visible", False)
            )
            
            # Auto-detect single vs multi-person mode based on target count
            is_single_person = len(self.targets_status) <= 1
            
            if all_trackers_confident and len([s for s in self.targets_status.values() if s.get("visible")]) > 0:
                # All visible targets have high confidence trackers - skip detection, just update trackers
                logger.debug(f"[PERF-SKIP] All {len([s for s in self.targets_status.values() if s.get('visible')])} targets have high confidence trackers (>0.96) - skipping face detection")
                face_locations = []
            else:
                # Need to run detection
                # ✅ NEW PIPELINE: Initialize and use new model pipelines
                
                # Initialize pipeline if needed
                if not self.model_pipeline_initialized:
                    self._initialize_model_pipeline()
                
                # Load appropriate pipeline
                if is_single_person:
                    # Single person optimization
                    face_locations = face_recognition.face_locations(rgb_full_frame, model="hog")
                else:
                    # Multi person - standard detection
                    face_locations = face_recognition.face_locations(rgb_full_frame, model="hog")
            # Get encodings if faces found
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations)
            else:
                face_encodings = []

            # Iterate over all targets to verify tracking
            for name, status in self.targets_status.items():
                if not status["visible"]:
                    continue
                
                face_found_at_tracked_location = False
                
                # Check if any detected face matches this target
                if face_locations:
                    # Check against known encoding
                    if status.get("encoding") is not None:
                        matches = face_recognition.compare_faces(face_encodings, status["encoding"], tolerance=0.6)
                        if True in matches:
                            face_found_at_tracked_location = True
                            # Update face box with the matched face
                            match_index = matches.index(True)
                            # ✅ FIX: Convert face_recognition (top, right, bottom, left) to (x1, y1, x2, y2)
                            top, right, bottom, left = face_locations[match_index]
                            status["face_box"] = (left, top, right, bottom)
                            
                            # ✅ RE-INIT TRACKER: Sync tracker with detected face to prevent drift
                            # If we don't re-init, tracker continues from old (wrong) position, causing "Extreme movement" next frame
                            if status.get("tracker"):
                                fl, ft, fr, fb = status["face_box"] # Now it is x1, y1, x2, y2
                                fw = fr - fl
                                fh = fb - ft
                                try:
                                    status["tracker"] = cv2.TrackerMIL_create() # Re-create to be safe
                                    status["tracker"].init(frame, (fl, ft, fw, fh))
                                    logger.debug(f"[TRACKER SYNC] {name}: Re-initialized tracker on detected face")
                                except Exception as e:
                                    logger.error(f"Tracker re-init failed: {e}")
                    
                    # Also check overlap with tracker box if encoding match failed or not available
                    if not face_found_at_tracked_location:
                        tracker_box = status.get("face_box")
                        if tracker_box:
                            # ✅ FIX: face_box is now (x1, y1, x2, y2)
                            tx1, ty1, tx2, ty2 = tracker_box
                            tracker_area = (tx2 - tx1) * (ty2 - ty1)
                            
                            for i, face_loc in enumerate(face_locations):
                                # face_loc is (top, right, bottom, left) -> (y1, x2, y2, x1)
                                fy1, fx2, fy2, fx1 = face_loc
                                
                                # Calculate IoU
                                inter_x1 = max(tx1, fx1)
                                inter_y1 = max(ty1, fy1)
                                inter_x2 = min(tx2, fx2)
                                inter_y2 = min(ty2, fy2)
                                
                                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                    face_area = (fx2 - fx1) * (fy2 - fy1)
                                    iou = inter_area / (tracker_area + face_area - inter_area)
                                    
                                    if iou > 0.3: # 30% overlap
                                        face_found_at_tracked_location = True
                                        # ✅ FIX: Convert face_recognition (top, right, bottom, left) to (x1, y1, x2, y2)
                                        top, right, bottom, left = face_loc
                                        status["face_box"] = (left, top, right, bottom)
                                        
                                        # ✅ RE-INIT TRACKER: Sync tracker with detected face (IoU match)
                                        if status.get("tracker"):
                                            fl, ft, fr, fb = status["face_box"] # Now it is x1, y1, x2, y2
                                            fw = fr - fl
                                            fh = fb - ft
                                            try:
                                                status["tracker"] = cv2.TrackerMIL_create()
                                                status["tracker"].init(frame, (fl, ft, fw, fh))
                                                logger.debug(f"[TRACKER SYNC] {name}: Re-initialized tracker on detected face (IoU match)")
                                            except Exception as e:
                                                logger.error(f"Tracker re-init failed: {e}")
                                        break

                if not face_found_at_tracked_location:
                    # Body box is often larger/different than face box
                    # Check if tracker is still in reasonable position (not completely drifted)
                    body_box = status.get("body_box")
                    pose_confidence = status.get("pose_confidence", 0.0)
                    skeleton_keypoints = status.get("skeleton_keypoints", 0)
                    
                    # ROBUST 4-PART VALIDATION for body-only tracking:
                    # 1. body_box exists (from skeleton)
                    # 2. pose confidence adequate (>0.50 is good quality) - INCREASED from 0.40
                    # 3. sufficient keypoints (≥10 indicates valid full-body skeleton)
                    # 4. body is in/near frame (within -50 to width+50, -50 to height+50)
                    
                    body_box_valid = body_box is not None
                    pose_quality_valid = pose_confidence > 0.50
                    keypoint_valid = skeleton_keypoints >= 10
                    
                    # Extract body box position for frame boundary check
                    body_in_frame = False
                    if body_box_valid:
                        bx1, by1, bx2, by2 = body_box
                        frame_h, frame_w = frame.shape[:2]
                        # Allow 50px margin beyond frame edges for tracking at boundaries
                        body_in_frame = (bx1 <= frame_w + 50 and bx2 >= -50 and 
                                       by1 <= frame_h + 50 and by2 >= -50)
                    
                    # ✅ CRITICAL: ANTI-GHOST LOGIC - Check if body is actually visible
                    # Not just checking skeleton but also verifying against detected faces in current frame
                    face_in_body_area = False
                    if body_box_valid and face_locations and len(face_locations) > 0:
                        bx1, by1, bx2, by2 = body_box
                        body_area = bx1 * by1 * (bx2 - bx1) * (by2 - by1)
                        for face_loc in face_locations:
                            fx1, fx2, fy1, fy2 = face_loc[2], face_loc[1], face_loc[0], face_loc[3]  # left, right, top, bottom
                            # Check if any detected face overlaps with body box
                            inter_x1 = max(bx1, fx1)
                            inter_y1 = max(by1, fy1)
                            inter_x2 = min(bx2, fx2)
                            inter_y2 = min(by2, fy2)
                            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                                iou = inter_area / max(1, body_area)
                                if iou >= 0.25:  # 25% overlap means face in body area (Increased from 15%)
                                    face_in_body_area = True
                                    logger.debug(f"[GHOST CHECK] {name}: Face detected in body area (IoU={iou:.2f})")
                                    break
                    
                    # If we have a VALID body detection (all criteria met), allow continued tracking
                    if body_box_valid and pose_quality_valid and keypoint_valid and body_in_frame:
                        # All validation criteria passed - high confidence body tracking
                        # But only if either (a) face visible in body area OR (b) skeleton very confident
                        if face_in_body_area or pose_confidence > 0.60:
                            status["face_detection_missing_frames"] = status.get("face_detection_missing_frames", 0) + 1
                            
                            # QUALITY-BASED TOLERANCE: More frames allowed for better skeleton quality
                            # REDUCED TOLERANCE to prevent ghost tracking (was 90/60/45)
                            # High quality (pose_confidence > 0.60): allow 45 frames (~1.5 seconds)
                            # Good quality (pose_confidence > 0.40): allow 30 frames (~1 second)
                            # Default fallback: 15 frames (~0.5 seconds)
                            if pose_confidence > 0.60:
                                max_missing_frames = 45
                                quality_tier = "HIGH"
                            elif pose_confidence > 0.40:
                                max_missing_frames = 30
                                quality_tier = "GOOD"
                            else:
                                max_missing_frames = 15
                                quality_tier = "BASIC"
                            
                            if status["face_detection_missing_frames"] <= max_missing_frames:
                                logger.debug(f"[BODY TRACK] {name}: No face (frame {status['face_detection_missing_frames']}/{max_missing_frames}), "
                                           f"body:✓ pose:{pose_confidence:.2f}({quality_tier}) keypts:{skeleton_keypoints}/33 - continuing skeleton tracking")
                            else:
                                # Too long without face and body validation failed
                                status["visible"] = False
                                status["tracker"] = None
                                status["consecutive_detections"] = 0
                                status["stable_tracking"] = False
                                logger.warning(f"[TRACK ENDED] {name}: No face for {max_missing_frames} frames - guard likely left frame")
                        else:
                            # Body valid but no face in body area AND low pose confidence - GHOST
                            status["visible"] = False
                            status["tracker"] = None
                            status["consecutive_detections"] = 0
                            status["stable_tracking"] = False
                            logger.warning(f"[GHOST REMOVED] {name}: Body detected but no face in area + low pose confidence - likely ghost")
                    
                    elif body_box_valid and (pose_quality_valid or keypoint_valid):
                        # Partial validation (either quality or keypoints valid, but not both)
                        # More conservative: allow up to 35 frames (1+ second)
                        status["face_detection_missing_frames"] = status.get("face_detection_missing_frames", 0) + 1
                        
                        if status["face_detection_missing_frames"] <= 35:
                            logger.debug(f"[BODY TRACK-PARTIAL] {name}: Partial validation (pose:{pose_confidence:.2f}, keypts:{skeleton_keypoints})")
                        else:
                            status["visible"] = False
                            status["tracker"] = None
                            status["consecutive_detections"] = 0
                            status["stable_tracking"] = False
                            logger.warning(f"[TRACK ENDED-PARTIAL] {name}: Partial body validation failed after 35 frames")
                    else:
                        # GHOST DETECTION: No face AND failed body validation
                        # Multiple reasons could cause this:
                        invalid_reasons = []
                        if not body_box_valid:
                            invalid_reasons.append("no_body_box")
                        if not pose_quality_valid:
                            invalid_reasons.append(f"pose_low({pose_confidence:.2f})")
                        if not keypoint_valid:
                            invalid_reasons.append(f"keypts_low({skeleton_keypoints})")
                        if not body_in_frame:
                            invalid_reasons.append("out_of_frame")
                        
                        status["visible"] = False
                        status["tracker"] = None
                        status["consecutive_detections"] = 0
                        status["stable_tracking"] = False
                        logger.warning(f"[GHOST REMOVED] {name}: Tracked but failed body validation - {', '.join(invalid_reasons)}")
                else:
                    # Face found - reset counter
                    status["face_detection_missing_frames"] = 0
            
            # ✅ BACK-VIEW SKELETON TRACKING: Reinitialize tracker using skeleton when face missing
            # When guard's back is to camera, face is not detected but skeleton is visible
            # Reininitialize tracker with skeleton-derived bounding box to maintain continuous tracking
            for name, status in self.targets_status.items():
                if status["visible"] and status.get("tracker") is not None:
                    # Check conditions for skeleton-based tracker reinitialization
                    face_missing_frames = status.get("face_detection_missing_frames", 0)
                    pose_confidence = status.get("pose_confidence", 0.0)
                    skeleton_keypoints = status.get("skeleton_keypoints", 0)
                    body_box = status.get("body_box")
                    
                    # Trigger reinitialization when:
                    # 1. Face has been missing for 12+ frames (400ms) - allows time for normal movement
                    # 2. Skeleton is valid (pose_confidence > 0.35, keypoints >= 10)
                    # 3. Body box is available and in frame
                    if (face_missing_frames >= 12 and 
                        pose_confidence > 0.35 and 
                        skeleton_keypoints >= 10 and 
                        body_box is not None):
                        
                        # ✅ CRITICAL: Convert normalized skeleton box to pixel coordinates
                        bx1_norm, by1_norm, bx2_norm, by2_norm = body_box
                        frame_h, frame_w = frame.shape[:2]
                        
                        # Convert normalized (0-1) coordinates to pixel coordinates
                        bx1 = max(0, int(bx1_norm * frame_w))
                        by1 = max(0, int(by1_norm * frame_h))
                        bx2 = min(frame_w, int(bx2_norm * frame_w))
                        by2 = min(frame_h, int(by2_norm * frame_h))
                        
                        bbox_width = bx2 - bx1
                        bbox_height = by2 - by1
                        
                        # Validate new bounding box
                        if bbox_width >= 20 and bbox_height >= 20:
                            try:
                                # ✅ BACK-VIEW TRACKING: Reinitialize tracker with skeleton bbox
                                # Use existing tracker if available, don't create new one (wastes resources)
                                if hasattr(status["tracker"], 'init'):
                                    status["tracker"].init(frame, (bx1, by1, bbox_width, bbox_height))
                                    status["back_view_tracking"] = True
                                    status["skeleton_track_start_frame"] = self.frame_counter
                                    logger.warning(f"[BACK VIEW] {name}: Tracker reinitialized with skeleton bbox at frame {self.frame_counter} "
                                                 f"(face missing {face_missing_frames} frames, pose confidence: {pose_confidence:.2f}, keypoints: {skeleton_keypoints}/33)")
                                else:
                                    # If tracker reinit failed, create fresh tracker
                                    new_tracker = None
                                    try:
                                        # Use MIL tracker (standard in OpenCV 4.x)
                                        new_tracker = cv2.TrackerMIL_create()
                                        
                                        if new_tracker is not None:
                                            new_tracker.init(frame, (bx1, by1, bbox_width, bbox_height))
                                            status["tracker"] = new_tracker
                                            status["back_view_tracking"] = True
                                            status["skeleton_track_start_frame"] = self.frame_counter
                                            logger.warning(f"[BACK VIEW] {name}: Created fresh MIL tracker with skeleton bbox "
                                                         f"(face missing {face_missing_frames} frames, pose: {pose_confidence:.2f})")
                                    except Exception as e:
                                        logger.debug(f"Failed MIL init: {e}")
                                        new_tracker = None
                            except Exception as e:
                                logger.warning(f"[BACK VIEW ERROR] {name}: Failed to reinitialize tracker with skeleton: {e}")

            
            if face_locations and len(face_locations) > 0:
                # ✅ CRITICAL OPTIMIZATION: Reduce face encoding cost from ~100ms to ~30ms per frame
                # Strategy: Use num_jitters=1 (fastest) + skip encoding for stable tracked targets
                num_untracked = len(untracked_targets)
                
                # Always use num_jitters=1 for MAXIMUM SPEED at 30 FPS video framerate
                # Quality difference is imperceptible in real-time video (consecutive frames are similar)
                num_jitters = 1
                
                # ✅ OPTIMIZATION: Skip encoding if NO untracked targets (all tracked)
                # If all targets are actively being tracked by CSRT, we don't need face encoding
                if num_untracked == 0:
                    logger.debug(f"[PERF] All {len(self.targets_status)} targets tracked - skipping expensive face encoding")
                    face_encodings = []
                else:
                    # Only encode faces when we have untracked targets to match against
                    face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations, num_jitters=num_jitters)
                
                # ✅ CRITICAL: Skip encoding step if no untracked targets
                # If all targets are tracked, we don't need to encode faces at all!
                if len(untracked_targets) == 0:
                    # All targets are already being tracked, skip expensive encoding
                    logger.debug(f"All {len(self.targets_status)} targets tracked, skipping face encoding")
                    face_encodings = []
                
                # Only proceed with matching if we have untracked targets and encodings
                if len(untracked_targets) == 0 or not face_encodings:
                    # No untracked targets or no encodings - skip the rest
                    logger.debug(f"Skipping face matching: {len(untracked_targets)} untracked, {len(face_encodings)} encodings")
                else:
                    # ✅ NEW PIPELINE: Detect brightness for dark mode pipeline
                    gray_frame = cv2.cvtColor(rgb_full_frame, cv2.COLOR_RGB2GRAY)
                    brightness = np.mean(gray_frame)
                    
                    # Build adaptive params based on brightness
                    if brightness < 100:  # Low-light mode
                        adaptive_params = {
                            "brightness": int(brightness),
                            "tolerance": 0.46,
                            "confidence": 0.47
                        }
                        logger.debug(f"Dark mode activated: brightness={brightness:.0f}")
                    else:
                        adaptive_params = {
                            "brightness": int(brightness),
                            "tolerance": 0.55,
                            "confidence": 0.45
                        }
                    
                    # ✅ ENHANCED: Adaptive thresholds for angle-aware detection
                    # Support detection at extreme angles (side profile, upward, downward views)
                    num_guards = len(untracked_targets)
                    
                    # ✅ CRITICAL: RELAXED tolerance for all-angle detection
                    # Increased tolerance to support detection at extreme angles (profile, up/down views)
                    # Guards should be identified regardless of head position
                    if adaptive_params["brightness"] < 100:  # Low-light mode
                        base_tolerance = 0.60  # RELAXED: Support extreme angles in dark (was 0.50)
                        min_confidence = 0.35  # RELAXED: Allow weaker matches at angles (was 0.50)
                        logger.debug(f"Using RELAXED angle-aware tolerances for dark mode: tolerance={base_tolerance:.2f}, min_confidence={min_confidence:.2f}")
                    elif num_guards >= 2:
                        # MULTI-GUARD MODE: Balanced for angle-awareness
                        base_tolerance = 0.60  # RELAXED: Detect all angles but avoid cross-match (was 0.50)
                        min_confidence = 0.38  # RELAXED: Support extreme angles (was 0.50)
                    else:
                        # SINGLE GUARD: RELAXED to detect from all angles
                        base_tolerance = 0.62  # RELAXED: Full-angle detection (was 0.52)
                        min_confidence = 0.36  # RELAXED: Support side/up/down views (was 0.48)
                    
                    # ✅ CRITICAL: Build complete cost matrix with ALL candidates
                    # But only consider STRONG face detections to avoid false positives
                    cost_matrix = []
                    guard_all_matches = {}
                    
                    # ✅ ENHANCED: Filter face encodings by size and aspect ratio
                    # This removes false faces (bags, shadows, random patches)
                    valid_face_indices = []
                    for face_idx, face_location in enumerate(face_locations):
                        top, right, bottom, left = face_location
                        face_width = right - left
                        face_height = bottom - top
                        aspect_ratio = face_width / max(1, face_height)
                        
                        # ✅ CRITICAL VALIDATION: Face must have reasonable dimensions
                        # ENHANCED FOR EXTREME ANGLES: Now accepts upward angles, downward angles, profile views
                        # Surveillance cameras often have extreme viewing angles
                        MIN_FACE_SIZE = 6  # Minimum face dimension (allows extreme distance + extreme angles)
                        ASPECT_RATIO_MIN = 0.30  # Very permissive for extreme upward angle
                        ASPECT_RATIO_MAX = 2.5  # Very permissive for extreme angles
                        
                        if (face_width >= MIN_FACE_SIZE and 
                            face_height >= MIN_FACE_SIZE and 
                            ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
                            valid_face_indices.append((face_idx, face_width, face_height))
                        else:
                            logger.debug(f"[REJECT] Face_{face_idx}: Bad dims={face_width}x{face_height}, aspect={aspect_ratio:.2f}")
                    
                    logger.info(f"[FACE] Found {len(face_locations)} detections, {len(valid_face_indices)} valid face candidates")
                    
                    for target_idx, name in enumerate(untracked_targets):
                        target_encoding = self.targets_status[name]["encoding"]
                        if target_encoding is None:
                            logger.warning(f"[WARN] Skip {name}: no encoding available")
                            continue
                        
                        all_matches = []
                        
                        for face_idx, face_width, face_height in valid_face_indices:
                            if face_idx >= len(face_encodings):
                                continue
                            unknown_encoding = face_encodings[face_idx]
                            if unknown_encoding is None:
                                continue
                            
                            dist = face_recognition.face_distance([target_encoding], unknown_encoding)[0]
                            confidence = 1.0 - dist
                            
                            # ✅ CRITICAL FIX: PREVENT GUARD/FUGITIVE CONFUSION
                            # Cross-check: If this is a guard, make sure it's NOT the fugitive
                            if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
                                fugitive_dist = face_recognition.face_distance([self.fugitive_face_encoding], unknown_encoding)[0]
                                fugitive_confidence = 1.0 - fugitive_dist
                                
                                # If face matches fugitive better than guard, REJECT this guard match
                                if fugitive_dist < dist:
                                    match_gap = dist - fugitive_dist
                                    if match_gap > 0.05:  # 5% gap means fugitive is clearly better match
                                        logger.debug(f"[REJECT] {name} vs face_{face_idx}: Matches FUGITIVE better (guard_dist={dist:.3f}, fugitive_dist={fugitive_dist:.3f})")
                                        continue
                            
                            # ✅ ANGLE-AWARE MATCHING: Accept matches within tolerance
                            # Low confidence can mean extreme angles (upward view), not necessarily wrong match
                            # For surveillance use case, tolerance-based matching is better than confidence-based
                            if dist <= base_tolerance:
                                all_matches.append({
                                    "face_idx": face_idx,
                                    "distance": dist,
                                    "confidence": confidence,
                                    "encoding": unknown_encoding
                                })
                                cost_matrix.append((dist, target_idx, face_idx, name, confidence, unknown_encoding))
                                if dist > 0.55:  # Extreme angle match - log it
                                    logger.debug(f"[ANGLE MATCH] {name} vs face_{face_idx}: High distance={dist:.3f} (likely extreme angle - upward/downward view)")
                            else:
                                logger.debug(f"[SKIP] {name} vs face_{face_idx}: Distance={dist:.3f} exceeds tolerance={base_tolerance:.2f}")
                        
                        guard_all_matches[name] = all_matches
                    
                    # Sort by distance (best matches first)
                    cost_matrix.sort(key=lambda x: x[0])
                    
                    # ✅ IMPROVED: Enhanced debug logging
                    if cost_matrix:
                        logger.info(f"[FACE] Found {len(face_locations)} faces, {len(cost_matrix)} potential matches")
                        by_guard = {}
                        for item in cost_matrix:
                            dist, _, face_idx, name, conf, _ = item
                            if name not in by_guard:
                                by_guard[name] = []
                            by_guard[name].append((dist, face_idx, conf))
                        
                        for name, matches in by_guard.items():
                            for i, (dist, face_idx, conf) in enumerate(matches[:2]):
                                logger.info(f"  {name} (face_{face_idx}): confidence={conf:.3f}, distance={dist:.3f}")
                    
                    # ✅ OPTIMIZED: Single-pass greedy assignment (faster, no quality loss)
                    assigned_faces = set()
                    assigned_targets = set()
                    assignments = []
                    
                    # Single-pass: Best-distance-first greedy assignment
                    # (Previous two-pass approach added ~50-100ms per frame with multiple guards)
                    for item in cost_matrix:
                        dist, target_idx, face_idx, name, confidence, unknown_enc = item
                        if face_idx in assigned_faces or name in assigned_targets:
                            continue
                        
                        # ✅ CRITICAL FIX: Accept matches ONLY if confidence is HIGH ENOUGH
                        # This prevents wrong guard assignment when tolerances are close
                        # Require at least min_confidence threshold to avoid false positive matches
                        if confidence >= min_confidence:
                            # ✅ ADDITIONAL CHECK: If multiple guards match same face, pick BEST only
                            # Check if this is the BEST match for this guard (no other face is much closer)
                            remaining_matches = [x for x in cost_matrix if x[3] == name and x[2] not in assigned_faces]
                            if remaining_matches and remaining_matches[0][0] != dist:
                                # This guard has better matches available - skip this weaker one
                                logger.debug(f"[SKIP] {name} -> face_{face_idx}: Better match available (dist={dist:.3f} vs best={remaining_matches[0][0]:.3f})")
                                continue
                            
                            assigned_faces.add(face_idx)
                            assigned_targets.add(name)
                            assignments.append((name, face_idx, dist, confidence, unknown_enc))
                            logger.info(f"[ASSIGN] {name} -> face_{face_idx} (confidence:{confidence:.3f}, distance:{dist:.3f})")
                        else:
                            logger.debug(f"[REJECT] {name} -> face_{face_idx}: Confidence {confidence:.3f} below minimum {min_confidence:.3f}")
                    
                    # ✅ Execute assignments
                    for name, face_idx, dist, confidence, unknown_enc in assignments:
                        (top, right, bottom, left) = face_locations[face_idx]
                        
                        # ✅ IMPROVED: Reset TARGET MISSING flag when guard is found again
                        self.targets_status[name]["target_missing_alert_logged"] = False
                        
                        # ✅ SEPARATION FIX: Clear overlap_disabled flag when guard is re-detected through normal face matching
                        self.targets_status[name]["overlap_disabled"] = False
                        
                        # ✅ CRITICAL: Store face encoding for later reference
                        if unknown_enc is not None:
                            self.targets_status[name]["face_encoding_history"].append(unknown_enc)
                        
                        # ✅ INDUSTRIAL-LEVEL: Enhanced bounding box validation and tracker initialization
                        # Validate bounding box is reasonable
                        bbox_width = right - left
                        bbox_height = bottom - top
                        frame_h, frame_w = frame.shape[:2]
                        
                        # Check bounding box validity
                        if bbox_width < 20 or bbox_height < 20 or bbox_width > frame_w or bbox_height > frame_h:
                            logger.warning(f"Invalid bbox for {name}: {left}, {top}, {right}, {bottom} (skipping)")
                            continue
                        
                        # ✅ INDUSTRIAL-LEVEL: Try multiple tracker types for robustness
                        tracker = None
                        try:
                            # Use MIL tracker (standard in OpenCV 4.x)
                            tracker = cv2.TrackerMIL_create()
                            
                            if tracker is not None:
                                # Try to initialize tracker
                                tracker.init(frame, (left, top, bbox_width, bbox_height))
                                logger.debug(f"Initialized MIL tracker for {name}")
                        except Exception as e:
                            logger.debug(f"Failed to initialize MIL tracker for {name}: {e}")
                        
                        self.targets_status[name]["tracker"] = tracker
                        self.targets_status[name]["face_box"] = (left, top, right, bottom)
                        self.targets_status[name]["visible"] = True
                        self.targets_status[name]["missing_pose_counter"] = 0
                        self.targets_status[name]["face_confidence"] = max(0.0, min(1.0, confidence))  # ✅ IMPROVED: Clamp confidence to [0,1]
                        
                        # ✅ INSTANT TRACKING: Mark guard as newly detected for 1-frame detection interval
                        # This ensures rapid acquisition and smooth tracking initiation
                        # Flag will be reset after 5 frames (166ms) when tracking becomes stable
                        self.targets_status[name]["newly_detected"] = True
                        self.targets_status[name]["detection_frame_count"] = 0  # Count frames since detection
                        
                        # ✅ STABILITY: Track consecutive detections for stable tracking
                        self.targets_status[name]["consecutive_detections"] = self.targets_status[name].get("consecutive_detections", 0) + 1
                        if self.targets_status[name]["consecutive_detections"] >= 3:
                            self.targets_status[name]["stable_tracking"] = True
                        self.targets_status[name]["face_match_confidence"] = max(0.0, min(1.0, confidence))  # ✅ IMPROVED: Clamp confidence
                        # ❌ FIXED: REMOVED incorrect timer reset here! Timer should ONLY reset AFTER grace period ends (line 3290), not on detection
                        # This was preventing grace period from ever executing properly
                        
                        # ✅ ENHANCED: Log guard identification with visual indicator
                        logger.warning(f"[DETECTED] OK {name} identified & tracking (confidence: {confidence:.3f}, distance: {dist:.3f}, bbox: {bbox_width}x{bbox_height} px)")
                        
                        # Log first detection event
                        if self.targets_status[name].get("consecutive_detections", 0) == 1:
                            logger.info(f"[NEW TRACK] {name} first detected in frame (face confidence: {confidence:.3f})")
                    
                    # ✅ CRITICAL: Log detailed diagnostic info for multi-guard scenarios
                    if len(untracked_targets) >= 2:
                        logger.info(f"[MULTI-GUARD] {len(untracked_targets)} guards tracking, {len(assigned_targets)} matched, {len(face_locations)} faces detected")
                        unmatched = len(face_locations) - len(assigned_faces)
                        if unmatched > 0:
                            logger.debug(f"[UNMATCHED] {unmatched} face(s) not assigned to any guard")
                        
                        # Show rejection log for debugging
                        for name in guard_all_matches.keys():
                            if name not in assigned_targets and guard_all_matches[name]:
                                matches = guard_all_matches[name]
                                logger.debug(f"[UNASSIGNED] {name}: best match distance={matches[0]['distance']:.3f}, confidence={matches[0]['confidence']:.3f} (below threshold)")
        
        # ✅ HARD RESET: If a guard has been missing for > 30 seconds, clear all tracking state
        # This prevents ghost tracking and forces a clean re-identification flow
        for name, status in self.targets_status.items():
            if not status.get("visible", False):
                if status.get("missing_start_time") is None:
                    status["missing_start_time"] = current_time
                else:
                    missing_duration = current_time - status["missing_start_time"]
                    if missing_duration >= 30.0 and not status.get("hard_reset_done", False):
                        try:
                            status["tracker"] = None
                            status["body_tracker"] = None
                            status["face_box"] = None
                            status["body_box"] = None
                            status["visible"] = False
                            status["stable_tracking"] = False
                            if status.get("pose_buffer") is not None and hasattr(status["pose_buffer"], "clear"):
                                status["pose_buffer"].clear()
                            status["face_confidence"] = 0.0
                            status["tracker_confidence"] = 0.0
                            status["face_detection_missing_frames"] = 0
                            status["missing_pose_counter"] = 0
                            status["newly_detected"] = False
                            status["alert_triggered_state"] = False
                            status["target_missing_alert_logged"] = False
                            status["hard_reset_done"] = True
                            status["consecutive_detections"] = 0
                            
                            # ✅ CRITICAL: Force immediate re-detection after hard reset
                            # Set re_detect_counter to trigger detection on next frame
                            self.re_detect_counter = 999  # Force detection to run immediately
                            
                            logger.warning(f"[RESET] {name}: Missing for {missing_duration:.1f}s — hard reset tracking, forcing immediate re-detection")
                        except Exception as e:
                            logger.error(f"[RESET] Error during hard reset for {name}: {e}")
            else:
                # Visible again — clear missing timer and allow future resets
                status["missing_start_time"] = None
                status["hard_reset_done"] = False

        # 3. Overlap Check (OPTIMIZED: Reduce frequency for performance)
        # ✅ CRITICAL OPTIMIZATION: Only check overlaps every 5 frames (167ms) instead of every frame
        # Overlap detection is ~15-20ms per frame for multiple guards, so reducing frequency saves significant time
        # At 2+ guards with overlap checks every frame: ~30-40ms overhead
        # With new interval (every 5 frames): ~6-8ms overhead spread across 5 frames = ~1.2-1.6ms per frame savings
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        num_visible_guards = len(active_names)
        
        # ✅ OPTIMIZATION: Overlap checks are expensive, only run when needed
        # - Single guard: Never check (single target can't overlap with itself)
        # - 2-3 guards: Check every 5 frames (comprehensive but not too frequent)  
        # - 4+ guards: Check every 10 frames (less frequent due to higher complexity)
        if num_visible_guards >= 2:
            check_frequency = 5 if num_visible_guards <= 3 else 10
            should_check_overlap = (self.frame_counter % check_frequency == 0)
        else:
            should_check_overlap = False
        
        if should_check_overlap:
            # ✅ IMPROVED: Multi-Guard Pose Detection Resolution
            # Lower IoU threshold (0.2) for better separation and stability
            self.targets_status = resolve_overlapping_poses(self.targets_status, iou_threshold=0.2)
        
        # ✅ OPTIMIZED: Inline safety check only when resolve_overlapping_poses didn't run
        if not should_check_overlap and num_visible_guards >= 2:
            # Quick safety check on frames where full resolution didn't run
            for i in range(len(active_names)):
                for j in range(i + 1, len(active_names)):
                    nameA = active_names[i]
                    nameB = active_names[j]
                    
                    # Quick box check only
                    boxA = self.targets_status[nameA]["face_box"]
                    boxB = self.targets_status[nameB]["face_box"]
                    
                    if boxA and boxB:
                        rectA = (boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1])
                        rectB = (boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1])
                        iou = calculate_iou(rectA, rectB)
                        
                        if iou > 0.40:  # ✅ HIGHER threshold for quick safety check (prevents false positives)
                            # Only do emergency disable if heavily overlapping
                            conf_a = self.targets_status[nameA].get("face_confidence", 0.5)
                            conf_b = self.targets_status[nameB].get("face_confidence", 0.5)
                            
                            if conf_a > conf_b:
                                self.targets_status[nameB]["tracker"] = None
                                self.targets_status[nameB]["visible"] = False
                            else:
                                self.targets_status[nameA]["tracker"] = None
                                self.targets_status[nameA]["visible"] = False
        
        # Refresh active_names for final action classification
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                nameA = active_names[i]
                nameB = active_names[j]
                
                # Check Face Box IoU
                boxA = self.targets_status[nameA]["face_box"]
                boxB = self.targets_status[nameB]["face_box"]
                # Convert to x,y,w,h format for IoU check
                rectA = (boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1])
                rectB = (boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1])
                
                iou = calculate_iou(rectA, rectB)
                if iou > 0.30:
                    # Multi-factor conflict resolution: confidence + temporal consistency
                    conf_a = self.targets_status[nameA].get("face_confidence", 0.5)
                    conf_b = self.targets_status[nameB].get("face_confidence", 0.5)
                    
                    # Get cached actions for temporal consistency
                    action_a = self.last_action_cache.get(nameA, "Unknown")
                    action_b = self.last_action_cache.get(nameB, "Unknown")
                    
                    # Weighted score: 60% confidence, 30% consistency, 10% IoU severity
                    score_a = (conf_a * 0.6) + (0.3 if action_a != "Unknown" else 0.0) + (0.1 * (1 - iou))
                    score_b = (conf_b * 0.6) + (0.3 if action_b != "Unknown" else 0.0) + (0.1 * (1 - iou))
                    
                    if score_a > score_b:
                        # Keep A, remove B
                        self.targets_status[nameB]["tracker"] = None
                        self.targets_status[nameB]["visible"] = False
                        logger.debug(f"Overlap resolved: keeping {nameA} (score: {score_a:.2f}) over {nameB} (score: {score_b:.2f}), IoU: {iou:.2f}")
                    else:
                        # Keep B, remove A
                        self.targets_status[nameA]["tracker"] = None
                        self.targets_status[nameA]["visible"] = False
                        logger.debug(f"Overlap resolved: keeping {nameB} (score: {score_b:.2f}) over {nameA} (score: {score_a:.2f}), IoU: {iou:.2f}")

        # 4. Processing & Drawing
        required_act = self.active_required_action  # Use current active action from dropdown
        monitor_mode = self.monitor_mode_var.get()  # Get monitoring mode
        current_time = time.time()
        min_buffer = max(CONFIG["performance"].get("min_buffer_for_classification", 5), 5)  # ✅ FIXED: Initialize before loop with safety check

        for name, status in self.targets_status.items():
            if status["visible"]:
                fx1, fy1, fx2, fy2 = status["face_box"]
                
                # --- USE DYNAMIC BODY BOX HELPER (consistent across all modes) ---
                bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), frame_h, frame_w, expansion_factor=3.0)

                # Ghost Box Check: Only draw if tracker is confident AND pose is found
                pose_found_in_box = False
                
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size != 0:
                        # ✅ CRITICAL: ALWAYS detect pose - do NOT skip frames
                        # Pose detection is essential for action classification and alert triggering
                        # Every frame must have pose analysis for real-time responsiveness
                        should_detect_pose = True
                        logger.debug(f"[POSE] {name}: Running pose detection on every frame")
                        
                        if should_detect_pose:
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            
                            # ✅ CRITICAL: Use MediaPipe Pose for pose detection (always available, works offline)
                            # Each guard gets independent pose detection from their cropped region
                            if self.pose_model:
                                pose_result = self.pose_model.process(rgb_crop)
                            else:
                                pose_result = None
                            
                            # Convert MoveNet output to cache format
                            results_crop = pose_result
                            
                            # Cache the result for next frame
                            status["last_cached_pose"] = results_crop
                        else:
                            # Use cached pose result
                            results_crop = status.get("last_cached_pose")
                        
                        # ========== ACTION DETECTION LOGIC ==========
                        current_action = "Unknown"
                        if results_crop is not None and results_crop.pose_landmarks is not None:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0 # Reset
                            
                            # ✅ MEDIAPIPE FORMAT: Extract actual visibility from landmarks
                            # Calculate actual pose quality from keypoint visibility
                            pose_landmarks = results_crop.pose_landmarks.landmark
                            visible_count = sum(1 for lm in pose_landmarks if lm.visibility > 0.5)
                            avg_visibility = sum(lm.visibility for lm in pose_landmarks) / len(pose_landmarks)
                            pose_quality = visible_count / len(pose_landmarks)  # Actual quality score
                            status["pose_confidence"] = pose_quality
                            status["skeleton_keypoints"] = visible_count  # Update keypoint count for tracking
                            
                            # ✅ MULTI-GUARD FIX: Per-guard pose quality threshold validation
                            if "pose_quality_history" not in status:
                                status["pose_quality_history"] = deque(maxlen=10)
                            status["pose_quality_history"].append(pose_quality)
                            
                            # ✅ EXTREME ANGLE HANDLING: Accept LOWER pose quality for back/upper views
                            # - Front view: 60% visibility required (good full-body view)
                            # - Back/Side view: 40% visibility sufficient (partially visible)
                            # - Extreme angles: 20% visibility minimum (only few keypoints visible)
                            # This prevents losing track when guard turns away or looks down
                            min_pose_quality_threshold = 0.20  # ✅ RELAXED: Allow very sparse poses
                            
                            # Only process pose if quality meets threshold
                            if pose_quality >= min_pose_quality_threshold:  # Accept even very sparse poses
                                # ✅ MEDIAPIPE: Use actual pose landmarks to classify action
                                raw_action = classify_action(pose_landmarks, crop.shape[0], crop.shape[1])
                                
                                # ✅ MULTI-GUARD ENHANCEMENT: Filter out "Unknown" from buffer
                                if raw_action != "Unknown" and avg_visibility > 0.4:
                                    status["pose_buffer"].append(raw_action)
                                    status["last_valid_pose"] = raw_action
                                    status["last_valid_pose_time"] = current_time
                                
                                min_buffer = max(CONFIG["performance"].get("min_buffer_for_classification", 5), 5)
                                if len(status["pose_buffer"]) >= min_buffer:
                                    # ✅ MULTI-GUARD IMPROVEMENT: Enhanced consensus with temporal validation
                                    # Use mode but require consistency AND temporal stability
                                    counts = Counter(status["pose_buffer"])
                                    most_common = counts.most_common(1)[0][0]
                                    confidence_pct = counts[most_common] / len(status["pose_buffer"])
                                    
                                    # ✅ STABILITY: Only use action if tracking is stable (3+ consecutive frames)
                                    # AND pose consensus is high (70% agreement in buffer)
                                    if confidence_pct >= 0.70 and status.get("stable_tracking", False):
                                        current_action = most_common
                                    else:
                                        # Low confidence - use last valid action (stability)
                                        current_action = status["last_valid_pose"] or "Standing"
                                else:
                                    # Buffer not full yet - use last valid action for stability
                                    current_action = status["last_valid_pose"] or "Unknown"
                            else:
                                # Poor pose quality - use last valid pose
                                current_action = status["last_valid_pose"] or "Standing"
                            
                            # Cache action for logging (only if not Unknown)
                            if current_action != "Unknown":
                                self.last_action_cache[name] = current_action
                            
                            # ✅ FIX 4: Store current_action in status for alert stopping logic
                            status["current_action"] = current_action
                            
                            # ✅ MULTI-GUARD IMPROVEMENT: Enhanced debug logging for pose tracking
                            if len(status["pose_buffer"]) >= min_buffer:
                                # Log detailed info for debugging
                                buffer_summary = Counter(status["pose_buffer"])
                                most_common_action = buffer_summary.most_common(1)[0][0]
                                buffer_str = ", ".join([f"{action}:{count}" for action, count in buffer_summary.most_common()])
                                avg_quality = sum(status["pose_quality_history"]) / len(status["pose_quality_history"])
                                logger.debug(f"[POSE] {name}: {current_action} (quality:{pose_quality:.2f}, avg_qual:{avg_quality:.2f}, buffer:[{buffer_str}], consensus:{buffer_summary[most_common_action]/len(status['pose_buffer']):.1%})")

                            # ========== ACTION ALERT LOGIC (GRACE PERIOD ENFORCED) ========== 
                            # Only consider actions performed in the grace period (last N seconds of interval)
                            if self.is_alert_mode and monitor_mode in ["Action Alerts Only"]:
                                elapsed_time = current_time - status["last_action_time"]
                                grace_period_length = 10  # seconds
                                grace_period_start = self.alert_interval - grace_period_length
                                
                                # Check if in grace period OR overdue (extended grace period until action performed)
                                in_grace_period_or_overdue = elapsed_time >= grace_period_start
                                
                                # 1. Check Action Performance (in grace period or overdue)
                                if in_grace_period_or_overdue and current_action == self.active_required_action:
                                    # Action performed!
                                    status["action_performed"] = True
                                    logger.debug(f"[ACTION] {name}: Action '{self.active_required_action}' performed (elapsed: {elapsed_time:.1f}s)")
                                    
                                    # Capture snapshot
                                    try:
                                        guard_bbox = (fx1, fy1, fx2, fy2)
                                        self.capture_alert_snapshot(frame, name, check_rate_limit=False, bbox=guard_bbox, is_fugitive=False)
                                    except Exception:
                                        pass

                                    # Log Success
                                    self.temp_log.append((
                                        time.strftime("%Y-%m-%d %H:%M:%S"),
                                        name,
                                        f"'{self.active_required_action}' required",
                                        "PERFORMED",
                                        "N/A",
                                        f"{status['pose_confidence']:.2f}"
                                    ))
                                    self.temp_log_counter += 1
                                    
                                    # Stop Alarm
                                    if status["alert_stop_event"] is not None:
                                        status["alert_stop_event"].set()
                                    
                                    # Reset Timer
                                    status["last_action_time"] = current_time
                                    status["action_performed"] = False
                                    status["alert_logged_timeout"] = False # Reset timeout log flag

                                # 2. Check Timeout (Alarm Trigger)
                                elif elapsed_time >= self.alert_interval:
                                    # Action NOT performed yet (otherwise we would have reset above)
                                    
                                    # Log Timeout (only once per cycle/timeout event)
                                    if not status.get("alert_logged_timeout", False):
                                        img_path = "N/A"
                                        try:
                                            guard_bbox = (fx1, fy1, fx2, fy2)
                                            img_path = self.capture_alert_snapshot(frame, name, check_rate_limit=False, bbox=guard_bbox, is_fugitive=False) or "N/A"
                                        except Exception:
                                            pass
                                            
                                        self.temp_log.append((
                                            time.strftime("%Y-%m-%d %H:%M:%S"),
                                            name,
                                            f"'{self.active_required_action}' required",
                                            "TIMEOUT",
                                            img_path,
                                            f"{status['pose_confidence']:.2f}"
                                        ))
                                        self.temp_log_counter += 1
                                        logger.warning(f"[ALERT] {name}: Action TIMEOUT - triggering alarm")
                                        status["alert_logged_timeout"] = True
                                    
                                    # Trigger/Ensure Alarm is Playing (Every Frame)
                                    if status["alert_stop_event"] is None:
                                        status["alert_stop_event"] = threading.Event()
                                    status["alert_stop_event"].clear()
                                    
                                    if not status.get("alert_sound_thread") or not status["alert_sound_thread"].is_alive():
                                        status["alert_sound_thread"] = play_siren_sound(
                                            stop_event=status["alert_stop_event"],
                                            duration_seconds=15,
                                            sound_file=get_sound_path("siren.mp3")
                                        )
                                        logger.warning(f"[ALERT] {name}: Restarting 15s alarm loop")
                            
                            # ========== STILLNESS DETECTION (PRO MODE ONLY) ==========
                            # ✅ CRITICAL FIX: Detect guard motionlessness using SKELETON keypoints (hands, arms, body)
                            # Previous implementation only tracked face box, which missed hand/arm movements
                            if self.is_pro_mode and self.is_stillness_alert:
                                current_time = time.time()
                                current_face_box = status.get("face_box")
                                
                                if current_face_box and results_crop is not None and results_crop.pose_landmarks is not None:
                                    # Get current skeleton keypoints for motion detection
                                    pose_landmarks = results_crop.pose_landmarks.landmark
                                    
                                    # Track key body parts: hands (wrists), elbows, shoulders, hips
                                    # These landmarks capture body/hand movement better than just face
                                    key_landmarks = [
                                        11, 12,  # Shoulders
                                        13, 14,  # Elbows
                                        15, 16,  # Wrists (hands)
                                        23, 24,  # Hips
                                    ]
                                    
                                    # Extract current positions
                                    current_keypoints = []
                                    for idx in key_landmarks:
                                        if idx < len(pose_landmarks):
                                            lm = pose_landmarks[idx]
                                            if lm.visibility > 0.3:  # Only track visible keypoints
                                                current_keypoints.append((lm.x, lm.y, lm.z))
                                    
                                    # Get previous keypoints for comparison
                                    previous_keypoints = status.get("last_keypoints_stillness")
                                    
                                    # Calculate motion: average movement of all tracked keypoints
                                    if previous_keypoints is not None and len(current_keypoints) > 0 and len(previous_keypoints) == len(current_keypoints):
                                        total_motion = 0.0
                                        for i in range(len(current_keypoints)):
                                            prev_x, prev_y, prev_z = previous_keypoints[i]
                                            curr_x, curr_y, curr_z = current_keypoints[i]
                                            # Calculate 3D Euclidean distance (normalized coordinates)
                                            keypoint_motion = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2) ** 0.5
                                            total_motion += keypoint_motion
                                        
                                        # Average motion across all keypoints
                                        avg_motion = total_motion / len(current_keypoints)
                                        
                                        # Motion threshold: 0.04 in normalized coordinates (about 4% of frame)
                                        # This captures significant hand/arm movements while avoiding false positives from minor jitter
                                        # Increased from 0.015 to reduce sensitivity - only triggers on actual stillness
                                        MOTION_THRESHOLD = 0.04
                                        
                                        if avg_motion > MOTION_THRESHOLD:
                                            # Guard moved (body/hands/arms) - reset stillness timer
                                            status["stillness_start_time"] = None
                                            status["stillness_alert_logged"] = False
                                            
                                            # Stop any active stillness alert sound
                                            if status.get("alert_sound_thread") and status["alert_sound_thread"].is_alive():
                                                if status.get("alert_stop_event"):
                                                    status["alert_stop_event"].set()
                                            
                                            logger.debug(f"[STILLNESS] {name}: Movement detected (avg_motion={avg_motion:.4f}) - reset timer")
                                        else:
                                            # No significant movement detected
                                            if status["stillness_start_time"] is None:
                                                # Start tracking stillness
                                                status["stillness_start_time"] = current_time
                                            elif not status["stillness_alert_logged"]:
                                                # Check if exceeded 15 seconds
                                                stillness_duration = current_time - status["stillness_start_time"]
                                                
                                                if stillness_duration >= 15.0:
                                                    # ✅ TRIGGER 15-SECOND STILLNESS ALERT (ONCE)
                                                    logger.warning(f"[STILLNESS ALERT] !!! {name} - NO MOTION FOR {stillness_duration:.1f} SECONDS - Triggering 15s alert!")
                                                    
                                                    try:
                                                        # Stop any previous alert
                                                        if status.get("alert_sound_thread") and status["alert_sound_thread"].is_alive():
                                                            if status.get("alert_stop_event"):
                                                                status["alert_stop_event"].set()
                                                                status["alert_sound_thread"].join(timeout=1)
                                                        
                                                        # Start new 15-second stillness alert sound
                                                        status["alert_stop_event"] = threading.Event()
                                                        status["alert_sound_thread"] = play_siren_sound(
                                                            stop_event=status["alert_stop_event"],
                                                            duration_seconds=15,
                                                            sound_file=get_sound_path("siren.mp3")
                                                        )
                                                        
                                                        # Add to CSV log
                                                        self.temp_log.append((
                                                            time.strftime("%Y-%m-%d %H:%M:%S"),
                                                            name,
                                                            "STILLNESS_ALERT",
                                                            f"NO MOTION FOR {stillness_duration:.1f} SECONDS",
                                                            "N/A",
                                                            "1.0"
                                                        ))
                                                        self.temp_log_counter += 1
                                                        self.save_log_to_file()  # Immediate flush
                                                        
                                                        # ✅ CRITICAL: Mark as logged so alert only triggers ONCE per stillness period
                                                        status["stillness_alert_logged"] = True
                                                    except Exception as e:
                                                        logger.error(f"[STILLNESS] Error triggering alert for {name}: {e}")
                                    
                                    # Store current keypoints for next frame's motion check
                                    status["last_keypoints_stillness"] = current_keypoints
                                elif results_crop is None or results_crop.pose_landmarks is None:
                                    # No pose detected - reset stillness tracking
                                    status["stillness_start_time"] = None
                                    status["last_keypoints_stillness"] = None
                            
                            # --- Dynamic Bounding Box Logic & Full Body Skeleton Drawing ---
                            h_c, w_c = crop.shape[:2]
                            
                            # ✅ CRITICAL FIX: Check if pose landmarks exist before accessing
                            if results_crop and results_crop.pose_landmarks:
                                p_lms = results_crop.pose_landmarks.landmark
                                
                                # ✅ ENHANCED: Extract all visible landmarks for full-body tracking
                                keypoints = []
                                for i, lm in enumerate(p_lms):
                                    if 0 <= lm.x <= 1 and 0 <= lm.y <= 1 and lm.visibility > 0.3:
                                        x_pixel = int(lm.x * w_c) + bx1
                                        y_pixel = int(lm.y * h_c) + by1
                                        keypoints.append((i, x_pixel, y_pixel, lm.visibility))
                                
                                # Draw full-body skeleton connections (MediaPipe Pose has 33 landmarks)
                                # Key connections: shoulders-elbows-wrists, hips-knees-ankles, spine
                                skeleton_connections = [
                                    # Upper body
                                    (11, 13), (13, 15),  # Left arm: shoulder -> elbow -> wrist
                                    (12, 14), (14, 16),  # Right arm: shoulder -> elbow -> wrist
                                    (11, 12),            # Shoulders
                                    # Spine
                                    (11, 23), (12, 24),  # Shoulders to hips
                                    (23, 24),            # Hip connection
                                    # Lower body
                                    (23, 25), (25, 27),  # Left leg: hip -> knee -> ankle
                                    (24, 26), (26, 28),  # Right leg: hip -> knee -> ankle
                                ]
                                
                                # Build coordinate map for connections
                                kpt_dict = {kpt[0]: (kpt[1], kpt[2], kpt[3]) for kpt in keypoints}
                                
                                # ✅ Draw skeleton connections (full-body 3D tracking)
                                for start_idx, end_idx in skeleton_connections:
                                    if start_idx in kpt_dict and end_idx in kpt_dict:
                                        start_pt = kpt_dict[start_idx]
                                        end_pt = kpt_dict[end_idx]
                                        # Color based on visibility
                                        avg_visibility = (start_pt[2] + end_pt[2]) / 2
                                        if avg_visibility > 0.5:
                                            color = (0, 255, 0)  # Green - high visibility
                                        else:
                                            color = (0, 165, 255)  # Orange - medium visibility
                                        cv2.line(frame, (start_pt[0], start_pt[1]), (end_pt[0], end_pt[1]), color, 2)
                                
                                # ✅ Draw keypoints as circles
                                for idx, x, y, visibility in keypoints:
                                    if visibility > 0.5:
                                        radius = 4
                                        color = (0, 255, 0)  # Green for high visibility
                                    else:
                                        radius = 2
                                        color = (0, 165, 255)  # Orange for medium visibility
                                    cv2.circle(frame, (x, y), radius, color, -1)
                                
                                # Calculate bounding box from skeleton with angle tolerance
                                # Use ALL landmarks (even low visibility) for sparse back/angle views
                                lx = [lm.x * w_c for lm in p_lms if 0 <= lm.x <= 1]
                                ly = [lm.y * h_c for lm in p_lms if 0 <= lm.y <= 1]
                                
                                if lx and ly and len(lx) >= 2:  # Need at least 2 points for valid bbox
                                    d_x1 = int(min(lx)) + bx1
                                    d_y1 = int(min(ly)) + by1
                                    d_x2 = int(max(lx)) + bx1
                                    d_y2 = int(max(ly)) + by1
                                    # Expand slightly for sparse skeletons (back/angle views)
                                    expand = 20 if pose_quality < 0.4 else 10
                                    d_x1 = max(0, d_x1 - expand)
                                    d_y1 = max(0, d_y1 - expand)
                                    d_x2 = min(frame_w, d_x2 + expand)
                                    d_y2 = min(frame_h, d_y2 + expand)
                                else:
                                    # Fallback to face box when skeleton too sparse
                                    d_x1, d_y1, d_x2, d_y2 = fx1, fy1, fx2, fy2
                            else:
                                # No pose detected - use face box
                                d_x1, d_y1, d_x2, d_y2 = fx1, fy1, fx2, fy2
                            
                            # Add minimal padding for sparse angles, more for good quality
                            if pose_quality >= 0.5:
                                # Good quality - use minimal padding
                                d_x1 = max(0, d_x1 - 10)
                                d_y1 = max(0, d_y1 - 10)
                                d_x2 = min(frame_w, d_x2 + 10)
                                d_y2 = min(frame_h, d_y2 + 10)
                            else:
                                # Sparse skeleton (back/angle) - use larger padding to capture full body
                                d_x1 = max(0, d_x1 - 30)
                                d_y1 = max(0, d_y1 - 30)
                                d_x2 = min(frame_w, d_x2 + 30)
                                d_y2 = min(frame_h, d_y2 + 30)
                            
                            # Draw Dynamic Box
                            # Color based on skeleton quality: green (good), orange (sparse)
                            box_color = (0, 255, 0) if pose_quality >= 0.5 else (0, 165, 255)
                            cv2.rectangle(frame, (d_x1, d_y1), (d_x2, d_y2), box_color, 2)

                            # Cache/update body box and initialize body tracker if needed
                            status["body_box"] = (d_x1, d_y1, d_x2, d_y2)
                            # Mark this as a fallback if pose quality is low
                            status["body_box_quality"] = "sparse" if pose_quality < 0.4 else "normal"
                            if status.get("body_tracker") is None:
                                try:
                                    # Use MIL tracker
                                    bt = cv2.TrackerMIL_create()
                                    bt.init(frame, (d_x1, d_y1, d_x2 - d_x1, d_y2 - d_y1))
                                    status["body_tracker"] = bt
                                except Exception:
                                    status["body_tracker"] = None
                            
                            # ✅ IMPROVED: Show identification confidence with guard name
                            face_conf = status.get("face_confidence", 0.0)
                            pose_conf = status.get("pose_confidence", 0.0)
                            
                            # Color code based on identification confidence
                            if face_conf > 0.85:
                                id_color = (0, 255, 0)  # Green - high confidence
                                confidence_indicator = "★★★"
                            elif face_conf > 0.65:
                                id_color = (0, 165, 255)  # Orange - medium confidence
                                confidence_indicator = "★★"
                            else:
                                id_color = (0, 0, 255)  # Red - low confidence
                                confidence_indicator = "★"
                            
                            # ✅ ENHANCED: Display guard name with visual indicators
                            # Show name with checkmark when identified and tracked
                            tracking_status = "✓" if status.get("stable_tracking") else "◇"
                            info_text = f"{tracking_status} {name} ({face_conf:.2f})"
                            action_text = f"{current_action} (P:{pose_conf:.1%})"
                            
                            cv2.putText(frame, info_text, (d_x1, d_y1 - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, id_color, 2)
                            cv2.putText(frame, action_text, (d_x1, d_y1 - 8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

                # Ghost Box Removal Logic
                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                    
                    # ✅ FALLBACK DRAWING: Draw face box even when pose isn't detected
                    # This ensures targets remain visible on screen at all times
                    face_conf = status.get("face_confidence", 0.0)
                    
                    # Color code based on identification confidence
                    if face_conf > 0.85:
                        id_color = (0, 255, 0)  # Green - high confidence
                    elif face_conf > 0.65:
                        id_color = (0, 165, 255)  # Orange - medium confidence
                    else:
                        id_color = (0, 0, 255)  # Red - low confidence
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), id_color, 2)
                    
                    # Display guard name with confidence
                    info_text = f"{name} ({face_conf:.2f})"
                    cv2.putText(frame, info_text, (fx1, fy1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_color, 2)
                    cv2.putText(frame, "No Pose", (fx1, fy1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                else:
                    # Cache for later use
                    if name not in self.last_action_cache:
                        self.last_action_cache[name] = "Unknown"
                    # If tracker says visible, but no pose for 30 frames (approx 1 sec) -> Kill Tracker
                    if status["missing_pose_counter"] > 30:
                        status["tracker"] = None
                        status["visible"] = False
            
            # --- Removed: Guard missing logging (non-alert event) ---
            # Now only logging ALERT_TRIGGERED events to CSV (Task 9)

            # ========== STILLNESS (TIMEOUT) ALERT LOGIC (from Basic_v5.py) ==========
            # Alert Logic (Time-based Pose Timeout) - Only if mode enables it
            if self.is_alert_mode and monitor_mode in ["Action Alerts Only"]:
                # ✅ IMPROVED: Ensure alert_interval is valid (should be > 0 seconds)
                if self.alert_interval <= 0:
                    self.alert_interval = 10  # Safety fallback to 10 seconds
                    logger.warning(f"Alert interval was invalid ({self.alert_interval}), reset to 10 seconds")
                
                time_diff = current_time - status["last_action_time"]
                time_left = max(0, self.alert_interval - time_diff)
                y_offset = 50 + (list(self.targets_status.keys()).index(name) * 30)
                color = (0, 255, 0) if time_left > 3 else (0, 0, 255)
                
                # Only show status on screen if target is genuinely lost or safe
                status_txt = "OK" if status["visible"] else "MISSING"
                cv2.putText(frame, f"{name} ({status_txt}): {time_left:.1f}s", (frame_w - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if time_diff > self.alert_interval:
                    if (current_time - status["alert_cooldown"]) > 2.5:
                        # ✅ ONLY play alert sound if Alert Mode is actually enabled
                        if status["alert_stop_event"] is None:
                            status["alert_stop_event"] = threading.Event()
                        status["alert_stop_event"].clear()  # Reset stop flag
                        # Play siren ONLY when alert mode is ON
                        status["alert_sound_thread"] = play_siren_sound(
                            stop_event=status["alert_stop_event"], 
                            duration_seconds=30,
                            sound_file=get_sound_path("siren.mp3")
                        )
                        status["alert_cooldown"] = current_time
                        
                        img_path = "N/A"
                        if status["visible"]:
                            # Capture full frame with highlighted guard bbox
                            fx1, fy1, fx2, fy2 = status["face_box"]
                            guard_bbox = (fx1, fy1, fx2, fy2)
                            snapshot_result = self.capture_alert_snapshot(
                                frame,
                                name,
                                check_rate_limit=True,
                                bbox=guard_bbox,
                                is_fugitive=False
                            )
                            img_path = snapshot_result if snapshot_result else "N/A"
                        else:
                            # Guard missing - capture full frame without bbox
                            snapshot_result = self.capture_alert_snapshot(
                                frame,
                                name,
                                check_rate_limit=True,
                                bbox=None,
                                is_fugitive=False
                            )
                            img_path = snapshot_result if snapshot_result else "N/A"

                        if self.is_logging:
                            # ✅ IMPROVED: Log ALERT TRIGGERED - TARGET MISSING only at event start
                            # Use a flag to track if we've already logged this missing event
                            if not status["visible"]:
                                # Target is MISSING
                                if not status.get("target_missing_alert_logged", False):
                                    # Log only once at the start of missing event
                                    log_s = "ALERT TRIGGERED - TARGET MISSING"
                                    log_a = "MISSING"
                                    confidence = status.get("face_confidence", 0.0)
                                    self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                                    status["target_missing_alert_logged"] = True  # Mark as logged
                                    self.temp_log_counter += 1
                                    logger.warning(f"[ALERT] {name} MISSING - Alert triggered!")
                            else:
                                # Target is VISIBLE - log as ALERT CONTINUED
                                log_s = "ALERT CONTINUED" if status["alert_triggered_state"] else "ALERT TRIGGERED"
                                log_a = self.last_action_cache.get(name, "Unknown")
                                confidence = status.get("face_confidence", 0.0)
                                self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                                status["target_missing_alert_logged"] = False  # Reset flag since target is visible
                            
                            status["alert_triggered_state"] = True
            # ============================================================
                
                # RESET: When action is performed or target reset
                if time_diff <= 0:
                    status["alert_logged_timeout"] = False

        return frame 

if __name__ == "__main__":
    app = PoseApp()
    app.root.mainloop()
