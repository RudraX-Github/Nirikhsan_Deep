import os
import json
import logging
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
import face_recognition
import time
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    try:
        config_path = os.path.join(SCRIPT_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Return defaults if file doesn't exist
            return get_default_config()
    except Exception as e:
        print(f"Config load error: {e}. Using defaults.")
        return get_default_config()

def get_default_config():
    return {
        "detection": {"min_detection_confidence": 0.5, "min_tracking_confidence": 0.5, 
                     "face_recognition_tolerance": 0.52, "re_detect_interval": 5},
        "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
        "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, "enable_frame_skipping": True, "min_buffer_for_classification": 5},
        "logging": {"log_directory": os.path.join(SCRIPT_DIR, "logs"), "max_log_size_mb": 10, "auto_flush_interval": 50},
        "storage": {"alert_snapshots_dir": os.path.join(SCRIPT_DIR, "alert_snapshots"), "snapshot_retention_days": 30,
                   "guard_profiles_dir": os.path.join(SCRIPT_DIR, "guard_profiles"), "capture_snapshots_dir": os.path.join(SCRIPT_DIR, "capture_snapshots"),
                   "audio_files_dir": os.path.join(SCRIPT_DIR, "audio_files")},
        "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8}
    }

CONFIG = load_config()

# Ensure directories exist
if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

# Setup Logger
logger = logging.getLogger("निराक्षण")
logger.setLevel(logging.WARNING)

# Avoid adding handlers multiple times if module is reloaded
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

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

class SafeLogger:
    """
    Wrapper for logger that sanitizes Unicode characters and replaces them with ASCII equivalents.
    Prevents UnicodeEncodeError on Windows console (cp1252 encoding).
    """
    
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
        try:
            text.encode('cp1252')
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

safe_logger = SafeLogger()

# ============================================================================
# STORAGE & SOUND UTILITIES
# ============================================================================

import cv2
import time
import threading
import gc
import csv
from datetime import datetime, timedelta

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

def get_storage_paths():
    """
    Get all organized storage directory paths.
    All paths are relative to the script directory.
    Structure:
    - guard_profiles/: Face images for recognition
    - capture_snapshots/: Timestamped captures
    - logs/: CSV events and session logs
    """
    paths = {
        "guard_profiles": CONFIG.get("storage", {}).get("guard_profiles_dir", os.path.join(SCRIPT_DIR, "guard_profiles")),
        "capture_snapshots": CONFIG.get("storage", {}).get("capture_snapshots_dir", os.path.join(SCRIPT_DIR, "capture_snapshots")),
        "logs": CONFIG.get("logging", {}).get("log_directory", os.path.join(SCRIPT_DIR, "logs")),
        "alert_snapshots": CONFIG.get("storage", {}).get("alert_snapshots_dir", os.path.join(SCRIPT_DIR, "alert_snapshots"))
    }
    
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    return paths

def save_guard_face(face_image, guard_name, angle="front"):
    """Save guard face image to guard_profiles directory with multi-angle support."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    
    if angle and angle != "front":
        guard_dir = os.path.join(paths["guard_profiles"], f"target_{safe_name}")
        os.makedirs(guard_dir, exist_ok=True)
        profile_path = os.path.join(guard_dir, f"{angle}.jpg")
    else:
        profile_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
        guard_dir = os.path.join(paths["guard_profiles"], f"target_{safe_name}")
        os.makedirs(guard_dir, exist_ok=True)
        front_path = os.path.join(guard_dir, "front.jpg")
        cv2.imwrite(front_path, face_image)
    
    cv2.imwrite(profile_path, face_image)
    logger.info(f"Saved guard face ({angle}): {profile_path}")
    return profile_path

def load_guard_angle_images(guard_name):
    """Load all angle reference images for a guard."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    guard_dir = os.path.join(paths["guard_profiles"], f"target_{safe_name}")
    
    angle_images = {}
    
    if os.path.exists(guard_dir):
        for angle in ["front", "left", "right", "back", "top"]:
            angle_path = os.path.join(guard_dir, f"{angle}.jpg")
            if os.path.exists(angle_path):
                angle_images[angle] = angle_path
                logger.debug(f"Found {angle} angle image for {guard_name}")
    
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

def get_sound_path(filename: str) -> str:
    """Return absolute path for a sound file in the audio_files directory."""
    audio_dir = CONFIG.get("storage", {}).get("audio_files_dir", os.path.join(SCRIPT_DIR, "audio_files"))
    return os.path.join(audio_dir, filename)

def play_siren_sound(stop_event=None, duration_seconds=30, sound_file="siren.mp3"):
    """Play alert sound looping for up to duration_seconds or until stop_event is set"""
    def _sound_worker():
        if os.path.isabs(sound_file) and os.path.exists(sound_file):
            mp3_path = sound_file
        else:
            audio_dir = CONFIG.get("storage", {}).get("audio_files_dir", os.path.join(SCRIPT_DIR, "audio_files"))
            mp3_path = os.path.join(audio_dir, os.path.basename(sound_file))
        
        if not os.path.exists(mp3_path):
            logger.warning(f"Sound file not found: {mp3_path}")
            fallback_path = os.path.join(SCRIPT_DIR, "audio_files", os.path.basename(sound_file))
            if os.path.exists(fallback_path):
                mp3_path = fallback_path
            else:
                logger.error(f"Sound file not found in audio_files: {os.path.basename(sound_file)}")
                return
        
        start_time = time.time()
        
        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.set_volume(1.0)
                
                pygame.mixer.music.play(-1)
                logger.info(f"Alert sound started via pygame (max {duration_seconds}s)")
                
                while True:
                    elapsed = time.time() - start_time
                    
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    time.sleep(0.1)
                
                pygame.mixer.music.stop()
                return
            except Exception as e:
                logger.warning(f"Pygame playback failed: {e}")
        
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                logger.info(f"Alert sound started via pydub (max {duration_seconds}s)")
                
                while True:
                    elapsed = time.time() - start_time
                    
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    play(audio)
                    # Note: pydub play is blocking, so loop logic is different. 
                    # For non-blocking loop with pydub, we'd need more complex logic.
                    # Assuming pygame is preferred.
                    break
            except Exception as e:
                logger.warning(f"Pydub playback failed: {e}")

    thread = threading.Thread(target=_sound_worker, daemon=True)
    thread.start()
    return thread

def optimize_memory():
    """Optimize memory usage by aggressive garbage collection at strategic points."""
    try:
        gc.collect()
        unreachable = gc.collect()
        if unreachable > 100:  # Log only if significant cleanup
            logger.debug(f"[MEMORY] Collected {unreachable} unreachable objects")
    except Exception as e:
        logger.debug(f"[MEMORY] GC optimization error: {e}")

def cleanup_old_snapshots():
    try:
        retention_days = CONFIG["storage"]["snapshot_retention_days"]
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        snapshot_dir = CONFIG["storage"]["alert_snapshots_dir"]
        
        if os.path.exists(snapshot_dir):
            for filename in os.listdir(snapshot_dir):
                filepath = os.path.join(snapshot_dir, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
    except Exception as e:
        logger.error(f"Snapshot cleanup error: {e}")

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
    
    if os.path.exists(guard_dir):
        for angle in ["front", "left", "right", "back", "top"]:
            angle_path = os.path.join(guard_dir, f"{angle}.jpg")
            if os.path.exists(angle_path):
                angle_images[angle] = angle_path
                logger.debug(f"Found {angle} angle image for {guard_name}")
    
    if not angle_images:
        main_face_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
        if os.path.exists(main_face_path):
            angle_images["front"] = main_face_path
            logger.debug(f"Using single face image for {guard_name} (legacy mode)")
    
    return angle_images

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


threading.Thread(target=cleanup_old_snapshots, daemon=True).start()

