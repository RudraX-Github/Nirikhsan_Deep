import os as _os_init
_os_init.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_os_init.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
    # FastMTCNN for face detection
    from facenet_pytorch import MTCNN
    FASTMTCNN_AVAILABLE = True
except ImportError:
    FASTMTCNN_AVAILABLE = False

try:
    # FaceNet for face embeddings
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False

try:
    # Deep SORT for tracking
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    try:
        config_path = os.path.join(SCRIPT_DIR, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Config load error: {e}. Using defaults.")
        return {
            "detection": {"min_detection_confidence": 0.5, "min_tracking_confidence": 0.5, 
                         "face_recognition_tolerance": 0.52, "re_detect_interval": 1},
            "alert": {"default_interval_seconds": 10, "alert_cooldown_seconds": 2.5},
            "performance": {"gui_refresh_ms": 30, "pose_buffer_size": 12, "frame_skip_interval": 2, "enable_frame_skipping": True, "min_buffer_for_classification": 5},
            "logging": {"log_directory": os.path.join(SCRIPT_DIR, "logs"), "max_log_size_mb": 10, "auto_flush_interval": 50},
            "storage": {"alert_snapshots_dir": os.path.join(SCRIPT_DIR, "alert_snapshots"), "snapshot_retention_days": 30,
                       "guard_profiles_dir": os.path.join(SCRIPT_DIR, "guard_profiles"), "capture_snapshots_dir": os.path.join(SCRIPT_DIR, "capture_snapshots"),
                       "audio_files_dir": os.path.join(SCRIPT_DIR, "audio_files")},
            "monitoring": {"mode": "pose", "session_restart_prompt_hours": 8}
        }

CONFIG = load_config()

if not os.path.exists(CONFIG["logging"]["log_directory"]):
    os.makedirs(CONFIG["logging"]["log_directory"])

logger = logging.getLogger("निराक्षण")
logger.setLevel(logging.WARNING)

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

def open_camera_with_timeout(camera_index, api=None, timeout_seconds=3):
    """
    Open a camera with timeout to prevent hanging on USB cameras.
    Returns None if timeout occurs.
    """
    cap = None
    result_holder = {'cap': None, 'error': None}
    
    def open_camera_thread():
        try:
            if api is not None:
                result_holder['cap'] = cv2.VideoCapture(camera_index, api)
            else:
                result_holder['cap'] = cv2.VideoCapture(camera_index)
        except Exception as e:
            result_holder['error'] = e
    
    thread = threading.Thread(target=open_camera_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        logger.warning(f"Camera {camera_index} opening timed out after {timeout_seconds}s - may be unavailable")
        return None
    
    if result_holder['error']:
        logger.error(f"Camera {camera_index} error: {result_holder['error']}")
        return None
    
    return result_holder['cap']

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
# ADVANCED PIPELINE: FastMTCNN + FaceNet + MoveNet + DeepSORT
# ============================================================================

class AdvancedDetectionPipeline:
    """
    Advanced detection pipeline using FastMTCNN for faces and MoveNet for poses.
    Replaces previous MediaPipe face detection with faster MTCNN.
    """
    
    def __init__(self):
        self.mtcnn = None
        self.facenet_model = None
        self.device = self._get_device()
        self._initialize()
    
    def _get_device(self):
        """Auto-detect available device (CUDA or CPU)"""
        try:
            if FASTMTCNN_AVAILABLE:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            pass
        return 'cpu'
    
    def _initialize(self):
        """Initialize FastMTCNN and FaceNet models"""
        try:
            if FASTMTCNN_AVAILABLE:
                self.mtcnn = MTCNN(device=self.device, keep_all=False, select_largest=False)
                self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
                if self.device == 'cuda':
                    self.facenet_model = self.facenet_model.to(self.device)
                logger.info(f"[PIPELINE] FastMTCNN and FaceNet initialized on {self.device}")
            else:
                logger.warning("[PIPELINE] FastMTCNN/FaceNet not available, falling back to legacy detection")
        except Exception as e:
            logger.error(f"[PIPELINE] Advanced pipeline initialization error: {e}")
            self.mtcnn = None
            self.facenet_model = None
    
    def detect_faces(self, frame, confidence_threshold=0.95):
        """
        Detect faces using FastMTCNN
        Returns: List of (x1, y1, x2, y2, confidence) tuples
        """
        try:
            if self.mtcnn is None:
                return []
            
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces with MTCNN
            detections = self.mtcnn.detect(rgb_frame, landmarks=False)
            
            if detections is None or len(detections) == 0:
                return []
            
            face_list = []
            for det in detections:
                if det is None:
                    continue
                box, conf = det
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = box
                    # Ensure coordinates are within bounds
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                    face_list.append((x1, y1, x2, y2, conf))
            
            return face_list
        except Exception as e:
            logger.debug(f"[DETECT] FastMTCNN error: {e}")
            return []
    
    def extract_face_embedding(self, frame, face_box):
        """
        Extract FaceNet embedding for a face region
        Returns: 512-dimensional embedding vector or None
        """
        try:
            if self.facenet_model is None or face_box is None:
                return None
            
            x1, y1, x2, y2 = map(int, face_box[:4])
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Resize to 160x160 (FaceNet input size)
            face_resized = cv2.resize(face_crop, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            try:
                import torch
                face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                # Only move to CUDA if available
                if self.device == 'cuda' and torch.cuda.is_available():
                    face_tensor = face_tensor.to(self.device)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = self.facenet_model(face_tensor)
                
                return embedding.cpu().numpy().flatten()
            except ImportError:
                logger.warning("[EMBED] Torch not available for tensor conversion")
                return None
        except Exception as e:
            logger.debug(f"[EMBED] FaceNet extraction error: {e}")
            return None
    
    def match_face_embedding(self, probe_embedding, reference_embedding, threshold=0.6):
        """
        Compare two FaceNet embeddings using L2 distance
        Returns: True if match, False otherwise
        """
        try:
            if probe_embedding is None or reference_embedding is None:
                return False
            
            # L2 distance
            distance = np.linalg.norm(probe_embedding - reference_embedding)
            match = distance < threshold
            
            return match, distance
        except Exception as e:
            logger.debug(f"[MATCH] Embedding comparison error: {e}")
            return False, 1.0


class SmartGuardTracker:
    """
    Smart tracking system using DeepSORT for stable guard tracking
    with re-identification capabilities.
    """
    
    def __init__(self, num_classes=1):
        self.tracker = None
        self.num_classes = num_classes
        self._initialize()
        self.tracked_guards = {}  # guard_id -> {embedding, last_seen, pose_data}
        self.frame_count = 0
    
    def _initialize(self):
        """Initialize DeepSORT tracker"""
        try:
            if DEEPSORT_AVAILABLE:
                self.tracker = DeepSort(
                    max_age=50,
                    n_init=3,
                    nn_budget=100,
                    embedder='mobilenet',
                    embedder_gpu=False
                )
                logger.info("[TRACKER] DeepSORT initialized successfully")
            else:
                logger.warning("[TRACKER] DeepSORT not available")
        except Exception as e:
            logger.error(f"[TRACKER] DeepSORT initialization error: {e}")
            self.tracker = None
    
    def update(self, detections, embeddings=None):
        """
        Update tracker with new detections and embeddings
        Returns: List of tracks with IDs
        """
        try:
            if self.tracker is None or len(detections) == 0:
                return []
            
            # Format detections for DeepSORT
            dets = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf = det[:5]
                w = x2 - x1
                h = y2 - y1
                
                # Format: (x1, y1, w, h, conf)
                dets.append([x1, y1, w, h, conf])
            
            dets = np.array(dets) if dets else np.empty((0, 5))
            
            # Update tracker
            tracks = self.tracker.update_tracks(dets, frame=None)
            
            self.frame_count += 1
            return tracks
        except Exception as e:
            logger.debug(f"[TRACKER] Update error: {e}")
            return []
    
    def add_guard_profile(self, guard_id, embedding, pose_data=None):
        """Store guard embedding and pose profile for re-identification"""
        self.tracked_guards[guard_id] = {
            'embedding': embedding,
            'last_seen_frame': self.frame_count,
            'pose_data': pose_data or {},
            'confidence': 1.0
        }
    
    def get_guard_profile(self, guard_id):
        """Retrieve stored guard profile"""
        return self.tracked_guards.get(guard_id)


class FaceAreaCropper:
    """
    Interactive GUI-based tool for selecting and cropping face areas from guard profile images.
    Uses CustomTkinter for cross-platform GUI support (works even without OpenCV display support).
    Creates face area embeddings for better recognition.
    """
    
    def __init__(self, image_path, detection_pipeline, guard_name="Guard"):
        self.image_path = image_path
        self.guard_name = guard_name
        self.detection_pipeline = detection_pipeline
        self.original_image = cv2.imread(image_path)
        self.cropped_face = None
        self.embedding = None
        self.selected = False
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        
        # Image for display in CTk (convert BGR to RGB)
        if self.original_image is not None:
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            # Scale to fit window
            max_width, max_height = 700, 600
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(pil_image)
            self.pil_image = pil_image
        else:
            self.display_image = None
            self.pil_image = None
    
    def select_face_area(self):
        """Launch CustomTkinter GUI for face area selection"""
        try:
            if self.original_image is None:
                logger.error("[CROP] Could not load image")
                return False
            
            # Create dialog window
            dialog = ctk.CTkToplevel()
            dialog.title(f"Face Area Selection - {self.guard_name}")
            dialog.geometry("900x700")
            dialog.resizable(True, True)
            
            # Title
            title_label = ctk.CTkLabel(
                dialog,
                text=f"Select face area for {self.guard_name}\nDrag rectangle around face, then click Confirm",
                text_color="#ecf0f1",
                font=("Arial", 12, "bold")
            )
            title_label.pack(pady=10)
            
            # Image frame with canvas
            frame = ctk.CTkFrame(dialog)
            frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create canvas for image
            canvas = tk.Canvas(
                frame,
                bg="#2b2b2b",
                highlightthickness=0,
                cursor="crosshair"
            )
            canvas.pack(fill="both", expand=True)
            
            # Display image on canvas
            if self.display_image:
                canvas_image = canvas.create_image(0, 0, image=self.display_image, anchor="nw")
            
            # Draw rectangle helper
            rect = None
            start_x = start_y = None
            
            def on_mouse_press(event):
                nonlocal start_x, start_y
                start_x, start_y = event.x, event.y
            
            def on_mouse_drag(event):
                nonlocal rect
                if start_x is None or start_y is None:
                    return
                
                if rect:
                    canvas.delete(rect)
                rect = canvas.create_rectangle(
                    start_x, start_y, event.x, event.y,
                    outline="#00ff00", width=2
                )
            
            def on_mouse_release(event):
                nonlocal start_x, start_y
                if start_x is not None and start_y is not None:
                    # Calculate scaled coordinates
                    scale_x = self.original_image.shape[1] / self.pil_image.width
                    scale_y = self.original_image.shape[0] / self.pil_image.height
                    
                    self.x1 = max(0, int(min(start_x, event.x) * scale_x))
                    self.y1 = max(0, int(min(start_y, event.y) * scale_y))
                    self.x2 = min(self.original_image.shape[1], int(max(start_x, event.x) * scale_x))
                    self.y2 = min(self.original_image.shape[0], int(max(start_y, event.y) * scale_y))
                    
                    if self.x1 < self.x2 and self.y1 < self.y2:
                        self.cropped_face = self.original_image[self.y1:self.y2, self.x1:self.x2]
                        logger.info(f"[CROP] Face area selected: ({self.x1}, {self.y1}) to ({self.x2}, {self.y2})")
                    
                    start_x = start_y = None
            
            # Bind mouse events
            canvas.bind("<Button-1>", on_mouse_press)
            canvas.bind("<B1-Motion>", on_mouse_drag)
            canvas.bind("<ButtonRelease-1>", on_mouse_release)
            
            # Button frame
            button_frame = ctk.CTkFrame(dialog)
            button_frame.pack(pady=10, fill="x", padx=10)
            
            confirm_btn = ctk.CTkButton(
                button_frame,
                text="Confirm Selection",
                fg_color="#27ae60",
                hover_color="#229954",
                command=lambda: self._on_confirm(dialog)
            )
            confirm_btn.pack(side="left", padx=5)
            
            cancel_btn = ctk.CTkButton(
                button_frame,
                text="Cancel",
                fg_color="#e74c3c",
                hover_color="#c0392b",
                command=lambda: self._on_cancel(dialog)
            )
            cancel_btn.pack(side="left", padx=5)
            
            # Instructions
            info_label = ctk.CTkLabel(
                dialog,
                text="1. Click and drag to draw rectangle around face\n2. Click Confirm when done, or Cancel to skip",
                text_color="#95a5a6",
                font=("Arial", 10)
            )
            info_label.pack(pady=5)
            
            # Wait for dialog to close
            dialog.wait_window()
            
            return self.selected
        except Exception as e:
            logger.error(f"[CROP] Selection error: {e}")
            return False
    
    def _on_confirm(self, dialog):
        """Handle confirm button"""
        if self.cropped_face is not None:
            self.selected = True
            logger.info("[CROP] Face area confirmed by user")
            dialog.destroy()
        else:
            messagebox.showwarning("No Selection", "Please draw a rectangle around the face first")
    
    def _on_cancel(self, dialog):
        """Handle cancel button"""
        logger.info("[CROP] Selection cancelled by user")
        dialog.destroy()
    
    def extract_embedding(self):
        """Extract FaceNet embedding from selected face area"""
        try:
            if self.cropped_face is None:
                logger.error("[CROP] No face area selected")
                return None
            
            # Use detection pipeline to extract embedding
            face_box = (0, 0, self.cropped_face.shape[1], self.cropped_face.shape[0])
            self.embedding = self.detection_pipeline.extract_face_embedding(self.cropped_face, face_box)
            
            if self.embedding is not None:
                logger.info(f"[CROP] Face embedding extracted successfully for {self.guard_name}")
            
            return self.embedding
        except Exception as e:
            logger.error(f"[CROP] Embedding extraction error: {e}")
            return None


class ThreadedIPCamera:
    """
    Threaded frame grabber for IP cameras (RTSP streams).
    Runs frame capture in a separate thread to prevent GUI lag.
    Always keeps the latest frame available for instant access.
    """
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.frame = None
        self.grabbed = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.cap = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_frame_time = time.time()
        
    def start(self):
        """Start the threaded frame grabber."""
        if self.running:
            return self
        
        # Open RTSP stream with optimized settings
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
            return None
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        time.sleep(0.5)
        return self
    
    def _update(self):
        """Continuously grab frames in background thread."""
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self._reconnect()
                    continue
                
                for _ in range(2):
                    self.cap.grab()
                
                grabbed, frame = self.cap.read()
                
                if grabbed and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
                        self.last_frame_time = time.time()
                    self.reconnect_attempts = 0
                else:
                    if time.time() - self.last_frame_time > 5:
                        logger.warning("IP Camera: Frame timeout, attempting reconnect...")
                        self._reconnect()
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"IP Camera thread error: {e}")
                time.sleep(0.1)
    
    def _reconnect(self):
        """Attempt to reconnect with Exponential Backoff Strategy."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("IP Camera: Max reconnection attempts reached. Pausing for 60s.")
            time.sleep(60) # Long pause before trying again to let network heal
            self.reconnect_attempts = 0
            
        # Calculate backoff: 2^attempts + random jitter to prevent thundering herd
        import random
        backoff_time = min(30, (2 ** self.reconnect_attempts)) + (random.randint(0, 1000) / 1000)
        
        logger.warning(f"IP Camera: Reconnecting in {backoff_time:.2f}s... (attempt {self.reconnect_attempts + 1})")
        time.sleep(backoff_time)
        
        self.reconnect_attempts += 1
        
        try:
            if self.cap:
                self.cap.release()
            
            # Re-initialize
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Optimization: Lower buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            
            if self.cap.isOpened():
                logger.info("IP Camera: Reconnection Successful!")
                self.reconnect_attempts = 0 # Reset on success
        except Exception as e:
            logger.error(f"IP Camera reconnection error: {e}")
    
    def read(self):
        """Get the latest frame (non-blocking)."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
    
    def isOpened(self):
        """Check if stream is active."""
        return self.running and self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Stop the threaded grabber and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def stop(self):
        """Alias for release() - stops the threaded grabber."""
        self.release()
    
    def set(self, prop, value):
        """Set camera property (compatibility with cv2.VideoCapture)."""
        if self.cap:
            return self.cap.set(prop, value)
        return False
    
    def get(self, prop):
        """Get camera property (compatibility with cv2.VideoCapture)."""
        if self.cap:
            return self.cap.get(prop)
        return 0


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
    """Save guard face image to guard_profiles directory with multi-angle support.
    
    Args:
        face_image: The face image to save
        guard_name: Guard name
        angle: Angle identifier (front, left, right, back, or None for legacy single image)
    """
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
                logger.debug(f"[PROFILE] Found {angle} angle image for {guard_name}: {angle_path}")
            else:
                logger.debug(f"[PROFILE] Missing {angle} angle for {guard_name}: {angle_path}")
    else:
        logger.debug(f"[PROFILE] Guard directory not found: {guard_dir}")
    
    if not angle_images:
        main_face_path = os.path.join(paths["guard_profiles"], f"target_{safe_name}_face.jpg")
        if os.path.exists(main_face_path):
            angle_images["front"] = main_face_path
            logger.debug(f"[PROFILE] Using single face image for {guard_name} (legacy mode): {main_face_path}")
        else:
            logger.warning(f"[PROFILE] No profile images found for {guard_name} at {main_face_path}")
    
    return angle_images

def save_capture_snapshot(face_image, guard_name):
    """Save timestamped capture snapshot to capture_snapshots directory."""
    paths = get_storage_paths()
    safe_name = guard_name.strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(paths["capture_snapshots"], f"{safe_name}_capture_{timestamp}.jpg")
    cv2.imwrite(snapshot_path, face_image)
    return snapshot_path

_storage_paths = get_storage_paths()

_audio_dir = CONFIG.get("storage", {}).get("audio_files_dir", os.path.join(SCRIPT_DIR, "audio_files"))
if not os.path.exists(_audio_dir):
    os.makedirs(_audio_dir)

csv_file = os.path.join(CONFIG["logging"]["log_directory"], "events.csv")
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Action", "Status", "Image_Path", "Confidence"])

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

def optimize_memory():
    """Optimize memory usage by aggressive garbage collection at strategic points."""
    try:
        gc.collect()
        unreachable = gc.collect()
        if unreachable > 100:  # Log only if significant cleanup
            logger.debug(f"[MEMORY] Collected {unreachable} unreachable objects")
    except Exception as e:
        logger.debug(f"[MEMORY] GC optimization error: {e}")

gc.set_threshold(1000, 15, 15)

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

mp_holistic = mp.solutions.holistic if MEDIAPIPE_AVAILABLE else None

def get_sound_path(filename: str) -> str:
    """Return absolute path for a sound file in the audio_files directory."""
    audio_dir = CONFIG.get("storage", {}).get("audio_files_dir", os.path.join(SCRIPT_DIR, "audio_files"))
    return os.path.join(audio_dir, filename)

def play_siren_sound(stop_event=None, duration_seconds=30, sound_file="siren.mp3"):
    """Play alert sound looping for up to duration_seconds or until stop_event is set
    
    Args:
        stop_event: threading.Event to signal stop playback
        duration_seconds: Maximum duration to play (default 30 seconds)
        sound_file: Path/name of audio file (full path or just filename for audio_files directory)
    """
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
                
                logger.info("Alert sound via pydub completed")
                return
            except Exception as e:
                logger.warning(f"Pydub playback failed: {e}")
        
        try:
            if platform.system() == "Windows":
                import winsound
                logger.info(f"Alert sound started via winsound (max {duration_seconds}s)")
                
                while True:
                    elapsed = time.time() - start_time
                    
                    if stop_event and stop_event.is_set():
                        logger.info(f"Alert sound stopped - action performed (elapsed: {elapsed:.1f}s)")
                        break
                    
                    if elapsed >= duration_seconds:
                        logger.info(f"Alert sound stopped - duration expired (elapsed: {elapsed:.1f}s)")
                        break
                    
                    winsound.Beep(2500, 150)
                    time.sleep(0.05)
                    winsound.Beep(1800, 150)
                    time.sleep(0.05)
            else:
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

def classify_action(landmarks, h, w):
    """
    Classify pose action with robust detection and confidence scoring.
    Supports: Hands Up, Hands Crossed, One Hand Raised (Left/Right), T-Pose, Sit, Standing
    Includes visibility and quality checks for stable detection.
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

        shoulder_to_hip_dist = abs(l_shoulder.y - l_hip.y)
        if shoulder_to_hip_dist < 0.01:
            return "Standing"
        
        shoulder_width = abs(r_shoulder.x - l_shoulder.x)
        body_scale = (shoulder_to_hip_dist + shoulder_width) / 2.0
        
        HANDS_UP_THRESHOLD = shoulder_to_hip_dist * 0.4
        HANDS_CROSSED_TOLERANCE = shoulder_to_hip_dist * 0.3
        ARM_EXTENSION = shoulder_to_hip_dist * 0.6
        WRIST_ARM_ALIGNMENT = shoulder_to_hip_dist * 0.25

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
        
        eyes_visible = l_eye.visibility > 0.40 and r_eye.visibility > 0.40
        
        if eyes_visible:
            visibility_adjustment = 1.0
            visibility_threshold_base = 0.45
        else:
            visibility_adjustment = 0.6
            visibility_threshold_base = 0.30
        
        l_wrist_visible = l_wrist.visibility > (0.45 * visibility_adjustment)
        r_wrist_visible = r_wrist.visibility > (0.45 * visibility_adjustment)
        l_elbow_visible = l_elbow.visibility > (0.45 * visibility_adjustment)
        r_elbow_visible = r_elbow.visibility > (0.45 * visibility_adjustment)
        nose_visible = nose.visibility > (0.40 * visibility_adjustment)
        l_shoulder_visible = l_shoulder.visibility > (0.40 * visibility_adjustment)
        r_shoulder_visible = r_shoulder.visibility > (0.40 * visibility_adjustment)
        l_knee_visible = l_knee.visibility > (0.30 * visibility_adjustment)
        r_knee_visible = r_knee.visibility > (0.30 * visibility_adjustment)
        l_hip_visible = l_hip.visibility > (0.35 * visibility_adjustment)
        r_hip_visible = r_hip.visibility > (0.35 * visibility_adjustment)
        
        visible_joints = sum([
            l_wrist_visible, r_wrist_visible, l_elbow_visible, r_elbow_visible,
            l_shoulder_visible, r_shoulder_visible, l_knee_visible, r_knee_visible,
            l_hip_visible, r_hip_visible, nose_visible
        ])
        
        if eyes_visible:
            min_joints_required = 7
        else:
            min_joints_required = 5
        
        if visible_joints < min_joints_required:  
            return "Standing"
        
        if (l_wrist_visible and r_wrist_visible and nose_visible):
            wrist_above_nose_l = (nose_y - lw_y) > (HANDS_UP_THRESHOLD * 0.8)
            wrist_above_nose_r = (nose_y - rw_y) > (HANDS_UP_THRESHOLD * 0.8)
            
            wrist_above_shoulder_l = (ls_y - lw_y) > (HANDS_UP_THRESHOLD * 0.5)
            wrist_above_shoulder_r = (rs_y - rw_y) > (HANDS_UP_THRESHOLD * 0.5)
            
            if ((wrist_above_nose_l and wrist_above_nose_r) or 
                (wrist_above_shoulder_l and wrist_above_shoulder_r)):
                return "Hands Up"
        
        if (l_wrist_visible and r_wrist_visible and l_shoulder_visible and r_shoulder_visible):
            chest_y = (ls_y + rs_y) / 2
            body_center_x = (ls_x + rs_x) / 2
            
            wrist_chest_dist_l = abs(lw_y - chest_y)
            wrist_chest_dist_r = abs(rw_y - chest_y)
            
            crossed_tolerance = HANDS_CROSSED_TOLERANCE * 1.2
            
            if (wrist_chest_dist_l < crossed_tolerance and 
                wrist_chest_dist_r < crossed_tolerance):
                left_hand_crossed = lw_x > body_center_x
                right_hand_crossed = rw_x < body_center_x
                
                if left_hand_crossed and right_hand_crossed:
                    return "Hands Crossed"
        
        if (l_wrist_visible and r_wrist_visible and l_elbow_visible and r_elbow_visible and
            l_shoulder_visible and r_shoulder_visible):
            
            lw_at_shoulder = abs(lw_y - ls_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            rw_at_shoulder = abs(rw_y - rs_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            le_at_shoulder = abs(l_elbow.y - ls_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            re_at_shoulder = abs(r_elbow.y - rs_y) < (WRIST_ARM_ALIGNMENT * 1.3)
            
            if lw_at_shoulder and rw_at_shoulder and le_at_shoulder and re_at_shoulder:
                shoulder_width = abs(rs_x - ls_x)
                if shoulder_width > 0:
                    left_extension = abs(lw_x - ls_x) / shoulder_width
                    right_extension = abs(rw_x - rs_x) / shoulder_width
                    
                    if left_extension > 0.6 and right_extension > 0.6:
                        return "T-Pose"
        
        if l_wrist_visible and r_wrist_visible and l_shoulder_visible and r_shoulder_visible:
            chest_y = (ls_y + rs_y) / 2
            
            left_raised = (nose_y - lw_y) > (HANDS_UP_THRESHOLD * 0.8)
            right_down = (rw_y - chest_y) > (HANDS_CROSSED_TOLERANCE * 0.8)
            
            if left_raised and right_down:
                return "One Hand Raised (Left)"
            
            right_raised = (nose_y - rw_y) > (HANDS_UP_THRESHOLD * 0.8)
            left_down = (lw_y - chest_y) > (HANDS_CROSSED_TOLERANCE * 0.8)
            
            if right_raised and left_down:
                return "One Hand Raised (Right)"
        
        if l_wrist_visible and not r_wrist_visible and nose_visible:
            if (nose_y - lw_y) > (HANDS_UP_THRESHOLD * 0.8):
                return "One Hand Raised (Left)"
        
        if r_wrist_visible and not l_wrist_visible and nose_visible:
            if (nose_y - rw_y) > (HANDS_UP_THRESHOLD * 0.8):
                return "One Hand Raised (Right)"
        
        if l_knee_visible and r_knee_visible and l_hip_visible and r_hip_visible:
            thigh_angle_l = abs(l_knee.y - l_hip.y)
            thigh_angle_r = abs(r_knee.y - r_hip.y)
            avg_thigh_angle = (thigh_angle_l + thigh_angle_r) / 2
            
            ankle_check_valid = False
            if landmarks[L_ANKLE].visibility > 0.30 and landmarks[R_ANKLE].visibility > 0.30:
                ankle_down_l = l_ankle.y > l_knee.y
                ankle_down_r = r_ankle.y > r_knee.y
                ankle_check_valid = ankle_down_l and ankle_down_r
            
            sit_threshold = shoulder_to_hip_dist * 0.15
            
            if avg_thigh_angle < sit_threshold:
                return "Sit"
            elif ankle_check_valid or avg_thigh_angle > sit_threshold:
                return "Standing"
            else:
                return "Standing"
        elif l_knee_visible or r_knee_visible:
            knee_y = l_knee.y if l_knee_visible else r_knee.y
            hip_y = l_hip.y if l_knee_visible else r_hip.y
            
            if abs(knee_y - hip_y) < (shoulder_to_hip_dist * 0.15):
                return "Sit"
            else:
                return "Standing"
        else:
            return "Standing"

        return "Standing" 

    except Exception as e:
        logger.debug(f"Pose classification error: {e}")
        return "Unknown"

def calculate_body_box(face_box, frame_h, frame_w, expansion_factor=3.0):
    """
    Calculate dynamic body bounding box from detected face box.
    
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
    
    if face_w < 20:
        adaptive_expansion = 5.0
    elif face_w > 150:
        adaptive_expansion = 3.0
    else:
        # Linear interpolation between 5.0 and 3.0
        ratio = (face_w - 20) / 130.0  # 0.0 to 1.0
        adaptive_expansion = 5.0 - (ratio * 2.0)
    
    # Expand horizontally based on adaptive face width
    bx1 = max(0, int(face_cx - (face_w * adaptive_expansion / 2)))
    bx2 = min(frame_w, int(face_cx + (face_w * adaptive_expansion / 2)))
    
    # Expand vertically: slightly above face, down to feet
    # Scale vertical expansion with face size too
    by1 = max(0, int(y1 - (face_h * 0.5)))
    
    # ✅ USER REQUEST: "track the guard as far as guard is visible"
    # ✅ SITTING GUARD FIX: "proper skalaton is only generating when guard is standing"
    # "make bb box dynamic so it can cover the body, not half skalton generation"
    # We use 8.0x face height as a safe default for standing bodies (head-to-body ratio ~1:8).
    # For sitting guards, the feedback loop in process_tracking_frame_optimized will
    # adjust this box to the actual skeleton size in subsequent frames.
    estimated_body_height = int(face_h * 8.0)
    by2 = min(frame_h, by1 + estimated_body_height)
    
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
        
        min_critical_keypoints = 8
        
        is_valid = (
            visible_count >= min_keypoints and
            critical_visible >= min_critical_keypoints
        )
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            avg_confidence = 0.0
        
        if len(visible_keypoints) >= 3:
            x_coords = [landmarks[idx].x for idx in visible_keypoints if idx < len(landmarks)]
            y_coords = [landmarks[idx].y for idx in visible_keypoints if idx < len(landmarks)]
            
            if x_coords and y_coords:
                x_spread = max(x_coords) - min(x_coords)
                y_spread = max(y_coords) - min(y_coords)
                spatial_spread = (x_spread + y_spread) / 2
                
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
    Extract skeleton-based features for robust body tracking.
    
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
        
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        
        min_vis = 0.25
        if (l_shoulder.visibility < min_vis or r_shoulder.visibility < min_vis or
            l_hip.visibility < min_vis or r_hip.visibility < min_vis):
            return None
        
        all_x = [p.x for p in landmarks if p.visibility > min_vis]
        all_y = [p.y for p in landmarks if p.visibility > min_vis]
        
        if not all_x or not all_y:
            return None
        
        skeleton_x_min = min(all_x)
        skeleton_x_max = max(all_x)
        skeleton_y_min = min(all_y)
        skeleton_y_max = max(all_y)
        
        expansion = 0.05
        skeleton_x_min = max(0, skeleton_x_min - expansion)
        skeleton_x_max = min(1, skeleton_x_max + expansion)
        skeleton_y_min = max(0, skeleton_y_min - expansion)
        skeleton_y_max = min(1, skeleton_y_max + expansion)
        
        torso_center_x = (l_shoulder.x + r_shoulder.x) / 2
        torso_center_y = (l_shoulder.y + l_hip.y) / 2
        
        body_scale = abs(l_hip.y - l_shoulder.y)
        
        features = {
            'skeleton_box': (skeleton_x_min, skeleton_y_min, skeleton_x_max, skeleton_y_max),
            'torso_center': (torso_center_x, torso_center_y),
            'body_scale': body_scale,
            'body_height': skeleton_y_max - skeleton_y_min,
            'body_width': skeleton_x_max - skeleton_x_min,
            'is_upright': body_scale > 0.2,
            'visibility_avg': sum(p.visibility for p in landmarks if p.visibility > 0) / max(1, len([p for p in landmarks if p.visibility > 0]))
        }
        
        return features
        
    except Exception as e:
        logger.debug(f"Skeleton feature extraction error: {e}")
        return None

def resolve_overlapping_poses(targets_status, iou_threshold=0.3):
    """
    Resolve conflicting pose detections when multiple guards overlap.
    Ensures each guard has independent, consistent pose detection.
    
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
            box_a = targets_status[name_a].get("face_box")
            if not box_a:
                continue
            
            for name_b in target_names[i+1:]:
                box_b = targets_status[name_b].get("face_box")
                if not box_b:
                    continue
                
                iou = calculate_iou(
                    (box_a[0], box_a[1], box_a[2] - box_a[0], box_a[3] - box_a[1]),
                    (box_b[0], box_b[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
                )
                
                if iou < iou_threshold:
                    was_disabled_a = not targets_status[name_a].get("visible")
                    was_disabled_b = not targets_status[name_b].get("visible")
                    
                    if was_disabled_a and targets_status[name_a].get("overlap_disabled"):
                        targets_status[name_a]["visible"] = True
                        targets_status[name_a]["overlap_disabled"] = False
                        targets_status[name_a]["tracker"] = None
                        logger.warning(f"[SEPARATION] Re-enabled {name_a} - guards separated (IoU: {iou:.2f})")
                    
                    if was_disabled_b and targets_status[name_b].get("overlap_disabled"):
                        targets_status[name_b]["visible"] = True
                        targets_status[name_b]["overlap_disabled"] = False
                        targets_status[name_b]["tracker"] = None
                        logger.warning(f"[SEPARATION] Re-enabled {name_b} - guards separated (IoU: {iou:.2f})")
                    
                    continue
                
                if targets_status[name_a].get("visible") and targets_status[name_b].get("visible"):
                    pose_conf_a = targets_status[name_a].get("pose_confidence", 0.0)
                    pose_conf_b = targets_status[name_b].get("pose_confidence", 0.0)
                    face_conf_a = targets_status[name_a].get("face_confidence", 0.0)
                    face_conf_b = targets_status[name_b].get("face_confidence", 0.0)
                    
                    pose_hist_a = targets_status[name_a].get("pose_quality_history", deque())
                    pose_hist_b = targets_status[name_b].get("pose_quality_history", deque())
                    recent_quality_a = sum(pose_hist_a) / len(pose_hist_a) if pose_hist_a else 0.0
                    recent_quality_b = sum(pose_hist_b) / len(pose_hist_b) if pose_hist_b else 0.0
                    
                    score_a = (pose_conf_a * 0.4) + (face_conf_a * 0.3) + (recent_quality_a * 0.3)
                    score_b = (pose_conf_b * 0.4) + (face_conf_b * 0.3) + (recent_quality_b * 0.3)
                    
                    if score_a < score_b:
                        targets_status[name_a]["visible"] = False
                        targets_status[name_a]["overlap_disabled"] = True
                        conflicts_resolved.append((name_a, name_b, score_a, score_b, iou))
                        logger.debug(f"[RESOLVE] Disabled {name_a} (score:{score_a:.3f}) - kept {name_b} (score:{score_b:.3f}), IoU:{iou:.2f}")
                    elif score_b < score_a:
                        targets_status[name_b]["visible"] = False
                        targets_status[name_b]["overlap_disabled"] = True
                        conflicts_resolved.append((name_b, name_a, score_b, score_a, iou))
                        logger.debug(f"[RESOLVE] Disabled {name_b} (score:{score_b:.3f}) - kept {name_a} (score:{score_a:.3f}), IoU:{iou:.2f}")
                    else:
                        if face_conf_a >= face_conf_b:
                            targets_status[name_b]["visible"] = False
                            targets_status[name_b]["overlap_disabled"] = True
                            logger.debug(f"[RESOLVE] Tie-break: Disabled {name_b} - kept {name_a} by face confidence")
                        else:
                            targets_status[name_a]["visible"] = False
                            targets_status[name_a]["overlap_disabled"] = True
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
        
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        
        person_resized = cv2.resize(person_crop, (64, 128))
        
        color_features = []
        for i in range(3):
            hist = cv2.calcHist([person_resized], [i], None, [8], [0, 256])
            color_features.extend(hist.flatten())
        
        gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
        
        features = np.array(color_features + edge_hist.flatten(), dtype=np.float32)
        
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
        f1 = np.array(features1).astype(np.float32)
        f2 = np.array(features2).astype(np.float32)
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        similarity = float(dot_product / (norm1 * norm2 + 1e-6))
        
        return max(0.0, min(1.0, float(similarity)))
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
            person_features_db[person_id] = {
                'features': new_features,
                'count': 1,
                'last_seen': time.time()
            }
            return person_id, 1.0
        
        stored_features = person_features_db[person_id]['features']
        similarity = calculate_feature_similarity(new_features, stored_features)
        
        if similarity >= confidence_threshold:
            alpha = 0.3
            person_features_db[person_id]['features'] = (
                alpha * new_features + 
                (1 - alpha) * stored_features
            )
            person_features_db[person_id]['count'] += 1
            person_features_db[person_id]['last_seen'] = time.time()
            return person_id, similarity
        else:
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

class PoseApp:
    def __init__(self, window_title=" निराक्षण - Niraakshan"):
        self.root = ctk.CTk()
        self.root.title(window_title)
        self.root.geometry("1800x1000")
        
        self.current_language = "Hindi"
        self.translations = self.load_translations()
        self.current_trans = self.translations["Hindi"]
        
        self.cap = None
        self.unprocessed_frame = None 
        self.is_running = False
        self.is_logging = False
        self.camera_index = 0
        self.is_ip_camera = False
        self.threaded_cam = None
        
        self.identity_confirmation_frames = 5
        self.guard_fugitive_margin = 0.12
        
        self.is_camera_running = False
        self.is_pro_mode = False
        self.is_alert_mode = False
        self.is_fugitive_detection = False
        self.is_stillness_alert = False
        self.guard_mode = "ADD"
        self.fugitive_add_mode = "ADD"
        
        self.alert_interval = 10
        self.monitor_mode_var = tk.StringVar(self.root)
        self.monitor_mode_var.set("Action Alerts Only")
        self.frame_w = 640 
        self.frame_h = 480 
        
        self.is_single_person_mode = False
        self.night_mode = False
        self.face_detector = None
        self.pose_model = None
        self.tracker = None
        self.model_pipeline_initialized = False
        
        self._initialize_model_pipeline()
        
        # ===== ADVANCED PIPELINE INITIALIZATION =====
        self.detection_pipeline = AdvancedDetectionPipeline()
        self.smart_tracker = SmartGuardTracker()
        self.guard_embeddings = {}  # Store guard face embeddings: guard_name -> embedding
        self.guard_selected_faces = {}  # Store cropped face regions: guard_name -> cropped_image
        self.current_tracked_guard_id = None
        self.last_reidentification_frame = 0
        self.reidentification_interval = 10  # Re-identify every N frames
        
        self.is_tracking = False
        self.active_required_action = "Hands Up"
        self.last_clock_second = -1

        self.target_map = {}
        self.targets_status = {} 
        self.selected_target_names = []
        self.re_detect_counter = 0    
        self.RE_DETECT_INTERVAL = CONFIG["detection"]["re_detect_interval"]
        self.RESIZE_SCALE = 1.0 
        self.temp_log = []
        self.temp_log_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.last_process_frame = None
        self.last_action_cache = {}
        self.session_start_time = time.time()
        self.onboarding_mode = False
        self.is_in_capture_mode = False
        self.onboarding_step = 0
        self.onboarding_name = None
        self.onboarding_poses = {}
        self.onboarding_detection_results = None
        self.onboarding_face_box = None
        
        self.fugitive_mode = False
        self.fugitive_image = None
        self.fugitive_face_encoding = None
        self.fugitive_name = "Unknown Fugitive"
        self.fugitive_detected_log_done = False
        self.last_fugitive_snapshot_time = 0
        self.fugitive_alert_sound_thread = None
        self.fugitive_alert_stop_event = None
        
        self.fugitive_currently_visible = False
        self.fugitive_alert_start_time = 0
        
        self.photo_storage = {}

        self.frame_timestamp_ms = 0 

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        
        self.sidebar_collapsed = False
        self.sidebar_width = 320
        
        self.main_container = ctk.CTkFrame(self.root, fg_color="#1a1a1a")
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=0)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        self.camera_controls_panel = ctk.CTkFrame(self.main_container, fg_color="#2b2b2b", height=50, corner_radius=0)
        self.camera_controls_panel.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.camera_controls_panel.grid_propagate(False)
        
        controls_frame = ctk.CTkFrame(self.camera_controls_panel, fg_color="transparent")
        controls_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
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
        
        self.root.protocol("WM_DELETE_WINDOW", self.graceful_exit)
        
        self.root.mainloop()
    
    def _initialize_model_pipeline(self):
        """Initialize appropriate model pipeline based on guard count and environment"""
        try:
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
                logger.info("MediaPipe Pose and Face Detection initialized successfully")
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
                return
            
            logger.info("Loading single-person pipeline with MediaPipe...")
            
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Pose loaded for single-person mode")
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
                return
            
            logger.info("Loading multi-person pipeline with MediaPipe...")
            
            if MEDIAPIPE_AVAILABLE:
                self.pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Pose loaded for multi-person mode")
            else:
                logger.warning("MediaPipe not available")
                self.pose_model = None
        except Exception as e:
            logger.error(f"Multi-person pipeline load error: {e}")
    
    def _detect_faces_blazeface(self, rgb_frame):
        """BlazeFace face detection for single-person mode"""
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            return face_locations
        except Exception as e:
            logger.debug(f"BlazeFace detection error: {e}")
            return []
    
    def _detect_faces_blazepose(self, rgb_frame):
        """BlazePose face detection for multi-person mode"""
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            return face_locations
        except Exception as e:
            logger.debug(f"BlazePose detection error: {e}")
            return []

    def detect_faces_fast(self, frame):
        """Fast face detection using MediaPipe"""
        try:
            if not MEDIAPIPE_AVAILABLE or not hasattr(self, 'mp_face_detection'):
                # Fallback to HOG if MediaPipe not available
                locs = face_recognition.face_locations(frame, model="hog")
                logger.debug(f"[DETECT] Using HOG fallback - found {len(locs)} faces")
                return locs
                
            results = self.mp_face_detection.process(frame) # frame is already RGB in usage
            face_locations = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    # Convert MediaPipe relative coords to dlib (top, right, bottom, left)
                    left, top, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    # Ensure coordinates are within frame
                    top = max(0, top)
                    left = max(0, left)
                    bottom = min(h, top + height)
                    right = min(w, left + width)
                    face_locations.append((top, right, bottom, left))
                logger.debug(f"[DETECT] MediaPipe found {len(face_locations)} faces")
            else:
                logger.debug(f"[DETECT] MediaPipe found no faces")
            return face_locations
        except Exception as e:
            logger.error(f"[ERROR] Face detection failed: {e}")
            return []
    
    def _detect_pose_movenet_lightning(self, rgb_frame):
        """MediaPipe Pose for single-person pose estimation"""
        try:
            if self.pose_model is None:
                return None
            
            results = self.pose_model.process(rgb_frame)
            return results
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None
    
    def _detect_pose_blazepose_multipose(self, rgb_frame):
        """MediaPipe Pose for multi-person pose estimation"""
        try:
            if self.pose_model is None:
                return None
            
            results = self.pose_model.process(rgb_frame)
            return results
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None
    
    def _apply_clahe_enhancement(self, frame):
        """Apply CLAHE + Gamma correction for dark environments"""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            inv_gamma = 1.0 / 1.5
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l_channel = cv2.LUT(l_channel, table)
            
            lab = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_frame
        except Exception as e:
            logger.debug(f"CLAHE enhancement error: {e}")
            return frame
    
    def toggle_sidebar(self):
        """Toggle sidebar visibility (not used in new design but kept for compatibility)"""
        pass
    
    def enhance_frame_for_low_light(self, frame):
        """
        Fast frame enhancement with night/day mode awareness
        Day mode: Skip enhancement entirely (fast)
        Night mode: Lightweight enhancement only if brightness is low
        """
        try:
            if not self.night_mode:
                return frame
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 100:
                return frame
            
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            if not hasattr(self, '_clahe_cache'):
                self._clahe_cache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel_clahe = self._clahe_cache.apply(l_channel)
            lab_enhanced = cv2.merge([l_channel_clahe, a, b])
            frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return frame_enhanced
        except Exception as e:
            logger.debug(f"Low-light enhancement failed: {e}, using original frame")
            return frame
    
    def get_adaptive_face_detection_params(self, frame):
        """
        Adaptive parameters based on lighting conditions
        Detects frame brightness and adjusts detection sensitivity accordingly
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50:
            face_tolerance = 0.52
            confidence_threshold = 0.48
            use_cnn_model = True
            logger.debug(f"Very dark conditions (brightness={brightness:.0f}) - Using stricter tolerance to prevent ghosts")
        elif brightness < 100:
            face_tolerance = 0.50
            confidence_threshold = 0.50
            use_cnn_model = True
            logger.debug(f"Dark conditions (brightness={brightness:.0f}) - Using CNN model")
        elif brightness < 150:
            face_tolerance = 0.52
            confidence_threshold = 0.50
            use_cnn_model = False
            logger.debug(f"Medium lighting (brightness={brightness:.0f}) - Using balanced detection")
        else:
            face_tolerance = 0.55
            confidence_threshold = 0.52
            use_cnn_model = False
            logger.debug(f"Good lighting (brightness={brightness:.0f}) - Using standard tolerance")
        
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
            guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", os.path.join(SCRIPT_DIR, "guard_profiles"))
            
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
            guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", os.path.join(SCRIPT_DIR, "guard_profiles"))
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
        guard_profiles_dir = CONFIG.get("storage", {}).get("guard_profiles_dir", os.path.join(SCRIPT_DIR, "guard_profiles"))
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
        
        # Load embeddings for all targets using new pipeline
        self.load_guard_embeddings()
        
        self.update_selected_preview()
    
    def load_guard_embeddings(self):
        """Load and cache FaceNet embeddings for all guards - auto-detection only"""
        try:
            if self.detection_pipeline.facenet_model is None:
                logger.warning("[EMBED] FaceNet not available, skipping embedding extraction")
                return
            
            for guard_name, image_path in self.target_map.items():
                if guard_name in self.guard_embeddings:
                    continue  # Already loaded
                
                try:
                    frame = cv2.imread(image_path)
                    if frame is None:
                        logger.warning(f"[EMBED] Could not load image for {guard_name}")
                        continue
                    
                    # Try auto-detection
                    faces = self.detection_pipeline.detect_faces(frame)
                    
                    if faces:
                        # Auto-detect successful - use the largest/first detected face
                        face_box = faces[0][:4]
                        embedding = self.detection_pipeline.extract_face_embedding(frame, face_box)
                        
                        if embedding is not None:
                            self.guard_embeddings[guard_name] = embedding
                            logger.info(f"[EMBED] Auto-loaded embedding for guard: {guard_name}")
                        else:
                            logger.warning(f"[EMBED] Could not extract embedding for {guard_name} - use Manual Select button")
                    else:
                        logger.warning(f"[EMBED] No faces detected in profile image for {guard_name} - use Manual Select button")
                except Exception as e:
                    logger.error(f"[EMBED] Error loading embedding for {guard_name}: {e}")
        except Exception as e:
            logger.error(f"[EMBED] Error in load_guard_embeddings: {e}")
    
    def select_guard_face_area_interactive(self, guard_name, image_path):
        """
        Interactive face area selection for a guard using FaceAreaCropper GUI.
        User manually selects face area from profile image using mouse in CTk dialog.
        Extracted embedding is cached for future use.
        """
        try:
            logger.info(f"[CROP-INTERACTIVE] Launching face selector for {guard_name}")
            
            # Create face cropper tool
            cropper = FaceAreaCropper(image_path, self.detection_pipeline, guard_name)
            
            # Launch interactive selection UI
            if cropper.select_face_area():
                # User confirmed selection - extract embedding
                embedding = cropper.extract_embedding()
                
                if embedding is not None:
                    self.guard_embeddings[guard_name] = embedding
                    logger.info(f"[CROP-INTERACTIVE] Successfully extracted embedding for {guard_name}")
                    messagebox.showinfo("Success", f"Face area saved for {guard_name}!")
                else:
                    logger.warning(f"[CROP-INTERACTIVE] Failed to extract embedding for {guard_name} after selection")
                    messagebox.showerror("Error", f"Could not extract embedding for {guard_name}")
            else:
                logger.warning(f"[CROP-INTERACTIVE] User cancelled face selection for {guard_name}")
        except Exception as e:
            logger.error(f"[CROP-INTERACTIVE] Error in interactive face selection for {guard_name}: {e}")
    
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
                    
                    # Guard name label 
                    name_label = ctk.CTkLabel(row_frame, text=guard_name, 
                                             text_color="#27ae60", font=("Segoe UI", 9), width=80)
                    name_label.pack(side="left", padx=5, pady=3)
                    
                    # Embedding status indicator
                    has_embedding = guard_name in self.guard_embeddings
                    status_color = "#27ae60" if has_embedding else "#e67e22"
                    status_text = "OK" if has_embedding else "MANUAL"
                    status_label = ctk.CTkLabel(row_frame, text=status_text, 
                                               text_color=status_color, font=("Segoe UI", 8, "bold"), width=50)
                    status_label.pack(side="left", padx=3, pady=3)
                    
                    # Manual select button for guards without embedding
                    if not has_embedding:
                        select_btn = ctk.CTkButton(
                            row_frame, 
                            text="Select Face", 
                            width=80,
                            height=24,
                            font=("Segoe UI", 8),
                            fg_color="#e67e22",
                            hover_color="#d35400",
                            command=lambda name=guard_name, fpath=filename: self.select_guard_face_area_interactive(name, fpath)
                        )
                        select_btn.pack(side="left", padx=3, pady=3)
                    
                    # Store reference and photo
                    self.guard_preview_grid[guard_name] = (img_label, imgtk)
                    self.photo_storage[f"guard_preview_{guard_name}"] = imgtk
                    
            except Exception as e:
                logger.error(f"Error loading guard preview for {guard_name}: {e}")
                # Show error row
                error_frame = ctk.CTkFrame(self.guard_preview_scroll_frame, fg_color="#2a2a2a", 
                                          border_width=1, border_color="#e74c3c", corner_radius=4)
                error_frame.pack(fill="x", padx=0, pady=2)
                ctk.CTkLabel(error_frame, text="[X]", text_color="#e74c3c", font=("Arial", 10)).pack(side="left", padx=3, pady=3)
                ctk.CTkLabel(error_frame, text=guard_name, text_color="#e74c3c", 
                            font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=5, pady=3)

    def apply_target_selection(self):
        self.targets_status = {} 
        if not self.selected_target_names:
            # No targets selected, tracking disabled
            logger.info("No targets selected - tracking disabled")
            return
        count = 0
        pose_buffer_size = max(CONFIG["performance"].get("pose_buffer_size", 5), 12)
        
        for name in self.selected_target_names:
            filename = self.target_map.get(name)
            if filename:
                try:
                    if not os.path.exists(filename):
                        logger.error(f"Guard profile file not found: {filename}")
                        continue
                    
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
                            "encoding": multi_angle_encodings[0],
                            "multi_angle_encodings": multi_angle_encodings,
                            "body_profile_landmarks": None,
                            "tracker": None,
                            "body_tracker": None,
                            "face_box": None, 
                            "body_box": None,
                            "visible": False,
                            "overlap_disabled": False,
                            "last_action_time": time.time(),
                            "action_performed": False,
                            "alert_cooldown": 0,
                            "alert_triggered_state": False,
                            "last_logged_action": None,
                            "pose_buffer": deque(maxlen=pose_buffer_size),
                            "missing_pose_counter": 0,
                            "face_confidence": 0.0,
                            "pose_confidence": 0.0,
                            "face_encoding_history": deque(maxlen=5),
                            "face_match_confidence": 0.0,
                            "last_valid_pose": None,
                            "last_valid_pose_time": time.time(),
                            "pose_quality_history": deque(maxlen=10),
                            "last_snapshot_time": 0,
                            "last_log_time": 0,
                            "alert_sound_thread": None,
                            "alert_stop_event": None,
                            "alert_logged_timeout": False,
                            "target_missing_alert_logged": False,
                            "last_stillness_check_time": time.time(),
                            "stillness_start_time": None,
                            "last_face_box_stillness": None,
                            "consecutive_detections": 0,
                            "stable_tracking": False,
                            "last_pose_vector": None,
                            "stillness_alert_logged": False,
                            "tracker_drift_counter": 0,
                            "needs_drift_verification": False,
                            "ghost_no_pose_frames": 0,
                        }
                        count += 1
                        logger.info(f"[OK] {name} initialized with {len(multi_angle_encodings)} face encoding(s) from {len(angle_images)} angle(s)")
                    else:
                        logger.error(f"[ERROR] {name}: No face encoding found in image(s) - guard profile may be invalid or image doesn't contain a face")
                        logger.error(f"[ERROR]   Checked angles: {list(angle_images.keys())}")
                        logger.error(f"[ERROR]   Guard profile directory: {guard_dir if 'guard_dir' in locals() else 'not found'}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load {name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        if count > 0:
            logger.warning(f"[OK] Tracking initialized for {count} targets (Pose Buffer: {pose_buffer_size} frames)")
            
            messagebox.showinfo("Tracking Updated", f"Now scanning for {count} selected targets")
        else:
            logger.error("[ERROR] No valid guard profiles found - ensure profile images contain faces")
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
                
                if h < 0 or m < 0 or m > 59:
                    messagebox.showerror("Error", "Please enter valid time values (H >= 0, 0 <= M <= 59)")
                    return
                
                total_seconds = h * 3600 + m * 60
                if total_seconds > 0:
                    self.alert_interval = total_seconds
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
        """Set alert interval with Hours, Minutes, Seconds dialog"""
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
            logger.warning(f"Night Mode - Face: FastMTCNN + FaceNet | Pose: MoveNet Lightning + CLAHE | Tracking: DeepSORT")
        else:
            self.btn_night_mode.configure(text="🌙 Night OFF", fg_color="#34495e")
            logger.warning(f"Day Mode - Face: FastMTCNN | Pose: MoveNet Lightning | Tracking: DeepSORT")

    # ===== ADVANCED PIPELINE METHODS =====
    
    def select_guard_face_area(self, guard_name):
        """
        Allow user to select and crop face area from guard profile image.
        Opens interactive UI for face selection.
        """
        try:
            guard_image_path = self.target_map.get(guard_name)
            if not guard_image_path or not os.path.exists(guard_image_path):
                logger.error(f"[SELECT] Guard image not found for {guard_name}")
                messagebox.showerror("Error", f"Guard image not found for {guard_name}")
                return False
            
            # Create face cropper
            cropper = FaceAreaCropper(guard_image_path, self.detection_pipeline)
            
            if cropper.select_face_area():
                embedding = cropper.extract_embedding()
                if embedding is not None:
                    self.guard_selected_faces[guard_name] = cropper.cropped_face
                    self.guard_embeddings[guard_name] = embedding
                    logger.info(f"[SELECT] Face area selected and embedding created for {guard_name}")
                    messagebox.showinfo("Success", f"Face area selected for {guard_name}")
                    return True
                else:
                    logger.error(f"[SELECT] Could not extract embedding for {guard_name}")
                    messagebox.showerror("Error", "Could not extract face embedding")
                    return False
            else:
                logger.warning(f"[SELECT] User cancelled face selection for {guard_name}")
                return False
        except Exception as e:
            logger.error(f"[SELECT] Face area selection error: {e}")
            messagebox.showerror("Error", f"Face selection failed: {e}")
            return False
    
    def recognize_face_advanced(self, frame, face_box):
        """
        Recognize guard using FaceNet embeddings (advanced pipeline)
        Returns: (guard_name, confidence, guard_embedding_distance)
        """
        try:
            if not self.guard_embeddings:
                logger.debug("[RECOGNIZE] No guard embeddings available")
                return None, 0.0, 1.0
            
            # Extract embedding from detected face
            probe_embedding = self.detection_pipeline.extract_face_embedding(frame, face_box)
            if probe_embedding is None:
                return None, 0.0, 1.0
            
            # Compare with all stored guard embeddings
            best_match = None
            best_distance = float('inf')
            best_confidence = 0.0
            
            for guard_name, guard_embedding in self.guard_embeddings.items():
                match, distance = self.detection_pipeline.match_face_embedding(
                    probe_embedding, guard_embedding, threshold=0.6
                )
                
                confidence = 1.0 - min(distance, 1.0)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = guard_name
                    best_confidence = confidence
            
            logger.debug(f"[RECOGNIZE] Best match: {best_match} (dist={best_distance:.3f}, conf={best_confidence:.3f})")
            
            return best_match, best_confidence, best_distance
        except Exception as e:
            logger.error(f"[RECOGNIZE] Face recognition error: {e}")
            return None, 0.0, 1.0
    
    def track_guard_advanced(self, frame, detections, embeddings):
        """
        Track guards using DeepSORT with smart re-identification
        Returns: List of tracked persons with IDs
        """
        try:
            if self.smart_tracker.tracker is None:
                logger.warning("[TRACK] DeepSORT not available")
                return []
            
            # Update tracker with detections and embeddings
            tracks = self.smart_tracker.update(detections, embeddings)
            
            # Store track info for re-identification
            for track in tracks:
                track_id = track.track_id
                if track_id not in self.smart_tracker.tracked_guards:
                    self.smart_tracker.tracked_guards[track_id] = {
                        'embedding': None,
                        'guard_name': None,
                        'last_seen_frame': self.frame_counter,
                        'confidence': 0.0
                    }
            
            return tracks
        except Exception as e:
            logger.error(f"[TRACK] Tracking error: {e}")
            return []
    
    def reidentify_guard_from_track(self, track_id):
        """
        Re-identify guard from tracking ID to handle brief occlusions
        or movement that might cause tracking drift
        """
        try:
            guard_profile = self.smart_tracker.get_guard_profile(track_id)
            if guard_profile and guard_profile.get('guard_name'):
                logger.info(f"[REIDENTIFY] Guard {guard_profile['guard_name']} re-identified (track_id={track_id})")
                return guard_profile['guard_name']
            return None
        except Exception as e:
            logger.error(f"[REIDENTIFY] Re-identification error: {e}")
            return None
    
    def update_guard_tracking_status(self, guard_name, is_visible, confidence=0.0):
        """
        Update tracking status for a guard with confidence score
        Enables smooth tracking even with brief occlusions
        """
        try:
            if guard_name not in self.targets_status:
                self.targets_status[guard_name] = {}
            
            self.targets_status[guard_name].update({
                'visible': is_visible,
                'confidence': confidence,
                'last_updated': time.time(),
                'frame_count': self.frame_counter
            })
            
            logger.debug(f"[TRACK_STATUS] {guard_name}: visible={is_visible}, conf={confidence:.3f}")
        except Exception as e:
            logger.error(f"[TRACK_STATUS] Error updating status: {e}")

    def switch_camera_dialog(self):
        """Allow switching to a different camera while app is running"""
        # Detect available cameras
        available_cameras = detect_available_cameras()
        
        if not available_cameras:
            messagebox.showerror("Camera Error", "No cameras detected!")
            return
        
        if len(available_cameras) == 1:
            messagebox.showinfo("Camera Switch", f"Only one camera detected (Camera {available_cameras[0]})")
            return
        
        # Create camera selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Switch Camera")
        dialog.geometry("400x550")
        dialog.configure(bg="#2c3e50")
        
        tk.Label(dialog, text="Select Camera:", font=('Helvetica', 14, 'bold'), 
                bg="#2c3e50", fg="white").pack(pady=15)
        
        # Variable to track camera type (0 = USB camera, 1 = IP camera)
        camera_type_var = tk.IntVar(value=0)
        
        # === USB CAMERAS SECTION ===
        usb_frame = tk.Frame(dialog, bg="#34495e")
        usb_frame.pack(pady=5, padx=20, fill="x")
        
        tk.Radiobutton(usb_frame, text="USB Camera", variable=camera_type_var,
                      value=0, font=('Helvetica', 12, 'bold'), bg="#34495e", fg="white",
                      selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=5)
        
        selected_camera = tk.IntVar(value=self.camera_index if hasattr(self, 'camera_index') else available_cameras[0])
        
        # Frame for radio buttons
        radio_frame = tk.Frame(usb_frame, bg="#34495e")
        radio_frame.pack(pady=5, padx=20, fill="both")
        
        for idx in available_cameras:
            current_label = " (Current)" if hasattr(self, 'camera_index') and idx == self.camera_index else ""
            tk.Radiobutton(radio_frame, text=f"Camera {idx}{current_label}", variable=selected_camera,
                          value=idx, font=('Helvetica', 11), bg="#34495e", fg="white",
                          selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=2)
        
        # === IP CAMERA SECTION ===
        ip_frame = tk.Frame(dialog, bg="#34495e")
        ip_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Radiobutton(ip_frame, text="IP Camera (RTSP)", variable=camera_type_var,
                      value=1, font=('Helvetica', 12, 'bold'), bg="#34495e", fg="white",
                      selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=5)
        
        # IP camera input fields
        input_frame = tk.Frame(ip_frame, bg="#34495e")
        input_frame.pack(pady=5, padx=30, fill="x")
        
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
        
        def on_switch():
            # Check if IP camera is selected
            if camera_type_var.get() == 1:
                # IP Camera selected - build RTSP URL
                ip_addr = ip_var.get().strip()
                port = port_var.get().strip()
                username = user_var.get().strip()
                password = pass_var.get().strip()
                
                # Validate required fields
                if not ip_addr:
                    messagebox.showerror("Error", "IP Address is required!")
                    return
                if not username:
                    messagebox.showerror("Error", "Username is required!")
                    return
                if not password:
                    messagebox.showerror("Error", "Password is required!")
                    return
                
                # Build RTSP URL (common format for most IP cameras)
                rtsp_url = f"rtsp://{username}:{password}@{ip_addr}:{port}/stream1"
                
                # Store running state
                was_running = self.is_camera_running
                
                # Stop current camera (USB or IP)
                if self.is_ip_camera and self.threaded_cam:
                    self.threaded_cam.stop()
                    self.threaded_cam = None
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                # Try to connect to IP camera using ThreadedIPCamera
                logger.warning(f"Attempting to connect to IP camera at {ip_addr}:{port}")
                self.camera_index = rtsp_url  # Store RTSP URL as camera index
                self.threaded_cam = ThreadedIPCamera(rtsp_url)
                result = self.threaded_cam.start()
                
                if result is None or not self.threaded_cam.isOpened():
                    # Try alternative RTSP paths
                    alt_paths = [
                        f"rtsp://{username}:{password}@{ip_addr}:{port}/Streaming/Channels/101",  # Hikvision
                        f"rtsp://{username}:{password}@{ip_addr}:{port}/cam/realmonitor?channel=1&subtype=0",  # Dahua
                        f"rtsp://{username}:{password}@{ip_addr}:{port}/live/ch00_0",  # Generic
                        f"rtsp://{username}:{password}@{ip_addr}:{port}/h264Preview_01_main",  # Some cameras
                    ]
                    
                    connected = False
                    for alt_url in alt_paths:
                        logger.warning(f"Trying alternative RTSP path: {alt_url}")
                        if self.threaded_cam:
                            self.threaded_cam.stop()
                        self.threaded_cam = ThreadedIPCamera(alt_url)
                        result = self.threaded_cam.start()
                        if result and self.threaded_cam.isOpened():
                            ret, _ = self.threaded_cam.read()
                            if ret:
                                self.camera_index = alt_url
                                connected = True
                                break
                    
                    if not connected:
                        if self.threaded_cam:
                            self.threaded_cam.stop()
                            self.threaded_cam = None
                        messagebox.showerror("Camera Error", 
                            f"Failed to connect to IP camera at {ip_addr}:{port}\n\n"
                            "Please check:\n"
                            "• IP address and port are correct\n"
                            "• Username and password are correct\n"
                            "• Camera is online and RTSP is enabled")
                        dialog.destroy()
                        return
                
                # Mark as IP camera
                self.is_ip_camera = True
                
                # Wait for first frame from threaded grabber
                logger.warning("Waiting for IP camera stream...")
                for _ in range(20):
                    ret, frame = self.threaded_cam.read()
                    if ret and frame is not None:
                        logger.warning("IP camera stream ready!")
                        break
                    time.sleep(0.1)
                
                # Update frame dimensions
                self.frame_w = int(self.threaded_cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                self.frame_h = int(self.threaded_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                
                # Resume if was running
                self.is_camera_running = was_running
                self.is_running = was_running
                
                if self.is_camera_running:
                    logger.warning(f"Connected to IP Camera at {ip_addr} - Restarting video feed")
                    self.update_video_feed()
                else:
                    logger.warning(f"IP Camera at {ip_addr} ready (feed paused)")
                
                messagebox.showinfo("Camera Switch", f"Connected to IP Camera at {ip_addr}")
                dialog.destroy()
                return
            
            # USB Camera selected - stop any IP camera first
            if self.is_ip_camera and self.threaded_cam:
                self.threaded_cam.stop()
                self.threaded_cam = None
                self.is_ip_camera = False
            
            new_camera = selected_camera.get()
            if hasattr(self, 'camera_index') and new_camera == self.camera_index and not self.is_ip_camera:
                messagebox.showinfo("Camera Switch", "Already using this camera")
                dialog.destroy()
                return
            
            # Store running state
            was_running = self.is_camera_running
            
            # Stop current camera if running
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Switch to new camera
            self.camera_index = new_camera
            # Use DirectShow API on Windows for better compatibility
            if platform.system() == "Windows":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_index}")
                dialog.destroy()
                return
            
            # Apply camera settings with Windows compatibility fixes
            try:
                logger.info(f"Configuring camera {self.camera_index} properties...")
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Windows fix: reduce CPU usage
                if platform.system() == "Windows":
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        logger.debug("MJPEG codec set successfully")
                    except Exception as codec_error:
                        logger.warning(f"Could not set MJPEG codec: {codec_error}")
                logger.debug("Camera settings applied successfully")
            except Exception as e:
                logger.warning(f"Some camera settings failed (non-critical): {e}")
            
            # Warm up camera
            for i in range(10):
                try:
                    ret, _ = self.cap.read()
                    time.sleep(0.05)
                    if not ret:
                        logger.debug(f"Camera warmup: frame read failed on attempt {i+1}, retrying...")
                        continue
                except Exception as e:
                    logger.debug(f"Camera warmup exception (attempt {i+1}): {e}")
                    continue
            
            # Update frame dimensions for new camera
            self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Resume if was running
            self.is_camera_running = was_running
            self.is_running = was_running
            
            if self.is_camera_running:
                # Restart video feed update loop for new camera
                logger.warning(f"Switched to Camera {self.camera_index} - Restarting video feed")
                self.update_video_feed()
            else:
                logger.warning(f"Camera {self.camera_index} ready (feed paused)")
            
            messagebox.showinfo("Camera Switch", f"Switched to Camera {self.camera_index}")
            dialog.destroy()
        
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

    def start_camera(self):
        if not self.is_running:
            try:
                # Detect available cameras
                available_cameras = detect_available_cameras()
                
                # Always show camera selection dialog (for USB and IP cameras)
                dialog = tk.Toplevel(self.root)
                dialog.title("Select Camera")
                dialog.geometry("400x550")
                dialog.configure(bg="#2c3e50")
                dialog.transient(self.root)
                dialog.grab_set()
                
                tk.Label(dialog, text="Select Camera:", font=('Helvetica', 14, 'bold'), 
                        bg="#2c3e50", fg="white").pack(pady=15)
                
                # Variable to track camera type (0 = USB camera, 1 = IP camera)
                camera_type_var = tk.IntVar(value=0)
                camera_selected = [False]  # Track if camera was selected
                
                # === USB CAMERAS SECTION ===
                usb_frame = tk.Frame(dialog, bg="#34495e")
                usb_frame.pack(pady=5, padx=20, fill="x")
                
                tk.Radiobutton(usb_frame, text="USB Camera", variable=camera_type_var,
                              value=0, font=('Helvetica', 12, 'bold'), bg="#34495e", fg="white",
                              selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=5)
                
                selected_camera = tk.IntVar(value=available_cameras[0] if available_cameras else 0)
                
                # Frame for USB camera radio buttons
                radio_frame = tk.Frame(usb_frame, bg="#34495e")
                radio_frame.pack(pady=5, padx=20, fill="both")
                
                if available_cameras:
                    for idx in available_cameras:
                        tk.Radiobutton(radio_frame, text=f"Camera {idx}", variable=selected_camera,
                                      value=idx, font=('Helvetica', 11), bg="#34495e", fg="white",
                                      selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=2)
                else:
                    tk.Label(radio_frame, text="No USB cameras detected", bg="#34495e", fg="#e74c3c",
                            font=('Helvetica', 10, 'italic')).pack(anchor="w", padx=10, pady=5)
                
                # === IP CAMERA SECTION ===
                ip_frame = tk.Frame(dialog, bg="#34495e")
                ip_frame.pack(pady=10, padx=20, fill="x")
                
                tk.Radiobutton(ip_frame, text="IP Camera (RTSP)", variable=camera_type_var,
                              value=1, font=('Helvetica', 12, 'bold'), bg="#34495e", fg="white",
                              selectcolor="#2c3e50", activebackground="#34495e").pack(anchor="w", padx=10, pady=5)
                
                # IP camera input fields
                input_frame = tk.Frame(ip_frame, bg="#34495e")
                input_frame.pack(pady=5, padx=30, fill="x")
                
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
                tk.Entry(input_frame, textvariable=user_var, font=('Helvetica', 10)).pack(fill="x", pady=(0, 10))
                
                # Password - REQUIRED
                tk.Label(input_frame, text="Password (Required):", anchor="w",
                        bg="#34495e", fg="#f39c12", font=('Helvetica', 10, 'bold')).pack(fill="x")
                pass_var = tk.StringVar()
                tk.Entry(input_frame, textvariable=pass_var, show="*", font=('Helvetica', 10)).pack(fill="x", pady=(0, 10))
                
                def on_select():
                    if camera_type_var.get() == 1:
                        # IP Camera selected
                        ip_addr = ip_var.get().strip()
                        port = port_var.get().strip()
                        username = user_var.get().strip()
                        password = pass_var.get().strip()
                        
                        if not ip_addr:
                            messagebox.showerror("Error", "IP Address is required!")
                            return
                        if not username:
                            messagebox.showerror("Error", "Username is required!")
                            return
                        if not password:
                            messagebox.showerror("Error", "Password is required!")
                            return
                        
                        # Build RTSP URL
                        rtsp_url = f"rtsp://{username}:{password}@{ip_addr}:{port}/stream1"
                        self.camera_index = rtsp_url
                    else:
                        # USB Camera selected
                        if not available_cameras:
                            messagebox.showerror("Error", "No USB cameras available!")
                            return
                        self.camera_index = selected_camera.get()
                    
                    camera_selected[0] = True
                    dialog.destroy()
                
                def on_cancel():
                    dialog.destroy()
                
                # Buttons
                btn_frame = tk.Frame(dialog, bg="#2c3e50")
                btn_frame.pack(pady=10)
                
                tk.Button(btn_frame, text="Connect", command=on_select, bg="#27ae60",
                         fg="white", font=('Helvetica', 11, 'bold'), width=10).pack(side="left", padx=5)
                tk.Button(btn_frame, text="Cancel", command=on_cancel, bg="#e74c3c",
                         fg="white", font=('Helvetica', 11, 'bold'), width=10).pack(side="left", padx=5)
                
                dialog.wait_window()
                
                # Check if camera was selected
                if not camera_selected[0]:
                    return
                
                # Open selected camera (USB or IP)
                if isinstance(self.camera_index, str) and self.camera_index.startswith("rtsp://"):
                    logger.warning(f"Connecting to IP camera with threaded grabber...")
                    
                    # Try main RTSP URL first
                    self.threaded_cam = ThreadedIPCamera(self.camera_index)
                    result = self.threaded_cam.start()
                    
                    if result is None or not self.threaded_cam.isOpened():
                        # Try alternative RTSP paths
                        ip_addr = ip_var.get().strip()
                        port = port_var.get().strip()
                        username = user_var.get().strip()
                        password = pass_var.get().strip()
                        
                        alt_paths = [
                            f"rtsp://{username}:{password}@{ip_addr}:{port}/Streaming/Channels/101",
                            f"rtsp://{username}:{password}@{ip_addr}:{port}/cam/realmonitor?channel=1&subtype=0",
                            f"rtsp://{username}:{password}@{ip_addr}:{port}/live/ch00_0",
                            f"rtsp://{username}:{password}@{ip_addr}:{port}/h264Preview_01_main",
                        ]
                        
                        connected = False
                        for alt_url in alt_paths:
                            logger.warning(f"Trying: {alt_url}")
                            if self.threaded_cam:
                                self.threaded_cam.stop()
                            self.threaded_cam = ThreadedIPCamera(alt_url)
                            result = self.threaded_cam.start()
                            if result and self.threaded_cam.isOpened():
                                ret, _ = self.threaded_cam.read()
                                if ret:
                                    self.camera_index = alt_url
                                    connected = True
                                    logger.warning(f"Connected successfully using: {alt_url}")
                                    break
                        
                        if not connected:
                            if self.threaded_cam:
                                self.threaded_cam.stop()
                                self.threaded_cam = None
                            messagebox.showerror("Camera Error", 
                                f"Failed to connect to IP camera\n\n"
                                "Please check:\n"
                                "• IP address and port are correct\n"
                                "• Username and password are correct\n"
                                "• Camera is online and RTSP is enabled")
                            return
                    
                    # Mark as IP camera for special handling in update loop
                    self.is_ip_camera = True
                    
                    # Wait for IP camera stream with SHORT timeout (don't block GUI)
                    logger.warning("Waiting for IP camera stream...")
                    ip_ready = False
                    for i in range(10):  # Reduced from 20 - only 1 second total
                        ret, frame = self.threaded_cam.read()
                        if ret and frame is not None:
                            logger.warning("IP camera stream ready!")
                            ip_ready = True
                            break
                        time.sleep(0.1)
                    
                    if not ip_ready:
                        logger.warning("IP camera stream not ready yet, but continuing anyway - may get frames on next update")
                else:
                    # USB Camera - standard VideoCapture with timeout
                    self.is_ip_camera = False
                    logger.info(f"Opening USB camera {self.camera_index} with 3 second timeout...")
                    
                    # Use timeout wrapper to prevent hanging
                    if platform.system() == "Windows":
                        self.cap = open_camera_with_timeout(self.camera_index, cv2.CAP_DSHOW, timeout_seconds=3)
                    else:
                        self.cap = open_camera_with_timeout(self.camera_index, timeout_seconds=3)
                    
                    if self.cap is None or not self.cap.isOpened():
                        messagebox.showerror("Camera Error", 
                            f"Failed to open camera {self.camera_index}\n\n"
                            "This may be due to:\n"
                            "• Camera is not available or disconnected\n"
                            "• Another application is using the camera\n"
                            "• USB camera driver issue\n\n"
                            "Try plugging in the camera again or restarting the application.")
                        return
                    
                    # USB camera settings
                    try:
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Windows MJPEG fix for stability
                        if platform.system() == "Windows":
                            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        logger.debug("Camera settings applied successfully")
                    except Exception as e:
                        logger.warning(f"Some camera settings failed (non-critical): {e}")
                    
                    # Warm up USB camera (non-blocking with timeout)
                    logger.info("Warming up camera...")
                    def warmup_camera():
                        """Background camera warmup to avoid blocking GUI"""
                        for i in range(5):  # Reduced from 10 for faster startup
                            try:
                                ret, _ = self.cap.read()
                                if ret:
                                    break
                            except:
                                pass
                            time.sleep(0.02)  # Reduced from 0.05
                        logger.info("Camera ready!")
                    
                    warmup_thread = threading.Thread(target=warmup_camera, daemon=True)
                    warmup_thread.start()
                    # Don't wait for warmup thread - let it run in background
                
                # Get frame dimensions from appropriate camera source
                if self.is_ip_camera and self.threaded_cam:
                    self.frame_w = int(self.threaded_cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    self.frame_h = int(self.threaded_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                else:
                    self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                self.is_running = True
                self.is_camera_running = True
                self.btn_camera_toggle.configure(text="🎥 Camera ON", fg_color="#27ae60")
                self.btn_guard_toggle.configure(state="normal")
                self.btn_alert_toggle.configure(state="normal")
                self.btn_fugitive_toggle.configure(state="normal")
                logger.warning(f"Camera {self.camera_index} started successfully")
                self.update_video_feed()
            except Exception as e:
                logger.error(f"Camera start error: {e}")
                messagebox.showerror("Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            
            if self.is_ip_camera and hasattr(self, 'threaded_cam') and self.threaded_cam:
                self.threaded_cam.stop()
                self.threaded_cam = None
                self.is_ip_camera = False
            
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.is_logging:
                self.save_log_to_file()
            
            # Stop Fugitive Mode if running
            if hasattr(self, 'is_fugitive_detection') and self.is_fugitive_detection:
                self.is_fugitive_detection = False
                self.fugitive_image = None
                self.fugitive_face_encoding = None
                self.fugitive_detected_log_done = False
                self.btn_fugitive_toggle.configure(text="🚨 Fugitive OFF", fg_color="#95a5a6")
                self.fugitive_preview_label.configure(image='', text="No Fugitive Selected")
            
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
        
        if self.frame_counter % 500 == 0:  # Every ~500 frames
            optimize_memory()
    
    def clear_caches(self):
        """Clear old action cache entries to free memory - different from global optimize_memory()"""
        try:
            # Clear old action cache - keep last 100 entries for multi-guard scenarios
            max_cache_size = 100
            if len(self.last_action_cache) > max_cache_size:
                # Keep only the most recent entries
                keys_to_remove = list(self.last_action_cache.keys())[:-max_cache_size]
                for key in keys_to_remove:
                    del self.last_action_cache[key]
            
            logger.debug("Action cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

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
            # Use threaded camera for IP cameras (smooth streaming)
            if self.is_ip_camera and hasattr(self, 'threaded_cam') and self.threaded_cam:
                if not self.threaded_cam.isOpened():
                    logger.error("IP Camera not available")
                    self.stop_camera()
                    return
                
                ret, frame = self.threaded_cam.read()
                if not ret or frame is None:
                    # IP camera: Just skip frame, don't block - next frame will come from buffer
                    if self.is_running:
                        self.root.after(10, self.update_video_feed)
                    return
            else:
                # USB camera: Original behavior
                if not self.cap or not self.cap.isOpened():
                    logger.error("Camera not available")
                    self.stop_camera()
                    return
                
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    logger.error(f"Camera read exception: {e}")
                    ret = False
                    frame = None
                
                if not ret or frame is None:
                    # Non-blocking recovery - don't freeze GUI with sleep loops
                    # Track consecutive failures for progressive recovery
                    self._frame_fail_count = getattr(self, '_frame_fail_count', 0) + 1
                    
                    if self._frame_fail_count <= 5:
                        # Quick retry - just skip this frame, schedule next update immediately
                        if self.is_running:
                            self.root.after(5, self.update_video_feed)
                        return
                    elif self._frame_fail_count <= 15:
                        # Moderate issues - try to clear buffer by grabbing without decoding
                        try:
                            self.cap.grab()  # Fast buffer clear
                        except Exception as grab_error:
                            logger.debug(f"Buffer grab failed: {grab_error}")
                        if self.is_running:
                            self.root.after(16, self.update_video_feed)
                        return
                    else:
                        # Persistent failure - schedule async reconnect with SHORT timeout
                        logger.warning(f"Camera frame failures: {self._frame_fail_count}, attempting reconnect...")
                        self._frame_fail_count = 0
                        
                        def async_reconnect():
                            """Fast camera reconnection with timeout safety"""
                            try:
                                logger.info("Releasing old camera instance...")
                                if self.cap:
                                    try:
                                        self.cap.release()
                                    except:
                                        pass
                                    self.cap = None
                                
                                # Minimal delay - camera release is quick
                                time.sleep(0.1)
                                
                                logger.info(f"Opening camera {self.camera_index} with timeout...")
                                # Use timeout wrapper for safe camera opening (2 second timeout)
                                if platform.system() == "Windows":
                                    self.cap = open_camera_with_timeout(self.camera_index, cv2.CAP_DSHOW, timeout_seconds=2)
                                else:
                                    self.cap = open_camera_with_timeout(self.camera_index, timeout_seconds=2)
                                
                                if self.cap is None:
                                    logger.error(f"Failed to open camera {self.camera_index} (timeout)")
                                    return
                                
                                if not self.cap.isOpened():
                                    logger.error(f"Failed to open camera {self.camera_index}")
                                    self.cap = None
                                    return
                                
                                # Apply minimal settings for fast recovery
                                try:
                                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                                except:
                                    pass
                                
                                # Fast warmup: just try to grab one frame
                                logger.debug("Testing camera...")
                                for attempt in range(3):
                                    try:
                                        ret = self.cap.grab()
                                        if ret:
                                            logger.info("Camera reconnected successfully")
                                            return
                                    except:
                                        pass
                                    time.sleep(0.05)
                                
                                logger.warning("Camera test frames failed - may be unstable")
                            except Exception as e:
                                logger.error(f"Async reconnection error: {e}")
                        
                        reconnect_thread = threading.Thread(target=async_reconnect, daemon=True)
                        reconnect_thread.start()
                        if self.is_running:
                            self.root.after(800, self.update_video_feed)  # Wait only 800ms for reconnect
                        return
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            logger.error(f"Exception type: {type(e).__name__}, Details: {str(e)}")
            # Don't immediately stop - try to recover
            self._frame_fail_count = getattr(self, '_frame_fail_count', 0) + 1
            if self._frame_fail_count > 20:  # Give it more chances before stopping
                logger.error("Too many camera failures - stopping camera")
                self.stop_camera()
            else:
                # Schedule retry
                if self.is_running:
                    self.root.after(100, self.update_video_feed)
            return
        
        # Reset failure counter on successful read
        self._frame_fail_count = 0
        self.unprocessed_frame = frame.copy()
        
        # Enhance frame for low-light conditions
        frame = self.enhance_frame_for_low_light(frame)
        
        # Frame skipping for performance
        self.frame_counter += 1
        skip_interval = CONFIG["performance"]["frame_skip_interval"]
        
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
        
        # Update GUI labels only every 30 frames (~1 second) instead of every frame
        if self.frame_counter % 30 == 0:  # Update labels every 30 frames
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.last_fps_time = current_time
            
            # Periodic memory optimization
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
        
        # Auto flush logs (only every 30 frames to reduce overhead)
        if self.frame_counter % 30 == 0:
            self.auto_flush_logs()
        
        # GUI VIDEO FEED RENDERING
        if self.video_label.winfo_exists():
            try:
                # Cache label dimensions - only recompute every 30 frames
                if self.frame_counter % 30 == 0 or not hasattr(self, '_cached_label_dims'):
                    self._cached_label_dims = (self.video_label.winfo_width(), self.video_label.winfo_height())
                
                lbl_w, lbl_h = self._cached_label_dims
                h, w = frame.shape[:2]
                
                # Use INTER_NEAREST (fastest) for real-time video
                if lbl_w > 10 and lbl_h > 10:
                    scale = min(lbl_w/w, lbl_h/h, 1.5)
                    new_w, new_h = int(w*scale), int(h*scale)
                    # INTER_NEAREST is 3-5x faster than INTER_LINEAR
                    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    frame_resized = frame
                    new_w, new_h = w, h
                
                # Convert BGR to RGB and create PhotoImage directly
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb, mode='RGB')
                photo = ImageTk.PhotoImage(image=pil_image)
                
                # Direct label update (configure is faster than recreating CTkImage)
                self.video_label.configure(image=photo, text="")
                self._current_photo = photo  # Keep reference to prevent GC
            except Exception as e:
                logger.debug(f"Frame display error: {e}")
        
        # Use constant 33ms refresh (30 FPS) for smooth real-time video
        self.root.after(33, self.update_video_feed)

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

    def process_tracking_frame_optimized(self, frame):
        # Fugitive detection runs FIRST, before checking if guards exist
        # This ensures fugitive alert works independently of guard tracking state
        
        # Only convert to RGB once per frame (reuse for all detection)
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # ==================== FUGITIVE MODE ====================
        # Fugitive detection must be INSTANT - run every frame like guard tracking
        fugitive_detected_this_frame = False
        if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
            # RUN EVERY FRAME - fugitive must be detected ASAP when appearing
            # Use FULL resolution for maximum accuracy (same as guard detection when guards missing)
            try:
                # FULL RESOLUTION: No downscaling for fugitive - accuracy is critical
                fugitive_face_locations = face_recognition.face_locations(rgb_full_frame, model="hog")
                if fugitive_face_locations:
                    # Use num_jitters=1 for speed
                    fugitive_face_encodings = face_recognition.face_encodings(rgb_full_frame, fugitive_face_locations, num_jitters=1)
                
                    for face_encoding, face_location in zip(fugitive_face_encodings, fugitive_face_locations):
                        # ✅ IDENTITY LOCK: If face belongs to a stable guard, skip fugitive check
                        # This prevents "Guard vs Fugitive" flickering
                        is_stable_guard_face = False
                        top, right, bottom, left = face_location
                        
                        for guard_name, guard_status in self.targets_status.items():
                            if guard_status.get("stable_tracking", False) and guard_status.get("face_box"):
                                gx1, gy1, gx2, gy2 = guard_status["face_box"]
                                # Calculate IoU with detected face
                                fx1, fy1, fx2, fy2 = left, top, right, bottom
                                
                                x_left = max(gx1, fx1)
                                y_top = max(gy1, fy1)
                                x_right = min(gx2, fx2)
                                y_bottom = min(gy2, fy2)
                                
                                if x_right > x_left and y_bottom > y_top:
                                    intersection = (x_right - x_left) * (y_bottom - y_top)
                                    guard_area = (gx2 - gx1) * (gy2 - gy1)
                                    face_area = (fx2 - fx1) * (fy2 - fy1)
                                    iou = intersection / float(guard_area + face_area - intersection)
                                    
                                    if iou > 0.5: # High overlap with stable guard
                                        is_stable_guard_face = True
                                        break
                        
                        if is_stable_guard_face:
                            continue

                        # Tolerance similar to guard detection for consistency
                        face_distance = face_recognition.face_distance([self.fugitive_face_encoding], face_encoding)
                        confidence = 1.0 - face_distance[0]
                        
                        # Same philosophy as guard - detect quickly but no false positives
                        # Use 0.55 tolerance (same as single guard) and 0.45 min confidence
                        fugitive_tolerance = 0.55
                        min_confidence_threshold = 0.45
                        
                        is_fugitive_match = (face_distance[0] <= fugitive_tolerance and 
                                           confidence >= min_confidence_threshold)
                        
                        if is_fugitive_match:
                            # Cross-check: Make sure this isn't a guard
                            is_likely_guard = False
                            margin = getattr(self, 'guard_fugitive_margin', 0.18)  # Increased to 0.18 for better separation
                            
                            if self.targets_status and len(self.targets_status) > 0:
                                for guard_name, guard_status in self.targets_status.items():
                                    # Check against ALL guard encodings, not just primary
                                    guard_encodings = guard_status.get("multi_angle_encodings", [])
                                    primary_enc = guard_status.get("encoding")
                                    if not guard_encodings and primary_enc is not None:
                                        guard_encodings = [primary_enc]
                                    
                                    if guard_encodings:
                                        # Find best match across all guard angles
                                        guard_distances = face_recognition.face_distance(guard_encodings, face_encoding)
                                        guard_dist = min(guard_distances)
                                        
                                        # Guard must be SIGNIFICANTLY better (0.08 margin)
                                        if guard_dist < face_distance[0] - 0.08:
                                            logger.debug(f"[SKIP FUGITIVE] Face is GUARD '{guard_name}' (guard_dist={guard_dist:.3f} << fugitive_dist={face_distance[0]:.3f})")
                                            is_likely_guard = True
                                            break
                                        
                                        # Skip if too close to any guard angle
                                        if abs(guard_dist - face_distance[0]) < margin:
                                            logger.debug(f"[SKIP FUGITIVE] Face TOO CLOSE to guard '{guard_name}' (guard={guard_dist:.3f}, fugitive={face_distance[0]:.3f}, margin={margin})")
                                            is_likely_guard = True
                                            break
                            
                            if is_likely_guard:
                                continue  # Skip this face - it's a guard, not fugitive
                            
                            # FUGITIVE CONFIRMED
                            fugitive_detected_this_frame = True
                            top, right, bottom, left = face_location
                            # No scaling needed - using full resolution
                            
                            # Draw bounding box - BRIGHT RED for fugitive alert
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                            cv2.putText(frame, f"FUGITIVE: {self.fugitive_name}", (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            
                            logger.warning(f"[FUGITIVE MATCH] Distance={face_distance[0]:.3f}, Confidence={confidence:.3f} - FUGITIVE DETECTED!")
                            
                            # Track if this is first appearance (for logging)
                            is_first_appearance = not getattr(self, 'fugitive_currently_visible', False)
                            self.fugitive_currently_visible = True
                            
                            # Play alert sound if not already playing
                            try:
                                alert_playing = (hasattr(self, 'fugitive_alert_sound_thread') and 
                                               self.fugitive_alert_sound_thread is not None and 
                                               self.fugitive_alert_sound_thread.is_alive())
                                
                                if not alert_playing:
                                    if self.fugitive_alert_stop_event is None:
                                        self.fugitive_alert_stop_event = threading.Event()
                                    
                                    self.fugitive_alert_stop_event = threading.Event()
                                    self.fugitive_alert_sound_thread = play_siren_sound(
                                        stop_event=self.fugitive_alert_stop_event,
                                        sound_file=get_sound_path("Fugitive.mp3"),
                                        duration_seconds=15
                                    )
                                    logger.warning(f"[FUGITIVE ALERT] !!! FUGITIVE DETECTED - {self.fugitive_name}")
                            except Exception as e:
                                logger.error(f"[FUGITIVE SOUND] Error playing alert: {e}")
                            
                            # Log only on first detection
                            if is_first_appearance and not getattr(self, 'fugitive_logged_once', False):
                                try:
                                    fugitive_bbox = (left, top, right, bottom)
                                    snapshot_path = self.capture_alert_snapshot(
                                        frame, 
                                        f"FUGITIVE_{self.fugitive_name}_APPEARED", 
                                        check_rate_limit=False,
                                        bbox=fugitive_bbox,
                                        is_fugitive=True
                                    )
                                    img_path = snapshot_path if snapshot_path else "N/A"
                                    
                                    self.temp_log.append((
                                        time.strftime("%Y-%m-%d %H:%M:%S"),
                                        f"FUGITIVE_{self.fugitive_name}",
                                        "FUGITIVE_DETECTED",
                                        f"FIRST_APPEARANCE (confidence: {confidence:.3f})",
                                        img_path,
                                        f"{confidence:.3f}"
                                    ))
                                    self.temp_log_counter += 1
                                    self.fugitive_logged_once = True
                                    self.save_log_to_file()
                                    logger.warning(f"[FUGITIVE DETECTED] {self.fugitive_name} FIRST APPEARANCE logged")
                                except Exception as e:
                                    logger.error(f"[FUGITIVE LOG] Error: {e}")
            except Exception as e:
                logger.debug(f"Fugitive detection error: {e}")
        
        # RESET FUGITIVE VISIBILITY FLAG if not detected this frame
        if self.is_fugitive_detection and not fugitive_detected_this_frame:
            if getattr(self, 'fugitive_currently_visible', False):
                self.fugitive_currently_visible = False
                logger.debug(f"[FUGITIVE] {self.fugitive_name} disappeared from view")
        
        # GUARD TRACKING: Only run if guards are selected
        if not self.targets_status:
            cv2.putText(frame, "SELECT TARGETS TO START", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        # Adaptive detection interval based on tracking stability
        # More frequent detection when targets are lost, less frequent when stable
        self.re_detect_counter += 1
        untracked_count = len([n for n, s in self.targets_status.items() if not s["visible"]])
        
        # CRITICAL: ALWAYS detect when guards are missing - NO INTERVAL DELAYS
        # This ensures untracked guards are identified ASAP when they appear
        if untracked_count > 0:
            # ANY guard missing - FORCE IMMEDIATE DETECTION (reset counter to run immediately)
            self.re_detect_counter = 0
        
        # Create rgb_full_frame here for detection
        rgb_full_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]
        
        # Get current time at the start of the tracking loop
        current_time = time.time()

        # 1. Update Trackers (PERSISTENT TRACKING - Keep tracking even at frame edges)
        # Keep tracking guards even if they partially leave the frame
        # Persistent tracking ensures continuous monitoring until guard fully exits
        for name, status in self.targets_status.items():
            # Skip invisible targets COMPLETELY - no tracker update, no drawing
            # This saves ~5-10ms per untracked target per frame
            if not status.get("visible", False):
                continue  # Skip to next target - invisible targets get zero processing
            
            if status.get("tracker") is None:
                continue  # No tracker to update
            
            # Only update tracker if it exists and target is visible
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
                        # ✅ CRITICAL: Force immediate re-detection on next frame
                        self.re_detect_counter = 999  # Force detection to run immediately
                        logger.warning(f"[TRACKER LOST] {name}: Extreme movement detected (dx={dx:.0f}, dy={dy:.0f}, size_change={size_change:.0f}) - FORCING re-detection")
                    else:
                        # ✅ ANTI-DRIFT: Mark tracker for identity verification (will be checked after face detection)
                        status["tracker_verify_counter"] = status.get("tracker_verify_counter", 0) + 1
                        
                        # ✅ DRIFT CHECK: Verify tracker is following the correct person
                        # USER REQUEST: "verify the guard with face every 5-10 seconds"
                        # 30 FPS * 5 seconds = 150 frames
                        # 30 FPS * 10 seconds = 300 frames
                        # We use 150 frames (5 seconds) as the base interval.
                        VERIFY_INTERVAL = 150
                        
                        if status["tracker_verify_counter"] >= VERIFY_INTERVAL:
                            status["needs_drift_verification"] = True
                            status["tracker_verify_counter"] = 0  # Reset counter
                            logger.info(f"[PERIODIC VERIFY] {name}: Triggering face verification (5s interval)")
                        
                        status["pending_tracker_box"] = new_box  # Store for later verification
                        
                        # Tracker still valid for now - apply smoothing
                        smoothed_box = smooth_bounding_box(new_box, status["face_box"], smoothing_factor=0.4)
                        status["face_box"] = smoothed_box
                        status["visible"] = True
                        status["needs_face_reverification"] = True
                        
                        if not status.get("consecutive_no_face_frames"):
                            status["consecutive_no_face_frames"] = 0
                        
                        if at_frame_edge:
                            logger.debug(f"[TRACKER OK] {name}: Persistent tracking at frame edge (movement: dx={dx:.0f}, dy={dy:.0f})")
                        else:
                            logger.debug(f"[TRACKER OK] {name}: Tracked (movement: dx={dx:.0f}, dy={dy:.0f})")
                else:
                    status["face_box"] = new_box
                    status["visible"] = True
            else:
                # Instead of immediate HARD RESET, enter "Coasting" mode
                status["consecutive_failures"] = status.get("consecutive_failures", 0) + 1
                
                # Allow 30 frames (1 second) of "bad data" before killing the track
                COASTING_TOLERANCE = 30 
                
                if status["consecutive_failures"] < COASTING_TOLERANCE:
                    # Keep the old box, mark as "Coasting"
                    status["visible"] = True # Keep it visible in UI
                    logger.debug(f"[COASTING] {name}: Signal lost, coasting... ({status['consecutive_failures']}/{COASTING_TOLERANCE})")
                    
                    # Optional: Draw the box in YELLOW to indicate weak signal
                    if status.get("body_box"):
                         bx1, by1, bx2, by2 = status["body_box"]
                         cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                         
                else:
                    # NOW trigger the Hard Reset
                    status["visible"] = False
                    status["tracker"] = None
                    status["consecutive_failures"] = 0
                    # ✅ CRITICAL: Force immediate re-detection on next frame
                    self.re_detect_counter = 999  # Force detection to run immediately
                    logger.warning(f"[TRACK LOST] {name}: Signal lost for {COASTING_TOLERANCE} frames - Resetting.")

        # 2. Detection (PARALLEL MATCHING) - ALWAYS run detection for guard identification
        # ✅ CRITICAL: Detection MUST run EVERY FRAME to catch guards immediately
        untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
        
        # ✅ ALWAYS RUN DETECTION - No skipping, every frame for fastest guard identification
        should_run_detection = True
        
        # ✅ DRIFT DETECTION: Initialize face detection results at outer scope
        # This ensures drift check can access them even if detection is skipped
        face_locations = []
        face_encodings = []
        
        if should_run_detection:
            # ✅ NEVER SKIP DETECTION when guards are missing
            # Only skip if ALL guards are visible AND all have high confidence trackers
            all_guards_visible = all(s.get("visible", False) for s in self.targets_status.values())
            all_trackers_confident = all(
                status.get("tracker_confidence", 0.0) > 0.96 and
                status.get("tracker_verify_counter", 0) < 150  # ✅ Force detection if verification due
                for status in self.targets_status.values()
                if status.get("visible", False)
            )
            
            # Auto-detect single vs multi-person mode based on target count
            is_single_person = len(self.targets_status) <= 1
            
            # ✅ INSTANT DETECTION: Run detection EVERY frame when ANY guard is missing
            # This ensures new guards are identified ASAP (within 1 frame / 33ms)
            untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
            
            if len(untracked_targets) > 0:
                # ANY guard is missing - ALWAYS run detection EVERY frame
                face_locations = self.detect_faces_fast(rgb_full_frame)
                if len(untracked_targets) > 0:
                    logger.debug(f"[DETECT] {len(untracked_targets)} guards missing - using Fast Detection, found {len(face_locations)} faces")
            elif all_guards_visible and all_trackers_confident and len(self.targets_status) > 0:
                # All visible targets have high confidence trackers - skip detection
                logger.debug(f"[PERF-SKIP] All {len(self.targets_status)} guards visible with high confidence - skipping face detection")
                face_locations = []
            else:
                # ✅ MUST DETECT: Either guards missing OR confidence low
                # Use MediaPipe for fast detection
                face_locations = self.detect_faces_fast(rgb_full_frame)
            
            # Get encodings if faces found (use original frame for accuracy)
            # ✅ ANTI-DRIFT: Always encode faces for identity verification (prevents tracker drift)
            # Encoding is expensive (~50-80ms) but necessary to detect when tracker follows wrong person
            untracked_targets = [name for name, s in self.targets_status.items() if not s["visible"]]
            
            # Count faces - if many unknown faces, we need encodings for drift detection
            num_faces = len(face_locations) if face_locations else 0
            num_guards = len(self.targets_status)
            has_potential_unknown = (num_faces > num_guards)  # More faces than guards = unknown persons
            
            # Check if any tracker needs identity verification this frame
            # USER REQUEST: "verify the guard with face every 5-10 seconds"
            # We check if the counter is approaching the 150-frame threshold (5 seconds)
            any_tracker_needs_verify = any(
                status.get("tracker_verify_counter", 0) >= 149  # Will trigger verify on next increment (150)
                for status in self.targets_status.values()
                if status.get("visible", False)
            )
            
            # ✅ CRITICAL FIX: ALWAYS encode when we have untracked targets and faces
            # This ensures matching can occur
            if face_locations and len(untracked_targets) > 0:
                # We have untracked guards AND detected faces - MUST encode for matching
                face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations, num_jitters=1)
                logger.debug(f"[ENCODING] Found {len(face_encodings)} face encodings for {len(untracked_targets)} untracked guards")
            elif face_locations and (has_potential_unknown or any_tracker_needs_verify):
                # Encode faces for drift detection even if all guards tracked
                face_encodings = face_recognition.face_encodings(rgb_full_frame, face_locations, num_jitters=1)
                if has_potential_unknown:
                    logger.debug(f"[ANTI-DRIFT] Encoding {num_faces} faces ({num_faces - num_guards} potential unknown persons)")
            else:
                # No faces or no need for encoding
                face_encodings = []

            # Iterate over all targets to verify tracking
            for name, status in self.targets_status.items():
                if not status["visible"]:
                    continue
                
                face_found_at_tracked_location = False
                imposter_detected = False  # Initialize here to avoid UnboundLocalError
                
                # ✅ PERFORMANCE: Skip verification if tracker confidence is very high
                # High confidence tracker = reliable tracking, no need for expensive face verification
                # OPTIMIZED: Raised threshold from 0.92 to 0.95 for more frequent verification
                tracker_conf = status.get("tracker_confidence", 0.0)
                
                # ✅ SAFETY: NEVER skip verification if multiple faces are present (risk of ID switch)
                # USER REQUEST: "if the unknown person is pass by the guard, the tracking forgot the guard and trakck that unknown person"
                # If multiple faces are detected, we MUST verify identity to prevent tracker from switching to the wrong person.
                multiple_faces_present = len(face_locations) > 1
                
                # ✅ PERIODIC VERIFICATION: Force verification if interval reached
                # USER REQUEST: "verify the guard with face every 5-10 seconds"
                needs_periodic_verify = status.get("needs_drift_verification", False)
                
                if tracker_conf > 0.95 and not multiple_faces_present and not needs_periodic_verify:
                    # Tracker is very stable AND scene is simple AND no periodic verify needed
                    face_found_at_tracked_location = True
                    status["face_detection_missing_frames"] = 0  # Reset counter
                    continue  # Skip to next target
                
                if needs_periodic_verify:
                    status["needs_drift_verification"] = False  # Reset flag after we proceed to verification
                    # logger.debug(f"[VERIFY] {name}: Performing periodic face verification")
                
                # Check if any detected face matches this target
                if face_locations and face_encodings and len(face_encodings) > 0:
                    # ✅ MULTI-ANGLE VERIFICATION: Check against ALL angle encodings
                    multi_angle_encodings = status.get("multi_angle_encodings", [])
                    primary_encoding = status.get("encoding")
                    
                    all_guard_encodings = multi_angle_encodings if multi_angle_encodings else ([primary_encoding] if primary_encoding is not None else [])
                    
                    if all_guard_encodings:
                        # ✅ BALANCED VERIFICATION: Use 0.52 tolerance - strict enough to prevent wrong match
                        # but relaxed enough to handle normal movement and slight angle changes
                        for face_idx, unknown_enc in enumerate(face_encodings):
                            # ✅ PERFORMANCE: Only check primary encoding first, use multi-angle if needed
                            primary_enc = all_guard_encodings[0]
                            primary_dist = face_recognition.face_distance([primary_enc], unknown_enc)[0]
                            
                            # Fast path: If primary matches well, use it
                            if primary_dist < 0.50:
                                best_dist = primary_dist
                            elif len(all_guard_encodings) > 1:
                                # Slow path: Check all angles only if primary didn't match well
                                distances = face_recognition.face_distance(all_guard_encodings, unknown_enc)
                                best_dist = min(distances)
                            else:
                                best_dist = primary_dist
                            
                            # ✅ BALANCED: Accept if distance < 0.52 (handles normal movement + slight angles)
                            if best_dist < 0.52:
                                face_found_at_tracked_location = True
                                # ✅ ANTI-DRIFT: Guard identity confirmed - reset drift counters
                                status["tracker_drift_count"] = 0
                                status["tracker_verify_counter"] = 0
                                
                                # Update face box with the matched face
                                # ✅ FIX: Convert face_recognition (top, right, bottom, left) to (x1, y1, x2, y2)
                                top, right, bottom, left = face_locations[face_idx]
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
                                        logger.debug(f"[TRACKER SYNC] {name}: Re-initialized tracker on detected face (dist={best_dist:.3f})")
                                    except Exception as e:
                                        logger.error(f"Tracker re-init failed: {e}")
                                break  # Found match, stop checking other faces
                    
                    # Also check overlap with tracker box if encoding match failed or not available
                    imposter_detected = False
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
                                        # ✅ STRICT VERIFICATION: Do NOT accept IoU match if encoding failed
                                        # USER REQUEST: "script is detecting unkonwn person as guard"
                                        # If we have encodings, and this face failed the encoding check (distance > 0.52),
                                        # we MUST REJECT it, even if it overlaps the tracker.
                                        # This prevents the tracker from "locking on" to an imposter who walked into the frame.
                                        
                                        if face_encodings and i < len(face_encodings) and all_guard_encodings:
                                            # Check distance again to be sure
                                            unknown_enc = face_encodings[i]
                                            
                                            # Calculate best distance to guard
                                            primary_enc = all_guard_encodings[0]
                                            dists = face_recognition.face_distance(all_guard_encodings, unknown_enc)
                                            best_dist = min(dists)
                                            
                                            # If distance is high (e.g. > 0.55), it's an imposter
                                            if best_dist > 0.55:
                                                imposter_detected = True
                                                logger.warning(f"[IMPOSTER REJECT] {name}: Overlapping face is NOT guard (dist={best_dist:.3f}) - Ignoring IoU match")
                                        
                                        if not imposter_detected:
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
                                        else:
                                            # Imposter detected at tracker location!
                                            # We should probably count this as a "missing face" for the guard
                                            # so that if it persists, we drop the track.
                                            pass

                if not face_found_at_tracked_location:
                    # ✅ GRACE PERIOD: Skip ghost check for newly detected guards (first 5 frames)
                    # This prevents false GHOST REMOVED immediately after detection
                    # Guards need a few frames to stabilize pose detection
                    consecutive_detections = status.get("consecutive_detections", 0)
                    if consecutive_detections < 5:
                        logger.debug(f"[GRACE PERIOD] {name}: Skipping ghost check (detection frame {consecutive_detections}/5)")
                        status["consecutive_detections"] = consecutive_detections + 1
                        continue  # Skip ghost check, continue tracking
                    
                    # Body box is often larger/different than face box
                    # Check if tracker is still in reasonable position (not completely drifted)
                    body_box = status.get("body_box")
                    pose_confidence = status.get("pose_confidence", 0.0)
                    skeleton_keypoints = status.get("skeleton_keypoints", 0)
                    
                    # ✅ ROBUST GHOST DETECTION: Track "face only, no pose" frames
                    # If tracker is tracking a face but NO pose is detected for 5 seconds (150 frames), it's a GHOST
                    # USER REQUEST: "at any point of time if there is only face verified with selected guard profile for more then 5 sec it shoud be hard reset"
                    if pose_confidence < 0.20 and skeleton_keypoints < 5:
                        # No valid pose detected - increment ghost counter
                        status["ghost_no_pose_frames"] = status.get("ghost_no_pose_frames", 0) + 1
                        
                        if status["ghost_no_pose_frames"] >= 150:  # 5 seconds @ 30 FPS
                            # ✅ GHOST CONFIRMED: Face tracked but NO POSE for 5+ seconds
                            # This is definitely a ghost (bag, poster, shadow, wrong person's face)
                            logger.warning(f"[GHOST - NO POSE] {name}: Face tracked but NO POSE for {status['ghost_no_pose_frames']} frames (5s) - HARD RESET")
                            
                            # Hard reset all tracking state
                            status["visible"] = False
                            status["tracker"] = None
                            status["body_tracker"] = None
                            status["face_box"] = None
                            status["body_box"] = None
                            status["consecutive_detections"] = 0
                            status["stable_tracking"] = False
                            status["face_confidence"] = 0.0
                            status["tracker_confidence"] = 0.0
                            status["face_detection_missing_frames"] = 0
                            status["missing_pose_counter"] = 0
                            status["ghost_no_pose_frames"] = 0  # Reset ghost counter
                            self.re_detect_counter = 999  # Force immediate re-detection
                            
                            # ✅ USER REQUEST: "make properly skalaton based on face"
                            # We force a re-initialization of the skeleton search in the next frame
                            # by ensuring the face is treated as a fresh detection candidate
                            status["force_skeleton_reinit"] = True
                            
                            continue  # Skip to next guard
                        else:
                            if status["ghost_no_pose_frames"] % 30 == 0:
                                logger.debug(f"[GHOST CHECK] {name}: No pose frame {status['ghost_no_pose_frames']}/150 (pose={pose_confidence:.2f}, keypts={skeleton_keypoints})")
                    else:
                        # Pose detected - reset ghost counter
                        status["ghost_no_pose_frames"] = 0
                    
                    # ROBUST 4-PART VALIDATION for body-only tracking:
                    # 1. body_box exists (from skeleton)
                    # 2. pose confidence adequate (>0.50 is good quality) - INCREASED from 0.40
                    # 3. sufficient keypoints (≥10 indicates valid full-body skeleton)
                    # 4. body is in/near frame (within -50 to width+50, -50 to height+50)
                    
                    # ✅ USE VALIDATE_SKELETON_QUALITY for robust verification
                    # This fixes "Verification mechanism not working properly"
                    # We need to get the landmarks from the last cached pose if available
                    last_pose = status.get("last_cached_pose")
                    skeleton_validation = {'valid': False, 'confidence': 0.0}
                    
                    if last_pose and last_pose.pose_landmarks:
                        skeleton_validation = validate_skeleton_quality(last_pose.pose_landmarks.landmark)
                        pose_confidence = skeleton_validation['confidence']
                        skeleton_keypoints = skeleton_validation['keypoint_count']
                        # Update status with more accurate metrics
                        status["pose_confidence"] = pose_confidence
                        status["skeleton_keypoints"] = skeleton_keypoints
                    
                    body_box_valid = body_box is not None
                    # Use the robust validation result instead of simple threshold
                    pose_quality_valid = skeleton_validation['valid'] or (pose_confidence > 0.50)
                    keypoint_valid = skeleton_keypoints >= 10
                    
                    # Extract body box position for frame boundary check
                    body_in_frame = False
                    if body_box_valid:
                        bx1, by1, bx2, by2 = body_box
                        frame_h, frame_w = frame.shape[:2]
                        # Allow 50px margin beyond frame edges for tracking at boundaries
                        body_in_frame = (bx1 <= frame_w + 50 and bx2 >= -50 and 
                                       by1 <= frame_h + 50 and by2 >= -50)
                    
                    # ✅ FIX "Fixated at one location" / "Considering chair as person"
                    # Check for static objects that look like skeletons (chairs, racks)
                    # If object is static (tracker not moving) AND skeleton validation is marginal
                    is_static = False
                    if status.get("tracker") and status.get("face_box"):
                        # Check movement history
                        if "position_history" not in status:
                            status["position_history"] = deque(maxlen=30) # 1 second history
                        
                        curr_center = ((status["face_box"][0] + status["face_box"][2])/2, 
                                     (status["face_box"][1] + status["face_box"][3])/2)
                        status["position_history"].append(curr_center)
                        
                        if len(status["position_history"]) >= 30:
                            # Calculate total movement over last second
                            start_pos = status["position_history"][0]
                            end_pos = status["position_history"][-1]
                            dist_moved = ((start_pos[0]-end_pos[0])**2 + (start_pos[1]-end_pos[1])**2)**0.5
                            
                            # If moved less than 5 pixels in 1 second -> STATIC
                            if dist_moved < 5:
                                is_static = True
                    
                    # If static AND skeleton is not high quality -> likely a chair/object
                    if is_static and pose_confidence < 0.65:
                        # Increase ghost counter faster for static low-quality objects
                        status["ghost_no_pose_frames"] = status.get("ghost_no_pose_frames", 0) + 2
                        logger.debug(f"[STATIC CHECK] {name}: Static object detected (conf={pose_confidence:.2f}) - accelerating ghost check")
                    
                    # ✅ CRITICAL: ANTI-GHOST LOGIC - Check if body is actually visible
                    # Not just checking skeleton but also verifying against detected faces in current frame
                    face_in_body_area = False
                    if body_box_valid and face_locations and len(face_locations) > 0:
                        bx1, by1, bx2, by2 = body_box
                        body_area = max(1, (bx2 - bx1) * (by2 - by1))  # FIXED: Correct area calculation
                        for face_loc in face_locations:
                            # face_locations format: (top, right, bottom, left)
                            fy1, fx2, fy2, fx1 = face_loc  # FIXED: Correct coordinate extraction
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
                        # ✅ SECURITY: If imposter was detected at tracker location, DO NOT continue tracking
                        if not imposter_detected and (face_in_body_area or pose_confidence > 0.60):
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
                            # ✅ HARD RESET on ghost detection - clear all state immediately
                            status["body_tracker"] = None
                            status["face_box"] = None
                            status["body_box"] = None
                            status["face_confidence"] = 0.0
                            status["tracker_confidence"] = 0.0
                            status["face_detection_missing_frames"] = 0
                            status["missing_pose_counter"] = 0
                            status["ghost_no_pose_frames"] = 0  # Reset ghost counter
                            self.re_detect_counter = 999  # Force immediate re-detection
                            logger.warning(f"[GHOST REMOVED + HARD RESET] {name}: Body detected but no face in area + low pose confidence - HARD RESET triggered")
                    
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
                        # ✅ HARD RESET on ghost detection - clear all state immediately
                        status["body_tracker"] = None
                        status["face_box"] = None
                        status["body_box"] = None
                        status["face_confidence"] = 0.0
                        status["tracker_confidence"] = 0.0
                        status["face_detection_missing_frames"] = 0
                        status["missing_pose_counter"] = 0
                        status["ghost_no_pose_frames"] = 0  # Reset ghost counter
                        self.re_detect_counter = 999  # Force immediate re-detection
                        logger.warning(f"[GHOST REMOVED + HARD RESET] {name}: Tracked but failed body validation - {', '.join(invalid_reasons)} - HARD RESET triggered")
                else:
                    # Face found - reset counters
                    status["face_detection_missing_frames"] = 0
                    status["ghost_no_pose_frames"] = 0  # Reset ghost counter when face is properly found
            
            # ✅ BACK-VIEW SKELETON TRACKING: DISABLED
            # USER REQUEST: "new movement have should never be only scleten without face"
            # "remember sceleten should always srart from face and should not consider as face"
            # We disable the logic that re-initializes the face tracker with the body box.
            # This prevents the system from tracking a skeleton independently of the face
            # and ensures that the "face box" always represents a face, not a body.
            
            # (Disabled block was here - lines 5800-5860)
            pass

            # ✅ OPTIMIZED: Proceed with guard matching only if we have untracked targets
            # face_encodings was already computed above (line ~5399) - reuse it here
            if face_locations and len(face_locations) > 0 and len(untracked_targets) > 0 and face_encodings and len(face_encodings) > 0:
                # We have faces and untracked guards - proceed with matching
                
                logger.info(f"[MATCHING] Starting match of {len(face_locations)} faces with {len(untracked_targets)} untracked guards")
                
                # ✅ NEW PIPELINE: Detect brightness for dark mode pipeline
                gray_frame = cv2.cvtColor(rgb_full_frame, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray_frame)
                
                # Build adaptive params based on brightness
                # ✅ BALANCED VALUES: Guard detection but NO ghost/false positives
                if brightness < 100:  # Low-light mode
                    adaptive_params = {
                        "brightness": int(brightness),
                        "tolerance": 0.52,  # STRICTER for dark to avoid ghosts
                        "confidence": 0.48  # HIGHER threshold to prevent false positives
                    }
                    logger.debug(f"Dark mode activated: brightness={brightness:.0f}")
                else:
                    adaptive_params = {
                        "brightness": int(brightness),
                        "tolerance": 0.55,  # BALANCED for normal
                        "confidence": 0.45  # MODERATE threshold
                    }
                
                # ✅ ENHANCED: Adaptive thresholds for angle-aware detection
                # Support detection at extreme angles (side profile, upward, downward views)
                num_guards = len(untracked_targets)
                
                # ✅ BALANCED TOLERANCE: Must detect guard quickly but AVOID ghost/false positives
                # Priority: Fast identification but NO false positives (ghosts are worse than slow detection)
                if adaptive_params["brightness"] < 100:  # Low-light mode
                    base_tolerance = 0.55  # BALANCED for dark conditions
                    min_confidence = 0.45  # Stricter to prevent ghosts in dark
                    logger.debug(f"Using LOW-LIGHT tolerances: tolerance={base_tolerance:.2f}, min_confidence={min_confidence:.2f}")
                elif num_guards >= 2:
                    # MULTI-GUARD MODE: Stricter to prevent cross-matching
                    base_tolerance = 0.52  # Strict for multi-guard
                    min_confidence = 0.48  # Higher threshold
                else:
                    # SINGLE GUARD: Balanced for fast ID but no ghosts
                    base_tolerance = 0.55  # Balanced tolerance
                    min_confidence = 0.45  # Moderate threshold to prevent ghost
                
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
                
                # ✅ CRITICAL FIX: This logging and matching loop was incorrectly INSIDE the face validation loop!
                # Now moved OUTSIDE to ensure all faces are validated BEFORE matching starts
                logger.info(f"[FACE] Found {len(face_locations)} detections, {len(valid_face_indices)} valid face candidates")
                
                for target_idx, name in enumerate(untracked_targets):
                    # ✅ CRITICAL FIX: Use ALL multi-angle encodings for matching, not just the first one
                    # This ensures guard is recognized from any angle (front, left, right, back, top)
                    multi_angle_encodings = self.targets_status[name].get("multi_angle_encodings", [])
                    target_encoding = self.targets_status[name]["encoding"]
                    
                    # Build list of all encodings to check against
                    all_guard_encodings = []
                    if multi_angle_encodings and len(multi_angle_encodings) > 0:
                        all_guard_encodings = multi_angle_encodings
                        logger.debug(f"[MULTI-ANGLE] {name}: Using {len(all_guard_encodings)} angle encodings for matching")
                    elif target_encoding is not None:
                        all_guard_encodings = [target_encoding]
                    
                    if not all_guard_encodings:
                        logger.warning(f"[WARN] Skip {name}: no encoding available")
                        continue
                    
                    all_matches = []
                    
                    for face_idx, face_width, face_height in valid_face_indices:
                        if face_idx >= len(face_encodings):
                            continue
                        unknown_encoding = face_encodings[face_idx]
                        if unknown_encoding is None:
                            continue
                        
                        # ✅ OPTIMIZED MULTI-ANGLE MATCHING: Check primary first, then others if needed
                        # This reduces computation when primary encoding matches well
                        primary_enc = all_guard_encodings[0]
                        primary_dist = face_recognition.face_distance([primary_enc], unknown_encoding)[0]
                        
                        # Fast path: If primary (front) matches well, use it directly
                        if primary_dist < 0.50 or len(all_guard_encodings) == 1:
                            best_dist = primary_dist
                            best_angle_idx = 0
                        else:
                            # Slow path: Primary didn't match well, check all angles
                            all_distances = face_recognition.face_distance(all_guard_encodings, unknown_encoding)
                            best_dist = min(all_distances)
                            best_angle_idx = list(all_distances).index(best_dist)
                            if best_angle_idx > 0:
                                logger.debug(f"[MULTI-ANGLE] {name} vs face_{face_idx}: Non-front angle {best_angle_idx+1} matched better (dist={best_dist:.3f} vs front={primary_dist:.3f})")
                        
                        confidence = 1.0 - best_dist
                        
                        # ✅ CRITICAL FIX: STRICT GUARD/FUGITIVE SEPARATION
                        # Cross-check: If this is a guard, make sure it's NOT the fugitive
                        # STRICT RULE: Never identify fugitive as guard, even at extreme angles
                        if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
                            fugitive_dist = face_recognition.face_distance([self.fugitive_face_encoding], unknown_encoding)[0]
                            fugitive_confidence = 1.0 - fugitive_dist
                            
                            # ✅ STRICT SEPARATION: Require LARGE margin to identify as guard
                            # If face matches fugitive within margin, REJECT guard match completely
                            # CRITICAL: When guard and fugitive have similar encodings (both ~0.4-0.5 distance),
                            # we need a MUCH larger margin to separate them reliably
                            margin = getattr(self, 'guard_fugitive_margin', 0.25)  # Increased from 0.18 to 0.25 for better separation
                            
                            # Case 1: Face matches fugitive better - REJECT (fugitive must be significantly worse)
                            if fugitive_dist < best_dist - 0.08:  # Guard must be at least 0.08 better (was 0.05)
                                logger.warning(f"[STRICT REJECT] {name} vs face_{face_idx}: Face is FUGITIVE (fugitive_dist={fugitive_dist:.3f} << guard_dist={best_dist:.3f})")
                                continue
                            
                            # Case 2: Face matches both similarly (within margin) - REJECT (too risky)
                            if abs(best_dist - fugitive_dist) < margin:
                                logger.warning(f"[STRICT REJECT] {name} vs face_{face_idx}: TOO CLOSE to fugitive (guard={best_dist:.3f}, fugitive={fugitive_dist:.3f}, margin={margin:.2f})")
                                continue
                            
                            # Case 3: Guard matches significantly better - OK to proceed
                            logger.debug(f"[GUARD OK] {name} vs face_{face_idx}: Guard confirmed (guard={best_dist:.3f}, fugitive={fugitive_dist:.3f}, gap={fugitive_dist-best_dist:.3f})")
                        
                        # ✅ BALANCED TOLERANCE: Prevent unknown persons while allowing angle variations
                        # Use 0.52 as max tolerance - strict enough to prevent false positives
                        # but relaxed enough for above-angle and side views
                        strict_tolerance = min(base_tolerance, 0.52)  # Balanced for accuracy + angle support
                        
                        # ✅ ANGLE-AWARE BOOST: If multiple angles available and best match is from non-front angle,
                        # allow slightly higher tolerance (guard captured from that exact angle)
                        if len(all_guard_encodings) > 1 and best_angle_idx > 0 and best_dist <= 0.55:
                            # Non-front angle matched - this is likely a valid above/side view
                            strict_tolerance = 0.55  # Allow up to 0.55 for non-front angle matches
                            logger.debug(f"[ANGLE BOOST] {name}: Using relaxed tolerance {strict_tolerance} for angle {best_angle_idx+1}")
                        
                        # ✅ SPATIAL BOOST & GLOBAL RECOVERY
                        # USER REQUEST: "now the script is loosing the guard if guard doing movements"
                        # "and only tring to detect the guard newar last location"
                        
                        spatial_boost = 0.0
                        last_box = self.targets_status[name].get("face_box")
                        is_lost = not self.targets_status[name].get("visible", False)
                        
                        if is_lost:
                            # Base recovery boost - look everywhere when lost
                            # This fixes "only trying to detect near last location"
                            spatial_boost = 0.05  # Global boost (0.52 -> 0.57)
                            
                            if last_box:
                                # Check distance to last known position
                                lx1, ly1, lx2, ly2 = last_box
                                lcx, lcy = (lx1+lx2)/2, (ly1+ly2)/2
                                
                                # Current face center
                                (top, right, bottom, left) = face_locations[face_idx]
                                fcx, fcy = (left+right)/2, (top+bottom)/2
                                
                                dist_px = ((lcx-fcx)**2 + (lcy-fcy)**2)**0.5
                                frame_w = self.unprocessed_frame.shape[1] if self.unprocessed_frame is not None else 1920
                                
                                # If within reasonable movement range (60% of screen width), apply EXTRA boost
                                # Increased from 300px to cover brisk walking across screen
                                if dist_px < (frame_w * 0.6): 
                                    spatial_boost = 0.09  # Stronger boost (0.52 -> 0.61) for likely candidates
                                    logger.debug(f"[SPATIAL BOOST] {name}: Face near last known pos (dist={dist_px:.0f}), boost={spatial_boost}")
                                else:
                                    logger.debug(f"[GLOBAL RECOVERY] {name}: Face far from last pos (dist={dist_px:.0f}), boost={spatial_boost}")

                        effective_tolerance = strict_tolerance + spatial_boost

                        if best_dist <= effective_tolerance:
                            all_matches.append({
                                "face_idx": face_idx,
                                "distance": best_dist,
                                "confidence": confidence,
                                "encoding": unknown_encoding,
                                "best_angle_idx": best_angle_idx
                            })
                            cost_matrix.append((best_dist, target_idx, face_idx, name, confidence, unknown_encoding))
                            if best_dist > 0.45:  # Higher distance - log it
                                logger.debug(f"[MATCH] {name} vs face_{face_idx}: Distance={best_dist:.3f} (tol={effective_tolerance:.2f})")
                        else:
                            logger.debug(f"[SKIP] {name} vs face_{face_idx}: Distance={best_dist:.3f} exceeds tolerance={effective_tolerance:.2f}")
                    
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
                
                # Single-pass: Best-distance-first greedy assignment with stability tracking
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
                        
                        # ✅ ANTI-GHOST: Additional verification - check encoding distance is reasonable
                        # If distance is too high (>0.52), require HIGHER confidence to prevent ghost
                        # RELAXED: Changed from 0.50/0.55 to 0.52/0.50 to allow guards with above-angle views
                        # Guards captured from ceiling cameras often have higher distances (0.48-0.52)
                        if dist > 0.52 and confidence < 0.50:
                            logger.warning(f"[GHOST PREVENTION] {name} -> face_{face_idx}: High distance ({dist:.3f}) + low confidence ({confidence:.3f}) - REJECT to prevent ghost")
                            continue
                        
                        # ✅ INSTANT IDENTIFICATION: Remove identity confirmation delay for untracked guards
                        # When guard is NOT visible, assign IMMEDIATELY (no waiting for confirmation)
                        status = self.targets_status[name]
                        
                        # Only require confirmation if guard is ALREADY being tracked
                        # This ensures fast initial identification while preventing identity switching
                        if status.get("visible", False) and status.get("stable_tracking", False):
                            # Guard already tracked - require confirmation before switching to different face
                            pending_face_idx = status.get("pending_face_idx", None)
                            pending_confirm_count = status.get("pending_confirm_count", 0)
                            confirm_threshold = getattr(self, 'identity_confirmation_frames', 3)  # Reduced from 5 to 3
                            
                            if pending_face_idx == face_idx:
                                # Same face detected again - increment confirmation
                                status["pending_confirm_count"] = pending_confirm_count + 1
                                if status["pending_confirm_count"] < confirm_threshold:
                                    logger.debug(f"[CONFIRM] {name} -> face_{face_idx}: Confirming ({status['pending_confirm_count']}/{confirm_threshold})")
                                    continue  # Don't reassign yet
                                else:
                                    # Confirmed - proceed with assignment
                                    logger.info(f"[CONFIRMED] {name} -> face_{face_idx}: Identity confirmed after {confirm_threshold} frames")
                                    status["pending_confirm_count"] = 0
                                    status["pending_face_idx"] = None
                            else:
                                # Different face - start new confirmation sequence
                                status["pending_face_idx"] = face_idx
                                status["pending_confirm_count"] = 1
                                logger.debug(f"[NEW PENDING] {name} -> face_{face_idx}: Starting confirmation (1/{confirm_threshold})")
                                continue  # Don't switch yet
                        else:
                            # ✅ RE-ENTRY STABILITY: Require confirmation for re-entering guards too
                            # USER REQUEST: "when guard it out of the frame script is identifying random person as guard"
                            # "when guard re-enter the frame script should stable track the guard again"
                            # We enforce a 2-frame confirmation for re-entry to prevent false positives from random people.
                            
                            pending_face_idx = status.get("pending_face_idx", None)
                            pending_confirm_count = status.get("pending_confirm_count", 0)
                            reentry_threshold = 1  # ✅ INSTANT: Identify on first frame appearance (33ms instead of 66ms)
                            
                            if pending_face_idx == face_idx:
                                status["pending_confirm_count"] = pending_confirm_count + 1
                                if status["pending_confirm_count"] < reentry_threshold:
                                    logger.debug(f"[RE-ENTRY] {name} -> face_{face_idx}: Confirming ({status['pending_confirm_count']}/{reentry_threshold})")
                                    continue  # Wait for confirmation
                                else:
                                    # Confirmed
                                    status["pending_confirm_count"] = 0
                                    status["pending_face_idx"] = None
                            else:
                                # First detection of re-entry
                                status["pending_face_idx"] = face_idx
                                status["pending_confirm_count"] = 1
                                logger.debug(f"[RE-ENTRY START] {name} -> face_{face_idx}: Starting confirmation")
                                continue  # Wait for confirmation
                        
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
                    self.targets_status[name]["face_confidence"] = max(0.0, min(1.0, confidence))
                    self.targets_status[name]["ghost_no_pose_frames"] = 0  # Reset ghost counter on successful detection
                    
                    # ✅ INSTANT TRACKING: Mark guard as newly detected for 1-frame detection interval
                    self.targets_status[name]["newly_detected"] = True
                    self.targets_status[name]["detection_frame_count"] = 0
                    
                    # ✅ STABILITY: Clear pending confirmation state on successful assignment
                    self.targets_status[name]["pending_face_idx"] = None
                    self.targets_status[name]["pending_confirm_count"] = 0
                    
                    # ✅ STABILITY: Track consecutive detections for stable tracking
                    self.targets_status[name]["consecutive_detections"] = self.targets_status[name].get("consecutive_detections", 0) + 1
                    if self.targets_status[name]["consecutive_detections"] >= 3:
                        self.targets_status[name]["stable_tracking"] = True
                    self.targets_status[name]["face_match_confidence"] = max(0.0, min(1.0, confidence))
                    
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
        
        # ✅ DRIFT VERIFICATION: Check trackers marked for drift verification
        # This runs AFTER face detection so face_locations and face_encodings are available
        for name, status in self.targets_status.items():
            if status.get("needs_drift_verification", False) and status.get("visible", False):
                # Clear the flag - we're checking now
                status["needs_drift_verification"] = False
                
                # Get the tracker's current position
                tracker_box = status.get("face_box")
                if tracker_box is None:
                    continue
                
                tx1, ty1, tx2, ty2 = tracker_box
                tracker_cx = (tx1 + tx2) / 2
                tracker_cy = (ty1 + ty2) / 2
                
                # Check if any face near tracker position matches the guard
                guard_found_near_tracker = False
                unknown_face_near_tracker = False
                
                # Get guard's encodings for comparison
                multi_angle_encodings = status.get("multi_angle_encodings", [])
                target_encoding = status.get("encoding")
                
                all_guard_encodings = []
                if multi_angle_encodings and len(multi_angle_encodings) > 0:
                    all_guard_encodings = multi_angle_encodings
                elif target_encoding is not None:
                    all_guard_encodings = [target_encoding]
                
                if not all_guard_encodings or len(face_locations) == 0:
                    continue
                
                # Check each face near the tracker
                for face_idx, (top, right, bottom, left) in enumerate(face_locations):
                    face_cx = (left + right) / 2
                    face_cy = (top + bottom) / 2
                    
                    # Check if face is near tracker (within 100 pixels)
                    distance_to_tracker = ((face_cx - tracker_cx) ** 2 + (face_cy - tracker_cy) ** 2) ** 0.5
                    
                    if distance_to_tracker < 100:
                        # Face is near tracker - check if it's the guard
                        if face_idx < len(face_encodings) and face_encodings[face_idx] is not None:
                            face_enc = face_encodings[face_idx]
                            
                            # Compare with all guard angles
                            all_distances = face_recognition.face_distance(all_guard_encodings, face_enc)
                            best_dist = min(all_distances)
                            
                            if best_dist < 0.50:  # Strict tolerance for drift check
                                guard_found_near_tracker = True
                                # Reset drift counter on successful verification
                                status["tracker_drift_counter"] = 0
                                logger.debug(f"[DRIFT CHECK] {name}: Guard confirmed near tracker (dist={best_dist:.3f})")
                                break
                            else:
                                # Face near tracker but NOT the guard - potential drift
                                unknown_face_near_tracker = True
                                logger.debug(f"[DRIFT CHECK] {name}: Unknown face near tracker (dist={best_dist:.3f})")
                
                # If unknown face detected near tracker but guard NOT found - increment drift counter
                if unknown_face_near_tracker and not guard_found_near_tracker:
                    drift_count = status.get("tracker_drift_counter", 0) + 1
                    status["tracker_drift_counter"] = drift_count
                    
                    logger.warning(f"[DRIFT DETECTED] {name}: Tracker may be following unknown person (drift_count={drift_count}/3)")
                    
                    if drift_count >= 3:
                        # HARD RESET - tracker has drifted to unknown person
                        status["tracker"] = None
                        status["body_tracker"] = None
                        status["face_box"] = None
                        status["body_box"] = None
                        status["visible"] = False
                        status["stable_tracking"] = False
                        status["consecutive_detections"] = 0
                        status["tracker_drift_counter"] = 0
                        self.re_detect_counter = 999  # Force immediate re-detection
                        logger.warning(f"[DRIFT RESET] {name}: Tracker drifted to unknown person — HARD RESET triggered")
        
        # ⏭️ TIME DECAY MECHANISM REMOVED
        # Previously: Hard reset after 5 seconds of missing
        # Removed per user request - guards should persist indefinitely while tracked

        # 3. Overlap Check (OPTIMIZED: Reduce frequency for performance)
        # ✅ CRITICAL OPTIMIZATION: Only check overlaps every 5 frames (167ms) instead of every frame
        # Overlap detection is ~15-20ms per frame for multiple guards, so reducing frequency saves significant time
        # At 2+ guards with overlap checks every frame: ~30-40ms overhead
        # With new interval (every 5 frames): ~6-8ms overhead spread across 5 frames = ~1.2-1.6ms per frame savings
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        num_visible_guards = len(active_names)
        
        if num_visible_guards >= 2:
            check_frequency = 5 if num_visible_guards <= 3 else 10
            should_check_overlap = (self.frame_counter % check_frequency == 0)
        else:
            should_check_overlap = False
        
        if should_check_overlap:
            self.targets_status = resolve_overlapping_poses(self.targets_status, iou_threshold=0.2)
        
        if not should_check_overlap and num_visible_guards >= 2:
            for i in range(len(active_names)):
                for j in range(i + 1, len(active_names)):
                    nameA = active_names[i]
                    nameB = active_names[j]
                    
                    boxA = self.targets_status[nameA]["face_box"]
                    boxB = self.targets_status[nameB]["face_box"]
                    
                    if boxA and boxB:
                        rectA = (boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1])
                        rectB = (boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1])
                        iou = calculate_iou(rectA, rectB)
                        
                        if iou > 0.40:
                            conf_a = self.targets_status[nameA].get("face_confidence", 0.5)
                            conf_b = self.targets_status[nameB].get("face_confidence", 0.5)
                            
                            if conf_a > conf_b:
                                self.targets_status[nameB]["tracker"] = None
                                self.targets_status[nameB]["visible"] = False
                            else:
                                self.targets_status[nameA]["tracker"] = None
                                self.targets_status[nameA]["visible"] = False
        
        active_names = [n for n, s in self.targets_status.items() if s["visible"]]
        
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                nameA = active_names[i]
                nameB = active_names[j]
                
                boxA = self.targets_status[nameA]["face_box"]
                boxB = self.targets_status[nameB]["face_box"]
                rectA = (boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1])
                rectB = (boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1])
                
                iou = calculate_iou(rectA, rectB)
                if iou > 0.30:
                    conf_a = self.targets_status[nameA].get("face_confidence", 0.5)
                    conf_b = self.targets_status[nameB].get("face_confidence", 0.5)
                    
                    action_a = self.last_action_cache.get(nameA, "Unknown")
                    action_b = self.last_action_cache.get(nameB, "Unknown")
                    
                    score_a = (conf_a * 0.6) + (0.3 if action_a != "Unknown" else 0.0) + (0.1 * (1 - iou))
                    score_b = (conf_b * 0.6) + (0.3 if action_b != "Unknown" else 0.0) + (0.1 * (1 - iou))
                    
                    if score_a > score_b:
                        self.targets_status[nameB]["tracker"] = None
                        self.targets_status[nameB]["visible"] = False
                        logger.debug(f"Overlap resolved: keeping {nameA} (score: {score_a:.2f}) over {nameB} (score: {score_b:.2f}), IoU: {iou:.2f}")
                    else:
                        self.targets_status[nameA]["tracker"] = None
                        self.targets_status[nameA]["visible"] = False
                        logger.debug(f"Overlap resolved: keeping {nameB} (score: {score_b:.2f}) over {nameA} (score: {score_a:.2f}), IoU: {iou:.2f}")

        required_act = self.active_required_action
        monitor_mode = self.monitor_mode_var.get()
        current_time = time.time()
        min_buffer = max(CONFIG["performance"].get("min_buffer_for_classification", 5), 5)

        for name, status in self.targets_status.items():
            if status["visible"]:
                fx1, fy1, fx2, fy2 = status["face_box"]
                
                use_prev_body_box = False
                prev_body_box = status.get("body_box")
                
                if prev_body_box and status.get("pose_confidence", 0) > 0.3:
                    pbx1, pby1, pbx2, pby2 = prev_body_box
                    if pbx2 > pbx1 and pby2 > pby1:
                        w_p = pbx2 - pbx1
                        h_p = pby2 - pby1
                        bx1 = max(0, int(pbx1 - w_p * 0.1))
                        by1 = max(0, int(pby1 - h_p * 0.1))
                        bx2 = min(frame_w, int(pbx2 + w_p * 0.1))
                        by2 = min(frame_h, int(pby2 + h_p * 0.1))
                        
                        face_cx = (fx1 + fx2) // 2
                        face_cy = (fy1 + fy2) // 2
                        if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by2:
                            use_prev_body_box = True
                            
                            box_h = by2 - by1
                            face_relative_y = face_cy - by1
                            if face_relative_y < (box_h * 0.3):
                                face_h = fy2 - fy1
                                min_standing_bottom = int(fy1 + (face_h * 7.0))
                                if by2 < min_standing_bottom:
                                    by2 = min(frame_h, min_standing_bottom)
                
                if not use_prev_body_box:
                    bx1, by1, bx2, by2 = calculate_body_box((fx1, fy1, fx2, fy2), frame_h, frame_w, expansion_factor=3.0)

                pose_found_in_box = False
                
                if bx1 < bx2 and by1 < by2:
                    crop = frame[by1:by2, bx1:bx2]
                    if crop.size != 0:
                        should_detect_pose = True
                        logger.debug(f"[POSE] {name}: Running pose detection on every frame")
                        
                        if should_detect_pose:
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            
                            if self.pose_model:
                                pose_result = self.pose_model.process(rgb_crop)
                            else:
                                pose_result = None
                            
                            results_crop = pose_result
                            
                            status["last_cached_pose"] = results_crop
                        else:
                            results_crop = status.get("last_cached_pose")
                        
                        current_action = "Unknown"
                        if results_crop is not None and results_crop.pose_landmarks is not None:
                            pose_found_in_box = True
                            status["missing_pose_counter"] = 0
                            
                            pose_landmarks = results_crop.pose_landmarks.landmark
                            visible_count = sum(1 for lm in pose_landmarks if lm.visibility > 0.5)
                            avg_visibility = sum(lm.visibility for lm in pose_landmarks) / len(pose_landmarks)
                            pose_quality = visible_count / len(pose_landmarks)
                            status["pose_confidence"] = pose_quality
                            status["skeleton_keypoints"] = visible_count
                            
                            if "pose_quality_history" not in status:
                                status["pose_quality_history"] = deque(maxlen=10)
                            status["pose_quality_history"].append(pose_quality)
                            
                            min_pose_quality_threshold = 0.20
                            
                            if pose_quality >= min_pose_quality_threshold:
                                raw_action = classify_action(pose_landmarks, crop.shape[0], crop.shape[1])
                                
                                if raw_action != "Unknown" and avg_visibility > 0.4:
                                    status["pose_buffer"].append(raw_action)
                                    status["last_valid_pose"] = raw_action
                                    status["last_valid_pose_time"] = current_time
                                
                                min_buffer = max(CONFIG["performance"].get("min_buffer_for_classification", 5), 5)
                                if len(status["pose_buffer"]) >= min_buffer:
                                    counts = Counter(status["pose_buffer"])
                                    most_common = counts.most_common(1)[0][0]
                                    confidence_pct = counts[most_common] / len(status["pose_buffer"])
                                    
                                    if confidence_pct >= 0.70 and status.get("stable_tracking", False):
                                        current_action = most_common
                                    else:
                                        current_action = status["last_valid_pose"] or "Standing"
                                else:
                                    current_action = status["last_valid_pose"] or "Unknown"
                            else:
                                current_action = status["last_valid_pose"] or "Standing"
                            
                            if current_action != "Unknown":
                                self.last_action_cache[name] = current_action
                            
                            status["current_action"] = current_action
                            
                            if len(status["pose_buffer"]) >= min_buffer:
                                buffer_summary = Counter(status["pose_buffer"])
                                most_common_action = buffer_summary.most_common(1)[0][0]
                                buffer_str = ", ".join([f"{action}:{count}" for action, count in buffer_summary.most_common()])
                                avg_quality = sum(status["pose_quality_history"]) / len(status["pose_quality_history"])
                                logger.debug(f"[POSE] {name}: {current_action} (quality:{pose_quality:.2f}, avg_qual:{avg_quality:.2f}, buffer:[{buffer_str}], consensus:{buffer_summary[most_common_action]/len(status['pose_buffer']):.1%})")

                            if self.is_alert_mode and monitor_mode in ["Action Alerts Only"]:
                                elapsed_time = current_time - status["last_action_time"]
                                grace_period_length = 10
                                grace_period_start = self.alert_interval - grace_period_length
                                
                                in_grace_period_or_overdue = elapsed_time >= grace_period_start
                                
                                if in_grace_period_or_overdue and current_action == self.active_required_action:
                                    status["action_performed"] = True
                                    logger.debug(f"[ACTION] {name}: Action '{self.active_required_action}' performed (elapsed: {elapsed_time:.1f}s)")
                                    
                                    try:
                                        guard_bbox = (fx1, fy1, fx2, fy2)
                                        self.capture_alert_snapshot(frame, name, check_rate_limit=False, bbox=guard_bbox, is_fugitive=False)
                                    except Exception:
                                        pass

                                    self.temp_log.append((
                                        time.strftime("%Y-%m-%d %H:%M:%S"),
                                        name,
                                        f"'{self.active_required_action}' required",
                                        "PERFORMED",
                                        "N/A",
                                        f"{status['pose_confidence']:.2f}"
                                    ))
                                    self.temp_log_counter += 1
                                    
                                    if status["alert_stop_event"] is not None:
                                        status["alert_stop_event"].set()
                                    
                                    status["last_action_time"] = current_time
                                    status["action_performed"] = False
                                    status["alert_logged_timeout"] = False

                                elif elapsed_time >= self.alert_interval:
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
                            
                            if self.is_pro_mode and self.is_stillness_alert:
                                current_time = time.time()
                                current_face_box = status.get("face_box")
                                
                                if current_face_box and results_crop is not None and results_crop.pose_landmarks is not None:
                                    pose_landmarks = results_crop.pose_landmarks.landmark
                                    
                                    key_landmarks = [
                                        11, 12,  # Shoulders
                                        13, 14,  # Elbows
                                        15, 16,  # Wrists (hands)
                                        23, 24,  # Hips
                                    ]
                                    
                                    current_keypoints = []
                                    for idx in key_landmarks:
                                        if idx < len(pose_landmarks):
                                            lm = pose_landmarks[idx]
                                            if lm.visibility > 0.3:
                                                current_keypoints.append((lm.x, lm.y, lm.z))
                                    
                                    previous_keypoints = status.get("last_keypoints_stillness")
                                    
                                    if previous_keypoints is not None and len(current_keypoints) > 0 and len(previous_keypoints) == len(current_keypoints):
                                        total_motion = 0.0
                                        for i in range(len(current_keypoints)):
                                            prev_x, prev_y, prev_z = previous_keypoints[i]
                                            curr_x, curr_y, curr_z = current_keypoints[i]
                                            keypoint_motion = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2) ** 0.5
                                            total_motion += keypoint_motion
                                        
                                        avg_motion = total_motion / len(current_keypoints)
                                        
                                        MOTION_THRESHOLD = 0.04
                                        
                                        if avg_motion > MOTION_THRESHOLD:
                                            status["stillness_start_time"] = None
                                            status["stillness_alert_logged"] = False
                                            
                                            if status.get("alert_sound_thread") and status["alert_sound_thread"].is_alive():
                                                if status.get("alert_stop_event"):
                                                    status["alert_stop_event"].set()
                                            
                                            logger.debug(f"[STILLNESS] {name}: Movement detected (avg_motion={avg_motion:.4f}) - reset timer")
                                        else:
                                            if status["stillness_start_time"] is None:
                                                status["stillness_start_time"] = current_time
                                            elif not status["stillness_alert_logged"]:
                                                stillness_duration = current_time - status["stillness_start_time"]
                                                
                                                if stillness_duration >= 15.0:
                                                    logger.warning(f"[STILLNESS ALERT] !!! {name} - NO MOTION FOR {stillness_duration:.1f} SECONDS - Triggering 15s alert!")
                                                    
                                                    try:
                                                        if status.get("alert_sound_thread") and status["alert_sound_thread"].is_alive():
                                                            if status.get("alert_stop_event"):
                                                                status["alert_stop_event"].set()
                                                                status["alert_sound_thread"].join(timeout=1)
                                                        
                                                        status["alert_stop_event"] = threading.Event()
                                                        status["alert_sound_thread"] = play_siren_sound(
                                                            stop_event=status["alert_stop_event"],
                                                            duration_seconds=15,
                                                            sound_file=get_sound_path("siren.mp3")
                                                        )
                                                        
                                                        self.temp_log.append((
                                                            time.strftime("%Y-%m-%d %H:%M:%S"),
                                                            name,
                                                            "STILLNESS_ALERT",
                                                            f"NO MOTION FOR {stillness_duration:.1f} SECONDS",
                                                            "N/A",
                                                            "1.0"
                                                        ))
                                                        self.temp_log_counter += 1
                                                        self.save_log_to_file()
                                                        
                                                        status["stillness_alert_logged"] = True
                                                    except Exception as e:
                                                        logger.error(f"[STILLNESS] Error triggering alert for {name}: {e}")
                                    
                                    status["last_keypoints_stillness"] = current_keypoints
                                elif results_crop is None or results_crop.pose_landmarks is None:
                                    status["stillness_start_time"] = None
                                    status["last_keypoints_stillness"] = None
                            
                            h_c, w_c = crop.shape[:2]
                            
                            if results_crop and results_crop.pose_landmarks:
                                p_lms = results_crop.pose_landmarks.landmark
                                
                                keypoints = []
                                for i, lm in enumerate(p_lms):
                                    if 0 <= lm.x <= 1 and 0 <= lm.y <= 1 and lm.visibility > 0.3:
                                        x_pixel = int(lm.x * w_c) + bx1
                                        y_pixel = int(lm.y * h_c) + by1
                                        keypoints.append((i, x_pixel, y_pixel, lm.visibility))
                                
                                skeleton_connections = [
                                    (11, 13), (13, 15),
                                    (12, 14), (14, 16),
                                    (11, 12),
                                    (11, 23), (12, 24),
                                    (23, 24),
                                    (23, 25), (25, 27),
                                    (24, 26), (26, 28),
                                ]
                                
                                kpt_dict = {kpt[0]: (kpt[1], kpt[2], kpt[3]) for kpt in keypoints}
                                
                                for start_idx, end_idx in skeleton_connections:
                                    if start_idx in kpt_dict and end_idx in kpt_dict:
                                        start_pt = kpt_dict[start_idx]
                                        end_pt = kpt_dict[end_idx]
                                        avg_visibility = (start_pt[2] + end_pt[2]) / 2
                                        if avg_visibility > 0.5:
                                            color = (0, 255, 0)
                                        else:
                                            color = (0, 165, 255)
                                        cv2.line(frame, (start_pt[0], start_pt[1]), (end_pt[0], end_pt[1]), color, 2)
                                
                                for idx, x, y, visibility in keypoints:
                                    if visibility > 0.5:
                                        radius = 4
                                        color = (0, 255, 0)
                                    else:
                                        radius = 2
                                        color = (0, 165, 255)
                                    cv2.circle(frame, (x, y), radius, color, -1)
                                
                                lx = [lm.x * w_c for lm in p_lms if 0 <= lm.x <= 1]
                                ly = [lm.y * h_c for lm in p_lms if 0 <= lm.y <= 1]
                                
                                if lx and ly and len(lx) >= 2:
                                    d_x1 = int(min(lx)) + bx1
                                    d_y1 = int(min(ly)) + by1
                                    d_x2 = int(max(lx)) + bx1
                                    d_y2 = int(max(ly)) + by1
                                    expand = 20 if pose_quality < 0.4 else 10
                                    d_x1 = max(0, d_x1 - expand)
                                    d_y1 = max(0, d_y1 - expand)
                                    d_x2 = min(frame_w, d_x2 + expand)
                                    d_y2 = min(frame_h, d_y2 + expand)
                                else:
                                    d_x1, d_y1, d_x2, d_y2 = fx1, fy1, fx2, fy2
                            else:
                                d_x1, d_y1, d_x2, d_y2 = fx1, fy1, fx2, fy2
                            
                            if pose_quality >= 0.5:
                                d_x1 = max(0, d_x1 - 10)
                                d_y1 = max(0, d_y1 - 10)
                                d_x2 = min(frame_w, d_x2 + 10)
                                d_y2 = min(frame_h, d_y2 + 10)
                            else:
                                d_x1 = max(0, d_x1 - 30)
                                d_y1 = max(0, d_y1 - 30)
                                d_x2 = min(frame_w, d_x2 + 30)
                                d_y2 = min(frame_h, d_y2 + 30)
                            
                            box_color = (0, 255, 0) if pose_quality >= 0.5 else (0, 165, 255)
                            cv2.rectangle(frame, (d_x1, d_y1), (d_x2, d_y2), box_color, 2)

                            status["body_box"] = (d_x1, d_y1, d_x2, d_y2)
                            status["body_box_quality"] = "sparse" if pose_quality < 0.4 else "normal"
                            if status.get("body_tracker") is None:
                                try:
                                    bt = cv2.TrackerMIL_create()
                                    bt.init(frame, (d_x1, d_y1, d_x2 - d_x1, d_y2 - d_y1))
                                    status["body_tracker"] = bt
                                except Exception:
                                    status["body_tracker"] = None
                            
                            face_conf = status.get("face_confidence", 0.0)
                            pose_conf = status.get("pose_confidence", 0.0)
                            
                            if face_conf > 0.85:
                                id_color = (0, 255, 0)
                                confidence_indicator = "★★★"
                            elif face_conf > 0.65:
                                id_color = (0, 165, 255)
                                confidence_indicator = "★★"
                            else:
                                id_color = (0, 0, 255)
                                confidence_indicator = "★"
                            
                            tracking_status = "✓" if status.get("stable_tracking") else "◇"
                            info_text = f"{tracking_status} {name} ({face_conf:.2f})"
                            action_text = f"{current_action} (P:{pose_conf:.1%})"
                            
                            cv2.putText(frame, info_text, (d_x1, d_y1 - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, id_color, 2)
                            cv2.putText(frame, action_text, (d_x1, d_y1 - 8), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

                if not pose_found_in_box:
                    status["missing_pose_counter"] += 1
                    
                    face_conf = status.get("face_confidence", 0.0)
                    
                    if face_conf > 0.85:
                        id_color = (0, 255, 0)
                    elif face_conf > 0.65:
                        id_color = (0, 165, 255)
                    else:
                        id_color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), id_color, 2)
                    
                    info_text = f"{name} ({face_conf:.2f})"
                    cv2.putText(frame, info_text, (fx1, fy1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, id_color, 2)
                    cv2.putText(frame, "No Pose", (fx1, fy1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                else:
                    if name not in self.last_action_cache:
                        self.last_action_cache[name] = "Unknown"
                    if status["missing_pose_counter"] > 30:
                        status["tracker"] = None
                        status["visible"] = False
            
            if self.is_alert_mode and monitor_mode in ["Action Alerts Only"]:
                if self.alert_interval <= 0:
                    self.alert_interval = 10
                    logger.warning(f"Alert interval was invalid ({self.alert_interval}), reset to 10 seconds")
                
                time_diff = current_time - status["last_action_time"]
                time_left = max(0, self.alert_interval - time_diff)
                y_offset = 50 + (list(self.targets_status.keys()).index(name) * 30)
                color = (0, 255, 0) if time_left > 3 else (0, 0, 255)
                
                status_txt = "OK" if status["visible"] else "MISSING"
                cv2.putText(frame, f"{name} ({status_txt}): {time_left:.1f}s", (frame_w - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if time_diff > self.alert_interval:
                    if (current_time - status["alert_cooldown"]) > 2.5:
                        if status["alert_stop_event"] is None:
                            status["alert_stop_event"] = threading.Event()
                        status["alert_stop_event"].clear()
                        status["alert_sound_thread"] = play_siren_sound(
                            stop_event=status["alert_stop_event"], 
                            duration_seconds=30,
                            sound_file=get_sound_path("siren.mp3")
                        )
                        status["alert_cooldown"] = current_time
                        
                        img_path = "N/A"
                        if status["visible"]:
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
                            snapshot_result = self.capture_alert_snapshot(
                                frame,
                                name,
                                check_rate_limit=True,
                                bbox=None,
                                is_fugitive=False
                            )
                            img_path = snapshot_result if snapshot_result else "N/A"

                        if self.is_logging:
                            if not status["visible"]:
                                if not status.get("target_missing_alert_logged", False):
                                    log_s = "ALERT TRIGGERED - TARGET MISSING"
                                    log_a = "MISSING"
                                    confidence = status.get("face_confidence", 0.0)
                                    self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                                    status["target_missing_alert_logged"] = True
                                    self.temp_log_counter += 1
                                    logger.warning(f"[ALERT] {name} MISSING - Alert triggered!")
                            else:
                                log_s = "ALERT CONTINUED" if status["alert_triggered_state"] else "ALERT TRIGGERED"
                                log_a = self.last_action_cache.get(name, "Unknown")
                                confidence = status.get("face_confidence", 0.0)
                                self.temp_log.append((time.strftime("%Y-%m-%d %H:%M:%S"), name, log_a, log_s, img_path, f"{confidence:.2f}"))
                                status["target_missing_alert_logged"] = False
                            
                            status["alert_triggered_state"] = True
            # ============================================================
                
                # RESET: When action is performed or target reset
                if time_diff <= 0:
                    status["alert_logged_timeout"] = False

        return frame 

if __name__ == "__main__":
    app = PoseApp()
    app.root.mainloop()
