import cv2
import time
import threading
import numpy as np
import random
from Utils import logger, safe_logger

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
            self.cap.set(prop, value)
    
    def get(self, prop):
        """Get camera property (compatibility with cv2.VideoCapture)."""
        if self.cap:
            return self.cap.get(prop)
        return 0

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

def apply_clahe_enhancement(frame):
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

_clahe_cache = None

def enhance_frame_for_low_light(frame, night_mode=False):
    """
    Fast frame enhancement with night/day mode awareness
    Day mode: Skip enhancement entirely (fast)
    Night mode: Lightweight enhancement only if brightness is low
    """
    global _clahe_cache
    try:
        if not night_mode:
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 100:
            return frame
        
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        if _clahe_cache is None:
            _clahe_cache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_clahe = _clahe_cache.apply(l_channel)
        lab_enhanced = cv2.merge([l_channel_clahe, a, b])
        frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return frame_enhanced
    except Exception as e:
        logger.debug(f"Low-light enhancement failed: {e}, using original frame")
        return frame

class CameraManager:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.cap = None
        self.is_running = False
        self.is_ip_camera = isinstance(camera_source, str) and (camera_source.startswith("rtsp://") or camera_source.startswith("http://"))
        self.threaded_cam = None
        self.night_mode = False

    def start(self):
        if self.is_running:
            return True

        try:
            if self.is_ip_camera:
                self.threaded_cam = ThreadedIPCamera(self.camera_source)
                self.threaded_cam.start()
                self.cap = self.threaded_cam
            else:
                # Ensure camera index is int if it's a number string
                if isinstance(self.camera_source, str) and self.camera_source.isdigit():
                    self.camera_source = int(self.camera_source)
                    
                self.cap = cv2.VideoCapture(self.camera_source)
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera source: {self.camera_source}")
                    return False
                
                # Set default resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False

    def stop(self):
        self.is_running = False
        if self.is_ip_camera and self.threaded_cam:
            self.threaded_cam.stop()
            self.threaded_cam = None
        elif self.cap:
            self.cap.release()
        self.cap = None

    def get_frame(self):
        if not self.is_running or not self.cap:
            return None

        try:
            if self.is_ip_camera:
                ret, frame = self.threaded_cam.read()
            else:
                ret, frame = self.cap.read()

            if ret and frame is not None:
                if self.night_mode:
                    return enhance_frame_for_low_light(frame, night_mode=True)
                return frame
            return None
        except Exception as e:
            logger.error(f"Get frame error: {e}")
            return None
            
    def set_night_mode(self, enabled):
        self.night_mode = enabled

