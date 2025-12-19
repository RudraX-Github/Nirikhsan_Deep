import cv2
import numpy as np
import os
import sys
import torch
import subprocess
import re
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from tkinter import Tk, filedialog

# ====================================================================
# --- CONFIGURATION ---
# ====================================================================

CAMERA_CONFIG = {
    "username": "admin",
    "password": "Demo12345678",
    "ip": "192.168.1.111",
    "port": "554"
}

# TUNING
MATCH_THRESHOLD = 0.60      
MAX_LOST_FRAMES = 30        

# HARDWARE CHECK
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    PROVIDERS = ['CUDAExecutionProvider']
else:
    print("⚠️  CRITICAL: GPU not found.")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
POSE_MODEL_NAME = "yolov8x-pose-p6.engine" 
POSE_MODEL_PATH = os.path.join(script_dir, POSE_MODEL_NAME)

SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

# ====================================================================
# --- HELPER: AUTO DETECT RESOLUTION ---
# ====================================================================

def get_stream_resolution(rtsp_url):
    """Uses ffprobe to detect the exact resolution of the stream."""
    print("🕵️  Probing camera resolution...")
    command = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'stream=width,height', 
        '-of', 'csv=s=x:p=0', 
        rtsp_url
    ]
    try:
        output = subprocess.check_output(command).decode('utf-8').strip()
        width, height = map(int, output.split('x'))
        print(f"✅ Camera Detected: {width}x{height}")
        return width, height
    except Exception as e:
        print(f"❌ Error probing camera: {e}")
        print("   Defaulting to 1280x720 (Might fail)")
        return 1280, 720

# ====================================================================
# --- GPU-ACCELERATED VIDEO CLASS ---
# ====================================================================

class FFmpegGPUReader:
    def __init__(self, rtsp_url):
        # 1. Auto-detect resolution first
        self.width, self.height = get_stream_resolution(rtsp_url)
        self.frame_size = self.width * self.height * 3 
        
        # 2. Launch FFmpeg
        command = [
            'ffmpeg',
            '-loglevel', 'error',
            '-hwaccel', 'cuda',         # GPU Decoding
            '-hwaccel_output_format', 'cuda', 
            '-rtsp_transport', 'tcp',   # TCP for stability
            '-fflags', 'nobuffer',      # Low latency
            '-flags', 'low_delay',
            '-i', rtsp_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',        # OpenCV Format
            '-'                         # Pipe to stdout
        ]
        
        try:
            print("🚀 Launching GPU Pipe...")
            # Increased bufsize to handle network hiccups
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**7)
        except Exception as e:
            print(f"FFmpeg Error: {e}")
            sys.exit(1)

    def read(self):
        # Read exact number of bytes
        raw_frame = self.process.stdout.read(self.frame_size)
        
        if len(raw_frame) != self.frame_size:
            print(f"⚠️  Incomplete read: Got {len(raw_frame)} bytes, expected {self.frame_size}")
            # Flushing helps recover from a bad packet
            self.process.stdout.flush()
            return False, None
            
        image = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, image

    def release(self):
        if self.process:
            self.process.terminate()

# ====================================================================
# --- LOGIC FUNCTIONS ---
# ====================================================================

def get_video_source():
    while True:
        print("\n===================================")
        print("   SELECT VIDEO SOURCE")
        print("===================================")
        print("1. Standard Webcam (USB)")
        print("2. CP Plus IP Camera (RTSP - GPU Accelerated)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            return 0
        elif choice == '2':
            user = CAMERA_CONFIG["username"]
            pwd  = CAMERA_CONFIG["password"]
            ip   = CAMERA_CONFIG["ip"]
            port = CAMERA_CONFIG["port"]
            # Subtype 0 = Main Stream (High Res)
            # Subtype 1 = Sub Stream (Low Res, Faster)
            # Try 1 if 0 is too slow/laggy
            rtsp = f"rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0"
            return rtsp

def compute_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None: return 0.0
    return np.dot(embedding1, embedding2)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def is_face_in_body(face_box, body_box):
    fx1, fy1, fx2, fy2 = face_box
    bx1, by1, bx2, by2 = body_box
    fcx = (fx1 + fx2) / 2
    fcy = (fy1 + fy2) / 2
    return (bx1 < fcx < bx2) and (by1 < fcy < by2)

def classify_action(keypoints):
    nose_y = keypoints[0][1]
    left_wrist_y, right_wrist_y = keypoints[9][1], keypoints[10][1]
    left_hip_y, right_hip_y = keypoints[11][1], keypoints[12][1]
    left_knee_y, right_knee_y = keypoints[13][1], keypoints[14][1]

    if left_hip_y == 0 or left_knee_y == 0: return "TRACKING"

    if (left_wrist_y > 0 and left_wrist_y < nose_y) or \
       (right_wrist_y > 0 and right_wrist_y < nose_y):
        return "HANDS UP!"

    thigh_len = ((left_knee_y - left_hip_y) + (right_knee_y - right_hip_y)) / 2
    torso_len = abs(((left_hip_y - keypoints[5][1]) + (right_hip_y - keypoints[6][1])) / 2)
    
    if torso_len > 0:
        ratio = thigh_len / torso_len
        if ratio < 0.6: return "SITTING"
        elif ratio > 0.6: return "STANDING"
    return "UNKNOWN"

def get_target_embedding_interactive():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select Target Person Image")
    if not file_path: sys.exit(0)
    
    img = cv2.imread(file_path)
    if img is None: sys.exit(1)

    print("\n--- INSTRUCTIONS ---")
    print("1. Draw a box around the Target's FACE.")
    print("2. Press ENTER.")
    r = cv2.selectROI("Select Target Face", img, showCrosshair=True)
    cv2.destroyAllWindows()
    
    x, y, w, h = [int(i) for i in r]
    if w == 0 or h == 0: sys.exit(0)

    face_crop = img[y:y+h, x:x+w]
    results = face_app.get(face_crop)
    
    if len(results) == 0:
        print("⚠️  Checking full image context...")
        all_faces = face_app.get(img)
        roi_center = (x + w/2, y + h/2)
        best_face = None
        min_dist = float('inf')
        for face in all_faces:
            box = face.bbox
            fc = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            dist = ((fc[0]-roi_center[0])**2 + (fc[1]-roi_center[1])**2)**0.5
            if dist < max(w, h) and dist < min_dist:
                min_dist = dist
                best_face = face
        if best_face: results = [best_face]

    if len(results) == 0:
        print("❌ FAILED. No face found. Try a clearer photo.")
        sys.exit(1)
    
    target_face = max(results, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    print(f"✅ Target ID Secured.")
    return target_face.embedding

# ====================================================================
# --- MAIN LOOP ---
# ====================================================================

# 1. INIT
if not os.path.isfile(POSE_MODEL_PATH):
    print(f"[ERROR] Engine file missing: {POSE_MODEL_PATH}")
    sys.exit(1)
pose_model = YOLO(POSE_MODEL_PATH, task='pose') 

try:
    print("🚀 Loading Identity System...")
    face_app = FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
    face_app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    print(f"[ERROR] InsightFace init failed: {e}")
    sys.exit(1)

# 2. SETUP
video_source = get_video_source()
target_embedding = get_target_embedding_interactive()

WINDOW_NAME = "Titan-Class Auto-Res"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 3. INITIALIZE CAPTURE
if isinstance(video_source, str):
    cap = FFmpegGPUReader(video_source) # Auto-detects size now
    is_rtsp = True
else:
    cap = cv2.VideoCapture(0)
    is_rtsp = False

locked_body_box = None 
frames_since_verification = 0

print("\n--- SURVEILLANCE ACTIVE ---")

while True:
    if is_rtsp:
        ret, frame = cap.read()
    else:
        ret, frame = cap.read()

    if not ret or frame is None:
        continue

    # --- INFERENCE ---
    face_results = face_app.get(frame)
    pose_results = pose_model(frame, verbose=False)
    
    bodies_detected = []
    keypoints_detected = []
    if pose_results[0].boxes is not None:
        bodies_detected = pose_results[0].boxes.xyxy.cpu().numpy()
        keypoints_detected = pose_results[0].keypoints.xy.cpu().numpy()

    # --- MATCHING ---
    current_body_index = -1
    highest_face_score = 0.0
    matched_face_box = None

    for face in face_results:
        score = compute_similarity(target_embedding, face.embedding)
        fx = int(face.bbox[0])
        fy = int(face.bbox[1])
        cv2.putText(frame, f"{score:.2f}", (fx, fy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        if score > MATCH_THRESHOLD and score > highest_face_score:
            highest_face_score = score
            matched_face_box = face.bbox.astype(int)

    if matched_face_box is not None:
        for i, body_box in enumerate(bodies_detected):
            if is_face_in_body(matched_face_box, body_box):
                current_body_index = i
                locked_body_box = body_box
                frames_since_verification = 0
                break
    
    elif locked_body_box is not None and frames_since_verification < MAX_LOST_FRAMES:
        best_iou = 0
        best_iou_index = -1
        for i, body_box in enumerate(bodies_detected):
            iou = calculate_iou(locked_body_box, body_box)
            if iou > best_iou:
                best_iou = iou
                best_iou_index = i
        
        if best_iou > 0.4:
            current_body_index = best_iou_index
            locked_body_box = bodies_detected[best_iou_index]
            frames_since_verification += 1
        else:
            frames_since_verification = MAX_LOST_FRAMES + 1

    # --- DRAWING ---
    if current_body_index != -1:
        is_verified = (frames_since_verification == 0)
        color = (0, 255, 0) if is_verified else (0, 165, 255)
        
        kpts = keypoints_detected[current_body_index]
        action = classify_action(kpts)
        
        for idx_a, idx_b in SKELETON_CONNECTIONS:
            pt_a = (int(kpts[idx_a][0]), int(kpts[idx_a][1]))
            pt_b = (int(kpts[idx_b][0]), int(kpts[idx_b][1]))
            if pt_a[0] > 0 and pt_a[1] > 0 and pt_b[0] > 0 and pt_b[1] > 0:
                cv2.line(frame, pt_a, pt_b, color, 3)
        
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        bx1, by1, bx2, by2 = map(int, bodies_detected[current_body_index])
        status_text = f"ID: {highest_face_score*100:.0f}%" if is_verified else "TRACKING"
        label = f"{action} [{status_text}]"
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (bx1, by1 - 30), (bx1 + w, by1), color, -1)
        cv2.putText(frame, label, (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
    else:
        if frames_since_verification > MAX_LOST_FRAMES:
            locked_body_box = None

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_rtsp: cap.release()
cv2.destroyAllWindows()
