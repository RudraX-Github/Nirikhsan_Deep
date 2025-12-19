import cv2
import numpy as np
import face_recognition
import time
import logging
import threading
import os
from collections import deque
from Utils import logger, safe_logger, CONFIG, get_sound_path, play_siren_sound, save_capture_snapshot, load_guard_angle_images, detect_faces_multiscale_distance, remove_duplicate_faces, match_body_silhouette

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

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

def smooth_bounding_box(current_box, previous_box, smoothing_factor=0.7):
    """
    Apply exponential moving average smoothing to bounding box to reduce jitter.
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

def validate_skeleton_quality(landmarks, visibility_threshold=0.35, min_keypoints=10):
    """
    ✅ ENHANCED: Comprehensive skeleton validation with quality scoring.
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

def extract_appearance_features(frame, face_box):
    """
    Extract appearance features from a person's image using color and edge histograms.
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
        if not MEDIAPIPE_AVAILABLE:
            return "Unknown"
        
        mp_holistic = mp.solutions.holistic

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

class TrackerEngine:
    def __init__(self):
        self.targets_status = {}
        self.pose_model = None
        self.mp_face_detection = None
        self.model_pipeline_initialized = False
        self.re_detect_counter = 0
        self.fugitive_face_encoding = None
        self.fugitive_name = None
        self.is_fugitive_detection = False
        self.fugitive_currently_visible = False
        self.fugitive_logged_once = False
        self.fugitive_alert_sound_thread = None
        self.fugitive_alert_stop_event = None
        self.guard_fugitive_margin = 0.12
        self.temp_log = []
        self.temp_log_counter = 0
        
        self._initialize_model_pipeline()

    def _initialize_model_pipeline(self):
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

    def detect_faces_fast(self, frame):
        """Fast face detection using MediaPipe"""
        try:
            if not MEDIAPIPE_AVAILABLE or not hasattr(self, 'mp_face_detection'):
                # Fallback to HOG if MediaPipe not available
                return face_recognition.face_locations(frame, model="hog")
                
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
            return face_locations
        except Exception as e:
            logger.debug(f"Fast face detection error: {e}")
            return []

    def capture_alert_snapshot(self, frame, target_name, check_rate_limit=False, bbox=None, is_fugitive=False):
        """
        Capture alert snapshot of FULL FRAME with highlighted bounding box.
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

    def process_frame(self, frame, known_face_encodings, known_face_names, process_this_frame=True):
        """
        Main processing loop: Face Detection -> Recognition -> Body Tracking -> Action Classification -> Alerts
        """
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 1. Face Detection & Recognition ---
        face_locations = []
        face_names = []
        face_confidences = []
        
        # Only run heavy face detection every N frames or if we need to re-detect
        if process_this_frame:
            face_locations = self.detect_faces_fast(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for face_encoding in face_encodings:
                    name = "Unknown"
                    confidence = 0.0
                    
                    # Check against known guards
                    if known_face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                                confidence = 1.0 - face_distances[best_match_index]
                            else:
                                # If not a match, confidence is low
                                confidence = 1.0 - min(face_distances) if len(face_distances) > 0 else 0.0
                    
                    # Check against Fugitive (Priority)
                    if self.is_fugitive_detection and self.fugitive_face_encoding is not None:
                        match = face_recognition.compare_faces([self.fugitive_face_encoding], face_encoding, tolerance=0.50)
                        if match[0]:
                            name = self.fugitive_name # Use raw name, label added later
                            confidence = 0.99
                            
                            if not self.fugitive_currently_visible:
                                logger.critical(f"!!! FUGITIVE DETECTED: {name} !!!")
                                self.capture_alert_snapshot(frame, name, is_fugitive=True)
                                play_siren_sound()
                            
                            self.fugitive_currently_visible = True
                    
                    face_names.append(name)
                    face_confidences.append(confidence)
        
        # --- 2. Update Targets (Guards & Fugitive) ---
        active_targets = set()
        
        for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidences):
            if name == "Unknown":
                continue
                
            # Initialize if new
            if name not in self.targets_status:
                self.targets_status[name] = {
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "pose_history": deque(maxlen=30),
                    "action_history": deque(maxlen=15),
                    "alert_cooldown": 0,
                    "face_box": (left, top, right, bottom),
                    "body_box": None,
                    "missing_frames": 0,
                    "is_tracked": True,
                    "pose_quality_history": deque(maxlen=10),
                    "last_snapshot_time": 0,
                    "is_fugitive": (name == self.fugitive_name)
                }
            
            target = self.targets_status[name]
            target["last_seen"] = time.time()
            target["face_box"] = (left, top, right, bottom)
            target["missing_frames"] = 0
            target["is_tracked"] = True
            
            # Update body box
            target["body_box"] = calculate_body_box((left, top, right, bottom), frame_h, frame_w)
            
            active_targets.add(name)
            
            # Draw Face Box & Name
            color = (0, 0, 255) if target.get("is_fugitive") else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Handle lost targets
        for name in list(self.targets_status.keys()):
            if name not in active_targets:
                self.targets_status[name]["missing_frames"] += 1
                # If lost for too long, stop tracking
                if self.targets_status[name]["missing_frames"] > 30:
                    self.targets_status[name]["is_tracked"] = False
                
                # If tracked but face lost, try to approximate body from last known
                if self.targets_status[name]["is_tracked"] and self.targets_status[name]["body_box"]:
                    # Keep the body box same or predict movement (simple: keep same)
                    pass

        # --- 3. Pose Estimation & Action Recognition ---
        if self.model_pipeline_initialized:
            for name, target in self.targets_status.items():
                if not target["is_tracked"] or not target["body_box"]:
                    continue
                
                # Extract ROI for Pose
                bx1, by1, bx2, by2 = target["body_box"]
                pad_x = int((bx2 - bx1) * 0.2)
                pad_y = int((by2 - by1) * 0.2)
                roi_x1 = max(0, bx1 - pad_x)
                roi_y1 = max(0, by1 - pad_y)
                roi_x2 = min(frame_w, bx2 + pad_x)
                roi_y2 = min(frame_h, by2 + pad_y)
                
                roi = rgb_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi.size == 0: continue
                
                # Run MediaPipe Pose
                results = self.pose_model.process(roi)
                
                if results.pose_landmarks:
                    # Draw Skeleton on Frame
                    # Need to adjust landmarks from ROI to Full Frame for drawing
                    # But draw_landmarks expects normalized coords (0-1)
                    
                    # Create a copy of landmarks for full frame
                    full_frame_landmarks = mp.solutions.pose.PoseLandmark
                    # This is tricky with the MP object. 
                    # Easier to draw on ROI and paste back? No, that overwrites background.
                    # We'll manually draw or adjust the landmarks object.
                    
                    # Let's just extract landmarks for logic first
                    adjusted_landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        abs_x = roi_x1 + lm.x * (roi_x2 - roi_x1)
                        abs_y = roi_y1 + lm.y * (roi_y2 - roi_y1)
                        lm.x = abs_x / frame_w
                        lm.y = abs_y / frame_h
                        adjusted_landmarks.append(lm)
                    
                    # Validate
                    validation = validate_skeleton_quality(adjusted_landmarks)
                    
                    if validation['valid']:
                        # Draw Skeleton
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, 
                            results.pose_landmarks, # These are now modified to full frame!
                            mp.solutions.pose.POSE_CONNECTIONS
                        )
                        
                        # Classify Action
                        action = classify_action(adjusted_landmarks, frame_h, frame_w)
                        target["action_history"].append(action)
                        
                        # Smooth Action
                        recent_actions = list(target["action_history"])
                        if recent_actions:
                            smoothed_action = Counter(recent_actions).most_common(1)[0][0]
                        else:
                            smoothed_action = "Unknown"
                        
                        # Display Action
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
                        cv2.putText(frame, f"Action: {smoothed_action}", (bx1, by2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Check Alerts
                        if smoothed_action in ["Hands Up", "Hands Crossed", "T-Pose", "One Hand Raised (Left)", "One Hand Raised (Right)"]:
                            if time.time() - target["alert_cooldown"] > 5:
                                logger.warning(f"ALERT: {name} performed {smoothed_action}")
                                self.capture_alert_snapshot(frame, name, bbox=(bx1, by1, bx2, by2))
                                target["alert_cooldown"] = time.time()
                                play_siren_sound()
                                
                        # Special Fugitive Alert
                        if target.get("is_fugitive"):
                             cv2.putText(frame, "FUGITIVE DETECTED", (50, 50), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return frame
