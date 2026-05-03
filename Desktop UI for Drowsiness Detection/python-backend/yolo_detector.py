import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import os

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

@dataclass
class PoseKeypoint:
    """Represents a single pose keypoint"""
    x: float
    y: float
    confidence: float
    visible: bool = True

@dataclass
class PersonDetection:
    """Represents a detected person with pose keypoints"""
    id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (full body)
    head_bbox: Tuple[float, float, float, float] = field(default_factory=lambda: (0, 0, 0, 0))  # x1, y1, x2, y2 (head only)
    confidence: float = 0.0
    keypoints: List[PoseKeypoint] = field(default_factory=list)
    drowsiness_score: float = 0.0
    drowsiness_state: str = "awake"  # awake, drowsy, sleeping
    last_update: float = 0.0
    track_id: Optional[int] = None  # Persistent tracking ID across frames

@dataclass
class DetectionResult:
    """Result of YOLO detection on a frame"""
    frame_id: int
    timestamp: float
    persons: List[PersonDetection]
    fps: float = 0.0
    processing_time: float = 0.0

def calculate_head_bbox(keypoints: List[PoseKeypoint], body_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Calculate head bounding box from keypoints, focused on head region to avoid overlapping"""
    x1, y1, x2, y2 = body_bbox
    body_height = y2 - y1
    body_width = x2 - x1
    
    # Find head-related keypoints (nose, eyes, ears)
    head_keypoints = []
    if len(keypoints) >= 5:
        # COCO pose: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        for idx in [0, 1, 2, 3, 4]:
            if idx < len(keypoints) and keypoints[idx].visible and keypoints[idx].confidence > 0.3:
                head_keypoints.append((keypoints[idx].x, keypoints[idx].y))
    
    if len(head_keypoints) > 0:
        # Use head keypoints to calculate head bbox
        head_x_coords = [kp[0] for kp in head_keypoints]
        head_y_coords = [kp[1] for kp in head_keypoints]
        
        # Head region: top 20-25% of body height
        head_x1 = min(head_x_coords) - body_width * 0.08
        head_y1 = y1 - body_height * 0.05  # Slightly above body top
        head_x2 = max(head_x_coords) + body_width * 0.08
        head_y2 = y1 + body_height * 0.25  # Top 25% of body
        
        # Ensure head bbox is within body bbox horizontally
        head_x1 = max(x1, head_x1)
        head_x2 = min(x2, head_x2)
        head_y1 = max(y1 - body_height * 0.05, y1)
        head_y2 = min(head_y2, y1 + body_height * 0.3)
    else:
        # Fallback: estimate head region (top 25% of body, centered horizontally)
        head_x1 = x1 + body_width * 0.2
        head_y1 = y1
        head_x2 = x2 - body_width * 0.2
        head_y2 = y1 + body_height * 0.25
    
    return (head_x1, head_y1, head_x2, head_y2)

def iou_xyxy(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes in xyxy format"""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    # Calculate intersection
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

class HeadFocusedTracker:
    """Multi-object tracker focused on head regions to avoid overlapping boxes"""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, head_iou_threshold: float = 0.25):
        """
        Args:
            iou_threshold: IoU threshold for matching (using head bbox)
            max_age: Maximum frames to keep a track without detection
            head_iou_threshold: Lower threshold for head bbox matching (more lenient)
        """
        self.iou_threshold = iou_threshold
        self.head_iou_threshold = head_iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Dict] = {}  # track_id -> track data
        self.next_id = 1
        
    def update(self, detections: List[PersonDetection]) -> List[PersonDetection]:
        """Update tracker with new detections, assign persistent IDs"""
        # Age all tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            
            # Remove old tracks
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # Match detections to tracks using head bbox IoU
        assignments: Dict[int, int] = {}  # track_id -> detection index
        used_detections = set()
        
        # Greedy matching: find best matches first
        while True:
            best_match = None
            best_iou = 0.0
            
            for track_id, track in self.tracks.items():
                if track_id in assignments:
                    continue
                if track['age'] > self.max_age:
                    continue
                    
                track_head_bbox = track.get('head_bbox', track.get('bbox'))
                
                for det_idx, detection in enumerate(detections):
                    if det_idx in used_detections:
                        continue
                    
                    # Use head bbox for matching if available, otherwise use body bbox
                    det_head_bbox = detection.head_bbox if detection.head_bbox[0] > 0 else detection.bbox
                    
                    iou = iou_xyxy(track_head_bbox, det_head_bbox)
                    
                    if iou > best_iou and iou > self.head_iou_threshold:
                        best_match = (track_id, det_idx, iou)
                        best_iou = iou
            
            if best_match is None:
                break
            
            track_id, det_idx, iou = best_match
            assignments[track_id] = det_idx
            used_detections.add(det_idx)
            
            # Update track with new detection
            detection = detections[det_idx]
            self.tracks[track_id]['bbox'] = detection.bbox
            self.tracks[track_id]['head_bbox'] = detection.head_bbox
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['last_update'] = time.time()
            detection.track_id = track_id
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx in used_detections:
                continue
            
            track_id = self.next_id
            self.next_id += 1
            
            head_bbox = detection.head_bbox if detection.head_bbox[0] > 0 else detection.bbox
            self.tracks[track_id] = {
                'bbox': detection.bbox,
                'head_bbox': head_bbox,
                'age': 0,
                'last_update': time.time()
            }
            detection.track_id = track_id
        
        return detections

def classify_pose_custom(k: np.ndarray, img_h: int, img_w: int, angle_thr: float = 15.0, drop_h_thr: float = 0.15, drop_sw_thr: float = 0.45):
    """Enhanced pose classification from gui_app.py"""
    if len(k) < 7:
        return "Bình thường", 0.0, 0.0
    nose, l_sh, r_sh = k[0], k[5], k[6]
    def valid(p):
        return p[0] > 0 and p[1] > 0
    have_l, have_r = valid(l_sh), valid(r_sh)
    if have_l and have_r:
        neck = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        shoulder_w = float(np.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1]))
    elif have_l:
        neck = (l_sh[0], l_sh[1]); shoulder_w = img_w * 0.18
    elif have_r:
        neck = (r_sh[0], r_sh[1]); shoulder_w = img_w * 0.18
    else:
        # CORRECTED: Neck should be BELOW the nose in image coordinates (Y increases down)
        neck = (nose[0], nose[1] + img_h * 0.12); shoulder_w = img_w * 0.2
    dx = nose[0] - neck[0]; dy = nose[1] - neck[1]
    angle_v = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    
    # Calculate drop ratios
    drop_pix = float(max(0, dy))  # Only count downward movement for "drop"
    drop_h_ratio = drop_pix / max(img_h, 1)
    drop_sw_ratio = drop_pix / max(shoulder_w, 1e-6)
    
    if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65:
        logging.debug(f"[POSE] → Gục xuống bàn (drop_h={drop_h_ratio:.3f}, drop_sw={drop_sw_ratio:.3f})")
        return "Gục xuống bàn", float(angle_v), float(drop_h_ratio)
    
    if angle_v > angle_thr or drop_h_ratio > drop_h_thr or drop_sw_ratio > drop_sw_thr:
        logging.debug(f"[POSE] → Ngủ gật (angle={angle_v:.1f}°, drop_h={drop_h_ratio:.3f}, drop_sw={drop_sw_ratio:.3f})")
        return "Ngủ gật", float(angle_v), float(drop_h_ratio)
    
    logging.debug(f"[POSE] → Bình thường")
    return "Bình thường", float(angle_v), float(drop_h_ratio)

class DrowsinessAnalyzer:
    """Analyzes pose detections to determine drowsiness state
    
    Supports two modes:
    1. CLASS-BASED (trained model): Uses model class predictions (binhthuong, ngugat, gucxuongban)
    2. KEYPOINT-BASED (pretrained model): Analyzes keypoints manually
    
    Enhanced with TIME-BASED temporal smoothing (2-3s continuous head drop)
    """
    
    def __init__(self, use_class_predictions: bool = True):
        """
        Args:
            use_class_predictions: If True, use model class predictions. If False, analyze keypoints manually.
        """
        self.use_class_predictions = use_class_predictions
        # Disable eye tracking - not reliable with classroom cameras
        self.use_eye_tracking = False
        
        # TIME-BASED drowsiness tracking (person_id -> tracking info)
        self.drowsiness_tracking = {}  # person_id -> {
                                        #   'state_history': [(timestamp, state), ...],
                                        #   'current_state': 'awake'|'drowsy'|'sleeping',
                                        #   'drowsy_since': timestamp or None,
                                        #   'sleeping_since': timestamp or None,
                                        #   'last_logged': timestamp or None
                                        # }
        
        # Time thresholds (seconds) - tuned to giảm false positive (cúi viết bài) và nhấp nháy state
        self.drowsy_time_threshold = 4.0    # Phải cúi đầu liên tục >= 4s mới coi là drowsy
        self.sleeping_time_threshold = 3.0  # Phải gục >= 3s mới coi là sleeping
        self.history_window = 1.5           # Cửa sổ smoothing 1.5s
        self.log_cooldown = 2.0             # Cooldown ngắn hơn để không bỏ sót sự kiện liên tiếp
        self.stale_track_timeout = 30.0     # Xoá tracking entry sau 30s không thấy person
        self.exit_hysteresis = 1.0          # Đang drowsy/sleeping: cần >=1s liên tục awake mới thật sự thoát
        
        # Class name mappings for trained model
        self.class_to_state = {
            'binhthuong': 'awake',
            'ngugat': 'drowsy',  # Slight head drop
            'gucxuongban': 'sleeping',  # Full head down on desk
            # Fallback for English names
            'awake': 'awake',
            'drowsy': 'drowsy',
            'sleeping': 'sleeping',
        }
        
        # State confidence scores
        self.state_scores = {
            'awake': 0.1,
            'drowsy': 0.6,
            'sleeping': 0.9
        }
        
    def analyze_person(self, person: PersonDetection, class_name: Optional[str] = None) -> PersonDetection:
        """Analyze a person's pose to determine drowsiness state with temporal smoothing
        
        Args:
            person: PersonDetection object
            class_name: Model-predicted class name (if available from trained model)
        
        Returns:
            Updated PersonDetection with drowsiness_state and drowsiness_score
        """
        
        # Get raw prediction from model or keypoint analysis
        raw_state = 'awake'
        raw_score = 0.1
        
        # MODE 1: Use class predictions from trained model (PREFERRED)
        # CRITICAL: Only use class predictions if class_name is a CUSTOM drowsiness class
        # Pretrained models return "person" which is NOT useful for drowsiness detection
        custom_classes = ['binhthuong', 'ngugat', 'gucxuongban', 'awake', 'drowsy', 'sleeping']
        is_custom_class = class_name and class_name.lower() in custom_classes
        
        if self.use_class_predictions and is_custom_class:
            # Map Vietnamese class to state
            predicted_state = self.class_to_state.get(class_name.lower(), 'awake')
            
            # CRITICAL FIX: Be more conservative
            # - 'ngugat' (slight head drop) could be writing → treat as awake unless confirmed
            # - 'gucxuongban' (head fully down) → likely sleeping
            if predicted_state == 'drowsy':
                # Require more evidence for drowsy state
                raw_state = 'drowsy'
                raw_score = 0.5  # Lower initial score
            elif predicted_state == 'sleeping':
                raw_state = 'sleeping'
                raw_score = 0.9
            else:
                raw_state = 'awake'
                raw_score = 0.1
                
            # Boost confidence if detection confidence is high
            if hasattr(person, 'confidence') and person.confidence > 0.8:
                raw_score = raw_score * 0.8 + person.confidence * 0.2
        
        # MODE 2: Fallback to keypoint-based analysis (for pretrained models)
        else:
            if len(person.keypoints) >= 7:
                # Skip nếu các keypoint then chốt (mũi + 2 vai) confidence quá thấp,
                # tránh fallback (0,0) gây nhầm "gục xuống bàn" khi quay mặt/khuất.
                key_kpts = person.keypoints[:7]
                core_idx = [0, 5, 6]  # nose, left shoulder, right shoulder
                core_ok = all(
                    i < len(key_kpts)
                    and key_kpts[i].confidence >= 0.3
                    and key_kpts[i].x >= 0
                    and key_kpts[i].y >= 0
                    for i in core_idx
                )

                if not core_ok:
                    raw_state = "awake"
                    raw_score = 0.1
                else:
                    k = np.array([[kpt.x, kpt.y] for kpt in key_kpts])
                    img_h = int(person.bbox[3] - person.bbox[1])
                    img_w = int(person.bbox[2] - person.bbox[0])

                    state_text, angle_v, drop_h_ratio = classify_pose_custom(
                        k, img_h, img_w,
                        angle_thr=25.0,
                        drop_h_thr=0.12,
                        drop_sw_thr=0.40
                    )

                    if state_text == "Gục xuống bàn":
                        raw_state = "sleeping"
                        raw_score = 0.9
                    elif state_text == "Ngủ gật":
                        raw_state = "drowsy"
                        raw_score = 0.5
                    else:
                        raw_state = "awake"
                        raw_score = 0.1
        
        # TIME-BASED TRACKING: Track continuous head drop duration
        person_id = person.track_id if person.track_id is not None else person.id
        current_time = time.time()
        
        # Initialize tracking for new person
        if person_id not in self.drowsiness_tracking:
            self.drowsiness_tracking[person_id] = {
                'state_history': [],
                'current_state': 'awake',
                'drowsy_since': None,
                'sleeping_since': None,
                'awake_since': None,   # mốc bắt đầu giai đoạn awake (để hysteresis exit)
                'last_logged': None,
                'last_seen': current_time,
            }

        tracking = self.drowsiness_tracking[person_id]
        tracking['last_seen'] = current_time

        # Cleanup stale tracking entries để tránh memory growth + state cũ.
        # Chạy mỗi ~5s thay vì mọi frame để không tốn CPU.
        if not hasattr(self, '_last_stale_cleanup'):
            self._last_stale_cleanup = current_time
        if current_time - self._last_stale_cleanup > 5.0:
            stale = [pid for pid, tr in self.drowsiness_tracking.items()
                     if current_time - tr.get('last_seen', current_time) > self.stale_track_timeout]
            for pid in stale:
                del self.drowsiness_tracking[pid]
            if stale:
                logging.info(f"Cleaned {len(stale)} stale tracking entries")
            self._last_stale_cleanup = current_time
        
        # Add current state to history
        tracking['state_history'].append((current_time, raw_state))
        
        # Remove old history (keep only last 1 second)
        tracking['state_history'] = [
            (t, s) for t, s in tracking['state_history'] 
            if current_time - t <= self.history_window
        ]
        
        # Count states in recent history for smoothing
        if len(tracking['state_history']) >= 3:  # Need at least 3 samples
            recent_states = [s for _, s in tracking['state_history']]
            awake_count = recent_states.count('awake')
            drowsy_count = recent_states.count('drowsy')
            sleeping_count = recent_states.count('sleeping')
            
            # Determine dominant state (with smoothing) - siết voting để giảm nhấp nháy
            if sleeping_count >= len(recent_states) * 0.7:   # >= 70% sleeping
                dominant_state = 'sleeping'
            elif drowsy_count >= len(recent_states) * 0.65:  # >= 65% drowsy
                dominant_state = 'drowsy'
            elif awake_count >= len(recent_states) * 0.5:    # >= 50% awake
                dominant_state = 'awake'
            else:
                dominant_state = raw_state  # Use raw if no clear dominant
        else:
            dominant_state = raw_state
        
        # Update continuous state tracking (có hysteresis exit để tránh nhấp nháy)
        if dominant_state == 'sleeping':
            if tracking['sleeping_since'] is None:
                tracking['sleeping_since'] = current_time
            tracking['drowsy_since'] = None
            tracking['awake_since'] = None  # đang sleeping, reset bộ đếm awake
        elif dominant_state == 'drowsy':
            if tracking['drowsy_since'] is None:
                tracking['drowsy_since'] = current_time
            tracking['sleeping_since'] = None
            tracking['awake_since'] = None
        else:  # dominant_state == 'awake'
            # Hysteresis: nếu đang ở drowsy/sleeping, KHÔNG reset timer ngay
            # mà đợi awake duy trì >= exit_hysteresis giây mới thật sự thoát.
            if tracking['current_state'] in ('drowsy', 'sleeping'):
                if tracking['awake_since'] is None:
                    tracking['awake_since'] = current_time
                awake_duration = current_time - tracking['awake_since']
                if awake_duration >= self.exit_hysteresis:
                    # Đã thật sự thoát: reset toàn bộ timer
                    tracking['drowsy_since'] = None
                    tracking['sleeping_since'] = None
                    tracking['awake_since'] = None
                    tracking['current_state'] = 'awake'
                # else: giữ nguyên drowsy_since / sleeping_since, current_state
            else:
                # Đang awake bình thường: chỉ reset bộ đếm timer ngủ gật
                tracking['drowsy_since'] = None
                tracking['sleeping_since'] = None
                tracking['awake_since'] = None
        
        # Make final decision based on continuous duration
        final_state = 'awake'
        final_score = 0.1
        
        # Check sleeping state (3s continuous)
        if tracking['sleeping_since'] is not None:
            sleeping_duration = current_time - tracking['sleeping_since']
            if sleeping_duration >= self.sleeping_time_threshold:
                final_state = 'sleeping'
                final_score = 0.9
                
                # Log sleeping detection (if not recently logged)
                if tracking['last_logged'] is None or \
                   (current_time - tracking['last_logged']) > self.log_cooldown:
                    logging.warning(
                        f"🚨 PHÁT HIỆN NGỦ GẬT: Người #{person_id} đã cúi đầu liên tục "
                        f"{sleeping_duration:.1f}s - Trạng thái: GỤC XUỐNG BÀN"
                    )
                    tracking['last_logged'] = current_time
                    tracking['current_state'] = 'sleeping'
            else:
                # Still in transition - need more time
                final_state = tracking['current_state']
                final_score = self.state_scores.get(final_state, 0.1)
        
        # Check drowsy state (4s continuous)
        elif tracking['drowsy_since'] is not None:
            drowsy_duration = current_time - tracking['drowsy_since']
            if drowsy_duration >= self.drowsy_time_threshold:
                final_state = 'drowsy'
                final_score = 0.6
                
                # Log drowsy detection (if not recently logged)
                if tracking['last_logged'] is None or \
                   (current_time - tracking['last_logged']) > self.log_cooldown:
                    logging.warning(
                        f"⚠️ PHÁT HIỆN NGỦ GẬT: Người #{person_id} đã cúi đầu liên tục "
                        f"{drowsy_duration:.1f}s - Trạng thái: NGỦ GẬT"
                    )
                    tracking['last_logged'] = current_time
                    tracking['current_state'] = 'drowsy'
            else:
                # Still in transition - need more time
                final_state = tracking['current_state']
                final_score = self.state_scores.get(final_state, 0.1)
        
        else:
            # Awake state
            final_state = 'awake'
            final_score = 0.1
            tracking['current_state'] = 'awake'
            final_score = 0.1
        
        # Set final state
        person.drowsiness_state = final_state
        person.drowsiness_score = final_score
        
        return person

class YOLODetector:
    """YOLO-based pose detection for drowsiness monitoring with head-focused tracking"""
    
    def __init__(self, model_path: Optional[str] = None):
        # Try to load trained model first, then fallback to default
        if model_path is None:
            # Check for trained models in order of preference
            import os
            here = os.path.dirname(__file__)
            
            # Priority: Use custom trained models for drowsiness detection
            trained_model_paths = [
                # Local trained models in models directory (HIGHEST PRIORITY)
                os.path.join(here, 'models', 'sleepy_pose_v11n_full_best.pt'),
                os.path.join(here, 'models', 'sleepy_pose_v11n3_best.pt'),
                # Backup: check old workspace
                os.path.join(here, '..', '..', 'DACN_PhatHienNguGat_Old', 'yolo-sleepy-allinone-final', 'runs', 'pose-train', 'sleepy_pose_v11n_full', 'weights', 'best.pt'),
                os.path.join(here, '..', '..', 'DACN_PhatHienNguGat_Old', 'yolo-sleepy-allinone-final', 'runs', 'pose-train', 'sleepy_pose_v11n3', 'weights', 'best.pt'),
            ]
            
            model_path = "yolo11n-pose.pt"  # Default fallback if no trained model found
            for path in trained_model_paths:
                if os.path.exists(path):
                    model_path = path
                    logging.info(f"✅ Using TRAINED drowsiness detection model: {path}")
                    break
            else:
                logging.warning(f"⚠️ No trained model found, using default pretrained model: {model_path}")
        self.model_path = model_path
        self.model = None
        self.drowsiness_analyzer = DrowsinessAnalyzer()
        # PERFORMANCE FIX: Use faster custom tracker or disable for YOLO built-in
        self.use_custom_tracker = False  # Set to True to use HeadFocusedTracker
        self.tracker = HeadFocusedTracker(iou_threshold=0.3, max_age=30, head_iou_threshold=0.25) if self.use_custom_tracker else None
        self.person_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        # Inference params (runtime adjustable)
        self.current_conf: float = 0.20  # Cân bằng: bỏ qua người quá xa/mờ để giảm false positive
        # Use smaller default input size for speed on CPU; can be adjusted at runtime
        self.current_imgsz: int = 640
        # Device/precision
        self.device = 'cpu'
        self.use_half = False
        
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO is required. Install with: pip install ultralytics")
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model with fallback chain and choose best device."""
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            logging.warning(f"Failed to load model {self.model_path}: {e}")
            # Try fallback models
            fallback_models = ["yolo11n-pose.pt", "yolov8n-pose.pt"]
            for fallback in fallback_models:
                try:
                    self.model = YOLO(fallback)
                    logging.info(f"Using fallback model: {fallback}")
                    self.model_path = fallback
                    break
                except Exception:
                    continue
            if self.model is None:
                raise ImportError(f"Failed to load any YOLO model. Tried: {self.model_path} and fallbacks")

        # Select device and precision
        try:
            if torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
                self.device = 'cuda'
                # Half precision can speed up inference on supported GPUs
                self.use_half = True
                logging.info("Using CUDA for YOLO inference (half precision enabled)")
            else:
                self.device = 'cpu'
                self.use_half = False
                logging.info("Using CPU for YOLO inference")
        except Exception:
            self.device = 'cpu'
            self.use_half = False
            logging.info("Using CPU for YOLO inference")
    
    def set_params(self, conf: Optional[float] = None, imgsz: Optional[int] = None):
        """Update inference parameters at runtime."""
        try:
            if conf is not None:
                # Clamp to sane range [0.05, 0.9]
                conf = float(conf)
                conf = max(0.05, min(conf, 0.9))
                self.current_conf = conf
            if imgsz is not None:
                imgsz = int(imgsz)
                if imgsz in (320, 480, 640, 800, 960, 1280):
                    self.current_imgsz = imgsz
        except Exception:
            # Ignore bad params
            pass

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect persons and analyze drowsiness in a frame"""
        start_time = time.time()
        self.frame_counter += 1
        
        # Run YOLO inference
        try:
            # PERFORMANCE FIX: Use YOLO built-in tracker for better speed
            # ByteTrack is much faster than custom HeadFocusedTracker
            if self.use_custom_tracker:
                # Old method: inference only, manual tracking
                results = self.model(
                    frame,
                    conf=self.current_conf,
                    iou=0.45,  # NMS IoU - tránh duplicate bbox cùng 1 người
                    imgsz=self.current_imgsz,
                    device=self.device,
                    half=self.use_half,
                    verbose=False,
                )
            else:
                # NEW: Use YOLO's built-in tracker (ByteTrack) - MUCH FASTER
                results = self.model.track(
                    frame,
                    conf=self.current_conf,
                    iou=0.45,  # NMS IoU explicit
                    imgsz=self.current_imgsz,
                    device=self.device,
                    half=self.use_half,
                    verbose=False,
                    persist=True,  # Persist tracks across frames
                    tracker="bytetrack.yaml"  # Use ByteTrack (fast & accurate)
                )
            
            # Log detection results periodically
            if self.frame_counter % 30 == 0:
                logging.info(f"[YOLO] Frame {self.frame_counter}: conf={self.current_conf}, imgsz={self.current_imgsz}, device={self.device}, tracker={'custom' if self.use_custom_tracker else 'bytetrack'}")
                
        except Exception as e:
            logging.error(f"YOLO inference failed: {e}")
            return DetectionResult(
                frame_id=self.frame_counter,
                timestamp=time.time(),
                persons=[]
            )
        
        persons: List[PersonDetection] = []
        # Temporarily collect detections and their class names so we can run tracking first
        pending: List[Tuple[PersonDetection, Optional[str]]] = []

        # Process results: build detections first (without final drowsiness state)
        for result in results:
            if result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints

                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        # Get bounding box
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        
                        # PERFORMANCE FIX: Get track ID from YOLO built-in tracker
                        track_id = None
                        if not self.use_custom_tracker and hasattr(boxes, 'id') and boxes.id is not None and i < len(boxes.id):
                            track_id = int(boxes.id[i].cpu().numpy())

                        # Get class name from trained model (if available)
                        class_name = None
                        if hasattr(boxes, 'cls') and boxes.cls is not None and i < len(boxes.cls):
                            class_id = int(boxes.cls[i].cpu().numpy())
                            if hasattr(self.model, 'names') and class_id in self.model.names:
                                class_name = self.model.names[class_id]
                                if self.frame_counter % 30 == 0 and i == 0:  # Log first detection periodically
                                    logging.info(f"[YOLO] Detected class: {class_name} (id={class_id}, conf={confidence:.2f})")

                        # Get keypoints
                        if i < len(keypoints.data):
                            kpts = keypoints.data[i].cpu().numpy()

                            # Flatten if 2D (shape: [17, 3] -> [51])
                            if kpts.ndim > 1:
                                kpts = kpts.flatten()

                            # Convert keypoints to PoseKeypoint objects
                            pose_keypoints = []
                            for j in range(0, len(kpts), 3):
                                if j + 2 < len(kpts):
                                    x, y, conf = float(kpts[j]), float(kpts[j+1]), float(kpts[j+2])
                                    pose_keypoints.append(PoseKeypoint(
                                        x=x,
                                        y=y,
                                        confidence=conf,
                                        visible=conf > 0.3  # thống nhất với pose classifier
                                    ))

                            # Create person detection with body bbox
                            body_bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

                            # Calculate head bbox from keypoints
                            head_bbox = calculate_head_bbox(pose_keypoints, body_bbox)

                            person = PersonDetection(
                                id=self.person_counter,
                                bbox=body_bbox,
                                head_bbox=head_bbox,
                                confidence=float(confidence),
                                keypoints=pose_keypoints,
                                track_id=track_id  # PERFORMANCE FIX: Use YOLO tracker ID
                            )

                            pending.append((person, class_name))
                            self.person_counter += 1

        # PERFORMANCE FIX: Use YOLO built-in tracker or custom tracker
        if self.use_custom_tracker and pending:
            # Old method: Use custom HeadFocusedTracker (slower)
            tracked_persons = self.tracker.update([p for p, _ in pending])
        else:
            # NEW: Track IDs already assigned by YOLO ByteTrack (faster!)
            tracked_persons = [p for p, _ in pending]

        # Now analyze drowsiness using stable track_id
        for idx, person in enumerate(tracked_persons):
            # pending list preserves ordering; guard against mismatch length
            class_name = None
            if idx < len(pending):
                _, class_name = pending[idx]
            analyzed = self.drowsiness_analyzer.analyze_person(person, class_name=class_name)
            analyzed.last_update = time.time()
            persons.append(analyzed)

        # Use track_id as person id if available
        for person in persons:
            if person.track_id is not None:
                person.id = person.track_id
        
        # Log detection results periodically
        if self.frame_counter % 30 == 0:
            logging.info(f"[YOLO] Detected {len(persons)} persons in frame {self.frame_counter}")
            if len(persons) > 0:
                for p in persons[:3]:  # Log first 3 persons
                    logging.info(f"  Person {p.id}: bbox={p.bbox}, state={p.drowsiness_state}, conf={p.confidence:.2f}")
        
        # Calculate FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            frame_id=self.frame_counter,
            timestamp=time.time(),
            persons=persons,
            fps=self.current_fps,
            processing_time=processing_time
        )
    
    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw detection results on frame with depth-aware scaling (adaptive box size based on distance)"""
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        for person in result.persons:
            # Use head_bbox if available (smaller, focused), otherwise use body bbox
            if hasattr(person, 'head_bbox') and person.head_bbox and person.head_bbox[0] > 0:
                x1, y1, x2, y2 = person.head_bbox
            else:
                x1, y1, x2, y2 = person.bbox
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 🔥 DEPTH-AWARE SCALING: Calculate relative distance from bbox size
            # Larger bbox = closer to camera, smaller bbox = farther from camera
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            frame_area = frame_width * frame_height
            
            # Normalized size (0.0 to 1.0)
            # 0.0 = very far (tiny box), 1.0 = very close (huge box)
            bbox_ratio = bbox_area / frame_area
            
            # Estimate relative depth (0-100 scale)
            # Closer person = higher depth score, farther = lower depth score
            if bbox_ratio > 0.3:  # Very close (>30% of frame)
                depth_level = 5  # Closest
                depth_text = "Very Close"
            elif bbox_ratio > 0.15:  # Close (15-30% of frame)
                depth_level = 4
                depth_text = "Close"
            elif bbox_ratio > 0.05:  # Medium (5-15% of frame)
                depth_level = 3
                depth_text = "Medium"
            elif bbox_ratio > 0.02:  # Far (2-5% of frame)
                depth_level = 2
                depth_text = "Far"
            else:  # Very far (<2% of frame)
                depth_level = 1
                depth_text = "Very Far"
            
            # 🎨 Scale visual elements based on depth
            # Line thickness: 1-4 pixels based on distance
            line_thickness = max(1, min(4, depth_level))
            
            # Font scale: 0.3-0.7 based on distance
            base_font_scale = 0.3 + (depth_level - 1) * 0.1  # 0.3 for far, 0.7 for close
            
            # Circle radius: 2-6 pixels based on distance
            circle_radius = max(2, min(6, depth_level + 1))
            
            # Choose color based on drowsiness state
            if person.drowsiness_state == "sleeping":
                color = (0, 0, 255)  # Red (BGR)
            elif person.drowsiness_state == "drowsy":
                color = (0, 165, 255)  # Orange (BGR)
            else:
                color = (0, 255, 0)  # Green (BGR)
            
            # Draw bounding box with adaptive thickness (thicker for closer persons)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw center point circle with adaptive radius (larger for closer persons)
            cv2.circle(annotated_frame, (center_x, center_y), circle_radius, color, -1)
            cv2.circle(annotated_frame, (center_x, center_y), circle_radius, (255, 255, 255), 1)
            
            # Use track_id if available
            track_id = getattr(person, 'track_id', None) or person.id
            
            # Draw person ID with depth indicator (top of box)
            # Format: "#ID [Depth]" e.g. "#1 [Close]"
            id_label = f"#{track_id}"
            id_font_scale = base_font_scale + 0.2  # Slightly larger for ID
            id_thickness = max(1, line_thickness - 1)  # Slightly thinner than box
            id_label_size, _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, id_thickness)
            
            # Background for ID label
            cv2.rectangle(annotated_frame, 
                         (center_x - id_label_size[0] // 2 - 4, y1 - id_label_size[1] - 8),
                         (center_x + id_label_size[0] // 2 + 4, y1),
                         (0, 0, 0), -1)
            cv2.putText(annotated_frame, id_label, 
                       (center_x - id_label_size[0] // 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, (255, 255, 255), id_thickness)
            
            # Draw drowsiness state only if not awake (bottom of box)
            if person.drowsiness_state != "awake":
                state_label = person.drowsiness_state.upper()
                state_font_scale = base_font_scale  # Scale with depth
                state_thickness = max(1, line_thickness - 1)
                state_label_size, _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, state_font_scale, state_thickness)
                
                # Background for state label
                padding = max(2, depth_level)  # Adaptive padding
                cv2.rectangle(annotated_frame,
                             (center_x - state_label_size[0] // 2 - padding, y2),
                             (center_x + state_label_size[0] // 2 + padding, y2 + state_label_size[1] + padding * 2),
                             color, -1)
                cv2.putText(annotated_frame, state_label,
                           (center_x - state_label_size[0] // 2, y2 + state_label_size[1] + padding - 1),
                           cv2.FONT_HERSHEY_SIMPLEX, state_font_scale, (255, 255, 255), state_thickness)
            
            # 🆕 DEPTH INDICATOR: Show estimated distance (optional - can be disabled)
            # Draw small depth badge next to ID label
            depth_badge = f"[{depth_text}]"
            depth_font_scale = base_font_scale * 0.6  # Smaller than ID
            depth_thickness = 1
            depth_badge_size, _ = cv2.getTextSize(depth_badge, cv2.FONT_HERSHEY_SIMPLEX, depth_font_scale, depth_thickness)
            
            # Position: right side of ID label
            depth_x = center_x + id_label_size[0] // 2 + 6
            depth_y = y1 - 4
            
            # Only show if there's enough space (don't overlap with edge)
            if depth_x + depth_badge_size[0] + 4 < frame_width:
                # Background
                cv2.rectangle(annotated_frame,
                             (depth_x - 2, y1 - depth_badge_size[1] - 8),
                             (depth_x + depth_badge_size[0] + 2, y1),
                             (80, 80, 80), -1)  # Dark gray background
                # Text
                cv2.putText(annotated_frame, depth_badge,
                           (depth_x, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, depth_font_scale, (200, 200, 200), depth_thickness)  # Light gray text
        
        # Draw FPS
        fps_text = f"FPS: {result.fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Draw processing time
        time_text = f"Process: {result.processing_time*1000:.1f}ms"
        cv2.putText(annotated_frame, time_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated_frame, time_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame

# Global detector instance
_detector: Optional[YOLODetector] = None

def get_detector() -> Optional[YOLODetector]:
    """Get the global YOLO detector instance"""
    return _detector

def initialize_detector(model_path: str = "yolo11n-pose.pt") -> bool:
    """Initialize the global YOLO detector"""
    global _detector
    try:
        _detector = YOLODetector(model_path)
        logging.info("YOLO detector initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize YOLO detector: {e}")
        return False

def detect_frame(frame: np.ndarray) -> DetectionResult:
    """Detect drowsiness in a frame using the global detector"""
    if _detector is None:
        return DetectionResult(
            frame_id=0,
            timestamp=time.time(),
            persons=[]
        )
    return _detector.detect(frame)

def draw_detections(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw detection results on frame using the global detector"""
    if _detector is None:
        return frame
    return _detector.draw_detections(frame, result)

# Helper to update detector settings from server/UI
def update_detection_settings(conf: Optional[float] = None, imgsz: Optional[int] = None):
    global _detector
    if _detector is None:
        return
    try:
        _detector.set_params(conf=conf, imgsz=imgsz)
        if conf is not None:
            logging.info(f"Updated detector confidence to {float(conf):.3f}")
    except Exception as e:
        logging.warning(f"Failed to update detection settings: {e}")
