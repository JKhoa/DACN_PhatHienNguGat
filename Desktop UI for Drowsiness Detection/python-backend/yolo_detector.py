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
        neck = (nose[0], nose[1] - img_h * 0.12); shoulder_w = img_w * 0.2
    dx = nose[0] - neck[0]; dy = nose[1] - neck[1]
    angle_v = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    drop_pix = dy
    drop_h_ratio = float(drop_pix) / max(img_h, 1)
    drop_sw_ratio = float(drop_pix) / max(shoulder_w, 1e-6)
    if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65:
        return "Gục xuống bàn", float(angle_v), float(drop_h_ratio)
    if angle_v > angle_thr or drop_h_ratio > drop_h_thr or drop_sw_ratio > drop_sw_thr:
        return "Ngủ gật", float(angle_v), float(drop_h_ratio)
    return "Bình thường", float(angle_v), float(drop_h_ratio)

class DrowsinessAnalyzer:
    """Analyzes pose detections to determine drowsiness state
    
    Supports two modes:
    1. CLASS-BASED (trained model): Uses model class predictions (binhthuong, ngugat, gucxuongban)
    2. KEYPOINT-BASED (pretrained model): Analyzes keypoints manually
    """
    
    def __init__(self, use_class_predictions: bool = True):
        """
        Args:
            use_class_predictions: If True, use model class predictions. If False, analyze keypoints manually.
        """
        self.use_class_predictions = use_class_predictions
        self.eye_closed_threshold = 0.3
        self.head_tilt_threshold = 30.0  # degrees
        self.drowsiness_history = {}  # person_id -> deque of states
        self.min_drowsiness_frames = 3  # frames to confirm drowsiness (faster response)
        
        # Class name mappings for trained model
        self.class_to_state = {
            'binhthuong': 'awake',
            'ngugat': 'drowsy', 
            'gucxuongban': 'sleeping',
            # Fallback for English names
            'awake': 'awake',
            'drowsy': 'drowsy',
            'sleeping': 'sleeping',
        }
        
    def analyze_person(self, person: PersonDetection, class_name: str = None) -> PersonDetection:
        """Analyze a person's pose to determine drowsiness state
        
        Args:
            person: PersonDetection object
            class_name: Model-predicted class name (if available from trained model)
        
        Returns:
            Updated PersonDetection with drowsiness_state and drowsiness_score
        """
        
        # MODE 1: Use class predictions from trained model (PREFERRED)
        if self.use_class_predictions and class_name:
            # Convert Vietnamese class name to English state
            state = self.class_to_state.get(class_name.lower(), 'awake')
            
            # Set state and score based on class
            person.drowsiness_state = state
            if state == 'sleeping':
                person.drowsiness_score = 0.9
            elif state == 'drowsy':
                person.drowsiness_score = 0.6
            else:  # awake
                person.drowsiness_score = 0.1
                
            # Additional confidence boost from detection confidence
            if hasattr(person, 'confidence'):
                person.drowsiness_score = person.drowsiness_score * 0.7 + person.confidence * 0.3
        
        # MODE 2: Fallback to keypoint-based analysis (for pretrained models)
        else:
            if len(person.keypoints) < 7:  # Need at least nose and shoulders
                person.drowsiness_state = "awake"
                person.drowsiness_score = 0.1
                return person
            
            # Convert keypoints to numpy array format for classify_pose_custom
            k = np.array([[kpt.x, kpt.y] for kpt in person.keypoints[:7]])
            
            # Get image dimensions from bounding box
            img_h = int(person.bbox[3] - person.bbox[1])
            img_w = int(person.bbox[2] - person.bbox[0])
            
            # Use enhanced classification logic
            state_text, angle_v, drop_h_ratio = classify_pose_custom(k, img_h, img_w)
            
            # Convert Vietnamese state to English and calculate score
            if state_text == "Gục xuống bàn":
                person.drowsiness_state = "sleeping"
                person.drowsiness_score = 0.9
            elif state_text == "Ngủ gật":
                person.drowsiness_state = "drowsy"
                person.drowsiness_score = 0.6
            else:
                person.drowsiness_state = "awake"
                person.drowsiness_score = 0.1
            
            # Additional eye closure detection for more accuracy
            if len(person.keypoints) >= 3:  # Have eyes
                left_eye = person.keypoints[1]
                right_eye = person.keypoints[2]
                
                if left_eye.visible and right_eye.visible:
                    eye_confidence = (left_eye.confidence + right_eye.confidence) / 2
                    if eye_confidence < self.eye_closed_threshold:
                        person.drowsiness_score = min(person.drowsiness_score + 0.3, 1.0)
                        if person.drowsiness_state == "awake":
                            person.drowsiness_state = "drowsy"
        
        # Update history for stability
        person_id = person.id
        if person_id not in self.drowsiness_history:
            self.drowsiness_history[person_id] = []
        
        history = self.drowsiness_history[person_id]
        history.append(person.drowsiness_score)
        
        # Keep only recent history
        if len(history) > 10:
            history.pop(0)
        
        # Smooth the state based on history for stability
        if len(history) >= self.min_drowsiness_frames:
            avg_score = sum(history[-self.min_drowsiness_frames:]) / self.min_drowsiness_frames
            
            if avg_score > 0.7:
                person.drowsiness_state = "sleeping"
            elif avg_score > 0.4:
                person.drowsiness_state = "drowsy"
            else:
                person.drowsiness_state = "awake"
        
        person.last_update = time.time()
        return person

class YOLODetector:
    """YOLO-based pose detection for drowsiness monitoring with head-focused tracking"""
    
    def __init__(self, model_path: str = None):
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
        self.tracker = HeadFocusedTracker(iou_threshold=0.3, max_age=30, head_iou_threshold=0.25)
        self.person_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        # Inference params (runtime adjustable)
        self.current_conf: float = 0.05  # VERY LOW threshold for maximum sensitivity (detect even far/small persons)
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
            # Use runtime-adjustable confidence/imgsz, default tuned for sensitivity
            # Pass device/precision hints; Ultralytics handles placement internally
            results = self.model(
                frame,
                conf=self.current_conf,
                imgsz=self.current_imgsz,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
            
            # Log detection results periodically
            if self.frame_counter % 30 == 0:
                logging.info(f"[YOLO] Frame {self.frame_counter}: conf={self.current_conf}, imgsz={self.current_imgsz}, device={self.device}")
                
        except Exception as e:
            logging.error(f"YOLO inference failed: {e}")
            return DetectionResult(
                frame_id=self.frame_counter,
                timestamp=time.time(),
                persons=[]
            )
        
        persons = []
        
        # Process results
        for result in results:
            if result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        # Get bounding box
                        box = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        
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
                                        visible=conf > 0.5
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
                                keypoints=pose_keypoints
                            )
                            
                            # Analyze drowsiness (pass class_name if available from trained model)
                            person = self.drowsiness_analyzer.analyze_person(person, class_name=class_name)
                            persons.append(person)
                            
                            self.person_counter += 1
        
        # Update tracker to assign persistent IDs
        persons = self.tracker.update(persons)
        
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
        """Draw detection results on frame with head-focused tracking boxes"""
        annotated_frame = frame.copy()
        
        for person in result.persons:
            # Use head_bbox if available (smaller, focused), otherwise use body bbox
            if hasattr(person, 'head_bbox') and person.head_bbox and person.head_bbox[0] > 0:
                x1, y1, x2, y2 = person.head_bbox
            else:
                x1, y1, x2, y2 = person.bbox
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color based on drowsiness state
            if person.drowsiness_state == "sleeping":
                color = (0, 0, 255)  # Red (BGR)
            elif person.drowsiness_state == "drowsy":
                color = (0, 165, 255)  # Orange (BGR)
            else:
                color = (0, 255, 0)  # Green (BGR)
            
            # Draw head-focused bounding box (thinner line for smaller boxes)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw center point circle
            cv2.circle(annotated_frame, (center_x, center_y), 4, color, -1)
            cv2.circle(annotated_frame, (center_x, center_y), 4, (255, 255, 255), 1)
            
            # Use track_id if available
            track_id = getattr(person, 'track_id', None) or person.id
            
            # Draw person ID (top of box)
            id_label = f"#{track_id}"
            id_font_scale = 0.5
            id_thickness = 1
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
                state_font_scale = 0.4
                state_thickness = 1
                state_label_size, _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, state_font_scale, state_thickness)
                
                # Background for state label
                cv2.rectangle(annotated_frame,
                             (center_x - state_label_size[0] // 2 - 3, y2),
                             (center_x + state_label_size[0] // 2 + 3, y2 + state_label_size[1] + 6),
                             color, -1)
                cv2.putText(annotated_frame, state_label,
                           (center_x - state_label_size[0] // 2, y2 + state_label_size[1] + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, state_font_scale, (255, 255, 255), state_thickness)
        
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
