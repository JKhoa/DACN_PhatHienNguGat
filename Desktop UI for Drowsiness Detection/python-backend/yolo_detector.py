import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

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
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    keypoints: List[PoseKeypoint]
    drowsiness_score: float = 0.0
    drowsiness_state: str = "awake"  # awake, drowsy, sleeping
    last_update: float = 0.0

@dataclass
class DetectionResult:
    """Result of YOLO detection on a frame"""
    frame_id: int
    timestamp: float
    persons: List[PersonDetection]
    fps: float = 0.0
    processing_time: float = 0.0

class DrowsinessAnalyzer:
    """Analyzes pose keypoints to determine drowsiness state"""
    
    def __init__(self):
        self.eye_closed_threshold = 0.3
        self.head_tilt_threshold = 30.0  # degrees
        self.drowsiness_history = {}  # person_id -> deque of states
        self.min_drowsiness_frames = 5  # frames to confirm drowsiness
        
    def analyze_person(self, person: PersonDetection) -> PersonDetection:
        """Analyze a person's pose to determine drowsiness state"""
        if len(person.keypoints) < 17:  # COCO pose has 17 keypoints
            return person
            
        # Get keypoint indices (COCO pose format)
        nose_idx = 0
        left_eye_idx = 1
        right_eye_idx = 2
        left_ear_idx = 3
        right_ear_idx = 4
        
        # Calculate drowsiness indicators
        drowsiness_score = 0.0
        indicators = []
        
        # 1. Eye closure detection
        if (left_eye_idx < len(person.keypoints) and right_eye_idx < len(person.keypoints)):
            left_eye = person.keypoints[left_eye_idx]
            right_eye = person.keypoints[right_eye_idx]
            
            if left_eye.visible and right_eye.visible:
                # Simple eye closure detection based on confidence
                eye_confidence = (left_eye.confidence + right_eye.confidence) / 2
                if eye_confidence < self.eye_closed_threshold:
                    drowsiness_score += 0.4
                    indicators.append("eyes_closed")
        
        # 2. Head tilt detection
        if (nose_idx < len(person.keypoints) and 
            left_ear_idx < len(person.keypoints) and 
            right_ear_idx < len(person.keypoints)):
            
            nose = person.keypoints[nose_idx]
            left_ear = person.keypoints[left_ear_idx]
            right_ear = person.keypoints[right_ear_idx]
            
            if (nose.visible and left_ear.visible and right_ear.visible):
                # Calculate head tilt angle
                ear_distance = abs(left_ear.x - right_ear.x)
                nose_offset = abs(nose.x - (left_ear.x + right_ear.x) / 2)
                
                if ear_distance > 0:
                    tilt_ratio = nose_offset / ear_distance
                    tilt_angle = np.arctan(tilt_ratio) * 180 / np.pi
                    
                    if tilt_angle > self.head_tilt_threshold:
                        drowsiness_score += 0.3
                        indicators.append("head_tilted")
        
        # 3. Head position (looking down)
        if nose_idx < len(person.keypoints):
            nose = person.keypoints[nose_idx]
            # If nose is significantly below the center of the bounding box
            bbox_center_y = (person.bbox[1] + person.bbox[3]) / 2
            if nose.y > bbox_center_y + 20:  # nose below center
                drowsiness_score += 0.3
                indicators.append("head_down")
        
        # Update person's drowsiness score
        person.drowsiness_score = min(drowsiness_score, 1.0)
        
        # Determine drowsiness state based on score and history
        person_id = person.id
        if person_id not in self.drowsiness_history:
            self.drowsiness_history[person_id] = []
        
        history = self.drowsiness_history[person_id]
        history.append(drowsiness_score)
        
        # Keep only recent history
        if len(history) > 10:
            history.pop(0)
        
        # Determine state based on recent history
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
    """YOLO-based pose detection for drowsiness monitoring"""
    
    def __init__(self, model_path: str = "yolo11n-pose.pt"):
        self.model_path = model_path
        self.model = None
        self.drowsiness_analyzer = DrowsinessAnalyzer()
        self.person_counter = 0
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO is required. Install with: pip install ultralytics")
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect persons and analyze drowsiness in a frame"""
        start_time = time.time()
        self.frame_counter += 1
        
        # Run YOLO inference
        try:
            results = self.model(frame, verbose=False)
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
                        
                        # Get keypoints
                        if i < len(keypoints.data):
                            kpts = keypoints.data[i].cpu().numpy()
                            
                            # Convert keypoints to PoseKeypoint objects
                            pose_keypoints = []
                            for j in range(0, len(kpts), 3):
                                if j + 2 < len(kpts):
                                    x, y, conf = kpts[j], kpts[j+1], kpts[j+2]
                                    pose_keypoints.append(PoseKeypoint(
                                        x=float(x),
                                        y=float(y),
                                        confidence=float(conf),
                                        visible=conf > 0.5
                                    ))
                            
                            # Create person detection
                            person = PersonDetection(
                                id=self.person_counter,
                                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                confidence=float(confidence),
                                keypoints=pose_keypoints
                            )
                            
                            # Analyze drowsiness
                            person = self.drowsiness_analyzer.analyze_person(person)
                            persons.append(person)
                            
                            self.person_counter += 1
        
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
        """Draw detection results on frame"""
        annotated_frame = frame.copy()
        
        for person in result.persons:
            # Draw bounding box
            x1, y1, x2, y2 = person.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color based on drowsiness state
            if person.drowsiness_state == "sleeping":
                color = (0, 0, 255)  # Red
            elif person.drowsiness_state == "drowsy":
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw person ID and state
            label = f"ID:{person.id} {person.drowsiness_state.upper()}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw drowsiness score
            score_text = f"Score: {person.drowsiness_score:.2f}"
            cv2.putText(annotated_frame, score_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw keypoints
            for kpt in person.keypoints:
                if kpt.visible:
                    cv2.circle(annotated_frame, (int(kpt.x), int(kpt.y)), 3, (255, 0, 0), -1)
        
        # Draw FPS
        fps_text = f"FPS: {result.fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw processing time
        time_text = f"Process: {result.processing_time*1000:.1f}ms"
        cv2.putText(annotated_frame, time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
