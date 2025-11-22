"""
Backup Server with Enhanced Multi-Student Tracking
Tích hợp tracking logic từ yolo-sleepy-allinone-final/gui_app.py
Tối ưu cho IP camera với 20+ học sinh
"""

import threading
import time
import base64
import io
import logging
import os
from typing import Dict, Optional, List, Tuple
from collections import deque

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, make_response, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

# Import YOLO detector
try:
    from yolo_detector import (
        initialize_detector, detect_frame, draw_detections, 
        DetectionResult, PersonDetection, get_detector, update_detection_settings
    )
    YOLO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"YOLO detector not available: {e}")
    YOLO_AVAILABLE = False

# Import Drowsiness Logger
try:
    from drowsiness_logger import (
        MultiCameraLogger, get_global_logger, init_logger
    )
    LOGGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Drowsiness logger not available: {e}")
    LOGGER_AVAILABLE = False

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    ping_interval=10,
    ping_timeout=30,
)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


# ==================== Enhanced Tracker từ gui_app.py ====================

def iou_xyxy(box1: Tuple[float, float, float, float], 
             box2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


class EnhancedTracker:
    """
    Enhanced tracker từ gui_app.py - tối ưu cho 20+ người
    Sử dụng greedy IoU matching với head-focused tracking
    """
    
    def __init__(self, iou_thr: float = 0.35, max_age: int = 25):
        """
        Args:
            iou_thr: IoU threshold for matching (0.35 là tốt cho nhiều người)
            max_age: Maximum frames to keep track without detection (25 frames)
        """
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks: Dict[int, Dict] = {}  # track_id -> {bbox, age, head_bbox, track_id}
        self.next_id = 1
        
    def update(self, detections: List[PersonDetection]) -> List[PersonDetection]:
        """
        Update tracker with new detections
        Returns detections with track_id assigned
        """
        # Age all tracks
        for tid in list(self.tracks.keys()):
            age_val = self.tracks[tid].get("age", 0)
            try:
                age_int = int(age_val)
            except Exception:
                age_int = 0
            self.tracks[tid]["age"] = age_int + 1
        
        # Extract bounding boxes for matching (use head_bbox if available, else body bbox)
        det_boxes = []
        for det in detections:
            # Use head_bbox for matching (smaller, more accurate for crowded scenes)
            if det.head_bbox and det.head_bbox[0] > 0:
                det_boxes.append(det.head_bbox)
            else:
                # Use top portion of body bbox as head approximation
                x1, y1, x2, y2 = det.bbox
                head_height = (y2 - y1) * 0.3  # Top 30% of body
                det_boxes.append((x1, y1, x2, y1 + head_height))
        
        # Greedy matching by IoU
        assignments: Dict[int, int] = {}  # track_id -> detection index
        used_dets = set()
        
        while True:
            best_tid, best_di, best_iou = None, None, 0.0
            
            for tid, tr in self.tracks.items():
                try:
                    age = int(tr.get("age", 0))
                except Exception:
                    age = 0
                if age > self.max_age:
                    continue
                
                # Get track bbox (prefer head_bbox)
                tb = tr.get("head_bbox", tr.get("bbox", None))
                if tb is None:
                    continue
                
                for di, dbox in enumerate(det_boxes):
                    if di in used_dets:
                        continue
                    ov = iou_xyxy(tb, dbox)
                    if ov > best_iou:
                        best_tid, best_di, best_iou = tid, di, ov
            
            if best_tid is None or best_di is None or best_iou < self.iou_thr:
                break
            
            # Update track with new detection
            detection = detections[best_di]
            self.tracks[best_tid]["bbox"] = detection.bbox
            self.tracks[best_tid]["head_bbox"] = detection.head_bbox if detection.head_bbox[0] > 0 else detection.bbox
            self.tracks[best_tid]["age"] = 0
            self.tracks[best_tid]["last_update"] = time.time()
            
            # Assign track_id to detection
            detection.track_id = best_tid
            
            assignments[best_tid] = best_di
            used_dets.add(best_di)
        
        # Create new tracks for unmatched detections
        for di, detection in enumerate(detections):
            if di in used_dets:
                continue
            
            tid = self.next_id
            self.next_id += 1
            
            head_bbox = detection.head_bbox if detection.head_bbox[0] > 0 else detection.bbox
            self.tracks[tid] = {
                "bbox": detection.bbox,
                "head_bbox": head_bbox,
                "age": 0,
                "last_update": time.time()
            }
            detection.track_id = tid
        
        # Prune old tracks (remove tracks that haven't been matched for max_age frames)
        for tid in list(self.tracks.keys()):
            try:
                age = int(self.tracks[tid].get("age", 0))
            except Exception:
                age = 0
            if tid not in assignments and age > self.max_age:
                del self.tracks[tid]
        
        return detections


# ==================== Enhanced Camera Worker ====================

class EnhancedCameraWorker(threading.Thread):
    """
    Camera worker với enhanced tracking cho 20+ học sinh
    Tích hợp tracking logic từ gui_app.py
    """

    def __init__(self, cam_id: str, url: str, enable_detection: bool = True):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.url = url
        self.enable_detection = enable_detection and YOLO_AVAILABLE
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._last_frame = None
        self._last_detection_result = None
        self._last_annotated_frame = None
        self._capture = None
        self._detection_enabled = self.enable_detection
        self._frame_width = 640
        self._frame_height = 360
        self._current_fps = 0.0

        # Enhanced tracker per camera
        self.tracker = EnhancedTracker(iou_thr=0.35, max_age=25)
        # WS throttle
        self._last_emit_ts = 0.0

        # Drowsiness state machine (standalone_app style)
        self._sleep_frames_required = 15
        self._awake_frames_required = 5
        self._per_id_sleep_count = {}
        self._per_id_awake_count = {}
        self._per_id_state = {}  # "Bình thường" | "Ngủ gật" | "Gục xuống bàn" | "Thức dậy"
        self._per_id_sleep_start = {}

        app.logger.info(f"[{self.cam_id}] EnhancedCameraWorker initialized with URL: {url}, detection: {self._detection_enabled}")

        # Register camera with drowsiness logger
        if LOGGER_AVAILABLE:
            logger = get_global_logger()
            # Extract camera name from cam_id (format: "id/name")
            camera_name = cam_id.split('/')[-1] if '/' in cam_id else cam_id
            logger.register_camera(cam_id, camera_name)
            app.logger.info(f"[{self.cam_id}] Camera registered with drowsiness logger as '{camera_name}'")

        # Auto-enable detection for webcam (device ID)
        if url.isdigit() or url.startswith('0'):
            self._detection_enabled = True and YOLO_AVAILABLE
            app.logger.info(f"[{self.cam_id}] Auto-enabling YOLO detection for webcam")

    def run(self):
        app.logger.info(f"[{self.cam_id}] Starting enhanced camera worker thread...")
        
        # Initialize camera once
        try:
            if self.url.isdigit():
                # Direct device ID - use DirectShow backend for Windows
                self._capture = cv2.VideoCapture(int(self.url), cv2.CAP_DSHOW)
                app.logger.info(f"[{self.cam_id}] Using DirectShow backend for device {self.url}")
            elif self.url.startswith('webcam-') or len(self.url) > 20:
                # Device ID string - try to find working webcam
                for i in range(5):
                    test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            test_cap.release()
                            self._capture = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                            app.logger.info(f"[{self.cam_id}] Using webcam device {i} with DirectShow")
                            break
                        test_cap.release()
                if self._capture is None:
                    self._capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    app.logger.info(f"[{self.cam_id}] Using default webcam 0 with DirectShow")
            else:
                # IP camera or RTSP URL
                self._capture = cv2.VideoCapture(self.url)
                app.logger.info(f"[{self.cam_id}] Using IP camera: {self.url}")
            
            if not self._capture.isOpened():
                app.logger.error(f"[{self.cam_id}] Failed to open camera")
                return
            
            # Get frame dimensions
            self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            app.logger.info(f"[{self.cam_id}] Camera opened successfully, frame size: {self._frame_width}x{self._frame_height}")
            
        except Exception as e:
            app.logger.error(f"[{self.cam_id}] Exception during camera initialization: {e}")
            return
        
        # Main loop
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        
        while self._running.is_set():
            try:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    app.logger.warning(f"[{self.cam_id}] Read failed, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Update frame dimensions
                h, w = frame.shape[:2]
                self._frame_width = w
                self._frame_height = h
                
                frame_count += 1
                fps_frame_count += 1
                
                # Calculate FPS every second
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()
                    with self._lock:
                        self._current_fps = current_fps
                    if frame_count % 30 == 0:  # Log every 30 frames
                        app.logger.info(f"[{self.cam_id}] Frame {frame_count}, FPS: {current_fps:.1f}, size: {frame.shape}")
                
                # Only keep the lock for quick state writes
                with self._lock:
                    self._last_frame = frame

                # Run YOLO detection with enhanced tracking if enabled (outside lock)
                if self._detection_enabled and YOLO_AVAILABLE:
                    try:
                        start_time = time.time()

                        # Detect with YOLO
                        detection_result = detect_frame(frame)

                        # Apply enhanced tracking
                        if detection_result and detection_result.persons:
                            tracked_persons = self.tracker.update(detection_result.persons)
                            detection_result.persons = tracked_persons
                            # Apply state machine and logs (no lock needed)
                            self._update_states_and_logs(tracked_persons)
                            if frame_count % 60 == 0:  # Log every 60 frames
                                unique_ids = len(set(p.track_id for p in tracked_persons if p.track_id))
                                app.logger.info(
                                    f"[{self.cam_id}] Tracked {len(tracked_persons)} persons "
                                    f"({unique_ids} unique IDs) in {time.time() - start_time:.3f}s"
                                )

                        # Set FPS/processing time
                        if detection_result:
                            detection_result.fps = current_fps
                            detection_result.processing_time = time.time() - start_time

                        # Prepare annotated frame (optional) outside lock
                        annotated = None
                        if frame_count % 30 == 0 and detection_result is not None:
                            try:
                                annotated = draw_detections(frame.copy(), detection_result)
                            except Exception:
                                annotated = None

                        # Commit results under short lock
                        with self._lock:
                            self._last_detection_result = detection_result
                            self._last_annotated_frame = annotated

                        # Emit realtime update to WS room for this camera (throttled)
                        try:
                            now_ts = time.time()
                            if now_ts - self._last_emit_ts >= 0.15:  # ~6-7 updates/sec
                                self._last_emit_ts = now_ts
                                persons_payload = []
                                if detection_result and detection_result.persons:
                                    for p in detection_result.persons:
                                        kpts = [{
                                            'x': float(k.x), 'y': float(k.y),
                                            'confidence': float(k.confidence), 'visible': bool(k.visible)
                                        } for k in p.keypoints]
                                        persons_payload.append({
                                            'id': int(getattr(p, 'id', 0) or 0),
                                            'track_id': int(getattr(p, 'track_id', getattr(p, 'id', 0)) or 0),
                                            'bbox': [float(v) for v in list(p.bbox)],
                                            'head_bbox': [float(v) for v in list(getattr(p, 'head_bbox', []) or [])] if getattr(p, 'head_bbox', None) is not None else None,
                                            'confidence': float(getattr(p, 'confidence', 0.0) or 0.0),
                                            'keypoints': kpts,
                                            'drowsiness_score': float(getattr(p, 'drowsiness_score', 0.0) or 0.0),
                                            'drowsiness_state': str(getattr(p, 'drowsiness_state', 'awake') or 'awake'),
                                            'last_update': float(getattr(p, 'last_update', time.time()))
                                        })
                                fw, fh = self.get_frame_dimensions()
                                socketio.emit('update', {
                                    'success': True,
                                    'schema': 'v1',
                                    'camera_id': self.cam_id,
                                    'frame_width': int(fw or 0),
                                    'frame_height': int(fh or 0),
                                    'fps': float(self._current_fps or 0.0),
                                    'processing_time': float(getattr(detection_result, 'processing_time', 0.0) or 0.0),
                                    'persons': persons_payload,
                                    'timestamp': now_ts,
                                }, namespace='/ws/camera', to=f'cam:{self.cam_id}')
                        except Exception as _emit_e:
                            app.logger.debug(f"[{self.cam_id}] WS emit skipped: {_emit_e}")

                    except Exception as e:
                        app.logger.error(f"[{self.cam_id}] Error during detection: {e}")
                        with self._lock:
                            self._last_detection_result = None
                            self._last_annotated_frame = None
                else:
                    with self._lock:
                        self._last_detection_result = None
                        self._last_annotated_frame = None
                
                # Small delay to prevent overwhelming CPU
                time.sleep(0.033)  # ~30 FPS max
                
            except Exception as e:
                app.logger.error(f"[{self.cam_id}] Error in camera worker main loop: {e}")
                time.sleep(1)
        
        # Cleanup
        if self._capture:
            self._capture.release()
        app.logger.info(f"[{self.cam_id}] Camera worker stopped")

    def stop(self):
        self._running.clear()
        if self._capture:
            self._capture.release()

    def get_detection_result(self) -> Optional[DetectionResult]:
        with self._lock:
            result = self._last_detection_result
            if result is not None:
                result.fps = self._current_fps
            return result

    def get_fps(self) -> float:
        with self._lock:
            return self._current_fps

    def get_frame_dimensions(self) -> Tuple[int, int]:
        with self._lock:
            return self._frame_width, self._frame_height

    def toggle_detection(self, enabled: bool):
        with self._lock:
            self._detection_enabled = enabled and YOLO_AVAILABLE
        app.logger.info(f"[{self.cam_id}] Detection toggled: {self._detection_enabled}")

    def is_detection_enabled(self) -> bool:
        with self._lock:
            return self._detection_enabled

    def get_last_jpeg(self, annotated: bool = False) -> Optional[bytes]:
        with self._lock:
            # If annotated requested but cached annotated frame is missing, render on-demand
            if annotated and self._last_annotated_frame is None and self._last_frame is not None and self._last_detection_result is not None:
                try:
                    self._last_annotated_frame = draw_detections(self._last_frame.copy(), self._last_detection_result)
                except Exception:
                    self._last_annotated_frame = None
            frame = self._last_annotated_frame if annotated else self._last_frame
            if frame is None:
                return None
            
            # Optionally downscale for UI stream to reduce CPU and bandwidth
            try:
                max_w = 960
                h, w = frame.shape[:2]
                if w > max_w:
                    scale = max_w / float(w)
                    nh = int(h * scale)
                    frame = cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)
            except Exception:
                pass

            # Encode as JPEG at a slightly lower quality to save CPU
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if not ret:
                return None
            return buffer.tobytes()

    def update_detection_result(self, result: DetectionResult):
        """Update detection result from external source (e.g., /api/detect/frame for webcam)."""
        with self._lock:
            self._last_detection_result = result
            # Also run state machine for logs if persons present
            if result and getattr(result, 'persons', None):
                self._update_states_and_logs(result.persons)

    def _update_states_and_logs(self, persons: List[PersonDetection]):
        """Standalone-like state machine to generate drowsiness logs per track id."""
        now = time.time()
        # Map backend states to UI Vietnamese for consistency
        def map_state(s: str) -> str:
            if s == 'drowsy':
                return 'Ngủ gật'
            if s == 'sleeping' or s == 'head_down':
                return 'Gục xuống bàn'
            return 'Bình thường'

        for p in persons:
            tid = int(getattr(p, 'track_id', getattr(p, 'id', 0)) or 0)
            if tid == 0:
                continue
            raw = getattr(p, 'drowsiness_state', 'awake') or 'awake'
            state_now = map_state(raw)

            prev = self._per_id_state.get(tid, 'Bình thường')
            sleep_cnt = self._per_id_sleep_count.get(tid, 0)
            awake_cnt = self._per_id_awake_count.get(tid, 0)

            if state_now in ('Ngủ gật', 'Gục xuống bàn'):
                sleep_cnt += 1
                awake_cnt = 0
            else:
                awake_cnt += 1
                sleep_cnt = 0

            eff_state = prev
            if prev in ('Ngủ gật', 'Gục xuống bàn'):
                if state_now not in ('Ngủ gật', 'Gục xuống bàn') and awake_cnt >= self._awake_frames_required:
                    eff_state = 'Thức dậy'
            elif prev == 'Thức dậy':
                if awake_cnt >= self._awake_frames_required:
                    eff_state = 'Bình thường'
            else:
                if state_now in ('Ngủ gật', 'Gục xuống bàn') and sleep_cnt >= self._sleep_frames_required:
                    eff_state = state_now

            # Emit logs on transition
            if eff_state != prev:
                if eff_state in ('Ngủ gật', 'Gục xuống bàn'):
                    self._per_id_sleep_start[tid] = now
                    append_log({'camera_id': self.cam_id, 'track_id': tid, 'type': 'sleepy' if eff_state == 'Ngủ gật' else 'head_down', 'state': eff_state, 'ts': now})
                    
                    # 🔥 NEW: Log to drowsiness logger
                    if LOGGER_AVAILABLE:
                        try:
                            logger = get_global_logger()
                            logger.update_student_state(self.cam_id, tid, True)
                        except Exception as log_err:
                            app.logger.debug(f"Logger error (start drowsy): {log_err}")
                            
                elif eff_state == 'Thức dậy':
                    dur = 0.0
                    if tid in self._per_id_sleep_start:
                        dur = now - self._per_id_sleep_start[tid]
                        del self._per_id_sleep_start[tid]
                    append_log({'camera_id': self.cam_id, 'track_id': tid, 'type': 'wake_up', 'state': eff_state, 'duration': dur, 'ts': now})
                    
                    # 🔥 NEW: Log wake up to drowsiness logger
                    if LOGGER_AVAILABLE:
                        try:
                            logger = get_global_logger()
                            logger.update_student_state(self.cam_id, tid, False)
                        except Exception as log_err:
                            app.logger.debug(f"Logger error (wake up): {log_err}")

            self._per_id_state[tid] = eff_state
            self._per_id_sleep_count[tid] = sleep_cnt
            self._per_id_awake_count[tid] = awake_cnt


# ==================== Camera Manager ====================

class CameraManager:
    def __init__(self):
        self._workers: Dict[str, EnhancedCameraWorker] = {}
        self._meta: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def list(self):
        with self._lock:
            out = []
            for cid, meta in self._meta.items():
                running = cid in self._workers and self._workers[cid].is_alive()
                out.append({
                    'id': cid,
                    'name': meta.get('name') or cid,
                    'type': meta.get('type', 'ip'),
                    'url': meta.get('url'),
                    'status': 'running' if running else 'stopped'
                })
            return out

    def add(self, cam_id: str, url: str, name: Optional[str] = None):
        with self._lock:
            if cam_id in self._meta:
                raise ValueError('Camera already exists')
            self._meta[cam_id] = {'url': url, 'name': name or cam_id, 'type': 'ip'}

    def update_camera_meta(self, cam_id: str, url: str, name: Optional[str] = None):
        """Update camera metadata"""
        with self._lock:
            if cam_id in self._meta:
                if url:
                    self._meta[cam_id]['url'] = url
                if name:
                    self._meta[cam_id]['name'] = name
                app.logger.info(f"[{cam_id}] Camera metadata updated")

    def remove(self, cam_id: str):
        """Remove a camera and stop its worker"""
        with self._lock:
            if cam_id in self._workers:
                try:
                    self._workers[cam_id].stop()
                    app.logger.info(f"[{cam_id}] Worker stopped")
                except Exception as e:
                    app.logger.warning(f"[{cam_id}] Error stopping worker: {e}")
            self._workers.pop(cam_id, None)
            if cam_id in self._meta:
                self._meta.pop(cam_id, None)
                app.logger.info(f"[{cam_id}] Camera metadata removed")

    def start(self, cam_id: str, enable_detection: bool = True):
        with self._lock:
            if cam_id not in self._meta:
                raise KeyError('Camera not found')
            
            # If worker exists and is alive, just ensure detection state is correct
            if cam_id in self._workers and self._workers[cam_id].is_alive():
                worker = self._workers[cam_id]
                if worker.is_detection_enabled() != enable_detection:
                    worker.toggle_detection(enable_detection)
                    app.logger.info(f"[{cam_id}] Toggled detection to {enable_detection}")
                app.logger.info(f"[{cam_id}] Camera worker already running and detection state updated.")
                return
            
            # If worker exists but is dead, remove it
            if cam_id in self._workers and not self._workers[cam_id].is_alive():
                app.logger.warning(f"[{cam_id}] Found dead worker, removing and restarting.")
                del self._workers[cam_id]
            
            try:
                worker = EnhancedCameraWorker(cam_id, self._meta[cam_id]['url'], enable_detection)
                self._workers[cam_id] = worker
                worker.start()
                app.logger.info(f"[{cam_id}] Enhanced camera worker started successfully with tracking")
            except Exception as e:
                app.logger.error(f"[{cam_id}] Failed to start camera worker: {e}")
                raise

    def stop(self, cam_id: str):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].stop()

    def get_jpeg(self, cam_id: str, annotated: bool = False) -> Optional[bytes]:
        with self._lock:
            worker = self._workers.get(cam_id)
        if not worker:
            return None
        return worker.get_last_jpeg(annotated)
    
    def get_detection_result(self, cam_id: str) -> Optional[DetectionResult]:
        with self._lock:
            worker = self._workers.get(cam_id)
        if not worker:
            return None
        result = worker.get_detection_result()
        if result is not None:
            if not hasattr(result, 'fps') or result.fps == 0.0:
                result.fps = worker.get_fps()
        return result
    
    def get_frame_dimensions(self, cam_id: str):
        with self._lock:
            worker = self._workers.get(cam_id)
        if not worker:
            return None, None
        return worker.get_frame_dimensions()
    
    def toggle_detection(self, cam_id: str, enabled: bool):
        with self._lock:
            worker = self._workers.get(cam_id)
        if worker:
            worker.toggle_detection(enabled)
    
    def has_camera(self, cam_id: str) -> bool:
        """Check if camera exists in metadata"""
        with self._lock:
            return cam_id in self._meta
    
    def get_worker(self, cam_id: str):
        """Get worker for a camera"""
        with self._lock:
            return self._workers.get(cam_id)


manager = CameraManager()
_start_time = time.time()


# ==================== Flask Routes ====================

@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,DELETE,OPTIONS'
    return resp


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'uptime_s': int(time.time() - _start_time)})


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    return jsonify({'success': True, 'cameras': manager.list()})


@app.route('/api/camera/add', methods=['POST'])
def add_camera():
    data = request.get_json(force=True, silent=True) or {}
    cam_id = data.get('id') or data.get('name')
    url = data.get('url')
    name = data.get('name')
    if not cam_id or not url:
        return jsonify({'success': False, 'error': 'id/name and url required'}), 400
    try:
        # If camera already exists, just update its metadata and ensure it's started
        if manager.has_camera(cam_id):
            manager.update_camera_meta(cam_id, url, name)
            app.logger.info(f"[{cam_id}] Camera already exists, metadata updated. Auto-starting with detection.")
            try:
                manager.start(cam_id, enable_detection=True)
            except Exception as e:
                app.logger.warning(f"[{cam_id}] Auto-start failed: {e}")
            return jsonify({'success': True, 'message': 'Camera already exists, metadata updated'}), 200
        
        manager.add(cam_id, url, name)
        # Auto-start newly added camera with detection
        try:
            manager.start(cam_id, enable_detection=True)
        except Exception as e:
            app.logger.warning(f"[{cam_id}] Auto-start failed after add: {e}")
        return jsonify({'success': True})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 409


@app.route('/api/camera/<cam_id>/start', methods=['POST'])
def start_camera(cam_id):
    try:
        data = request.get_json(force=True, silent=True) or {}
        enable_detection = data.get('enable_detection', True)
        manager.start(cam_id, enable_detection)
        return jsonify({'success': True})
    except KeyError:
        return jsonify({'success': False, 'error': 'not found'}), 404


@app.route('/api/camera/<cam_id>/stop', methods=['POST'])
def stop_camera(cam_id):
    """Stop a camera"""
    try:
        if not manager.has_camera(cam_id):
            return jsonify({'success': False, 'error': f'Camera {cam_id} not found'}), 404
        manager.stop(cam_id)
        return jsonify({'success': True, 'message': f'Camera {cam_id} stopped'})
    except Exception as e:
        app.logger.error(f"[{cam_id}] Error stopping camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/<cam_id>/remove', methods=['DELETE'])
def remove_camera(cam_id):
    """Remove a camera from the system"""
    try:
        if not manager.has_camera(cam_id):
            return jsonify({'success': False, 'error': f'Camera {cam_id} not found'}), 404
        
        manager.remove(cam_id)
        app.logger.info(f"[{cam_id}] Camera removed successfully")
        return jsonify({'success': True, 'message': f'Camera {cam_id} removed successfully'})
    except Exception as e:
        app.logger.error(f"[{cam_id}] Error removing camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/<cam_id>/stream', methods=['GET'])
def stream_frame(cam_id):
    """Return latest frame as base64 JSON for easy <img src="data:"> usage in the UI."""
    annotated = request.args.get('annotated', 'false').lower() == 'true'
    jpeg = manager.get_jpeg(cam_id, annotated=annotated)
    if jpeg is None:
        return jsonify({'success': False, 'error': 'No frame available'}), 404
    
    b64 = base64.b64encode(jpeg).decode('utf-8')
    return jsonify({
        'success': True,
        'frame': f'data:image/jpeg;base64,{b64}',
        'timestamp': time.time()
    })


@app.route('/api/camera/<cam_id>/detection', methods=['GET'])
def get_detection_results(cam_id):
    """Get detection results for a specific camera with enhanced tracking"""
    try:
        # Check if camera exists
        if not manager.has_camera(cam_id):
            return jsonify({
                'success': False,
                'error': f'Camera {cam_id} not found',
                'frame_width': 640,
                'frame_height': 360,
                'fps': 0.0,
                'persons': [],
                'timestamp': time.time()
            }), 404
        
        # Check if camera worker exists and is running
        worker = manager.get_worker(cam_id)
        if not worker:
            return jsonify({
                'success': True,
                'frame_width': 640,
                'frame_height': 360,
                'fps': 0.0,
                'persons': [],
                'timestamp': time.time()
            })
        
        result = manager.get_detection_result(cam_id)
        if result is None:
            # Return empty result instead of 404 to avoid noisy errors while detection warms up
            frame_width, frame_height = manager.get_frame_dimensions(cam_id) or (640, 360)
            return jsonify({
                'success': True,
                'frame_width': frame_width,
                'frame_height': frame_height,
                'fps': worker.get_fps() if worker else 0.0,
                'persons': [],
                'timestamp': time.time()
            })
        
        # Get frame dimensions from worker
        frame_width, frame_height = manager.get_frame_dimensions(cam_id) or (640, 360)
        
        # Convert DetectionResult to JSON-serializable format
        persons_data = []
        for person in result.persons:
            keypoints_data = []
            for kpt in person.keypoints:
                keypoints_data.append({
                    'x': kpt.x,
                    'y': kpt.y,
                    'confidence': kpt.confidence,
                    'visible': kpt.visible
                })
            
            persons_data.append({
                'id': person.id,
                'track_id': getattr(person, 'track_id', person.id),
                'bbox': person.bbox,
                'head_bbox': getattr(person, 'head_bbox', None),
                'confidence': person.confidence,
                'keypoints': keypoints_data,
                'drowsiness_score': person.drowsiness_score,
                'drowsiness_state': person.drowsiness_state,
                'last_update': person.last_update
            })
        
        return jsonify({
            'success': True,
            'detection_result': {
                'frame_id': result.frame_id,
                'timestamp': result.timestamp,
                'persons': persons_data,
                'fps': result.fps if hasattr(result, 'fps') and result.fps else worker.get_fps() if worker else 0.0,
                'processing_time': result.processing_time if hasattr(result, 'processing_time') else 0.0,
                'frame_width': frame_width,
                'frame_height': frame_height
            },
            'frame_width': frame_width,
            'frame_height': frame_height,
            'fps': result.fps if hasattr(result, 'fps') and result.fps else worker.get_fps() if worker else 0.0,
            'persons': persons_data,
            'timestamp': result.timestamp if hasattr(result, 'timestamp') else time.time()
        })
    except Exception as e:
        app.logger.error(f"[{cam_id}] Error getting detection results: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'frame_width': 640,
            'frame_height': 360,
            'fps': 0.0,
            'persons': [],
            'timestamp': time.time()
        }), 500


# ==================== Drowsiness Logging API Endpoints ====================

@app.route('/api/logs/cameras', methods=['GET'])
def get_cameras_list():
    """Lấy danh sách tất cả camera đã đăng ký"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        cameras = []
        for camera_id, cam_logger in logger.cameras.items():
            cameras.append({
                'camera_id': camera_id,
                'camera_name': cam_logger.camera_name,
                'active_drowsy_count': len(cam_logger.active_events)
            })
        
        return jsonify({
            'success': True,
            'cameras': cameras,
            'total': len(cameras)
        })
    except Exception as e:
        app.logger.error(f"Error getting cameras list: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/stats/<camera_id>', methods=['GET'])
def get_camera_stats(camera_id: str):
    """Lấy thống kê cho một camera
    
    Query params:
        period: 'today', 'week', 'month', hoặc 'YYYY-MM-DD_YYYY-MM-DD'
    """
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        period = request.args.get('period', 'today')
        
        stats = logger.get_camera_stats(camera_id, period)
        
        if 'error' in stats:
            return jsonify({'success': False, 'error': stats['error']}), 404
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        app.logger.error(f"Error getting camera stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/stats', methods=['GET'])
def get_all_stats():
    """Lấy thống kê tất cả camera
    
    Query params:
        period: 'today', 'week', 'month', hoặc 'YYYY-MM-DD_YYYY-MM-DD'
    """
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        period = request.args.get('period', 'today')
        
        stats = logger.get_all_cameras_stats(period)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'period': period
        })
    except Exception as e:
        app.logger.error(f"Error getting all stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/summary', methods=['GET'])
def get_summary_stats():
    """Lấy thống kê tổng hợp tất cả camera
    
    Query params:
        period: 'today', 'week', 'month', hoặc 'YYYY-MM-DD_YYYY-MM-DD'
    """
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        period = request.args.get('period', 'today')
        
        summary = logger.get_summary_stats(period)
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        app.logger.error(f"Error getting summary stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/events/<camera_id>', methods=['GET'])
def get_camera_events(camera_id: str):
    """Lấy log chi tiết các sự kiện ngủ gật của camera
    
    Query params:
        period: 'today', 'week', 'month', hoặc 'YYYY-MM-DD_YYYY-MM-DD'
    """
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        period = request.args.get('period', 'today')
        
        events = logger.get_camera_events(camera_id, period)
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'period': period,
            'events': events,
            'total_events': len(events)
        })
    except Exception as e:
        app.logger.error(f"Error getting camera events: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/active', methods=['GET'])
def get_active_drowsy():
    """Lấy danh sách học sinh đang ngủ gật (tất cả camera)"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        active = logger.get_active_drowsy_all_cameras()
        
        # Count total
        total_active = sum(len(students) for students in active.values())
        
        return jsonify({
            'success': True,
            'active_drowsy': active,
            'total_active': total_active,
            'cameras_with_drowsy': len(active)
        })
    except Exception as e:
        app.logger.error(f"Error getting active drowsy students: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/save', methods=['POST'])
def save_logs():
    """Lưu logs ra file JSON"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        logger = get_global_logger()
        data = request.get_json(silent=True) or {}
        filepath = data.get('filepath')
        
        logger.save_to_file(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Logs saved successfully'
        })
    except Exception as e:
        app.logger.error(f"Error saving logs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/export/pdf', methods=['POST'])
def export_pdf_report():
    """Xuất báo cáo PDF"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        from report_generator import get_report_generator
        
        logger = get_global_logger()
        data = request.get_json() or {}
        
        period = data.get('period', 'today')
        camera_ids = data.get('camera_ids', None)
        
        # Get summary statistics
        summary = logger.get_summary_stats(period)
        
        # Get camera statistics
        if camera_ids:
            camera_stats = [logger.get_camera_stats(cam_id, period) for cam_id in camera_ids]
        else:
            all_stats = logger.get_all_cameras_stats(period)
            camera_stats = all_stats.get('camera_stats', [])
        
        # Get detailed events
        all_events = []
        for cam_id in logger.cameras.keys():
            if camera_ids is None or cam_id in camera_ids:
                cam_logger = logger.cameras[cam_id]
                start_time, end_time = logger._parse_period(period)
                events = cam_logger.get_detailed_events(start_time, end_time)
                # Add camera info to each event
                for event in events:
                    event['camera_id'] = cam_id
                    event['camera_name'] = cam_logger.camera_name
                all_events.extend(events)
        
        # Sort by start time
        all_events.sort(key=lambda x: x['start_time'], reverse=True)
        
        # Generate PDF
        report_gen = get_report_generator()
        pdf_path = report_gen.generate_pdf_report(
            period=period,
            camera_stats=camera_stats,
            summary=summary,
            events=all_events,
            camera_ids=camera_ids
        )
        
        # Return file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=os.path.basename(pdf_path)
        )
        
    except Exception as e:
        app.logger.error(f"Error generating PDF report: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/export/excel', methods=['POST'])
def export_excel_report():
    """Xuất báo cáo Excel"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({'success': False, 'error': 'Logger not available'}), 503
        
        from report_generator import get_report_generator
        
        logger = get_global_logger()
        data = request.get_json() or {}
        
        period = data.get('period', 'today')
        camera_ids = data.get('camera_ids', None)
        
        # Get summary statistics
        summary = logger.get_summary_stats(period)
        
        # Get camera statistics
        if camera_ids:
            camera_stats = [logger.get_camera_stats(cam_id, period) for cam_id in camera_ids]
        else:
            all_stats = logger.get_all_cameras_stats(period)
            camera_stats = all_stats.get('camera_stats', [])
        
        # Get detailed events
        all_events = []
        for cam_id in logger.cameras.keys():
            if camera_ids is None or cam_id in camera_ids:
                cam_logger = logger.cameras[cam_id]
                start_time, end_time = logger._parse_period(period)
                events = cam_logger.get_detailed_events(start_time, end_time)
                # Add camera info
                for event in events:
                    event['camera_id'] = cam_id
                    event['camera_name'] = cam_logger.camera_name
                all_events.extend(events)
        
        # Sort by start time
        all_events.sort(key=lambda x: x['start_time'], reverse=True)
        
        # Generate Excel
        report_gen = get_report_generator()
        excel_path = report_gen.generate_excel_report(
            period=period,
            camera_stats=camera_stats,
            summary=summary,
            events=all_events
        )
        
        # Return file
        return send_file(
            excel_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=os.path.basename(excel_path)
        )
        
    except Exception as e:
        app.logger.error(f"Error generating Excel report: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect/frame', methods=['POST', 'OPTIONS'])
def detect_frame_endpoint():
    """Detect persons and drowsiness from a frame sent from frontend (for webcam)"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    if not YOLO_AVAILABLE:
        return jsonify({'success': False, 'error': 'YOLO detector not available'}), 503
    
    try:
        data = request.get_json(force=True, silent=True) or {}
        frame_base64 = data.get('frame')
        camera_id = data.get('camera_id', 'webcam')
        
        if not frame_base64:
            return jsonify({'success': False, 'error': 'frame data required'}), 400
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if ',' in frame_base64:
                frame_base64 = frame_base64.split(',')[1]
            
            frame_bytes = base64.b64decode(frame_base64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
            
            h, w = frame.shape[:2]
            
        except Exception as e:
            app.logger.error(f"Error decoding frame: {e}")
            return jsonify({'success': False, 'error': f'Failed to decode frame: {str(e)}'}), 400
        
        # Optional dynamic settings and simple preprocessing (mirror WS behavior)
        try:
            conf = data.get('conf', None)
            if conf is not None:
                update_detection_settings(conf=float(conf))
        except Exception:
            pass

        try:
            preprocess = (data.get('preprocess') if isinstance(data, dict) else None) or {}
            enabled = preprocess.get('enabled', True)
            if enabled:
                gamma = float(preprocess.get('gamma', 1.2))
                beta = float(preprocess.get('beta', 15.0))
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
                if abs(gamma - 1.0) > 1e-3:
                    inv_gamma = 1.0 / max(gamma, 1e-6)
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                    frame = cv2.LUT(frame, table)
        except Exception:
            pass

        # Run detection with enhanced tracker
        result = detect_frame(frame)
        
        # Store result for this camera (webcam) if it's registered
        if manager.has_camera(camera_id):
            # Update detection result for webcam
            worker = manager.get_worker(camera_id)
            if worker:
                worker.update_detection_result(result)
        
        # Convert DetectionResult to JSON-serializable format
        persons_data = []
        for person in result.persons:
            keypoints_data = []
            for kpt in person.keypoints:
                keypoints_data.append({
                    'x': float(kpt.x),
                    'y': float(kpt.y),
                    'confidence': float(kpt.confidence),
                    'visible': bool(kpt.visible)
                })
            
            persons_data.append({
                'id': int(getattr(person, 'id', 0) or 0),
                'track_id': int(getattr(person, 'track_id', getattr(person, 'id', 0)) or 0),
                'bbox': [float(v) for v in list(person.bbox)],
                'head_bbox': [float(v) for v in list(getattr(person, 'head_bbox', []) or [])] if getattr(person, 'head_bbox', None) is not None else None,
                'confidence': float(getattr(person, 'confidence', 0.0) or 0.0),
                'keypoints': keypoints_data,
                'drowsiness_score': float(getattr(person, 'drowsiness_score', 0.0) or 0.0),
                'drowsiness_state': str(getattr(person, 'drowsiness_state', 'awake') or 'awake'),
                'last_update': float(getattr(person, 'last_update', time.time()))
            })
        
        return jsonify({
            'success': True,
            'detection_result': {
                'frame_id': int(getattr(result, 'frame_id', 0) or 0),
                'timestamp': float(getattr(result, 'timestamp', time.time())),
                'persons': persons_data,
                'fps': float(getattr(result, 'fps', 0.0) or 0.0),
                'processing_time': float(getattr(result, 'processing_time', 0.0) or 0.0),
                'frame_width': w,
                'frame_height': h
            },
            'frame_width': w,
            'frame_height': h,
            'fps': float(getattr(result, 'fps', 0.0) or 0.0),
            'persons': persons_data,
            'timestamp': float(getattr(result, 'timestamp', time.time()))
        })
        
    except Exception as e:
        app.logger.error(f"Error in detect_frame_endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'frame_width': 640,
            'frame_height': 360,
            'fps': 0.0,
            'persons': [],
            'timestamp': time.time()
        }), 500


@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    """Get system statistics"""
    cameras = manager.list()
    running_cameras = [c for c in cameras if c['status'] == 'running']
    
    total_students = 0
    drowsy_students = 0
    sleeping_students = 0
    
    for cam in running_cameras:
        result = manager.get_detection_result(cam['id'])
        if result and result.persons:
            total_students += len(result.persons)
            for person in result.persons:
                if person.drowsiness_state == 'drowsy':
                    drowsy_students += 1
                elif person.drowsiness_state == 'sleeping':
                    sleeping_students += 1
    
    return jsonify({
        'success': True,
        'stats': {
            'total_cameras': len(cameras),
            'running_cameras': len(running_cameras),
            'total_students': total_students,
            'drowsy_students': drowsy_students,
            'sleeping_students': sleeping_students,
            'yolo_available': YOLO_AVAILABLE,
            'detector_initialized': YOLO_AVAILABLE
        }
    })


# ==================== Realtime Logs ====================
_log_buffer = deque(maxlen=500)
_log_lock = threading.Lock()

def append_log(entry: Dict):
    with _log_lock:
        entry['ts'] = entry.get('ts', time.time())
        _log_buffer.append(entry)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        since = float(request.args.get('since', '0') or 0)
        with _log_lock:
            items = list(_log_buffer)
        if since > 0:
            items = [e for e in items if e.get('ts', 0) > since]
        return jsonify({'success': True, 'logs': items, 'now': time.time()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== WebSocket: Realtime detection ====================

@socketio.on('connect', namespace='/ws/detect')
def ws_connect():
    app.logger.info('🔌 [WS /ws/detect] Client CONNECTED')
    emit('hello', {'msg': 'connected'})


@socketio.on('disconnect', namespace='/ws/detect')
def ws_disconnect():
    app.logger.info('🔌 [WS /ws/detect] Client DISCONNECTED')


# 🔥 WebSocket state tracking (per track_id)
_ws_per_id_state = {}  # track_id -> "awake" | "drowsy" | "sleeping"
_ws_per_id_sleep_start = {}  # track_id -> timestamp
_ws_sleep_frames_required = 8  # ~1.6s at 5fps (same as camera worker)
_ws_awake_frames_required = 5  # ~1s at 5fps
_ws_per_id_sleep_count = {}  # track_id -> int
_ws_per_id_awake_count = {}  # track_id -> int

@socketio.on('frame', namespace='/ws/detect')
def ws_frame(data):
    """Receive base64 frame, run detection, emit result immediately with logging support."""
    try:
        app.logger.info(f"📥 [WS /ws/detect] Received 'frame' event from client")
        
        if not YOLO_AVAILABLE:
            emit('result', {'success': False, 'error': 'YOLO not available'})
            return
        # Ensure detector is initialized as a fallback
        try:
            if get_detector() is None:
                app.logger.warning("Detector not initialized yet; initializing on first WS frame...")
                # Pass None so detector can auto-pick user's trained weights if present
                initialize_detector(None)
        except Exception as _e:
            app.logger.error(f"Failed to lazy-initialize detector: {_e}")
        frame_b64 = data.get('frame') if isinstance(data, dict) else None
        cam_id = data.get('camera_id', 'webcam') if isinstance(data, dict) else 'webcam'
        # Optional dynamic settings and preprocessing controls
        try:
            conf = data.get('conf', None)
            if conf is not None:
                update_detection_settings(conf=float(conf))
        except Exception:
            pass
        if not frame_b64:
            emit('result', {'success': False, 'error': 'frame required'})
            return
        # strip prefix
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        frame_bytes = base64.b64decode(frame_b64)
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            emit('result', {'success': False, 'error': 'decode failed'})
            return
        # Simple low-light enhancement: brightness + gamma
        try:
            preprocess = (data.get('preprocess') if isinstance(data, dict) else None) or {}
            enabled = preprocess.get('enabled', True)
            if enabled:
                gamma = float(preprocess.get('gamma', 1.2))
                beta = float(preprocess.get('beta', 15.0))  # brightness shift
                # Brightness/contrast
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
                # Gamma correction via LUT (fast)
                if abs(gamma - 1.0) > 1e-3:
                    inv_gamma = 1.0 / max(gamma, 1e-6)
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                    frame = cv2.LUT(frame, table)
        except Exception as _pp_e:
            app.logger.debug(f"Preprocess error ignored: {_pp_e}")
        h, w = frame.shape[:2]
        # detect
        det = detect_frame(frame)
        # update worker if exists
        if manager.has_camera(cam_id):
            worker = manager.get_worker(cam_id)
            if worker:
                worker.update_detection_result(det)
        # 🔥 Track state changes and log drowsiness events
        now = time.time()
        for p in det.persons:
            tid = int(getattr(p, 'track_id', getattr(p, 'id', 0)) or 0)
            state_now = str(getattr(p, 'drowsiness_state', 'awake') or 'awake')
            
            # Initialize tracking for new person
            if tid not in _ws_per_id_state:
                _ws_per_id_state[tid] = 'awake'
                _ws_per_id_sleep_count[tid] = 0
                _ws_per_id_awake_count[tid] = 0
            
            prev_state = _ws_per_id_state[tid]
            
            # Update counters
            if state_now in ('drowsy', 'sleeping'):
                _ws_per_id_sleep_count[tid] = _ws_per_id_sleep_count.get(tid, 0) + 1
                _ws_per_id_awake_count[tid] = 0
            else:  # awake
                _ws_per_id_awake_count[tid] = _ws_per_id_awake_count.get(tid, 0) + 1
                _ws_per_id_sleep_count[tid] = 0
            
            sleep_cnt = _ws_per_id_sleep_count[tid]
            awake_cnt = _ws_per_id_awake_count[tid]
            
            # Determine effective state with temporal smoothing
            eff_state = prev_state
            if prev_state in ('drowsy', 'sleeping'):
                # Currently drowsy/sleeping → check if waking up
                if state_now == 'awake' and awake_cnt >= _ws_awake_frames_required:
                    eff_state = 'wake_up'
            elif prev_state == 'wake_up':
                # Just woke up → transition to fully awake
                if awake_cnt >= _ws_awake_frames_required:
                    eff_state = 'awake'
            else:
                # Currently awake → check if falling asleep
                if state_now in ('drowsy', 'sleeping') and sleep_cnt >= _ws_sleep_frames_required:
                    eff_state = state_now
            
            # 🔥 LOG STATE TRANSITIONS
            if eff_state != prev_state:
                if eff_state in ('drowsy', 'sleeping'):
                    # Started drowsiness
                    _ws_per_id_sleep_start[tid] = now
                    append_log({
                        'camera_id': cam_id,
                        'track_id': tid,
                        'type': 'sleepy' if eff_state == 'drowsy' else 'head_down',
                        'state': 'Ngủ gật' if eff_state == 'drowsy' else 'Gục xuống bàn',
                        'ts': now
                    })
                    
                    # 🔥 Log to drowsiness logger
                    if LOGGER_AVAILABLE:
                        try:
                            logger = get_global_logger()
                            # Register camera if not already (webcam detection)
                            if not hasattr(logger, '_registered_webcam'):
                                logger.register_camera(cam_id, "WebSocket Camera")
                                logger._registered_webcam = True
                                app.logger.info(f"[WS] Registered webcam '{cam_id}' with drowsiness logger")
                            
                            logger.update_student_state(cam_id, tid, True)
                            app.logger.info(f"[WS] 🔴 Học sinh #{tid} BẮT ĐẦU {eff_state} (camera: {cam_id})")
                        except Exception as log_err:
                            app.logger.debug(f"[WS] Logger error (start drowsy): {log_err}")
                
                elif eff_state == 'wake_up':
                    # Waking up
                    dur = 0.0
                    if tid in _ws_per_id_sleep_start:
                        dur = now - _ws_per_id_sleep_start[tid]
                        del _ws_per_id_sleep_start[tid]
                    
                    append_log({
                        'camera_id': cam_id,
                        'track_id': tid,
                        'type': 'wake_up',
                        'state': 'Thức dậy',
                        'duration': dur,
                        'ts': now
                    })
                    
                    # 🔥 Log wake up to drowsiness logger
                    if LOGGER_AVAILABLE:
                        try:
                            logger = get_global_logger()
                            logger.update_student_state(cam_id, tid, False)
                            app.logger.info(f"[WS] 🟢 Học sinh #{tid} THỨC DẬY sau {dur:.1f}s (camera: {cam_id})")
                        except Exception as log_err:
                            app.logger.debug(f"[WS] Logger error (wake up): {log_err}")
            
            # Update state
            _ws_per_id_state[tid] = eff_state
        
        # serialize persons
        persons = []
        for p in det.persons:
            kpts = [{
                'x': float(k.x), 'y': float(k.y),
                'confidence': float(k.confidence), 'visible': bool(k.visible)
            } for k in p.keypoints]
            persons.append({
                'id': int(getattr(p, 'id', 0) or 0),
                'track_id': int(getattr(p, 'track_id', getattr(p, 'id', 0)) or 0),
                'bbox': [float(v) for v in list(p.bbox)],
                'head_bbox': [float(v) for v in list(getattr(p, 'head_bbox', []) or [])] if getattr(p, 'head_bbox', None) is not None else None,
                'confidence': float(getattr(p, 'confidence', 0.0) or 0.0),
                'keypoints': kpts,
                'drowsiness_score': float(getattr(p, 'drowsiness_score', 0.0) or 0.0),
                'drowsiness_state': str(getattr(p, 'drowsiness_state', 'awake') or 'awake'),
                'last_update': float(getattr(p, 'last_update', time.time()))
            })
        # Debug: log detection summary
        try:
            app.logger.info(f"WS detect: persons={len(persons)} size={w}x{h} fps={float(getattr(det, 'fps', 0.0) or 0.0):.2f}")
        except Exception:
            pass
        emit('result', {
            'success': True,
            'camera_id': cam_id,
            'schema': 'v1',
            'frame_width': w,
            'frame_height': h,
            'fps': float(getattr(det, 'fps', 0.0) or 0.0),
            'processing_time': float(getattr(det, 'processing_time', 0.0) or 0.0),
            'persons': persons,
            'timestamp': float(getattr(det, 'timestamp', time.time()))
        })
    except Exception as e:
        app.logger.error(f"WS frame error: {e}")
        emit('result', {'success': False, 'error': str(e)})


# ==================== Initialize YOLO on Startup ====================

# Realtime WS for camera workers (rooms per camera)
@socketio.on('connect', namespace='/ws/camera')
def ws_cam_connect():
    app.logger.info('WS client connected to /ws/camera')
    emit('hello', {'msg': 'connected'})


@socketio.on('disconnect', namespace='/ws/camera')
def ws_cam_disconnect():
    app.logger.info('WS client disconnected from /ws/camera')


@socketio.on('subscribe', namespace='/ws/camera')
def ws_cam_subscribe(data):
    try:
        cam_id = (data or {}).get('camera_id')
        if not cam_id:
            emit('error', {'success': False, 'error': 'camera_id required'})
            return
        join_room(f'cam:{cam_id}')
        app.logger.info(f"WS subscribed to camera room cam:{cam_id}")
        emit('subscribed', {'success': True, 'camera_id': cam_id})

        # Proactively send a snapshot update so the client can render immediately
        try:
            worker = manager.get_worker(cam_id)
            frame_width, frame_height = manager.get_frame_dimensions(cam_id) or (640, 360)
            fps = worker.get_fps() if worker else 0.0
            result = manager.get_detection_result(cam_id)
            persons_payload = []
            if result and getattr(result, 'persons', None):
                for p in result.persons:
                    kpts = [{
                        'x': float(k.x), 'y': float(k.y),
                        'confidence': float(k.confidence), 'visible': bool(k.visible)
                    } for k in p.keypoints]
                    persons_payload.append({
                        'id': int(getattr(p, 'id', 0) or 0),
                        'track_id': int(getattr(p, 'track_id', getattr(p, 'id', 0)) or 0),
                        'bbox': [float(v) for v in list(p.bbox)],
                        'head_bbox': [float(v) for v in list(getattr(p, 'head_bbox', []) or [])] if getattr(p, 'head_bbox', None) is not None else None,
                        'confidence': float(getattr(p, 'confidence', 0.0) or 0.0),
                        'keypoints': kpts,
                        'drowsiness_score': float(getattr(p, 'drowsiness_score', 0.0) or 0.0),
                        'drowsiness_state': str(getattr(p, 'drowsiness_state', 'awake') or 'awake'),
                        'last_update': float(getattr(p, 'last_update', time.time()))
                    })
            socketio.emit('update', {
                'success': True,
                'schema': 'v1',
                'camera_id': cam_id,
                'frame_width': int(frame_width or 0),
                'frame_height': int(frame_height or 0),
                'fps': float(fps or 0.0),
                'processing_time': float(getattr(result, 'processing_time', 0.0) or 0.0),
                'persons': persons_payload,
                'timestamp': time.time(),
            }, namespace='/ws/camera', to=f'cam:{cam_id}')
        except Exception as _snap_e:
            app.logger.debug(f"WS snapshot emit failed for cam:{cam_id}: {_snap_e}")
    except Exception as e:
        emit('error', {'success': False, 'error': str(e)})


@socketio.on('unsubscribe', namespace='/ws/camera')
def ws_cam_unsubscribe(data):
    try:
        cam_id = (data or {}).get('camera_id')
        if not cam_id:
            emit('error', {'success': False, 'error': 'camera_id required'})
            return
        leave_room(f'cam:{cam_id}')
        app.logger.info(f"WS unsubscribed from camera room cam:{cam_id}")
        emit('unsubscribed', {'success': True, 'camera_id': cam_id})
    except Exception as e:
        emit('error', {'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Initialize Drowsiness Logger
    if LOGGER_AVAILABLE:
        app.logger.info("Initializing Drowsiness Logger...")
        try:
            log_dir = os.path.join(os.path.dirname(__file__), 'drowsiness_logs')
            init_logger(log_dir)
            app.logger.info(f"✅ Drowsiness Logger initialized successfully (log_dir: {log_dir})")
        except Exception as e:
            app.logger.error(f"❌ Failed to initialize Drowsiness Logger: {e}")
    
    # Initialize YOLO detector
    if YOLO_AVAILABLE:
        app.logger.info("Initializing YOLO detector...")
        try:
            # Resolve model weights path robustly
            backend_dir = os.path.dirname(__file__)
            root_dir = os.path.dirname(os.path.dirname(backend_dir))
            
            # PRIORITY ORDER (HIGHEST TO LOWEST):
            # 1. Custom trained models for drowsiness detection (BEST - trained on 1000 epochs)
            # 2. Backup trained models from old workspace
            # 3. Generic pose models (fallback if no trained model)
            variants = [
                # Custom trained models in backend/models directory (HIGHEST PRIORITY)
                os.path.join(backend_dir, 'models', 'sleepy_pose_v11n_full_best.pt'),
                os.path.join(backend_dir, 'models', 'sleepy_pose_v11n3_best.pt'),
                # Backup: old workspace
                os.path.join(root_dir, '..', 'DACN_PhatHienNguGat_Old', 'yolo-sleepy-allinone-final', 'runs', 'pose-train', 'sleepy_pose_v11n_full', 'weights', 'best.pt'),
                os.path.join(root_dir, '..', 'DACN_PhatHienNguGat_Old', 'yolo-sleepy-allinone-final', 'runs', 'pose-train', 'sleepy_pose_v11n3', 'weights', 'best.pt'),
                # Legacy paths (old structure)
                os.path.join(root_dir, 'yolo-sleepy-allinone-final', 'best.pt'),
                os.path.join(root_dir, 'yolo-sleepy-allinone-final', 'runs', 'pose', 'train', 'weights', 'best.pt'),
                os.path.join(root_dir, 'yolo-sleepy-allinone-final', 'weights', 'best.pt'),
                # Generic pretrained models (LOWEST PRIORITY - fallback only)
                'yolo11m-pose.pt', 'yolo11s-pose.pt', 'yolo11n-pose.pt'
            ]
            search_dirs = [backend_dir, root_dir, os.path.dirname(backend_dir)]
            model_path = None
            for name in variants:
                # Absolute paths may already be combined above
                candidate_paths = [name] if os.path.isabs(name) else [os.path.join(d, name) for d in search_dirs]
                for p in candidate_paths:
                    if os.path.exists(p):
                        model_path = p
                        if 'sleepy_pose' in p or 'best.pt' in p:
                            app.logger.info(f"✅ Using TRAINED drowsiness detection model: {model_path}")
                        else:
                            app.logger.info(f"Found local model weights: {model_path}")
                        break
                if model_path:
                    break
            if model_path:
                initialize_detector(model_path)
            else:
                app.logger.warning("⚠️ No trained model found, using default pretrained model (less accurate for drowsiness)")
                app.logger.info("No local weights found; using alias 'yolo11n-pose.pt' (Ultralytics may auto-download)")
                initialize_detector('yolo11n-pose.pt')
            app.logger.info("✅ YOLO detector initialized successfully")
        except Exception as e:
            app.logger.error(f"❌ Failed to initialize YOLO detector: {e}")
    
    # Start Flask server
    app.logger.info("Starting Flask+SocketIO server with enhanced tracking...")
    # Werkzeug safety guard (Flask 3+): allow in this desktop dev context
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)

