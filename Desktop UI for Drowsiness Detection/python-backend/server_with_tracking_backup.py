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
from flask import Flask, request, jsonify, Response, make_response
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
                                    'camera_id': self.cam_id,
                                    'frame_width': int(fw or 0),
                                    'frame_height': int(fh or 0),
                                    'fps': float(self._current_fps or 0.0),
                                    'persons': persons_payload,
                                    'timestamp': now_ts,
                                }, namespace='/ws/camera', room=f'cam:{self.cam_id}')
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
                elif eff_state == 'Thức dậy':
                    dur = 0.0
                    if tid in self._per_id_sleep_start:
                        dur = now - self._per_id_sleep_start[tid]
                        del self._per_id_sleep_start[tid]
                    append_log({'camera_id': self.cam_id, 'track_id': tid, 'type': 'wake_up', 'state': eff_state, 'duration': dur, 'ts': now})

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


@socketio.on('frame', namespace='/ws/detect')
def ws_frame(data):
    """Receive base64 frame, run detection, emit result immediately."""
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
            'frame_width': w,
            'frame_height': h,
            'fps': float(getattr(det, 'fps', 0.0) or 0.0),
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
                'camera_id': cam_id,
                'frame_width': int(frame_width or 0),
                'frame_height': int(frame_height or 0),
                'fps': float(fps or 0.0),
                'persons': persons_payload,
                'timestamp': time.time(),
            }, namespace='/ws/camera', room=f'cam:{cam_id}')
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
    # Initialize YOLO detector
    if YOLO_AVAILABLE:
        app.logger.info("Initializing YOLO detector...")
        try:
            # Resolve model weights path robustly, prefer larger models for better accuracy
            backend_dir = os.path.dirname(__file__)
            root_dir = os.path.dirname(os.path.dirname(backend_dir))
            # preference order: user's trained best.pt > 11m > 11s > 11n
            variants = [
                os.path.join(root_dir, 'yolo-sleepy-allinone-final', 'best.pt'),
                os.path.join(root_dir, 'yolo-sleepy-allinone-final', 'runs', 'pose', 'train', 'weights', 'best.pt'),
                os.path.join(root_dir, 'yolo-sleepy-allinone-final', 'weights', 'best.pt'),
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
                        app.logger.info(f"Found local model weights: {model_path}")
                        break
                if model_path:
                    break
            if model_path:
                initialize_detector(model_path)
            else:
                app.logger.info("No local weights found; using alias 'yolo11n-pose.pt' (Ultralytics may auto-download)")
                initialize_detector('yolo11n-pose.pt')
            app.logger.info("✅ YOLO detector initialized successfully")
        except Exception as e:
            app.logger.error(f"❌ Failed to initialize YOLO detector: {e}")
    
    # Start Flask server
    app.logger.info("Starting Flask+SocketIO server with enhanced tracking...")
    # Werkzeug safety guard (Flask 3+): allow in this desktop dev context
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)

