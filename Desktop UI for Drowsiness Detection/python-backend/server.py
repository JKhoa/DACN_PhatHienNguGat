import sys
# Force UTF-8 stdout/stderr so print('✅') etc. don't crash when piped or on
# terminals whose default encoding is cp1252 (Windows). Must run before any
# module that uses print() with non-ASCII characters.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import threading
import time
import base64
import io
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import YOLO detector
try:
    from yolo_detector import initialize_detector, detect_frame, draw_detections, DetectionResult, get_detector
    YOLO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"YOLO detector not available: {e}")
    YOLO_AVAILABLE = False

# Drowsiness logger (per-student state + DB persistence)
try:
    from drowsiness_logger import get_global_logger, init_logger
    LOGGER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"drowsiness_logger not available: {e}")
    LOGGER_AVAILABLE = False

# Chatbot (natural-language queries over logs DB)
try:
    from chatbot import handle_question as _chatbot_handle
    CHATBOT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"chatbot not available: {e}")
    CHATBOT_AVAILABLE = False

app = Flask(__name__)
CORS(app)

_log_format = '[%(asctime)s] %(levelname)s: %(message)s'
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
# Console handler (giữ nguyên hành vi cũ)
if not any(isinstance(h, logging.StreamHandler) for h in _root_logger.handlers):
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter(_log_format))
    _root_logger.addHandler(_console)
# Rotating file handler: logs/server.log, 10MB x 5 backup
try:
    _logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(_logs_dir, exist_ok=True)
    _file_handler = RotatingFileHandler(
        os.path.join(_logs_dir, 'server.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    _file_handler.setFormatter(logging.Formatter(_log_format))
    _root_logger.addHandler(_file_handler)
except Exception as _e:
    print(f"[WARN] Could not init rotating file log: {_e}")

# Register /api/v1/detect/... blueprint (Vietnamese ensemble pipeline)
try:
    from api_v1 import bp_api_v1
    app.register_blueprint(bp_api_v1)
    app.logger.info("Registered api_v1 blueprint")
except Exception as e:
    app.logger.warning(f"api_v1 blueprint not registered: {e}")

# Initialize logger + pre-register historical cameras from DB
if LOGGER_AVAILABLE:
    try:
        _log_dir = os.path.join(os.path.dirname(__file__), 'drowsiness_logs')
        init_logger(_log_dir)
        _glogger = get_global_logger()
        try:
            from db_helper import get_database
            _db = get_database()
            _cursor = _db._get_connection().cursor()
            _cursor.execute("SELECT DISTINCT camera_id, camera_name FROM drowsy_events")
            for _row in _cursor.fetchall():
                _glogger.register_camera(_row[0], _row[1])
            app.logger.info(f"Registered {len(_glogger.cameras)} cameras from database")
        except Exception as _e:
            app.logger.warning(f"Could not pre-load cameras from DB: {_e}")
    except Exception as e:
        app.logger.error(f"Failed to init drowsiness logger: {e}")

# Initialize SocketIO for WebSocket support
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# Register api_v1 realtime websocket namespace (if blueprint was loaded)
try:
    from api_v1 import register_realtime_ws
    register_realtime_ws(socketio)
    app.logger.info("Registered api_v1 realtime WS namespace")
except Exception as e:
    app.logger.warning(f"api_v1 realtime WS not registered: {e}")


# ── Enhanced Tracker (IoU-based multi-student tracking) ─────────────────────
def iou_xyxy(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
    return inter / union if union > 0 else 0.0


class EnhancedTracker:
    def __init__(self, iou_thr: float = 0.25, max_age: int = 25):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1

    def update(self, detections):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] = self.tracks[tid].get("age", 0) + 1
        det_boxes = []
        for det in detections:
            hb = getattr(det, 'head_bbox', None)
            if hb and hb[0] > 0:
                det_boxes.append(tuple(hb))
            else:
                x1, y1, x2, y2 = det.bbox
                det_boxes.append((x1, y1, x2, y1 + (y2 - y1) * 0.3))
        used_dets = set()
        while True:
            best_tid, best_di, best_iou = None, None, 0.0
            for tid, tr in self.tracks.items():
                if tr.get("age", 0) > self.max_age:
                    continue
                tb = tr.get("head_bbox", tr.get("bbox"))
                for di, dbox in enumerate(det_boxes):
                    if di in used_dets:
                        continue
                    ov = iou_xyxy(tb, dbox)
                    if ov > best_iou:
                        best_tid, best_di, best_iou = tid, di, ov
            if best_tid is None or best_iou < self.iou_thr:
                break
            det = detections[best_di]
            hb = getattr(det, 'head_bbox', None)
            self.tracks[best_tid].update({
                "bbox": det.bbox,
                "head_bbox": hb if (hb and hb[0] > 0) else det.bbox,
                "age": 0,
            })
            det.track_id = best_tid
            used_dets.add(best_di)
        for di, det in enumerate(detections):
            if di not in used_dets:
                tid = self.next_id
                self.next_id += 1
                hb = getattr(det, 'head_bbox', None)
                self.tracks[tid] = {
                    "bbox": det.bbox,
                    "head_bbox": hb if (hb and hb[0] > 0) else det.bbox,
                    "age": 0,
                }
                det.track_id = tid
        active_tids = {d.track_id for d in detections}
        for tid in list(self.tracks.keys()):
            if tid not in active_tids and self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return detections


class CameraWorker(threading.Thread):
    def __init__(self, cam_id: str, url: str, enable_detection: bool = True):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.url = url
        self.enable_detection = enable_detection and YOLO_AVAILABLE
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_detection_result: Optional[DetectionResult] = None
        self._last_annotated_frame: Optional[np.ndarray] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._detection_enabled = self.enable_detection
        self._frame_width = 640
        self._frame_height = 360
        self._current_fps = 0.0
        self._tracker = EnhancedTracker()
        self._per_id_state: Dict[int, str] = {}

        if LOGGER_AVAILABLE:
            try:
                get_global_logger().register_camera(cam_id, cam_id.split('/')[-1])
            except Exception as e:
                app.logger.warning(f"[{cam_id}] logger register failed: {e}")
        
        app.logger.info(f"[{self.cam_id}] CameraWorker initialized with URL: {url}, detection: {self._detection_enabled}")
        
        # Auto-enable detection for webcam (device ID)
        if url.isdigit() or url.startswith('0'):
            self._detection_enabled = True and YOLO_AVAILABLE
            app.logger.info(f"[{self.cam_id}] Auto-enabling YOLO detection for webcam")

    def _open_capture(self) -> bool:
        """Mở capture từ url. Trả True nếu mở thành công và đọc được frame."""
        try:
            if self.url.isdigit():
                self._capture = cv2.VideoCapture(int(self.url), cv2.CAP_DSHOW)
                app.logger.info(f"[{self.cam_id}] Using DirectShow backend for device {self.url}")
            elif self.url.startswith('webcam-') or len(self.url) > 20:
                self._capture = None
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
                self._capture = cv2.VideoCapture(self.url)
                app.logger.info(f"[{self.cam_id}] Using IP camera: {self.url}")

            if not self._capture or not self._capture.isOpened():
                app.logger.error(f"[{self.cam_id}] Failed to open camera")
                return False

            self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._frame_width
            self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._frame_height
            app.logger.info(f"[{self.cam_id}] Camera opened, frame size: {self._frame_width}x{self._frame_height}")
            return True
        except Exception as e:
            app.logger.error(f"[{self.cam_id}] Exception opening capture: {e}")
            return False

    def _release_capture(self):
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def run(self):
        app.logger.info(f"[{self.cam_id}] Starting camera worker thread...")
        if not self._open_capture():
            return

        # Main loop
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        consecutive_failures = 0
        reopen_attempts = 0
        FAIL_THRESHOLD = 30        # ~3s ở 10Hz read fail
        MAX_REOPEN = 5             # bỏ cuộc sau 5 lần reopen liên tiếp

        while self._running.is_set():
            try:
                ok, frame = self._capture.read() if self._capture is not None else (False, None)
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures % 10 == 1:
                        app.logger.warning(f"[{self.cam_id}] Read failed ({consecutive_failures})")
                    if consecutive_failures >= FAIL_THRESHOLD:
                        reopen_attempts += 1
                        if reopen_attempts > MAX_REOPEN:
                            app.logger.error(f"[{self.cam_id}] Reached {MAX_REOPEN} reopen attempts, giving up")
                            break
                        backoff = min(2 ** (reopen_attempts - 1), 30)
                        app.logger.warning(f"[{self.cam_id}] Attempting reopen #{reopen_attempts} after {backoff}s")
                        self._release_capture()
                        time.sleep(backoff)
                        if self._open_capture():
                            consecutive_failures = 0
                            app.logger.info(f"[{self.cam_id}] Reopen #{reopen_attempts} succeeded")
                        else:
                            consecutive_failures = 0  # reset, vòng sau sẽ thử lại nếu read tiếp tục fail
                    else:
                        time.sleep(0.1)
                    continue

                # Read thành công - reset counter
                if consecutive_failures > 0 or reopen_attempts > 0:
                    consecutive_failures = 0
                    reopen_attempts = 0
                
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
                    with self._lock:
                        self._current_fps = current_fps
                    fps_frame_count = 0
                    fps_start_time = time.time()
                if frame_count % 30 == 0:  # Log every 30 frames
                        app.logger.info(f"[{self.cam_id}] Frame {frame_count}, FPS: {current_fps:.1f}, size: {frame.shape}")
                
                with self._lock:
                    self._last_frame = frame
                    
                    # Run YOLO detection if enabled
                    if self._detection_enabled and YOLO_AVAILABLE:
                        try:
                            detection_result = detect_frame(frame)
                            # Update FPS in detection result
                            if hasattr(detection_result, 'fps'):
                                detection_result.fps = self._current_fps
                            # Apply IoU tracker + forward state transitions to logger
                            if detection_result and getattr(detection_result, 'persons', None):
                                detection_result.persons = self._tracker.update(detection_result.persons)
                                self._process_logs(detection_result.persons)
                            self._last_detection_result = detection_result

                            # Create annotated frame with detections
                            annotated_frame = draw_detections(frame, detection_result)
                            self._last_annotated_frame = annotated_frame
                        except Exception as e:
                            app.logger.error(f"[{self.cam_id}] Detection failed: {e}")
                            self._last_detection_result = None
                            self._last_annotated_frame = None
                
                # Small sleep to reduce CPU usage
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                app.logger.error(f"[{self.cam_id}] Exception in main loop: {e}")
                time.sleep(0.1)
        
        # Cleanup
        if self._capture is not None:
            try:
                self._capture.release()
                app.logger.info(f"[{self.cam_id}] Camera released")
            except Exception as e:
                app.logger.error(f"[{self.cam_id}] Error releasing camera: {e}")
        
        app.logger.info(f"[{self.cam_id}] Worker stopped")

    def stop(self):
        self._running.clear()

    def _process_logs(self, persons):
        """Forward per-student drowsiness state transitions to the global logger."""
        if not LOGGER_AVAILABLE:
            return
        try:
            logger = get_global_logger()
        except Exception:
            return
        for p in persons:
            tid = int(getattr(p, 'track_id', 0) or 0)
            if tid <= 0:
                continue
            state = str(getattr(p, 'drowsiness_state', 'awake'))
            prev = self._per_id_state.get(tid, 'awake')
            if state == prev:
                continue
            drowsy_now = state in ('drowsy', 'sleeping')
            drowsy_prev = prev in ('drowsy', 'sleeping')
            if drowsy_now and not drowsy_prev:
                try:
                    logger.update_student_state(self.cam_id, tid, True)
                except Exception as e:
                    app.logger.warning(f"[{self.cam_id}] logger update (enter) failed: {e}")
            elif drowsy_prev and not drowsy_now:
                try:
                    logger.update_student_state(self.cam_id, tid, False)
                except Exception as e:
                    app.logger.warning(f"[{self.cam_id}] logger update (exit) failed: {e}")
            self._per_id_state[tid] = state

    def emit_ws_update(self):
        """Emit WebSocket update for this camera (called by background thread)"""
        try:
            with self._lock:
                result = self._last_detection_result
                frame_width = self._frame_width
                frame_height = self._frame_height
                fps = self._current_fps
            
            if result is None:
                return
            
            # Convert persons to JSON
            persons_payload = []
            if hasattr(result, 'persons'):
                for p in result.persons:
                    kpts = [{'x': float(k.x), 'y': float(k.y), 'confidence': float(k.confidence), 'visible': bool(k.visible)} for k in p.keypoints]
                    persons_payload.append({
                        'id': int(p.id),
                        'track_id': int(getattr(p, 'track_id', p.id)),
                        'bbox': [float(v) for v in p.bbox],
                        'head_bbox': [float(v) for v in getattr(p, 'head_bbox', [])] if getattr(p, 'head_bbox', None) else None,
                        'confidence': float(p.confidence),
                        'keypoints': kpts,
                        'drowsiness_score': float(p.drowsiness_score),
                        'drowsiness_state': str(p.drowsiness_state),
                        'last_update': float(p.last_update)
                    })
            
            # Emit to WebSocket room
            socketio.emit('update', {
                'success': True,
                'camera_id': self.cam_id,
                'frame_width': int(frame_width),
                'frame_height': int(frame_height),
                'fps': float(fps),
                'processing_time': float(getattr(result, 'processing_time', 0.0)),
                'persons': persons_payload,
                'timestamp': time.time()
            }, namespace='/ws/camera', to=f'cam:{self.cam_id}')
            
        except Exception as e:
            app.logger.error(f"[{self.cam_id}] WS emit error: {e}")

    def get_last_jpeg(self, annotated: bool = False) -> Optional[bytes]:
        with self._lock:
            frame_to_encode = None
            
            if annotated and self._last_annotated_frame is not None:
                frame_to_encode = self._last_annotated_frame
            elif self._last_frame is not None:
                frame_to_encode = self._last_frame
            else:
                return None
                
            ok, buf = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            return buf.tobytes()
    
    def get_detection_result(self) -> Optional[DetectionResult]:
        with self._lock:
            result = self._last_detection_result
            if result is not None and hasattr(result, 'fps'):
                result.fps = self._current_fps
            return result
    
    def get_frame_dimensions(self):
        return self._frame_width, self._frame_height
    
    def get_fps(self) -> float:
        with self._lock:
            return self._current_fps
    
    def toggle_detection(self, enabled: bool):
        self._detection_enabled = enabled and YOLO_AVAILABLE


class CameraManager:
    def __init__(self):
        self._workers: Dict[str, CameraWorker] = {}
        self._meta: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._ws_broadcaster_running = False
        self._ws_broadcaster_thread = None
    
    def start_ws_broadcaster(self):
        """Start background thread to emit WebSocket updates for all IP cameras"""
        if self._ws_broadcaster_running:
            return
        
        self._ws_broadcaster_running = True
        
        def broadcast_loop():
            app.logger.info("🔄 WebSocket broadcaster thread started")
            while self._ws_broadcaster_running:
                try:
                    with self._lock:
                        workers_snapshot = list(self._workers.items())
                    
                    for cam_id, worker in workers_snapshot:
                        if worker.is_alive():
                            worker.emit_ws_update()
                    
                    time.sleep(0.15)  # Emit at ~6-7 Hz
                except Exception as e:
                    app.logger.error(f"WS broadcaster error: {e}")
                    time.sleep(1.0)
            
            app.logger.info("🛑 WebSocket broadcaster thread stopped")
        
        self._ws_broadcaster_thread = threading.Thread(target=broadcast_loop, daemon=True)
        self._ws_broadcaster_thread.start()
    
    def stop_ws_broadcaster(self):
        """Stop the WebSocket broadcaster thread"""
        self._ws_broadcaster_running = False
        if self._ws_broadcaster_thread:
            self._ws_broadcaster_thread.join(timeout=2.0)

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
            # If worker exists but is dead, remove it first
            if cam_id in self._workers:
                if not self._workers[cam_id].is_alive():
                    app.logger.info(f"[{cam_id}] Removing dead worker")
                    self._workers.pop(cam_id, None)
                else:
                    # Worker is alive - just update detection if needed
                    app.logger.info(f"[{cam_id}] Camera worker already running, toggling detection: {enable_detection}")
                    self._workers[cam_id].toggle_detection(enable_detection)
                return
            try:
                worker = CameraWorker(cam_id, self._meta[cam_id]['url'], enable_detection)
                self._workers[cam_id] = worker
                worker.start()
                app.logger.info(f"[{cam_id}] Camera worker started successfully with detection={enable_detection}")
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
            # Ensure FPS is set
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


def _is_local_webcam_url(url: str) -> bool:
    """Reject local webcam device IDs — browser holds webcam in web mode,
    so backend opening it would contend for the device."""
    if not url:
        return False
    u = str(url).strip()
    if u.isdigit():
        return True
    if u.startswith('webcam-') or u.startswith('device-'):
        return True
    return False


@app.route('/api/camera/add', methods=['POST'])
def add_camera():
    data = request.get_json(force=True, silent=True) or {}
    cam_id = data.get('id') or data.get('name')
    url = data.get('url')
    name = data.get('name')
    if not cam_id or not url:
        return jsonify({'success': False, 'error': 'id/name and url required'}), 400
    if _is_local_webcam_url(url):
        return jsonify({
            'success': False,
            'error': 'Local webcam is handled by the frontend (/ws/detect). Only IP/RTSP URLs are accepted here.',
        }), 400
    try:
        manager.add(cam_id, url, name)
        return jsonify({'success': True})
    except ValueError as e:
        # Camera already exists - update URL and name if different, then return success
        with manager._lock:
            if cam_id in manager._meta:
                # Update URL and name if provided
                if url:
                    manager._meta[cam_id]['url'] = url
                if name:
                    manager._meta[cam_id]['name'] = name
                app.logger.info(f"[{cam_id}] Camera already exists, updated metadata")
                return jsonify({'success': True, 'message': 'Camera already exists, updated'})
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
    try:
        if not manager.has_camera(cam_id):
            return jsonify({'success': False, 'error': f'Camera {cam_id} not found'}), 404
        
        annotated = request.args.get('annotated', 'false').lower() == 'true'
        jpeg = manager.get_jpeg(cam_id, annotated)
        if jpeg is None:
            return jsonify({'success': False, 'error': 'no frame available'}), 404
        b64 = base64.b64encode(jpeg).decode('utf-8')
        return jsonify({'success': True, 'frame': b64, 'ts': time.time()})
    except Exception as e:
        app.logger.error(f"[{cam_id}] Error streaming frame: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/<cam_id>/detection', methods=['GET'])
def get_detection_results(cam_id):
    """Get detection results for a specific camera"""
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
    app.logger.info("Received request to /api/detect/frame")
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    if not YOLO_AVAILABLE:
        app.logger.error("YOLO detector not available")
        return jsonify({'success': False, 'error': 'YOLO detector not available'}), 503
    
    try:
        data = request.get_json(force=True, silent=True) or {}
        frame_base64 = data.get('frame')
        
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
            app.logger.info(f"Decoded frame: {w}x{h}")
            
        except Exception as e:
            app.logger.error(f"Error decoding frame: {e}")
            return jsonify({'success': False, 'error': f'Failed to decode frame: {str(e)}'}), 400
        
        # Run detection
        app.logger.info(f"Running detection on frame {frame.shape}")
        result = detect_frame(frame)
        app.logger.info(f"Detection completed: {len(result.persons)} persons detected")
        
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
                'fps': result.fps,
                'processing_time': result.processing_time,
                'frame_width': w,
                'frame_height': h
            }
        })
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in detect_frame_endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    uptime = int(time.time() - _start_time)
    cams = manager.list()
    
    # Get detection stats
    detection_stats = {
        'yolo_available': YOLO_AVAILABLE,
        'detector_initialized': get_detector() is not None,
        'total_detections': 0,
        'drowsy_count': 0,
        'sleeping_count': 0
    }
    
    # Count detection results across all cameras
    for cam in cams:
        result = manager.get_detection_result(cam['id'])
        if result:
            detection_stats['total_detections'] += len(result.persons)
            for person in result.persons:
                if person.drowsiness_state == 'drowsy':
                    detection_stats['drowsy_count'] += 1
                elif person.drowsiness_state == 'sleeping':
                    detection_stats['sleeping_count'] += 1
    
    return jsonify({
        'success': True, 
        'uptime_s': uptime, 
        'cameras': len(cams),
        'detection_stats': detection_stats
    })


@app.route('/api/camera/<cam_id>/detection/toggle', methods=['POST'])
def toggle_detection(cam_id):
    """Toggle detection on/off for a specific camera"""
    data = request.get_json(force=True, silent=True) or {}
    enabled = data.get('enabled', True)
    
    manager.toggle_detection(cam_id, enabled)
    return jsonify({'success': True, 'detection_enabled': enabled})


@app.route('/api/detection/initialize', methods=['POST'])
def initialize_detection():
    """Initialize the YOLO detector"""
    if not YOLO_AVAILABLE:
        return jsonify({'success': False, 'error': 'YOLO not available'}), 400
    
    data = request.get_json(force=True, silent=True) or {}
    model_path = data.get('model_path', 'yolo11n-pose.pt')
    
    success = initialize_detector(model_path)
    if success:
        return jsonify({'success': True, 'message': 'YOLO detector initialized'})
    else:
        return jsonify({'success': False, 'error': 'Failed to initialize detector'}), 500


# ==================== Logs & Chatbot Routes ====================

@app.route('/api/logs/summary', methods=['GET'])
def api_logs_summary():
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    period = request.args.get('period', 'today')
    return jsonify({'success': True, 'summary': get_global_logger().get_summary_stats(period)})


@app.route('/api/logs/stats', methods=['GET'])
def api_logs_stats():
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    period = request.args.get('period', 'today')
    stats = get_global_logger().get_all_cameras_stats(period)
    return jsonify({'success': True, 'stats': stats, 'camera_stats': stats})


@app.route('/api/logs/events/<path:camera_id>', methods=['GET'])
def api_logs_events(camera_id):
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    period = request.args.get('period', 'today')
    try:
        events = get_global_logger().get_camera_events(camera_id, period)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True, 'camera_id': camera_id, 'events': events})


@app.route('/api/logs/cameras', methods=['GET'])
def api_logs_cameras():
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    logger = get_global_logger()
    cams = []
    try:
        for cid, cam_logger in logger.cameras.items():
            cams.append({
                'id': cid,
                'name': getattr(cam_logger, 'camera_name', cid),
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True, 'cameras': cams})


@app.route('/api/logs/active', methods=['GET'])
def api_logs_active():
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    active = get_global_logger().get_active_drowsy_all_cameras()
    flat = []
    for cid, sts in active.items():
        for s in sts:
            flat.append({
                'camera_id': cid,
                'student_id': s.get('student_id'),
                'current_duration_display': s.get('duration_display'),
                'duration_display': s.get('duration_display'),
            })
    # Emit both keys so UI (App.tsx uses active_students, DashboardPanel uses active_drowsy_students) works unchanged.
    return jsonify({'success': True, 'active_students': flat, 'active_drowsy_students': flat})


def _parse_iso(ts: str, default=None):
    """Parse ISO8601 with tolerance for trailing Z."""
    if not ts:
        return default
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts.replace('Z', '+00:00')).replace(tzinfo=None)
    except Exception:
        return default


def _time_slot(hour: int) -> str:
    if 6 <= hour < 12:
        return 'morning'
    if 12 <= hour < 18:
        return 'afternoon'
    return 'evening'


_DROWSY_DURATION_THRESHOLD_SEC = 60.0  # < this = "drowsy" (chớm); >= = "sleeping" (ngủ thực sự)


@app.route('/api/logs/export/<fmt>', methods=['POST'])
def api_logs_export(fmt):
    """Generate and return a PDF/Excel report as a binary attachment.
    Body: {period: 'today'|'week'|'month'|..., camera_ids?: [str,...]}
    """
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    fmt = (fmt or '').lower()
    if fmt not in ('pdf', 'excel', 'xlsx'):
        return jsonify({'success': False, 'error': 'format must be pdf or excel'}), 400
    try:
        body = request.get_json(force=True, silent=True) or {}
        period = body.get('period', 'today')
        camera_ids = body.get('camera_ids') or None

        logger = get_global_logger()
        camera_stats = logger.get_all_cameras_stats(period)
        if camera_ids:
            camera_stats = [s for s in camera_stats if s.get('camera_id') in camera_ids]
        summary = logger.get_summary_stats(period)

        # Gather events across requested cameras
        events: list = []
        target_ids = camera_ids if camera_ids else list(logger.cameras.keys())
        for cid in target_ids:
            try:
                for ev in logger.get_camera_events(cid, period):
                    ev_with_cam = dict(ev)
                    ev_with_cam.setdefault('camera_id', cid)
                    ev_with_cam.setdefault('camera_name', getattr(logger.cameras.get(cid), 'camera_name', cid))
                    events.append(ev_with_cam)
            except Exception as e:
                app.logger.warning(f"export: skipping camera {cid}: {e}")

        from report_generator import get_report_generator
        gen = get_report_generator(os.path.join(os.path.dirname(__file__), 'reports'))
        if fmt == 'pdf':
            filepath = gen.generate_pdf_report(period, camera_stats, summary, events, camera_ids)
            mimetype = 'application/pdf'
        else:
            filepath = gen.generate_excel_report(period, camera_stats, summary, events)
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

        from flask import send_file
        download_name = os.path.basename(filepath)
        return send_file(filepath, mimetype=mimetype, as_attachment=True, download_name=download_name)
    except Exception as e:
        app.logger.error(f"/api/logs/export/{fmt} error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    """Aggregated statistics for StatisticsPanel.
    Query params: start_time (ISO), end_time (ISO), camera_id (optional).
    """
    if not LOGGER_AVAILABLE:
        return jsonify({'success': False, 'error': 'logger unavailable'}), 503
    try:
        from datetime import datetime, timedelta
        start_ts = _parse_iso(request.args.get('start_time'), datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
        end_ts = _parse_iso(request.args.get('end_time'), datetime.now())
        camera_filter = request.args.get('camera_id')
        start_str = start_ts.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_ts.strftime('%Y-%m-%d %H:%M:%S')

        from db_helper import get_database
        conn = get_database()._get_connection()
        cur = conn.cursor()

        where = "WHERE start_time BETWEEN ? AND ?"
        params = [start_str, end_str]
        if camera_filter:
            where += " AND camera_id = ?"
            params.append(camera_filter)

        # Per-event rows in range
        cur.execute(
            f"SELECT camera_id, camera_name, start_time, duration_seconds, is_active FROM drowsy_events {where}",
            params,
        )
        rows = cur.fetchall()

        # Per-camera name lookup from the logger (to cover cameras with 0 events in range)
        logger = get_global_logger()
        camera_names: Dict[str, str] = {cid: getattr(cl, 'camera_name', cid) for cid, cl in logger.cameras.items()}
        for r in rows:
            if r[0] not in camera_names:
                camera_names[r[0]] = r[1] or r[0]

        # Aggregate
        per_cam: Dict[str, Dict] = {}
        per_date: Dict[str, Dict[str, int]] = {}
        per_slot: Dict[str, Dict[str, int]] = {'morning': {'drowsy': 0, 'sleeping': 0}, 'afternoon': {'drowsy': 0, 'sleeping': 0}, 'evening': {'drowsy': 0, 'sleeping': 0}}

        total_drowsy = 0
        total_sleeping = 0

        for cam_id, cam_name, start_time, duration, is_active in rows:
            is_sleeping = (duration or 0) >= _DROWSY_DURATION_THRESHOLD_SEC
            bucket_key = 'sleeping' if is_sleeping else 'drowsy'
            if is_sleeping:
                total_sleeping += 1
            else:
                total_drowsy += 1

            # per camera
            pc = per_cam.setdefault(cam_id, {
                'cameraId': cam_id,
                'cameraName': camera_names.get(cam_id, cam_name or cam_id),
                'drowsy': 0, 'sleeping': 0, 'wakeUps': 0, 'currentDrowsy': 0,
            })
            pc[bucket_key] += 1
            pc['wakeUps'] += 0 if is_active else 1

            # per date
            try:
                day = str(start_time)[:10]
            except Exception:
                day = ''
            if day:
                pd = per_date.setdefault(day, {'drowsy': 0, 'sleeping': 0})
                pd[bucket_key] += 1

            # per time slot
            try:
                hour = int(str(start_time)[11:13])
                per_slot[_time_slot(hour)][bucket_key] += 1
            except Exception:
                pass

        # Currently drowsy across all cameras (live, not date-bound)
        current_drowsy_count = 0
        try:
            for cid, sts in logger.get_active_drowsy_all_cameras().items():
                n = len(sts)
                current_drowsy_count += n
                if cid in per_cam:
                    per_cam[cid]['currentDrowsy'] = n
                elif not camera_filter or camera_filter == cid:
                    per_cam.setdefault(cid, {
                        'cameraId': cid,
                        'cameraName': camera_names.get(cid, cid),
                        'drowsy': 0, 'sleeping': 0, 'wakeUps': 0, 'currentDrowsy': n,
                    })
        except Exception:
            pass

        # Build byDate list sorted
        by_date = [
            {'date': d, 'drowsy': v['drowsy'], 'sleeping': v['sleeping']}
            for d, v in sorted(per_date.items())
        ]

        # Build byTimeSlot with totals (totalStudents approximated from distinct student_ids per slot)
        cur.execute(
            f"SELECT start_time, COUNT(DISTINCT student_id) FROM drowsy_events {where} GROUP BY start_time",
            params,
        )
        # Simpler: just count events as a proxy for totalStudents per slot
        by_time_slot = []
        for slot, counts in per_slot.items():
            total = counts['drowsy'] + counts['sleeping']
            rate = (counts['sleeping'] / total * 100.0) if total > 0 else 0.0
            by_time_slot.append({
                'timeSlot': slot,
                'drowsy': counts['drowsy'],
                'sleeping': counts['sleeping'],
                'totalStudents': total,
                'drowsyRate': round(rate, 2),
            })

        # Alerts: camera rooms with >=60% of events being "sleeping"
        alerts = []
        for cam in per_cam.values():
            total = cam['drowsy'] + cam['sleeping']
            if total == 0:
                continue
            rate = cam['sleeping'] / total * 100.0
            cam['drowsyRate'] = round(rate, 2)
            if rate >= 60.0:
                alerts.append({
                    'cameraId': cam['cameraId'],
                    'cameraName': cam['cameraName'],
                    'message': f"Tỷ lệ ngủ gật cao ({rate:.0f}%)",
                    'severity': 'critical' if rate >= 80.0 else 'warning',
                    'drowsyRate': round(rate, 2),
                })

        statistics = {
            'totalDrowsy': total_drowsy,
            'totalSleeping': total_sleeping,
            'totalWakeUps': total_drowsy + total_sleeping,  # events that finished in range
            'byCamera': list(per_cam.values()),
            'byDate': by_date,
            'byTimeSlot': by_time_slot,
            'currentDrowsyCount': current_drowsy_count,
            'alerts': alerts,
        }
        return jsonify({'success': True, 'statistics': statistics})
    except Exception as e:
        app.logger.error(f"/api/statistics error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chatbot', methods=['POST'])
@app.route('/api/chatbot/query', methods=['POST'])
def api_chatbot():
    if not CHATBOT_AVAILABLE:
        return jsonify({'success': False, 'error': 'chatbot unavailable'}), 503
    data = request.get_json(force=True, silent=True) or {}
    question = data.get('question', '')
    try:
        res = _chatbot_handle(question) or {}
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({
        'success': True,
        'summary_text': res.get('summary_text'),
        'rows': res.get('rows'),
        'column_names': res.get('column_names'),
    })


# ==================== WebSocket Handlers ====================

@socketio.on('connect', namespace='/ws/detect')
def ws_detect_connect():
    app.logger.info('✅ WS client connected to /ws/detect')
    emit('hello', {'msg': 'connected to /ws/detect'})


@socketio.on('disconnect', namespace='/ws/detect')
def ws_detect_disconnect():
    app.logger.info('❌ WS client disconnected from /ws/detect')


@socketio.on('frame', namespace='/ws/detect')
def ws_detect_frame(data):
    """Receive webcam frame, run YOLO detection, emit result immediately"""
    try:
        if not YOLO_AVAILABLE:
            emit('result', {'success': False, 'error': 'YOLO not available'})
            return
        
        # Ensure detector is initialized
        if get_detector() is None:
            app.logger.warning("Detector not initialized, initializing...")
            initialize_detector('yolo11n-pose.pt')
        
        # Extract frame data
        frame_b64 = data.get('frame') if isinstance(data, dict) else None
        cam_id = data.get('camera_id', 'webcam') if isinstance(data, dict) else 'webcam'
        
        if not frame_b64:
            emit('result', {'success': False, 'error': 'frame required'})
            return
        
        # Decode base64 frame
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        
        frame_bytes = base64.b64decode(frame_b64)
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('result', {'success': False, 'error': 'decode failed'})
            return
        
        h, w = frame.shape[:2]
        
        # Run YOLO detection
        det = detect_frame(frame)
        
        # Convert DetectionResult to JSON-serializable format
        persons_data = []
        for person in det.persons:
            keypoints_data = [{
                'x': float(kpt.x),
                'y': float(kpt.y),
                'confidence': float(kpt.confidence),
                'visible': bool(kpt.visible)
            } for kpt in person.keypoints]
            
            persons_data.append({
                'id': int(person.id),
                'track_id': int(getattr(person, 'track_id', person.id)),
                'bbox': [float(v) for v in person.bbox],
                'head_bbox': [float(v) for v in getattr(person, 'head_bbox', [])] if getattr(person, 'head_bbox', None) else None,
                'confidence': float(person.confidence),
                'keypoints': keypoints_data,
                'drowsiness_score': float(person.drowsiness_score),
                'drowsiness_state': str(person.drowsiness_state),
                'last_update': float(person.last_update)
            })
        
        # Emit result back to client
        emit('result', {
            'success': True,
            'frame_id': int(det.frame_id),
            'frame_width': int(w),
            'frame_height': int(h),
            'fps': float(getattr(det, 'fps', 0.0)),
            'processing_time': float(det.processing_time),
            'persons': persons_data,
            'timestamp': float(det.timestamp)
        })
        
        app.logger.info(f"📤 [WS /ws/detect] Emitted result: {len(persons_data)} persons detected")
        
    except Exception as e:
        app.logger.error(f"❌ [WS /ws/detect] Error processing frame: {e}")
        emit('result', {'success': False, 'error': str(e)})


@socketio.on('connect', namespace='/ws/camera')
def ws_camera_connect():
    app.logger.info('✅ WS client connected to /ws/camera')
    emit('hello', {'msg': 'connected to /ws/camera'})


@socketio.on('disconnect', namespace='/ws/camera')
def ws_camera_disconnect():
    app.logger.info('❌ WS client disconnected from /ws/camera')


@socketio.on('subscribe', namespace='/ws/camera')
def ws_camera_subscribe(data):
    """Subscribe to IP camera updates"""
    try:
        cam_id = (data or {}).get('camera_id')
        if not cam_id:
            emit('error', {'success': False, 'error': 'camera_id required'})
            return
        
        join_room(f'cam:{cam_id}')
        app.logger.info(f"📡 [WS /ws/camera] Subscribed to room cam:{cam_id}")
        emit('subscribed', {'success': True, 'camera_id': cam_id})
        
        # Send immediate snapshot if camera exists
        if manager.has_camera(cam_id):
            worker = manager.get_worker(cam_id)
            result = manager.get_detection_result(cam_id)
            frame_width, frame_height = manager.get_frame_dimensions(cam_id) or (640, 360)
            fps = worker.get_fps() if worker else 0.0
            
            persons_payload = []
            if result and hasattr(result, 'persons'):
                for p in result.persons:
                    kpts = [{
                        'x': float(k.x),
                        'y': float(k.y),
                        'confidence': float(k.confidence),
                        'visible': bool(k.visible)
                    } for k in p.keypoints]
                    
                    persons_payload.append({
                        'id': int(p.id),
                        'track_id': int(getattr(p, 'track_id', p.id)),
                        'bbox': [float(v) for v in p.bbox],
                        'head_bbox': [float(v) for v in getattr(p, 'head_bbox', [])] if getattr(p, 'head_bbox', None) else None,
                        'confidence': float(p.confidence),
                        'keypoints': kpts,
                        'drowsiness_score': float(p.drowsiness_score),
                        'drowsiness_state': str(p.drowsiness_state),
                        'last_update': float(p.last_update)
                    })
            
            socketio.emit('update', {
                'success': True,
                'camera_id': cam_id,
                'frame_width': int(frame_width),
                'frame_height': int(frame_height),
                'fps': float(fps),
                'processing_time': float(getattr(result, 'processing_time', 0.0) if result else 0.0),
                'persons': persons_payload,
                'timestamp': time.time()
            }, namespace='/ws/camera', to=f'cam:{cam_id}')
            
            app.logger.info(f"📤 [WS /ws/camera] Sent snapshot to cam:{cam_id}: {len(persons_payload)} persons")
        
    except Exception as e:
        app.logger.error(f"❌ [WS /ws/camera] Subscribe error: {e}")
        emit('error', {'success': False, 'error': str(e)})


@socketio.on('unsubscribe', namespace='/ws/camera')
def ws_camera_unsubscribe(data):
    """Unsubscribe from IP camera updates"""
    try:
        cam_id = (data or {}).get('camera_id')
        if not cam_id:
            emit('error', {'success': False, 'error': 'camera_id required'})
            return
        
        leave_room(f'cam:{cam_id}')
        app.logger.info(f"📡 [WS /ws/camera] Unsubscribed from room cam:{cam_id}")
        emit('unsubscribed', {'success': True, 'camera_id': cam_id})
        
    except Exception as e:
        app.logger.error(f"❌ [WS /ws/camera] Unsubscribe error: {e}")
        emit('error', {'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Initialize YOLO detector on startup
    if YOLO_AVAILABLE:
        print("Initializing YOLO detector...")
        success = initialize_detector('yolo11n-pose.pt')
        if success:
            print("✅ YOLO detector initialized successfully")
        else:
            print("❌ Failed to initialize YOLO detector")
    else:
        print("⚠️ YOLO not available - detection features disabled")
    
    # Start WebSocket broadcaster for IP cameras
    print("📡 Starting WebSocket broadcaster thread...")
    manager.start_ws_broadcaster()
    
    # Start Flask-SocketIO server (supports both HTTP and WebSocket)
    print("🚀 Starting Flask+SocketIO server on http://127.0.0.1:5000")
    print("   - HTTP REST API: /api/*")
    print("   - WebSocket /ws/detect: Webcam detection")
    print("   - WebSocket /ws/camera: IP camera streaming")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)
