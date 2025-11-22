import threading
import time
import base64
import io
import logging
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

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

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
        
        app.logger.info(f"[{self.cam_id}] CameraWorker initialized with URL: {url}, detection: {self._detection_enabled}")
        
        # Auto-enable detection for webcam (device ID)
        if url.isdigit() or url.startswith('0'):
            self._detection_enabled = True and YOLO_AVAILABLE
            app.logger.info(f"[{self.cam_id}] Auto-enabling YOLO detection for webcam")

    def run(self):
        app.logger.info(f"[{self.cam_id}] Starting camera worker thread...")
        
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


@app.route('/api/camera/add', methods=['POST'])
def add_camera():
    data = request.get_json(force=True, silent=True) or {}
    cam_id = data.get('id') or data.get('name')
    url = data.get('url')
    name = data.get('name')
    if not cam_id or not url:
        return jsonify({'success': False, 'error': 'id/name and url required'}), 400
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
