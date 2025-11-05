#!/usr/bin/env python3
"""
Enhanced server for drowsiness detection with real camera streaming
"""
import threading
import time
import base64
import logging
from typing import Optional

import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import YOLO detector helpers
try:
    from yolo_detector import (
        initialize_detector,
        detect_frame,
        draw_detections,
        DetectionResult,
        get_detector,
    )
    YOLO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"YOLO detector not available: {e}")
    YOLO_AVAILABLE = False

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class CameraWorker(threading.Thread):
    """Background reader for a single camera source with optional detection/annotation."""

    def __init__(self, cam_id: str, url: str, enable_detection: bool = True):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.url = url
        self._detection_enabled = enable_detection

        # Runtime state
        self._cap = None
        self._status = 'stopped'
        self._last_error = ''
        self._last_frame = None
        self._last_annotated_frame = None
        self._last_detection_result: Optional[DetectionResult] = None

        # FPS helpers
        self._frame_count = 0
        self._last_time = time.time()
        self._last_fps = 0.0

        # Control + safety
        self._stop_flag = False
        self._paused = False
        self._lock = threading.Lock()

    def _open_capture(self):
        """Open the capture with robust backend fallbacks on Windows for device indices."""
        if self.url.isdigit():
            idx = int(self.url)
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(idx, backend)
                if cap and cap.isOpened():
                    return cap
            return None
        else:
            cap = cv2.VideoCapture(self.url)
            return cap if cap and cap.isOpened() else None

    def run(self):
        self._status = 'starting'
        while not self._stop_flag:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_capture()
                if self._cap is None:
                    self._status = 'error'
                    self._last_error = f'Failed to open source: {self.url}'
                    time.sleep(1.0)
                    continue
                self._status = 'running'
                self._frame_count = 0
                self._last_time = time.time()

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._status = 'error'
                time.sleep(0.2)
                continue

            with self._lock:
                self._last_frame = frame

            # FPS calc
            now = time.time()
            self._frame_count += 1
            if now - self._last_time >= 1.0:
                self._last_fps = self._frame_count / max(1e-6, (now - self._last_time))
                self._frame_count = 0
                self._last_time = now

            # Detection + annotation
            try:
                if self._detection_enabled and YOLO_AVAILABLE and get_detector() is not None:
                    result = detect_frame(frame)
                    annotated = draw_detections(frame, result)
                    with self._lock:
                        self._last_detection_result = result
                        self._last_annotated_frame = annotated
            except Exception as e:
                logging.error(f"[{self.cam_id}] Detection failed: {e}")
                with self._lock:
                    self._last_detection_result = None
                    self._last_annotated_frame = None

            # Reduce CPU usage
            time.sleep(0.01)

        # Cleanup
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._status = 'stopped'

    def stop(self):
        self._stop_flag = True

    def get_last_jpeg(self, annotated: bool = False) -> Optional[bytes]:
        with self._lock:
            frame = self._last_annotated_frame if annotated and self._last_annotated_frame is not None else self._last_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return buf.tobytes()

    def get_detection_result(self) -> Optional[DetectionResult]:
        with self._lock:
            return self._last_detection_result

    def toggle_detection(self, enabled: bool):
        self._detection_enabled = enabled
        logging.info(f"[{self.cam_id}] Detection toggled: {enabled}")

class CameraManager:
    def __init__(self):
        self._cameras = {}
        self._workers = {}
        self._lock = threading.Lock()
    
    def add(self, cam_id: str, url: str, name: str = None):
        with self._lock:
            if cam_id in self._cameras:
                raise ValueError('Camera already exists')
            self._cameras[cam_id] = {'url': url, 'name': name or cam_id}
    
    def start(self, cam_id: str, enable_detection: bool = True):
        with self._lock:
            if cam_id not in self._cameras:
                raise KeyError('Camera not found')
            if cam_id in self._workers and self._workers[cam_id].is_alive():
                app.logger.info(f"[{cam_id}] Camera worker already running")
                return
            try:
                worker = CameraWorker(cam_id, self._cameras[cam_id]['url'], enable_detection)
                self._workers[cam_id] = worker
                worker.start()
                app.logger.info(f"[{cam_id}] Camera worker started successfully")
            except Exception as e:
                app.logger.error(f"[{cam_id}] Failed to start camera worker: {e}")
                raise
    
    def stop(self, cam_id: str):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].stop()
    
    def list(self):
        with self._lock:
            result = []
            for cam_id, meta in self._cameras.items():
                running = cam_id in self._workers and self._workers[cam_id].is_alive()
                result.append({
                    'id': cam_id,
                    'name': meta['name'],
                    'type': 'webcam' if meta['url'].isdigit() else 'ip',
                    'url': meta['url'],
                    'status': 'running' if running else 'stopped'
                })
            return result
    
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
        return worker.get_detection_result()
    
    def toggle_detection(self, cam_id: str, enabled: bool):
        with self._lock:
            worker = self._workers.get(cam_id)
        if worker:
            worker.toggle_detection(enabled)

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

@app.route('/api/camera/<cam_id>/stream', methods=['GET'])
def stream_frame(cam_id):
    """Return latest frame as base64 JSON"""
    annotated = request.args.get('annotated', 'false').lower() == 'true'
    jpeg = manager.get_jpeg(cam_id, annotated)
    if jpeg is None:
        return jsonify({'success': False, 'error': 'no frame'}), 404
    b64 = base64.b64encode(jpeg).decode('utf-8')
    return jsonify({'success': True, 'frame': b64, 'ts': time.time()})

@app.route('/api/camera/<cam_id>/detection', methods=['GET'])
def get_detection_results(cam_id):
    """Get detection results for a specific camera"""
    result = manager.get_detection_result(cam_id)
    if result is None:
        # Return empty result with 200 to avoid noisy 404s while detection warms up
        return jsonify({
            'success': True,
            'detection_result': {
                'frame_id': 0,
                'timestamp': time.time(),
                'persons': [],
                'fps': 0.0,
                'processing_time': 0.0
            }
        })
    
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
            'bbox': person.bbox,
            'confidence': person.confidence,
            'keypoints': keypoints_data,
            'head_bbox': getattr(person, 'head_bbox', None),
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
            'processing_time': result.processing_time
        }
    })

@app.route('/api/camera/<cam_id>/detection/toggle', methods=['POST'])
def toggle_detection(cam_id):
    """Toggle detection on/off for a specific camera"""
    data = request.get_json(force=True, silent=True) or {}
    enabled = data.get('enabled', True)
    
    manager.toggle_detection(cam_id, enabled)
    return jsonify({'success': True, 'detection_enabled': enabled})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify routing works"""
    return jsonify({'success': True, 'message': 'Test endpoint works'})

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

    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

