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

    def run(self):
        backoff = 1.0
        while self._running.is_set():
            if self._capture is None:
                app.logger.info(f"[{self.cam_id}] Opening stream: {self.url}")
                self._capture = cv2.VideoCapture(self.url)
                if not self._capture.isOpened():
                    app.logger.warning(f"[{self.cam_id}] Failed to open stream. Retry in {backoff:.1f}s")
                    self._capture.release()
                    self._capture = None
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 10.0)
                    continue
                backoff = 1.0

            ok, frame = self._capture.read()
            if not ok or frame is None:
                app.logger.warning(f"[{self.cam_id}] Read failed. Reopening...")
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
                time.sleep(0.5)
                continue

            with self._lock:
                self._last_frame = frame
                
                # Run YOLO detection if enabled
                if self._detection_enabled and YOLO_AVAILABLE:
                    try:
                        detection_result = detect_frame(frame)
                        self._last_detection_result = detection_result
                        
                        # Create annotated frame with detections
                        annotated_frame = draw_detections(frame, detection_result)
                        self._last_annotated_frame = annotated_frame
                    except Exception as e:
                        app.logger.error(f"[{self.cam_id}] Detection failed: {e}")
                        self._last_detection_result = None
                        self._last_annotated_frame = None

            # Small sleep to reduce CPU usage on fast sources
            time.sleep(0.001)

        # Cleanup
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        app.logger.info(f"[{self.cam_id}] Worker stopped")

    def stop(self):
        self._running.clear()

    def get_last_jpeg(self, annotated: bool = False) -> Optional[bytes]:
        with self._lock:
            frame_to_encode = None
            
            if annotated and self._last_annotated_frame is not None:
                frame_to_encode = self._last_annotated_frame
            elif self._last_frame is not None:
                frame_to_encode = self._last_frame
            
            if frame_to_encode is None:
                return None
                
            ok, buf = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            return buf.tobytes()
    
    def get_detection_result(self) -> Optional[DetectionResult]:
        with self._lock:
            return self._last_detection_result
    
    def toggle_detection(self, enabled: bool):
        self._detection_enabled = enabled and YOLO_AVAILABLE


class CameraManager:
    def __init__(self):
        self._workers: Dict[str, CameraWorker] = {}
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

    def remove(self, cam_id: str):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].stop()
            self._workers.pop(cam_id, None)
            self._meta.pop(cam_id, None)

    def start(self, cam_id: str, enable_detection: bool = True):
        with self._lock:
            if cam_id not in self._meta:
                raise KeyError('Camera not found')
            if cam_id in self._workers and self._workers[cam_id].is_alive():
                return
            worker = CameraWorker(cam_id, self._meta[cam_id]['url'], enable_detection)
            self._workers[cam_id] = worker
            worker.start()

    def stop(self, cam_id: str):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].stop()
                # Let the thread exit asynchronously

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


@app.route('/api/camera/<cam_id>/stop', methods=['POST'])
def stop_camera(cam_id):
    manager.stop(cam_id)
    return jsonify({'success': True})


@app.route('/api/camera/<cam_id>/remove', methods=['DELETE'])
def remove_camera(cam_id):
    manager.remove(cam_id)
    return jsonify({'success': True})


@app.route('/api/camera/<cam_id>/stream', methods=['GET'])
def stream_frame(cam_id):
    """Return latest frame as base64 JSON for easy <img src="data:"> usage in the UI."""
    annotated = request.args.get('annotated', 'false').lower() == 'true'
    jpeg = manager.get_jpeg(cam_id, annotated)
    if jpeg is None:
        return jsonify({'success': False, 'error': 'no frame'}), 404
    b64 = base64.b64encode(jpeg).decode('utf-8')
    return jsonify({'success': True, 'frame': b64, 'ts': time.time()})


@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    # Keep it lightweight without extra deps
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


@app.route('/api/camera/<cam_id>/detection', methods=['GET'])
def get_detection_results(cam_id):
    """Get detection results for a specific camera"""
    result = manager.get_detection_result(cam_id)
    if result is None:
        return jsonify({'success': False, 'error': 'no detection data'}), 404
    
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


if __name__ == '__main__':
    # Bind to localhost only; Electron opens from file://
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)




