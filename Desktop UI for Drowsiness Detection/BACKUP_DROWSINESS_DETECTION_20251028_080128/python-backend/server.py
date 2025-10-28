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

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class CameraWorker(threading.Thread):
    def __init__(self, cam_id: str, url: str):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.url = url
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._capture: Optional[cv2.VideoCapture] = None

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

    def get_last_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._last_frame is None:
                return None
            ok, buf = cv2.imencode('.jpg', self._last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            return buf.tobytes()


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

    def start(self, cam_id: str):
        with self._lock:
            if cam_id not in self._meta:
                raise KeyError('Camera not found')
            if cam_id in self._workers and self._workers[cam_id].is_alive():
                return
            worker = CameraWorker(cam_id, self._meta[cam_id]['url'])
            self._workers[cam_id] = worker
            worker.start()

    def stop(self, cam_id: str):
        with self._lock:
            if cam_id in self._workers:
                self._workers[cam_id].stop()
                # Let the thread exit asynchronously

    def get_jpeg(self, cam_id: str) -> Optional[bytes]:
        with self._lock:
            worker = self._workers.get(cam_id)
        if not worker:
            return None
        return worker.get_last_jpeg()


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
        manager.start(cam_id)
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
    jpeg = manager.get_jpeg(cam_id)
    if jpeg is None:
        return jsonify({'success': False, 'error': 'no frame'}), 404
    b64 = base64.b64encode(jpeg).decode('utf-8')
    return jsonify({'success': True, 'frame': b64, 'ts': time.time()})


@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    # Keep it lightweight without extra deps
    uptime = int(time.time() - _start_time)
    cams = manager.list()
    return jsonify({'success': True, 'uptime_s': uptime, 'cameras': len(cams)})


if __name__ == '__main__':
    # Bind to localhost only; Electron opens from file://
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)


