# DACN PhatHienNguGat - 2026
"""Flask blueprint ``/api/v1/detect/...`` cho pipeline ensemble VN.

Endpoints:

- ``POST /api/v1/detect/image``   — ảnh tĩnh, trả ``objects[]`` + ``top_k``.
- ``POST /api/v1/detect/video``   — sample N frame, báo khi drowsy >2s liên tục.
- ``GET  /api/v1/detect/health``  — ping + info model.
- WebSocket ``/api/v1/detect/realtime`` (SocketIO namespace) — realtime webcam,
  primary YOLO only (KHÔNG HF fallback).

Đăng ký:
    from api_v1 import bp_api_v1, register_realtime_ws
    app.register_blueprint(bp_api_v1)
    register_realtime_ws(socketio)
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from flask import Blueprint, jsonify, request

from detectors.ensemble import EnsembleDetector

LOGGER = logging.getLogger("api_v1")

bp_api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1/detect")

VIDEO_FRAME_SAMPLE: int = 10       # sample mỗi N frame
DROWSY_ALERT_SECONDS: float = 2.0  # >2s liên tục → alert
MAX_UPLOAD_MB: int = 25

_detector_lock = Lock()
_detector: EnsembleDetector | None = None


def _get_detector(enable_hf: bool = True) -> EnsembleDetector:
    """Singleton — load 1 lần, chia sẻ cho mọi request."""
    global _detector
    with _detector_lock:
        if _detector is None:
            LOGGER.info("Khởi tạo EnsembleDetector (enable_hf=%s)", enable_hf)
            _detector = EnsembleDetector(enable_hf_fallback=enable_hf)
        return _detector


def _decode_image_from_request() -> tuple[np.ndarray | None, str | None]:
    """Đọc ảnh từ multipart ``file`` hoặc JSON ``image_base64``.

    Returns:
        (ndarray BGR, error_msg). Đúng 1 trong 2 là None.
    """
    import cv2

    if "file" in request.files:
        f = request.files["file"]
        data = f.read()
        if len(data) > MAX_UPLOAD_MB * (1 << 20):
            return None, f"File > {MAX_UPLOAD_MB}MB"
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return (img, None) if img is not None else (None, "Không decode được ảnh")

    body = request.get_json(silent=True) or {}
    b64 = body.get("image_base64")
    if b64:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            return None, "image_base64 không hợp lệ"
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return (img, None) if img is not None else (None, "Không decode được base64")

    return None, "Cần 'file' (multipart) hoặc 'image_base64' (JSON)"


@bp_api_v1.route("/health", methods=["GET"])
def health() -> Any:  # type: ignore[valid-type]
    """Ping endpoint — cũng trả tên weight đang load."""
    try:
        det = _get_detector(enable_hf=False)
        info = {
            "status": "ok",
            "hybrid_mode": det.hybrid_mode,
            "primary": det.primary_path.name,
            "secondary": det.secondary_path.name if det.secondary_path else None,
            "hf_fallback": list(det.hf_cls_pipelines.keys()),
        }
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "error": str(exc)}), 500
    return jsonify(info)


@bp_api_v1.route("/image", methods=["POST"])
def detect_image() -> Any:  # type: ignore[valid-type]
    """Phát hiện trên 1 ảnh. Body: multipart ``file`` hoặc JSON ``image_base64``.

    Query param:
        conf (float): ngưỡng primary, mặc định 0.35.
        use_secondary (0/1): bật secondary YOLO phone.
        use_hf (0/1): bật HF classifier fallback.
    """
    img, err = _decode_image_from_request()
    if err:
        return jsonify({"error": err}), 400

    conf = float(request.args.get("conf", 0.35))
    use_secondary = request.args.get("use_secondary", "1") != "0"
    use_hf = request.args.get("use_hf", "1") != "0"

    det = _get_detector(enable_hf=use_hf)
    res = det.detect_image(img, conf=conf, use_secondary=use_secondary, use_hf_fallback=use_hf)
    return jsonify(res.to_dict())


@bp_api_v1.route("/video", methods=["POST"])
def detect_video() -> Any:  # type: ignore[valid-type]
    """Sample N frame của video, alert khi ``ngu_gat`` > ``DROWSY_ALERT_SECONDS``.

    Body: multipart ``file`` (mp4/avi/webm).
    Query param:
        conf (float), frame_stride (int, mặc định ``VIDEO_FRAME_SAMPLE``).
    """
    import cv2

    raw: bytes | None = None
    filename = "video.mp4"
    if "file" in request.files:
        upload = request.files["file"]
        raw = upload.read()
        filename = upload.filename or filename
    else:
        body = request.get_json(silent=True) or {}
        b64 = body.get("video_base64")
        filename = body.get("filename") or filename
        if b64:
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                raw = base64.b64decode(b64)
            except Exception:  # noqa: BLE001
                return jsonify({"error": "video_base64 không hợp lệ"}), 400
    if raw is None:
        return jsonify({"error": "Cần 'file' (multipart) hoặc 'video_base64' (JSON)"}), 400
    if len(raw) > 200 * (1 << 20):
        return jsonify({"error": "Video > 200MB"}), 400

    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        conf = float(request.args.get("conf", 0.35))
        stride = int(request.args.get("frame_stride", VIDEO_FRAME_SAMPLE))
        det = _get_detector(enable_hf=False)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Không mở được video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames_result: list[dict] = []
        drowsy_streak = 0.0
        drowsy_start_time: float | None = None
        alerts: list[dict] = []
        idx = 0
        t_start = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                t_frame = idx / fps
                res = det.detect_image(frame, conf=conf, use_secondary=True, use_hf_fallback=False)
                has_drowsy = any(o.class_name == "ngu_gat" for o in res.objects)
                if has_drowsy:
                    if drowsy_start_time is None:
                        drowsy_start_time = t_frame
                    drowsy_streak = t_frame - drowsy_start_time
                    if drowsy_streak >= DROWSY_ALERT_SECONDS:
                        alerts.append({
                            "type": "ngu_gat",
                            "display_name": "Ngủ gật",
                            "since_seconds": round(drowsy_start_time, 2),
                            "duration_seconds": round(drowsy_streak, 2),
                        })
                else:
                    drowsy_start_time = None
                    drowsy_streak = 0.0
                frames_result.append({
                    "frame": idx,
                    "time_seconds": round(t_frame, 2),
                    "objects": [asdict(o) for o in res.objects],
                })
            idx += 1

        cap.release()
        # Gom alert liên tục
        merged_alerts: list[dict] = []
        for a in alerts:
            if merged_alerts and a["since_seconds"] == merged_alerts[-1]["since_seconds"]:
                merged_alerts[-1] = a
            else:
                merged_alerts.append(a)

        return jsonify({
            "total_frames": total,
            "fps": fps,
            "sampled": len(frames_result),
            "elapsed_seconds": round(time.perf_counter() - t_start, 2),
            "frames": frames_result,
            "alerts": merged_alerts,
        })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ==================== WebSocket realtime ====================

_WS_NAMESPACE = "/api/v1/detect/realtime"


def register_realtime_ws(socketio: Any) -> None:
    """Đăng ký SocketIO handler cho realtime webcam.

    Client emit ``frame`` với payload ``{"image_base64": "...", "conf": 0.35}``,
    server reply ``result`` với ``objects[]`` + ``inference_time_ms``.
    """
    import cv2

    @socketio.on("connect", namespace=_WS_NAMESPACE)
    def _on_connect():  # noqa: ANN202
        LOGGER.info("WS realtime client connected")
        socketio.emit("ready", {"status": "ok"}, namespace=_WS_NAMESPACE)

    @socketio.on("frame", namespace=_WS_NAMESPACE)
    def _on_frame(data):  # noqa: ANN001, ANN202
        b64 = (data or {}).get("image_base64", "")
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            socketio.emit("error", {"error": "base64 sai"}, namespace=_WS_NAMESPACE)
            return
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            socketio.emit("error", {"error": "decode fail"}, namespace=_WS_NAMESPACE)
            return
        conf = float(data.get("conf", 0.35))
        det = _get_detector(enable_hf=False)
        res = det.detect_image(img, conf=conf, use_secondary=True, use_hf_fallback=False)
        socketio.emit("result", res.to_dict(), namespace=_WS_NAMESPACE)


