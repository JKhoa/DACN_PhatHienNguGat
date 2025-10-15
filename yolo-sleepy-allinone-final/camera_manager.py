import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, List

import cv2


@dataclass
class CameraConfig:
    name: str
    type: str  # 'webcam' | 'ip'
    source: Optional[int] = None
    brand: str = "generic"
    username: str = ""
    password: str = ""
    ip: str = ""
    port: int = 554
    stream_quality: str = "main"
    enabled: bool = True
    inference_fps: int = 10
    reconnect_max: int = 5
    notes: str = ""


class CameraStatus:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECT_WAIT = "reconnect_wait"


@dataclass
class CameraSession:
    config: CameraConfig
    capture: Optional[cv2.VideoCapture] = None
    last_frame_time: float = 0.0
    reconnect_attempts: int = 0
    status: str = CameraStatus.DISCONNECTED
    error_msg: str = ""
    running: bool = False
    frame: Optional[object] = None
    fps: float = 0.0
    thread: Optional[threading.Thread] = None


def _open_capture(cfg: CameraConfig) -> Optional[cv2.VideoCapture]:
    if cfg.type == "webcam":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(int(cfg.source or 0), backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
        return None
    # IP camera RTSP
    if cfg.ip:
        url = f"rtsp://{cfg.username}:{cfg.password}@{cfg.ip}:{cfg.port}/"
        cap = cv2.VideoCapture(url)
        return cap if cap.isOpened() else None
    return None


class MultiCameraController:
    def __init__(self):
        self.sessions: List[CameraSession] = []
        self.lock = threading.Lock()
        self.on_status_change: Optional[Callable[[CameraSession], None]] = None
        self.on_new_frame: Optional[Callable[[CameraSession, object], None]] = None
        self.on_error: Optional[Callable[[CameraSession, Exception], None]] = None

    def add(self, cfg: CameraConfig) -> CameraSession:
        s = CameraSession(config=cfg)
        with self.lock:
            self.sessions.append(s)
        return s

    def remove(self, s: CameraSession) -> None:
        self.stop(s)
        with self.lock:
            if s in self.sessions:
                self.sessions.remove(s)

    def start(self, s: CameraSession) -> None:
        if s.running:
            return
        s.running = True
        s.thread = threading.Thread(target=self._capture_loop, args=(s,), daemon=True)
        s.thread.start()

    def stop(self, s: CameraSession) -> None:
        s.running = False
        if s.thread:
            s.thread.join(timeout=1.5)
        if s.capture:
            try:
                s.capture.release()
            except Exception:
                pass
            s.capture = None
        s.status = CameraStatus.DISCONNECTED
        s.frame = None

    def start_all(self) -> None:
        with self.lock:
            for s in self.sessions:
                if s.config.enabled:
                    self.start(s)

    def stop_all(self) -> None:
        with self.lock:
            for s in self.sessions:
                self.stop(s)

    def _emit_status(self, s: CameraSession):
        if self.on_status_change:
            try:
                self.on_status_change(s)
            except Exception:
                pass

    def _emit_frame(self, s: CameraSession, frame):
        if self.on_new_frame:
            try:
                self.on_new_frame(s, frame)
            except Exception:
                pass

    def _emit_error(self, s: CameraSession, exc: Exception):
        if self.on_error:
            try:
                self.on_error(s, exc)
            except Exception:
                pass

    def _capture_loop(self, s: CameraSession):
        backoff_seq = [2, 4, 8, 16, 30]
        s.reconnect_attempts = 0
        s.status = CameraStatus.CONNECTING
        self._emit_status(s)

        while s.running:
            try:
                if s.capture is None:
                    s.status = CameraStatus.CONNECTING
                    self._emit_status(s)
                    s.capture = _open_capture(s.config)
                    if s.capture is None:
                        s.status = CameraStatus.ERROR
                        s.error_msg = "Open failed"
                        self._emit_status(s)
                        # schedule reconnect
                        delay = backoff_seq[min(s.reconnect_attempts, len(backoff_seq) - 1)]
                        s.reconnect_attempts += 1
                        s.status = CameraStatus.RECONNECT_WAIT
                        self._emit_status(s)
                        t0 = time.time()
                        while s.running and time.time() - t0 < delay:
                            time.sleep(0.1)
                        continue
                    s.status = CameraStatus.CONNECTED
                    s.reconnect_attempts = 0
                    self._emit_status(s)

                # Read frame with simple timeout
                t_read0 = time.time()
                ok, frame = s.capture.read()
                if not ok or frame is None:
                    raise RuntimeError("read failed")
                s.last_frame_time = time.time()
                self._emit_frame(s, frame)
                # Throttle display; inference is handled elsewhere
                time.sleep(max(0.0, 1.0 / 25.0))
                # frame timeout: if no frame for 3s, force reconnect
                if time.time() - s.last_frame_time > 3.0:
                    raise RuntimeError("frame timeout")
            except Exception as e:
                self._emit_error(s, e)
                # Force reconnect
                try:
                    if s.capture:
                        s.capture.release()
                except Exception:
                    pass
                s.capture = None
                s.status = CameraStatus.ERROR
                self._emit_status(s)
                # Backoff wait before next iteration
                delay = backoff_seq[min(s.reconnect_attempts, len(backoff_seq) - 1)]
                s.reconnect_attempts += 1
                s.status = CameraStatus.RECONNECT_WAIT
                self._emit_status(s)
                t0 = time.time()
                while s.running and time.time() - t0 < delay:
                    time.sleep(0.1)







