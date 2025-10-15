#!/usr/bin/env python3
"""
Shared Camera Core Module
Core classes for camera management used by both standalone and multi-camera apps
"""

import cv2
import numpy as np
import threading
from typing import Any, Optional
from dataclasses import dataclass

import queue
import time


@dataclass
class CameraConfig:
    """Camera configuration"""
    name: str
    type: str  # 'webcam' or 'ip'
    source: Any  # int for webcam, str for IP
    brand: str = "generic"
    username: str = ""
    password: str = ""
    ip: str = ""
    port: int = 554
    stream_quality: str = "main"  # main or sub
    enabled: bool = True


@dataclass
class CameraStream:
    """Camera stream data"""
    config: CameraConfig
    capture: Optional[cv2.VideoCapture] = None
    frame: Optional[np.ndarray] = None
    detection_result: Optional[Any] = None
    fps: float = 0.0
    status: str = "disconnected"  # disconnected, connecting, connected, error
    error_msg: str = ""
    thread: Optional[threading.Thread] = None
    running: bool = False
    frame_count: int = 0
    detection_count: int = 0
    sleepy_count: int = 0
    frame_queue: Optional[queue.Queue] = None
    capture_thread: Optional["CameraCaptureThread"] = None


# Threaded camera capture with frame queue for high FPS and low latency
class CameraCaptureThread(threading.Thread):
    def __init__(self, camera_stream: CameraStream, queue_size: int = 3, target_fps: float = 30.0):
        super().__init__(daemon=True)
        self.camera_stream = camera_stream
        self.capture = camera_stream.capture
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.target_fps = target_fps
        self.last_frame_time = 0

    def run(self):
        self.running = True
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.capture.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            # Put frame in queue, drop oldest if full
            try:
                if self.frame_queue.full():
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            # FPS limiting
            if self.target_fps > 0:
                now = time.time()
                elapsed = now - self.last_frame_time
                min_interval = 1.0 / self.target_fps
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                self.last_frame_time = time.time()

    def stop(self):
        self.running = False

    def get_latest_frame(self):
        frame = None
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
        return frame

def start_threaded_capture(camera_stream: CameraStream, target_fps: float = 30.0):
    """Start threaded capture for a camera stream"""
    if camera_stream.capture is None:
        return
    if camera_stream.capture_thread is not None and camera_stream.capture_thread.running:
        return
    camera_stream.frame_queue = queue.Queue(maxsize=3)
    camera_stream.capture_thread = CameraCaptureThread(camera_stream, queue_size=3, target_fps=target_fps)
    camera_stream.capture_thread.start()
    camera_stream.running = True

def stop_threaded_capture(camera_stream: CameraStream):
    """Stop threaded capture for a camera stream"""
    if camera_stream.capture_thread is not None:
        camera_stream.capture_thread.stop()
        camera_stream.capture_thread.join(timeout=1.0)
        camera_stream.capture_thread = None
    camera_stream.running = False
    camera_stream.frame_queue = None

def generate_rtsp_url(config: CameraConfig) -> str:
    """Generate RTSP URL for IP camera based on brand"""
    rtsp_paths = {
        "imou": "/cam/realmonitor?channel=1&subtype=0",
        "hikvision": "/Streaming/Channels/101" if config.stream_quality == "main" else "/Streaming/Channels/102",
        "dahua": "/cam/realmonitor?channel=1&subtype=0" if config.stream_quality == "main" else "/cam/realmonitor?channel=1&subtype=1",
        "tapo": "/stream1" if config.stream_quality == "main" else "/stream2",
        "tplink": "/stream1" if config.stream_quality == "main" else "/stream2",
        "xiaomi": "/live/ch00_0" if config.stream_quality == "main" else "/live/ch00_1",
        "mijia": "/live/ch00_0" if config.stream_quality == "main" else "/live/ch00_1",
        "reolink": "/h264Preview_01_main" if config.stream_quality == "main" else "/h264Preview_01_sub",
        "foscam": "/videoMain" if config.stream_quality == "main" else "/videoSub",
        "axis": "/axis-media/media.amp?videocodec=h264",
        "bosch": "/rtsp_tunnel?h264&unicast&line=1",
        "sony": "/media/video1",
        "panasonic": "/MediaInput/stream_1",
        "vivotek": "/live.sdp",
        "dlink": "/play1.sdp" if config.stream_quality == "main" else "/play2.sdp",
        "arlo": "/rtspstream/video",
        "netgear": "/rtspstream/video",
        "generic": "/stream1",
        "onvif": "/onvif1",
        "standard": "/video.mjpg"
    }
    
    path = rtsp_paths.get(config.brand.lower(), "/stream1")
    
    if config.username and config.password:
        return f"rtsp://{config.username}:{config.password}@{config.ip}:{config.port}{path}"
    else:
        return f"rtsp://{config.ip}:{config.port}{path}"


def connect_camera(config: CameraConfig) -> Optional[cv2.VideoCapture]:
    """Connect to a camera and return VideoCapture object"""
    try:
        # Determine source
        if config.type == "webcam":
            source = config.source
        else:  # IP camera
            source = generate_rtsp_url(config)
        
        # Open capture
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            return None
        
        # Test read
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        
        return cap
        
    except Exception as e:
        print(f"Error connecting to camera {config.name}: {e}")
        return None


# Supported camera brands for UI display
SUPPORTED_BRANDS = [
    "generic",
    "imou",
    "hikvision", 
    "dahua",
    "tapo",
    "tplink",
    "xiaomi",
    "mijia",
    "reolink",
    "foscam",
    "axis",
    "bosch",
    "sony",
    "panasonic",
    "vivotek",
    "dlink",
    "arlo",
    "netgear",
    "onvif",
    "standard"
]
