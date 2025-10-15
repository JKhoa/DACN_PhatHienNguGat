#!/usr/bin/env python3
"""
Multi-Camera Monitoring Application
Hỗ trợ không giới hạn số lượng camera với YOLO detection
Support unlimited cameras with real-time YOLO detection
"""

import cv2
import numpy as np
import argparse
import time
import yaml
import threading
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import math

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
    stream_quality: str = "main"
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

class MultiCameraMonitor:
    """Multi-camera monitoring system"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, 
                 process_stride: int = 1, max_fps: int = 30):
        """
        Initialize multi-camera monitor
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detection
            process_stride: Process every N frames (1 = every frame)
            max_fps: Maximum FPS per camera
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.process_stride = process_stride
        self.max_fps = max_fps
        self.frame_delay = 1.0 / max_fps if max_fps > 0 else 0
        
        self.streams: List[CameraStream] = []
        self.running = False
        self.display_mode = "grid"  # grid, single, hud
        self.selected_camera = 0
        
        # Performance tracking
        self.total_fps = 0.0
        self.last_update = time.time()
        
    def generate_rtsp_url(self, config: CameraConfig) -> str:
        """Generate RTSP URL for IP camera"""
        rtsp_paths = {
            "imou": "/cam/realmonitor?channel=1&subtype=0",
            "hikvision": "/Streaming/Channels/101",
            "dahua": "/cam/realmonitor?channel=1&subtype=0",
            "tapo": "/stream1",
            "tplink": "/stream1",
            "xiaomi": "/live/ch00_0",
            "mijia": "/live/ch00_0",
            "reolink": "/h264Preview_01_main",
            "foscam": "/videoMain",
            "axis": "/axis-media/media.amp?videocodec=h264",
            "bosch": "/rtsp_tunnel?h264&unicast&line=1",
            "sony": "/media/video1",
            "panasonic": "/MediaInput/stream_1",
            "vivotek": "/live.sdp",
            "dlink": "/play1.sdp",
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
    
    def add_camera(self, config: CameraConfig) -> int:
        """Add a camera to monitoring"""
        stream = CameraStream(config=config)
        self.streams.append(stream)
        return len(self.streams) - 1
    
    def connect_camera(self, stream: CameraStream) -> bool:
        """Connect to a camera"""
        try:
            stream.status = "connecting"
            
            # Determine source
            if stream.config.type == "webcam":
                source = stream.config.source
            else:  # IP camera
                source = self.generate_rtsp_url(stream.config)
            
            # Open capture
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                stream.status = "error"
                stream.error_msg = "Cannot open camera"
                return False
            
            # Test read
            ret, frame = cap.read()
            if not ret or frame is None:
                stream.status = "error"
                stream.error_msg = "Cannot read frame"
                cap.release()
                return False
            
            stream.capture = cap
            stream.frame = frame
            stream.status = "connected"
            return True
            
        except Exception as e:
            stream.status = "error"
            stream.error_msg = str(e)
            return False
    
    def camera_thread(self, stream: CameraStream):
        """Thread for capturing and processing camera stream"""
        fps_queue = deque(maxlen=30)
        
        while stream.running:
            try:
                start_time = time.time()
                
                # Capture frame
                if stream.capture is None or not stream.capture.isOpened():
                    if not self.connect_camera(stream):
                        time.sleep(5)  # Wait before retry
                        continue
                
                ret, frame = stream.capture.read()
                if not ret or frame is None:
                    stream.status = "error"
                    stream.error_msg = "Frame read failed"
                    time.sleep(1)
                    continue
                
                stream.frame_count += 1
                
                # Run detection on every Nth frame
                if stream.frame_count % self.process_stride == 0:
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    stream.detection_result = results[0]
                    
                    # Count detections
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        stream.detection_count = len(results[0].boxes)
                
                stream.frame = frame
                stream.status = "connected"
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)
                stream.fps = sum(fps_queue) / len(fps_queue)
                
                # Limit FPS
                if self.frame_delay > 0:
                    sleep_time = self.frame_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except Exception as e:
                stream.status = "error"
                stream.error_msg = f"Thread error: {e}"
                time.sleep(1)
    
    def start_monitoring(self):
        """Start monitoring all cameras"""
        self.running = True
        
        # Start thread for each camera
        for stream in self.streams:
            if stream.config.enabled:
                stream.running = True
                stream.thread = threading.Thread(target=self.camera_thread, args=(stream,))
                stream.thread.daemon = True
                stream.thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring all cameras"""
        self.running = False
        
        # Stop all threads
        for stream in self.streams:
            stream.running = False
            if stream.thread:
                stream.thread.join(timeout=2)
            if stream.capture:
                stream.capture.release()
    
    def calculate_grid_layout(self, num_cameras: int) -> Tuple[int, int]:
        """Calculate optimal grid layout"""
        if num_cameras == 0:
            return (1, 1)
        elif num_cameras == 1:
            return (1, 1)
        elif num_cameras == 2:
            return (1, 2)
        elif num_cameras <= 4:
            return (2, 2)
        elif num_cameras <= 6:
            return (2, 3)
        elif num_cameras <= 9:
            return (3, 3)
        elif num_cameras <= 12:
            return (3, 4)
        elif num_cameras <= 16:
            return (4, 4)
        else:
            # Dynamic calculation for larger grids
            cols = math.ceil(math.sqrt(num_cameras))
            rows = math.ceil(num_cameras / cols)
            return (rows, cols)
    
    def create_mosaic_view(self, display_width: int = 1920, display_height: int = 1080) -> np.ndarray:
        """Create mosaic view of all cameras"""
        active_streams = [s for s in self.streams if s.config.enabled]
        num_cameras = len(active_streams)
        
        if num_cameras == 0:
            # Empty display
            canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(canvas, "No active cameras", (display_width//2 - 200, display_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            return canvas
        
        # Calculate grid layout
        rows, cols = self.calculate_grid_layout(num_cameras)
        
        # Calculate cell size
        cell_width = display_width // cols
        cell_height = display_height // rows
        
        # Create canvas
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Place each camera
        for idx, stream in enumerate(active_streams):
            if idx >= rows * cols:
                break
            
            row = idx // cols
            col = idx % cols
            
            x = col * cell_width
            y = row * cell_height
            
            # Get frame or create placeholder
            if stream.frame is not None:
                frame = stream.frame.copy()
                
                # Draw detection if available
                if stream.detection_result is not None:
                    frame = stream.detection_result.plot()
                
                # Resize to cell size
                frame = cv2.resize(frame, (cell_width, cell_height))
                
                # Add info overlay
                self.draw_camera_info(frame, stream, compact=True)
                
                # Place on canvas
                canvas[y:y+cell_height, x:x+cell_width] = frame
            else:
                # Placeholder
                placeholder = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                status_color = {
                    "disconnected": (128, 128, 128),
                    "connecting": (0, 255, 255),
                    "connected": (0, 255, 0),
                    "error": (0, 0, 255)
                }.get(stream.status, (128, 128, 128))
                
                cv2.putText(placeholder, stream.config.name, (10, cell_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(placeholder, stream.status, (10, cell_height//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
                
                canvas[y:y+cell_height, x:x+cell_width] = placeholder
            
            # Draw border
            cv2.rectangle(canvas, (x, y), (x+cell_width-1, y+cell_height-1), (50, 50, 50), 2)
        
        # Draw overall info
        self.draw_overall_info(canvas)
        
        return canvas
    
    def draw_camera_info(self, frame: np.ndarray, stream: CameraStream, compact: bool = False):
        """Draw camera info overlay"""
        h, w = frame.shape[:2]
        
        if compact:
            # Compact info for grid view
            info_text = f"{stream.config.name} | FPS:{stream.fps:.1f}"
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if stream.detection_count > 0:
                cv2.putText(frame, f"Detections: {stream.detection_count}", (5, h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Full info for single view
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(frame, stream.config.name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {stream.fps:.1f} | Detections: {stream.detection_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def draw_overall_info(self, canvas: np.ndarray):
        """Draw overall system info"""
        h, w = canvas.shape[:2]
        
        # Count active cameras
        active = sum(1 for s in self.streams if s.status == "connected")
        total = len([s for s in self.streams if s.config.enabled])
        
        # Calculate total FPS
        total_fps = sum(s.fps for s in self.streams if s.status == "connected")
        
        # Draw info bar at bottom
        info_h = 40
        cv2.rectangle(canvas, (0, h-info_h), (w, h), (30, 30, 30), -1)
        
        info_text = f"Cameras: {active}/{total} | Total FPS: {total_fps:.1f} | Mode: {self.display_mode.upper()}"
        cv2.putText(canvas, info_text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = "Q:Quit | G:Grid | S:Single | H:HUD | N/P:Next/Prev"
        cv2.putText(canvas, instructions, (w-650, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def create_single_view(self, display_width: int = 1920, display_height: int = 1080) -> np.ndarray:
        """Create single camera view"""
        active_streams = [s for s in self.streams if s.config.enabled and s.status == "connected"]
        
        if not active_streams or self.selected_camera >= len(active_streams):
            canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            cv2.putText(canvas, "No camera selected", (display_width//2 - 150, display_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return canvas
        
        stream = active_streams[self.selected_camera]
        
        if stream.frame is None:
            canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            return canvas
        
        # Get frame
        frame = stream.frame.copy()
        
        # Draw detection
        if stream.detection_result is not None:
            frame = stream.detection_result.plot()
        
        # Resize to display
        frame = cv2.resize(frame, (display_width, display_height))
        
        # Draw info
        self.draw_camera_info(frame, stream, compact=False)
        
        # Draw navigation
        nav_text = f"Camera {self.selected_camera + 1}/{len(active_streams)}"
        cv2.putText(frame, nav_text, (display_width - 250, display_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run_gui_mode(self, display_width: int = 1920, display_height: int = 1080):
        """Run in GUI mode with window"""
        cv2.namedWindow("Multi-Camera Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Camera Monitor", display_width, display_height)
        
        print("🎥 Multi-Camera Monitor Started")
        print("Controls:")
        print("  Q - Quit")
        print("  G - Grid view")
        print("  S - Single view")
        print("  N - Next camera (in single view)")
        print("  P - Previous camera (in single view)")
        
        while self.running:
            try:
                # Create display based on mode
                if self.display_mode == "grid":
                    display = self.create_mosaic_view(display_width, display_height)
                elif self.display_mode == "single":
                    display = self.create_single_view(display_width, display_height)
                else:  # hud
                    display = self.create_mosaic_view(display_width, display_height)
                
                cv2.imshow("Multi-Camera Monitor", display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                    break
                elif key == ord('g') or key == ord('G'):
                    self.display_mode = "grid"
                elif key == ord('s') or key == ord('S'):
                    self.display_mode = "single"
                elif key == ord('h') or key == ord('H'):
                    self.display_mode = "hud"
                elif key == ord('n') or key == ord('N'):
                    active_count = len([s for s in self.streams if s.config.enabled and s.status == "connected"])
                    if active_count > 0:
                        self.selected_camera = (self.selected_camera + 1) % active_count
                elif key == ord('p') or key == ord('P'):
                    active_count = len([s for s in self.streams if s.config.enabled and s.status == "connected"])
                    if active_count > 0:
                        self.selected_camera = (self.selected_camera - 1) % active_count
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in display loop: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def run_cli_mode(self):
        """Run in CLI mode (stats only)"""
        print("🎥 Multi-Camera Monitor - CLI Mode")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
                print("=" * 80)
                print("MULTI-CAMERA MONITORING SYSTEM")
                print("=" * 80)
                print()
                
                for idx, stream in enumerate(self.streams):
                    if not stream.config.enabled:
                        continue
                    
                    status_icon = {
                        "disconnected": "⚪",
                        "connecting": "🟡",
                        "connected": "🟢",
                        "error": "🔴"
                    }.get(stream.status, "⚪")
                    
                    print(f"{status_icon} Camera {idx+1}: {stream.config.name}")
                    print(f"   Status: {stream.status}")
                    if stream.status == "connected":
                        print(f"   FPS: {stream.fps:.1f}")
                        print(f"   Detections: {stream.detection_count}")
                        print(f"   Frames: {stream.frame_count}")
                    elif stream.status == "error":
                        print(f"   Error: {stream.error_msg}")
                    print()
                
                # Overall stats
                active = sum(1 for s in self.streams if s.status == "connected")
                total = len([s for s in self.streams if s.config.enabled])
                total_fps = sum(s.fps for s in self.streams if s.status == "connected")
                
                print("=" * 80)
                print(f"Active Cameras: {active}/{total} | Total FPS: {total_fps:.1f}")
                print("=" * 80)
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")

def load_cameras_from_yaml(yaml_file: str) -> List[CameraConfig]:
    """Load camera configurations from YAML file"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        cameras = []
        for cam_data in data.get('cameras', []):
            config = CameraConfig(
                name=cam_data.get('name', 'Unknown'),
                type=cam_data.get('type', 'webcam'),
                source=cam_data.get('source', 0),
                brand=cam_data.get('brand', 'generic'),
                username=cam_data.get('username', ''),
                password=cam_data.get('password', ''),
                ip=cam_data.get('ip', ''),
                port=cam_data.get('port', 554),
                stream_quality=cam_data.get('stream_quality', 'main'),
                enabled=cam_data.get('enabled', True)
            )
            cameras.append(config)
        
        return cameras
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Monitoring System")
    
    # Model settings
    parser.add_argument("--model", default="yolov11_1000ep_best.pt", 
                       help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--stride", type=int, default=1,
                       help="Process every N frames (1=all frames)")
    parser.add_argument("--max-fps", type=int, default=30,
                       help="Maximum FPS per camera (0=unlimited)")
    
    # Camera configuration
    parser.add_argument("--config", default="cameras.yaml",
                       help="Camera configuration YAML file")
    parser.add_argument("--add-webcam", action="store_true",
                       help="Add default webcam (camera 0)")
    
    # Display settings
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui",
                       help="Display mode")
    parser.add_argument("--width", type=int, default=1920,
                       help="Display width")
    parser.add_argument("--height", type=int, default=1080,
                       help="Display height")
    parser.add_argument("--view", choices=["grid", "single"], default="grid",
                       help="Initial view mode")
    
    args = parser.parse_args()
    
    print("🎥 Multi-Camera Monitoring System")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"Process stride: {args.stride}")
    print(f"Max FPS: {args.max_fps if args.max_fps > 0 else 'Unlimited'}")
    print(f"Display: {args.width}x{args.height}")
    print("=" * 60)
    
    # Initialize monitor
    monitor = MultiCameraMonitor(
        model_path=args.model,
        conf_threshold=args.conf,
        process_stride=args.stride,
        max_fps=args.max_fps
    )
    
    monitor.display_mode = args.view
    
    # Load cameras
    cameras_loaded = 0
    
    # From YAML file
    if args.config:
        cameras = load_cameras_from_yaml(args.config)
        for cam in cameras:
            monitor.add_camera(cam)
            cameras_loaded += 1
            print(f"✅ Added: {cam.name} ({cam.type})")
    
    # Add webcam if requested
    if args.add_webcam:
        webcam_config = CameraConfig(
            name="Webcam 0",
            type="webcam",
            source=0,
            enabled=True
        )
        monitor.add_camera(webcam_config)
        cameras_loaded += 1
        print(f"✅ Added: Webcam 0")
    
    if cameras_loaded == 0:
        print("❌ No cameras configured!")
        print("💡 Use --config cameras.yaml or --add-webcam")
        return
    
    print(f"\n📊 Total cameras: {cameras_loaded}")
    print("\n🚀 Starting monitoring...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Wait for cameras to connect
    time.sleep(2)
    
    # Run display
    try:
        if args.mode == "gui":
            monitor.run_gui_mode(args.width, args.height)
        else:
            monitor.run_cli_mode()
    finally:
        print("\n🛑 Stopping monitoring...")
        monitor.stop_monitoring()
        print("✅ Stopped")

if __name__ == "__main__":
    main()
