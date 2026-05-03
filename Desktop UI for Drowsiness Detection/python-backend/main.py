import cv2
import numpy as np
import json
import time
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import base64
import io
from PIL import Image
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrowsinessDetector:
    def __init__(self, model_path="yolo-sleepy-allinone-final/best.pt"):
        """Initialize YOLO model for drowsiness detection"""
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to default YOLO pose model
            self.model = YOLO('yolo11n-pose.pt')
            logger.info("Using default YOLO pose model")
        
        self.confidence_threshold = 0.5
        self.sleep_threshold = 3.0  # seconds
        self.student_tracking = {}  # Track students across frames
        
    def detect_drowsiness(self, frame):
        """Detect drowsiness in frame using YOLO model - Focus on head detection"""
        try:
            # Run YOLO detection with focus on head region
            results = self.model(frame, conf=self.confidence_threshold)
            
            detections = []
            current_time = time.time()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Get class prediction
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Determine state based on class
                        if cls == 0:  # Normal
                            state = 'normal'
                        elif cls == 1:  # Sleepy
                            state = 'sleepy'
                        elif cls == 2:  # Head down
                            state = 'head_down'
                        else:
                            state = 'normal'
                        
                        # Focus on head region - adjust bounding box to focus on upper part
                        head_height = (y2 - y1) * 0.4  # Focus on top 40% (head region)
                        head_y1 = y1
                        head_y2 = y1 + head_height
                        
                        # Calculate center point of head region
                        center_x = int((x1 + x2) / 2)
                        center_y = int((head_y1 + head_y2) / 2)
                        
                        # Generate student ID based on head position (smaller grid for better separation)
                        student_id = f"student-{center_x//30}-{center_y//30}"
                        
                        # Update tracking
                        if student_id not in self.student_tracking:
                            self.student_tracking[student_id] = {
                                'first_seen': current_time,
                                'sleep_start': None,
                                'sleep_duration': 0,
                                'last_state': 'normal',
                                'position_history': [],
                                'head_bbox': [int(x1), int(head_y1), int(x2), int(head_y2)]
                            }
                        
                        # Update sleep duration
                        tracking = self.student_tracking[student_id]
                        if state in ['sleepy', 'head_down']:
                            if tracking['sleep_start'] is None:
                                tracking['sleep_start'] = current_time
                            tracking['sleep_duration'] = current_time - tracking['sleep_start']
                        else:
                            tracking['sleep_start'] = None
                            tracking['sleep_duration'] = 0
                        
                        tracking['last_state'] = state
                        tracking['position_history'].append({
                            'x': center_x,
                            'y': center_y,
                            'timestamp': current_time
                        })
                        
                        # Keep only last 10 positions
                        if len(tracking['position_history']) > 10:
                            tracking['position_history'] = tracking['position_history'][-10:]
                        
                        # Update head bounding box
                        tracking['head_bbox'] = [int(x1), int(head_y1), int(x2), int(head_y2)]
                        
                        detection = {
                            'id': student_id,
                            'position': {'x': center_x, 'y': center_y},
                            'state': state,
                            'confidence': float(confidence),
                            'sleepDuration': tracking['sleep_duration'],
                            'lastUpdate': datetime.now().isoformat(),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Full body bbox
                            'headBbox': [int(x1), int(head_y1), int(x2), int(head_y2)]  # Head-only bbox
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in drowsiness detection: {e}")
            return []

class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.detector = DrowsinessDetector()
        self.running = False
        
    def add_camera(self, camera_id, camera_config):
        """Add a new camera"""
        try:
            if camera_config['type'] == 'ip':
                # IP Camera
                cap = cv2.VideoCapture(camera_config['rtspUrl'])
                if not cap.isOpened():
                    raise Exception(f"Cannot open RTSP stream: {camera_config['rtspUrl']}")
            elif camera_config['type'] == 'webcam':
                # Webcam - try different backends for better compatibility
                device_id = int(camera_config.get('deviceId', 0))
                
                # Try different backends
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                cap = None
                
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(device_id, backend)
                        if cap.isOpened():
                            # Test if we can read a frame
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                logger.info(f"Successfully opened webcam {device_id} with backend {backend}")
                                break
                            else:
                                cap.release()
                                cap = None
                    except Exception as e:
                        logger.warning(f"Backend {backend} failed for device {device_id}: {e}")
                        if cap:
                            cap.release()
                            cap = None
                
                if not cap or not cap.isOpened():
                    raise Exception(f"Cannot open webcam device: {device_id}")
            else:
                raise Exception(f"Unsupported camera type: {camera_config['type']}")
            
            self.cameras[camera_id] = {
                'cap': cap,
                'config': camera_config,
                'last_frame': None,
                'students': [],
                'fps': 0,
                'last_fps_time': time.time(),
                'frame_count': 0,
                'running': False
            }
            
            logger.info(f"Added camera {camera_id}: {camera_config['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add camera {camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id):
        """Remove a camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id]['cap'].release()
            del self.cameras[camera_id]
            logger.info(f"Removed camera {camera_id}")
    
    def start_camera(self, camera_id):
        """Start camera processing"""
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        camera = self.cameras[camera_id]
        camera['running'] = True
        camera['last_fps_time'] = time.time()
        camera['frame_count'] = 0
        
        # Start processing thread
        thread = threading.Thread(target=self._process_camera, args=(camera_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started camera {camera_id} processing thread")
        return True
    
    def stop_camera(self, camera_id):
        """Stop camera processing"""
        if camera_id in self.cameras:
            self.cameras[camera_id]['running'] = False
    
    def _process_camera(self, camera_id):
        """Process camera frames"""
        camera = self.cameras[camera_id]
        cap = camera['cap']
        
        logger.info(f"Starting camera processing for {camera_id}")
        
        while camera.get('running', False):
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Log first successful frame
                if camera.get('frame_count', 0) == 0:
                    logger.info(f"Successfully reading frames from camera {camera_id}, frame shape: {frame.shape}")
                
                # Detect drowsiness
                detections = self.detector.detect_drowsiness(frame)
                
                # Update camera data
                camera['students'] = detections
                camera['last_frame'] = frame
                
                # Calculate FPS
                current_time = time.time()
                camera['frame_count'] += 1
                if current_time - camera['last_fps_time'] >= 1.0:
                    camera['fps'] = camera['frame_count'] / (current_time - camera['last_fps_time'])
                    logger.info(f"Camera {camera_id} FPS: {camera['fps']:.1f}, Students: {len(detections)}")
                    camera['frame_count'] = 0
                    camera['last_fps_time'] = current_time
                
                # Small delay to prevent overwhelming
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Stopped camera processing for {camera_id}")
    
    def get_camera_status(self, camera_id):
        """Get camera status and data"""
        if camera_id not in self.cameras:
            return None
        
        camera = self.cameras[camera_id]
        students = camera['students']
        sleepy_students = len([s for s in students if s['state'] in ['sleepy', 'head_down']])
        
        return {
            'id': camera_id,
            'name': camera['config']['name'],
            'status': 'online' if camera.get('running', False) else 'offline',
            'fps': camera['fps'],
            'students': students,
            'totalStudents': len(students),
            'sleepyStudents': sleepy_students,
            'lastUpdate': datetime.now().isoformat()
        }
    
    def get_all_cameras_status(self):
        """Get status of all cameras"""
        status = {}
        for camera_id in self.cameras:
            status[camera_id] = self.get_camera_status(camera_id)
        return status

# Global camera manager
camera_manager = CameraManager()

def start_backend():
    """Start the Python backend server"""
    logger.info("Starting Python backend for drowsiness detection...")
    
    # Example: Add a test camera (you can remove this)
    # camera_manager.add_camera('test-cam', {
    #     'name': 'Test Camera',
    #     'type': 'webcam',
    #     'deviceId': '0'
    # })
    
    return camera_manager

if __name__ == "__main__":
    # Start backend
    manager = start_backend()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down backend...")
        for camera_id in list(manager.cameras.keys()):
            manager.remove_camera(camera_id)

