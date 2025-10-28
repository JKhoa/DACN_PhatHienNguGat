#!/usr/bin/env python3
"""
Test script for YOLO drowsiness detection system
"""

import cv2
import time
import sys
import os
from pathlib import Path

# Add the python-backend directory to the path
backend_dir = Path(__file__).parent / "Desktop UI for Drowsiness Detection" / "python-backend"
sys.path.insert(0, str(backend_dir))

try:
    from yolo_detector import initialize_detector, detect_frame, draw_detections
    print("✅ YOLO detector imported successfully")
except ImportError as e:
    print(f"❌ Failed to import YOLO detector: {e}")
    sys.exit(1)

def test_yolo_detection():
    """Test YOLO detection with webcam"""
    print("🚀 Starting YOLO detection test...")
    
    # Initialize detector
    model_path = backend_dir / "yolo11n-pose.pt"
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure yolo11n-pose.pt is in the python-backend directory")
        return False
    
    print(f"📁 Using model: {model_path}")
    
    success = initialize_detector(str(model_path))
    if not success:
        print("❌ Failed to initialize YOLO detector")
        return False
    
    print("✅ YOLO detector initialized successfully")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return False
    
    print("📹 Webcam opened successfully")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            # Run detection
            detection_result = detect_frame(frame)
            
            # Draw detections
            annotated_frame = draw_detections(frame, detection_result)
            
            # Display frame
            cv2.imshow('YOLO Drowsiness Detection Test', annotated_frame)
            
            # Print detection info every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"📊 Frame {frame_count}: {len(detection_result.persons)} persons detected, FPS: {fps:.1f}")
                
                for person in detection_result.persons:
                    print(f"   Person {person.id}: {person.drowsiness_state} (score: {person.drowsiness_score:.2f})")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"test_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Saved frame: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"📈 Test completed: {frame_count} frames processed, avg FPS: {avg_fps:.1f}")
    
    return True

def test_model_loading():
    """Test if the YOLO model can be loaded"""
    print("🔍 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        model_path = backend_dir / "yolo11n-pose.pt"
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"📁 Loading model from: {model_path}")
        model = YOLO(str(model_path))
        print("✅ YOLO model loaded successfully")
        
        # Test inference on a dummy image
        import numpy as np
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✅ Model inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 YOLO Drowsiness Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Model loading
    print("\n1️⃣ Testing model loading...")
    if not test_model_loading():
        print("❌ Model loading test failed. Please check your YOLO installation.")
        return
    
    # Test 2: Full detection test
    print("\n2️⃣ Testing full detection pipeline...")
    if not test_yolo_detection():
        print("❌ Detection test failed.")
        return
    
    print("\n🎉 All tests passed! YOLO detection system is working correctly.")
    print("\n📋 Next steps:")
    print("   1. Start the Python backend: python server.py")
    print("   2. Start the Electron app: npm run electron")
    print("   3. Enable YOLO detection in the camera settings")

if __name__ == "__main__":
    main()
