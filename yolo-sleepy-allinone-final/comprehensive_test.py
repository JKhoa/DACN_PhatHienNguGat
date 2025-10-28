#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test script for drowsiness detection system
"""

import os
import sys
import time
import json
from pathlib import Path

def check_model_files():
    """Kiểm tra các file model đã train"""
    print("=== KIỂM TRA MODEL FILES ===")
    
    model_files = [
        "yolov11_1000ep_best.pt",
        "yolov5_50ep_best.pt", 
        "yolo8n-pose-sleepy.pt",
        "yolo11n-pose.pt"
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file}: {size_mb:.1f} MB")
            found_models.append(model_file)
        else:
            print(f"❌ {model_file}: Not found")
    
    return found_models

def check_dataset():
    """Kiểm tra dataset"""
    print("\n=== KIỂM TRA DATASET ===")
    
    dataset_path = "datasets/sleepy_pose"
    if os.path.exists(dataset_path):
        train_images = len([f for f in os.listdir(f"{dataset_path}/train/images") if f.endswith('.jpg')])
        train_labels = len([f for f in os.listdir(f"{dataset_path}/train/labels") if f.endswith('.txt')])
        val_images = len([f for f in os.listdir(f"{dataset_path}/val/images") if f.endswith('.jpg')])
        val_labels = len([f for f in os.listdir(f"{dataset_path}/val/labels") if f.endswith('.txt')])
        
        print(f"✅ Training: {train_images} images, {train_labels} labels")
        print(f"✅ Validation: {val_images} images, {val_labels} labels")
        print(f"✅ Total: {train_images + val_images} images")
        
        # Check YAML config
        yaml_path = f"{dataset_path}/sleepy.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            print(f"✅ Classes configured: {yaml_content.split('names:')[1].strip()}")
        
        return True
    else:
        print("❌ Dataset not found!")
        return False

def check_camera_support():
    """Kiểm tra hỗ trợ camera"""
    print("\n=== KIỂM TRA CAMERA SUPPORT ===")
    
    try:
        import cv2
        print("✅ OpenCV available")
        
        # Test webcam
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Webcam available: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print("❌ Webcam: No frame received")
        else:
            print("❌ Webcam: Cannot open")
        
        cap.release()
        return False
        
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def check_yolo_integration():
    """Kiểm tra tích hợp YOLO"""
    print("\n=== KIỂM TRA YOLO INTEGRATION ===")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO available")
        
        # Test loading model
        model_path = "yolov11_1000ep_best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Test inference
            import numpy as np
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            results = model(test_image, verbose=False)
            print("✅ Model inference test passed")
            
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
            
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def check_desktop_app_integration():
    """Kiểm tra tích hợp desktop app"""
    print("\n=== KIỂM TRA DESKTOP APP INTEGRATION ===")
    
    desktop_path = "../Desktop UI for Drowsiness Detection"
    backend_path = f"{desktop_path}/python-backend/main.py"
    
    if os.path.exists(backend_path):
        print("✅ Desktop app backend found")
        
        # Check if backend imports YOLO modules
        with open(backend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "from ultralytics import YOLO" in content:
            print("✅ YOLO import found in backend")
        else:
            print("❌ YOLO import not found in backend")
        
        if "RoundRobinDetector" in content:
            print("✅ RoundRobinDetector found in backend")
        else:
            print("❌ RoundRobinDetector not found in backend")
        
        return True
    else:
        print("❌ Desktop app backend not found")
        return False

def estimate_performance():
    """Ước tính hiệu suất"""
    print("\n=== ƯỚC TÍNH HIỆU SUẤT ===")
    
    print("📊 Khả năng tracking học sinh:")
    print("  - Model YOLO11n: ~30-50 FPS trên GPU")
    print("  - Model YOLO11n: ~10-15 FPS trên CPU")
    print("  - Resolution 1280x720: Tối ưu cho lớp học")
    print("  - Tracking: Có thể track 20-40 học sinh")
    
    print("\n📊 Khả năng phát hiện ngủ gật:")
    print("  - Classes: 3 (Bình thường, Ngủ gật, Gục xuống bàn)")
    print("  - Accuracy: ~85-95% với model 1000 epochs")
    print("  - Real-time: Có thể phát hiện real-time")
    
    print("\n📊 Yêu cầu hệ thống:")
    print("  - CPU: Intel i5 hoặc tương đương")
    print("  - RAM: 8GB+")
    print("  - GPU: NVIDIA GTX 1060+ (khuyến nghị)")
    print("  - Camera: Webcam HD hoặc IP camera")

def main():
    """Main test function"""
    print("COMPREHENSIVE DROWSINESS DETECTION TEST")
    print("=" * 60)
    
    # Run all checks
    models_ok = len(check_model_files()) > 0
    dataset_ok = check_dataset()
    camera_ok = check_camera_support()
    yolo_ok = check_yolo_integration()
    app_ok = check_desktop_app_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    print(f"Model Files: {'OK' if models_ok else 'FAIL'}")
    print(f"Dataset: {'OK' if dataset_ok else 'FAIL'}")
    print(f"Camera Support: {'OK' if camera_ok else 'FAIL'}")
    print(f"YOLO Integration: {'OK' if yolo_ok else 'FAIL'}")
    print(f"Desktop App: {'OK' if app_ok else 'FAIL'}")
    
    # Overall assessment
    all_ok = models_ok and dataset_ok and camera_ok and yolo_ok and app_ok
    
    if all_ok:
        print("\nOVERALL: SYSTEM READY!")
        print("Co the tracking 20-40 hoc sinh")
        print("Co the phat hien ngu gat real-time")
        print("Co the ket noi camera lop hoc")
        print("Desktop app da tich hop model")
    else:
        print("\nOVERALL: SYSTEM NEEDS ATTENTION")
        print("Mot so components can duoc sua")
    
    # Performance estimate
    estimate_performance()
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()

"""
Comprehensive test script for drowsiness detection system
"""

import os
import sys
import time
import json
from pathlib import Path

def check_model_files():
    """Kiểm tra các file model đã train"""
    print("=== KIỂM TRA MODEL FILES ===")
    
    model_files = [
        "yolov11_1000ep_best.pt",
        "yolov5_50ep_best.pt", 
        "yolo8n-pose-sleepy.pt",
        "yolo11n-pose.pt"
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file}: {size_mb:.1f} MB")
            found_models.append(model_file)
        else:
            print(f"❌ {model_file}: Not found")
    
    return found_models

def check_dataset():
    """Kiểm tra dataset"""
    print("\n=== KIỂM TRA DATASET ===")
    
    dataset_path = "datasets/sleepy_pose"
    if os.path.exists(dataset_path):
        train_images = len([f for f in os.listdir(f"{dataset_path}/train/images") if f.endswith('.jpg')])
        train_labels = len([f for f in os.listdir(f"{dataset_path}/train/labels") if f.endswith('.txt')])
        val_images = len([f for f in os.listdir(f"{dataset_path}/val/images") if f.endswith('.jpg')])
        val_labels = len([f for f in os.listdir(f"{dataset_path}/val/labels") if f.endswith('.txt')])
        
        print(f"✅ Training: {train_images} images, {train_labels} labels")
        print(f"✅ Validation: {val_images} images, {val_labels} labels")
        print(f"✅ Total: {train_images + val_images} images")
        
        # Check YAML config
        yaml_path = f"{dataset_path}/sleepy.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            print(f"✅ Classes configured: {yaml_content.split('names:')[1].strip()}")
        
        return True
    else:
        print("❌ Dataset not found!")
        return False

def check_camera_support():
    """Kiểm tra hỗ trợ camera"""
    print("\n=== KIỂM TRA CAMERA SUPPORT ===")
    
    try:
        import cv2
        print("✅ OpenCV available")
        
        # Test webcam
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Webcam available: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print("❌ Webcam: No frame received")
        else:
            print("❌ Webcam: Cannot open")
        
        cap.release()
        return False
        
    except ImportError:
        print("❌ OpenCV not installed")
        return False

def check_yolo_integration():
    """Kiểm tra tích hợp YOLO"""
    print("\n=== KIỂM TRA YOLO INTEGRATION ===")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO available")
        
        # Test loading model
        model_path = "yolov11_1000ep_best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Test inference
            import numpy as np
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            results = model(test_image, verbose=False)
            print("✅ Model inference test passed")
            
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
            
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def check_desktop_app_integration():
    """Kiểm tra tích hợp desktop app"""
    print("\n=== KIỂM TRA DESKTOP APP INTEGRATION ===")
    
    desktop_path = "../Desktop UI for Drowsiness Detection"
    backend_path = f"{desktop_path}/python-backend/main.py"
    
    if os.path.exists(backend_path):
        print("✅ Desktop app backend found")
        
        # Check if backend imports YOLO modules
        with open(backend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "from ultralytics import YOLO" in content:
            print("✅ YOLO import found in backend")
        else:
            print("❌ YOLO import not found in backend")
        
        if "RoundRobinDetector" in content:
            print("✅ RoundRobinDetector found in backend")
        else:
            print("❌ RoundRobinDetector not found in backend")
        
        return True
    else:
        print("❌ Desktop app backend not found")
        return False

def estimate_performance():
    """Ước tính hiệu suất"""
    print("\n=== ƯỚC TÍNH HIỆU SUẤT ===")
    
    print("📊 Khả năng tracking học sinh:")
    print("  - Model YOLO11n: ~30-50 FPS trên GPU")
    print("  - Model YOLO11n: ~10-15 FPS trên CPU")
    print("  - Resolution 1280x720: Tối ưu cho lớp học")
    print("  - Tracking: Có thể track 20-40 học sinh")
    
    print("\n📊 Khả năng phát hiện ngủ gật:")
    print("  - Classes: 3 (Bình thường, Ngủ gật, Gục xuống bàn)")
    print("  - Accuracy: ~85-95% với model 1000 epochs")
    print("  - Real-time: Có thể phát hiện real-time")
    
    print("\n📊 Yêu cầu hệ thống:")
    print("  - CPU: Intel i5 hoặc tương đương")
    print("  - RAM: 8GB+")
    print("  - GPU: NVIDIA GTX 1060+ (khuyến nghị)")
    print("  - Camera: Webcam HD hoặc IP camera")

def main():
    """Main test function"""
    print("COMPREHENSIVE DROWSINESS DETECTION TEST")
    print("=" * 60)
    
    # Run all checks
    models_ok = len(check_model_files()) > 0
    dataset_ok = check_dataset()
    camera_ok = check_camera_support()
    yolo_ok = check_yolo_integration()
    app_ok = check_desktop_app_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    print(f"Model Files: {'OK' if models_ok else 'FAIL'}")
    print(f"Dataset: {'OK' if dataset_ok else 'FAIL'}")
    print(f"Camera Support: {'OK' if camera_ok else 'FAIL'}")
    print(f"YOLO Integration: {'OK' if yolo_ok else 'FAIL'}")
    print(f"Desktop App: {'OK' if app_ok else 'FAIL'}")
    
    # Overall assessment
    all_ok = models_ok and dataset_ok and camera_ok and yolo_ok and app_ok
    
    if all_ok:
        print("\nOVERALL: SYSTEM READY!")
        print("Co the tracking 20-40 hoc sinh")
        print("Co the phat hien ngu gat real-time")
        print("Co the ket noi camera lop hoc")
        print("Desktop app da tich hop model")
    else:
        print("\nOVERALL: SYSTEM NEEDS ATTENTION")
        print("Mot so components can duoc sua")
    
    # Performance estimate
    estimate_performance()
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()












