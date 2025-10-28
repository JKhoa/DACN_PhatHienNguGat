#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích kết quả training model phát hiện ngủ gật
"""

import json
import os
from pathlib import Path

def analyze_training_results():
    """Phân tích kết quả training"""
    
    print("=== PHAN TICH KET QUA TRAINING MODEL PHAT HIEN NGU GAT ===\n")
    
    # Tìm các file training results
    results_dirs = [
        "training_results_1000ep",
        "training_results_100ep", 
        "training_results_50ep",
        "training_results_30ep"
    ]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            print(f"📁 {results_dir.upper()}:")
            
            # Tìm file JSON trong thư mục
            for file in os.listdir(results_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(results_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        print(f"  📄 {file}")
                        
                        # Hiển thị thông tin cơ bản
                        if 'training_session' in data:
                            session = data['training_session']
                            print(f"    ⏱️  Thời gian: {session.get('total_duration_hours', 'N/A')} giờ")
                            print(f"    📅 Bắt đầu: {session.get('start_time', 'N/A')}")
                            print(f"    📅 Kết thúc: {session.get('end_time', 'N/A')}")
                        
                        # Hiển thị metrics
                        if 'final_metrics' in data:
                            metrics = data['final_metrics']
                            print(f"    📊 Metrics cuối cùng:")
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    print(f"      - {key}: {value:.4f}")
                                else:
                                    print(f"      - {key}: {value}")
                        
                        # Hiển thị class performance
                        if 'class_performance' in data:
                            class_perf = data['class_performance']
                            print(f"    🎯 Hiệu suất theo class:")
                            for class_name, perf in class_perf.items():
                                print(f"      - {class_name}:")
                                for metric, value in perf.items():
                                    if isinstance(value, (int, float)):
                                        print(f"        * {metric}: {value:.4f}")
                                    else:
                                        print(f"        * {metric}: {value}")
                        
                        print()
                        
                    except Exception as e:
                        print(f"    ❌ Lỗi đọc file {file}: {e}")
                        print()
    
    # Kiểm tra dataset
    print("THONG TIN DATASET:")
    dataset_path = "datasets/sleepy_pose"
    if os.path.exists(dataset_path):
        train_images = len([f for f in os.listdir(f"{dataset_path}/train/images") if f.endswith('.jpg')])
        train_labels = len([f for f in os.listdir(f"{dataset_path}/train/labels") if f.endswith('.txt')])
        val_images = len([f for f in os.listdir(f"{dataset_path}/val/images") if f.endswith('.jpg')])
        val_labels = len([f for f in os.listdir(f"{dataset_path}/val/labels") if f.endswith('.txt')])
        
        print(f"  🏋️  Training: {train_images} ảnh, {train_labels} labels")
        print(f"  ✅ Validation: {val_images} ảnh, {val_labels} labels")
        print(f"  📈 Tổng: {train_images + val_images} ảnh")
        
        # Đọc config dataset
        yaml_path = f"{dataset_path}/sleepy.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            print(f"  📋 Classes: {yaml_content.split('names:')[1].strip()}")
    
    print()
    
    # Kiểm tra model files
    print("🤖 MODEL FILES:")
    model_files = [
        "yolov11_1000ep_best.pt",
        "yolov5_50ep_best.pt", 
        "yolo8n-pose-sleepy.pt",
        "yolo11n-pose.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {model_file} (không tìm thấy)")
    
    print()
    print("=== KET LUAN ===")
    print("Model da duoc train voi 3 classes:")
    print("   0: binhthuong (binh thuong)")
    print("   1: ngugat (ngu gat)")  
    print("   2: gucxuongban (guc xuong ban)")
    print()
    print("Model tot nhat: yolov11_1000ep_best.pt (1000 epochs)")
    print("Dataset: 420 anh (282 train + 138 val)")
    print("Ho tro: Webcam + IP Camera (RTSP)")

if __name__ == "__main__":
    analyze_training_results()

"""
Phân tích kết quả training model phát hiện ngủ gật
"""

import json
import os
from pathlib import Path

def analyze_training_results():
    """Phân tích kết quả training"""
    
    print("=== PHAN TICH KET QUA TRAINING MODEL PHAT HIEN NGU GAT ===\n")
    
    # Tìm các file training results
    results_dirs = [
        "training_results_1000ep",
        "training_results_100ep", 
        "training_results_50ep",
        "training_results_30ep"
    ]
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            print(f"📁 {results_dir.upper()}:")
            
            # Tìm file JSON trong thư mục
            for file in os.listdir(results_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(results_dir, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        print(f"  📄 {file}")
                        
                        # Hiển thị thông tin cơ bản
                        if 'training_session' in data:
                            session = data['training_session']
                            print(f"    ⏱️  Thời gian: {session.get('total_duration_hours', 'N/A')} giờ")
                            print(f"    📅 Bắt đầu: {session.get('start_time', 'N/A')}")
                            print(f"    📅 Kết thúc: {session.get('end_time', 'N/A')}")
                        
                        # Hiển thị metrics
                        if 'final_metrics' in data:
                            metrics = data['final_metrics']
                            print(f"    📊 Metrics cuối cùng:")
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    print(f"      - {key}: {value:.4f}")
                                else:
                                    print(f"      - {key}: {value}")
                        
                        # Hiển thị class performance
                        if 'class_performance' in data:
                            class_perf = data['class_performance']
                            print(f"    🎯 Hiệu suất theo class:")
                            for class_name, perf in class_perf.items():
                                print(f"      - {class_name}:")
                                for metric, value in perf.items():
                                    if isinstance(value, (int, float)):
                                        print(f"        * {metric}: {value:.4f}")
                                    else:
                                        print(f"        * {metric}: {value}")
                        
                        print()
                        
                    except Exception as e:
                        print(f"    ❌ Lỗi đọc file {file}: {e}")
                        print()
    
    # Kiểm tra dataset
    print("THONG TIN DATASET:")
    dataset_path = "datasets/sleepy_pose"
    if os.path.exists(dataset_path):
        train_images = len([f for f in os.listdir(f"{dataset_path}/train/images") if f.endswith('.jpg')])
        train_labels = len([f for f in os.listdir(f"{dataset_path}/train/labels") if f.endswith('.txt')])
        val_images = len([f for f in os.listdir(f"{dataset_path}/val/images") if f.endswith('.jpg')])
        val_labels = len([f for f in os.listdir(f"{dataset_path}/val/labels") if f.endswith('.txt')])
        
        print(f"  🏋️  Training: {train_images} ảnh, {train_labels} labels")
        print(f"  ✅ Validation: {val_images} ảnh, {val_labels} labels")
        print(f"  📈 Tổng: {train_images + val_images} ảnh")
        
        # Đọc config dataset
        yaml_path = f"{dataset_path}/sleepy.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            print(f"  📋 Classes: {yaml_content.split('names:')[1].strip()}")
    
    print()
    
    # Kiểm tra model files
    print("🤖 MODEL FILES:")
    model_files = [
        "yolov11_1000ep_best.pt",
        "yolov5_50ep_best.pt", 
        "yolo8n-pose-sleepy.pt",
        "yolo11n-pose.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {model_file} (không tìm thấy)")
    
    print()
    print("=== KET LUAN ===")
    print("Model da duoc train voi 3 classes:")
    print("   0: binhthuong (binh thuong)")
    print("   1: ngugat (ngu gat)")  
    print("   2: gucxuongban (guc xuong ban)")
    print()
    print("Model tot nhat: yolov11_1000ep_best.pt (1000 epochs)")
    print("Dataset: 420 anh (282 train + 138 val)")
    print("Ho tro: Webcam + IP Camera (RTSP)")

if __name__ == "__main__":
    analyze_training_results()












