#!/usr/bin/env python3
"""
YOLOv8 Pose Training Script for Multi-Person Sleepy Detection
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def train_yolov8():
    """Train YOLOv8 pose model with sleepy dataset"""
    
    # Setup paths
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    dataset_yaml = root_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
    
    print(f"Current directory: {current_dir}")
    print(f"Root directory: {root_dir}")
    print(f"Dataset config: {dataset_yaml}")
    
    # Check if dataset config exists
    if not dataset_yaml.exists():
        print(f"Error: Dataset config not found at {dataset_yaml}")
        return False
    
    try:
        # Load YOLOv8n-pose model
        print("Loading YOLOv8n-pose model...")
        model = YOLO('yolov8n-pose.pt')
        print("Model loaded successfully!")
        
        # Training configuration
        train_kwargs = {
            'data': str(dataset_yaml),
            'epochs': 50,
            'batch': 4,
            'imgsz': 640,
            'project': str(root_dir / 'runs' / 'pose-train'),
            'name': 'sleepy_pose_v8n',
            'save': True,
            'device': 'cpu',
            'workers': 0,
            'patience': 20,
            'plots': True,
            'verbose': True,
            'seed': 42
        }
        
        print("Starting YOLOv8 training...")
        print(f"Training parameters: {train_kwargs}")
        
        # Start training
        results = model.train(**train_kwargs)
        
        print("✅ YOLOv8 training completed successfully!")
        print(f"Best model saved at: {results.save_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during YOLOv8 training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Starting YOLOv8 Pose Training for Multi-Person Sleepy Detection")
    print("=" * 60)
    
    success = train_yolov8()
    
    if success:
        print("\n🎉 YOLOv8 training process completed successfully!")
    else:
        print("\n❌ YOLOv8 training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()