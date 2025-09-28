#!/usr/bin/env python3
"""
YOLOv5 Pose Training Script for Multi-Person Sleepy Detection
============================================================

This script trains a YOLOv5 pose detection model on the multi-person sleepy dataset.
It's designed to work as part of the complete training pipeline with YOLOv8 and YOLOv11.

Features:
- YOLOv5n-pose model training
- Custom dataset configuration
- CPU training support  
- Progress monitoring
- Model validation
- Enhanced pose detection for sleepiness classification

Usage:
    python train_yolo5.py

Author: AI Assistant
Date: December 2024
"""

import os
import sys
import subprocess
from pathlib import Path

def install_yolov5():
    """Install YOLOv5 if not already installed"""
    print("📦 Installing YOLOv5...")
    try:
        # Clone YOLOv5 repository if not exists
        yolov5_dir = Path("yolov5")
        if not yolov5_dir.exists():
            subprocess.run([
                "git", "clone", "https://github.com/ultralytics/yolov5.git"
            ], check=True)
            
        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"
        ], check=True)
        
        print("✅ YOLOv5 installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing YOLOv5: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def train_yolov5():
    """Train YOLOv5 pose model"""
    print("\n🚀 Starting YOLOv5 Pose Training for Multi-Person Sleepy Detection")
    print("=" * 60)
    
    # Get paths
    current_dir = Path.cwd()
    root_dir = current_dir.parent
    dataset_config = root_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
    
    print(f"Current directory: {current_dir}")
    print(f"Root directory: {root_dir}")
    print(f"Dataset config: {dataset_config}")
    
    # Verify dataset exists
    if not dataset_config.exists():
        print(f"❌ Dataset configuration not found: {dataset_config}")
        return False
    
    # Setup YOLOv5 if needed
    yolov5_dir = current_dir / "yolov5"
    if not yolov5_dir.exists():
        if not install_yolov5():
            return False
    
    try:
        print("Loading YOLOv5n-pose model...")
        
        # Training parameters optimized for our custom dataset
        train_cmd = [
            sys.executable,
            "yolov5/train.py",
            "--img", "640",
            "--batch", "4", 
            "--epochs", "50",
            "--data", str(dataset_config),
            "--weights", "yolov5n-pose.pt",  # This will auto-download if not exists
            "--project", str(root_dir / "runs" / "pose-train"),
            "--name", "sleepy_pose_v5n",
            "--save-period", "10",
            "--device", "cpu",  # Use CPU for compatibility
            "--workers", "0",   # Reduce workers for stability
            "--patience", "20", # Early stopping
            "--plots"           # Generate training plots
        ]
        
        print("Starting YOLOv5 training...")
        print(f"Training command: {' '.join(train_cmd)}")
        
        # Run training
        result = subprocess.run(train_cmd, cwd=current_dir, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ YOLOv5 training completed successfully!")
            
            # Check for trained model
            model_path = root_dir / "runs" / "pose-train" / "sleepy_pose_v5n" / "weights" / "best.pt"
            if model_path.exists():
                print(f"✅ Trained model saved: {model_path}")
                
                # Copy model to main directory for easy access
                target_path = root_dir / "yolo5n-pose-sleepy.pt"
                import shutil
                shutil.copy2(model_path, target_path)
                print(f"✅ Model copied to: {target_path}")
                
            return True
        else:
            print(f"❌ YOLOv5 training failed with code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error during YOLOv5 training: {e}")
        return False

def main():
    """Main function"""
    print("🎯 YOLOv5 Pose Training for Multi-Person Sleepy Detection")
    print("=" * 60)
    
    success = train_yolov5()
    
    if success:
        print("\n🎉 YOLOv5 Training Session Completed!")
        print("=" * 50)
        print("✅ Model: YOLOv5n-pose")
        print("✅ Task: Multi-person sleepy detection")
        print("✅ Dataset: Custom sleepy pose dataset (60 images)")
        print("✅ Training: Completed successfully")
        print("\n📝 Next Steps:")
        print("- Test model with: python ../standalone_app.py --model yolo5n-pose-sleepy.pt")
        print("- Compare with YOLOv8/v11 models")
        print("- Run full model comparison benchmark")
    else:
        print("\n❌ YOLOv5 training failed!")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()