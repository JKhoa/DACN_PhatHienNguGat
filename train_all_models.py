#!/usr/bin/env python3
"""
Multi-Model Training Script for Multi-Person Sleepy Detection
Train YOLOv5, YOLOv8, and YOLOv11 models with expanded dataset
"""

import subprocess
import time
import os
from pathlib import Path
import shutil

class MultiModelTrainer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.yolo_dir = self.base_dir / "yolo-sleepy-allinone-final"
        self.tools_dir = self.yolo_dir / "tools"
        self.dataset_config = self.yolo_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
        
        # Model configurations
        self.models = {
            'yolov11': {
                'model_path': self.yolo_dir / 'yolo11n-pose.pt',
                'train_script': self.tools_dir / 'train_pose.py',
                'epochs': 50,
                'batch_size': 8
            },
            'yolov8': {
                'model_path': self.yolo_dir / 'yolov8n-pose.pt', 
                'train_script': None,  # Will use ultralytics directly
                'epochs': 50,
                'batch_size': 8
            },
            'yolov5': {
                'model_path': self.yolo_dir / 'yolov5nu.pt',
                'train_script': None,  # Will use YOLOv5 training
                'epochs': 50,
                'batch_size': 8
            }
        }
    
    def check_dataset(self):
        """Verify dataset is ready for training"""
        print("[CHECK] Verifying dataset...")
        
        if not self.dataset_config.exists():
            print(f"[ERROR] Dataset config not found: {self.dataset_config}")
            return False
        
        dataset_dir = self.dataset_config.parent
        train_dir = dataset_dir / "train" / "images"
        val_dir = dataset_dir / "val" / "images"
        
        train_images = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        val_images = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
        
        print(f"[INFO] Dataset ready:")
        print(f"  - Train images: {train_images}")
        print(f"  - Val images: {val_images}")
        print(f"  - Total: {train_images + val_images}")
        
        return train_images > 0 and val_images > 0
    
    def train_yolov11(self):
        """Train YOLOv11 model"""
        print("\n" + "="*60)
        print("[TRAIN] Starting YOLOv11 Training")
        print("="*60)
        
        try:
            # Change to tools directory
            original_cwd = Path.cwd()
            os.chdir(self.tools_dir)
            
            result = subprocess.run([
                "python", "train_pose.py"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                print("[SUCCESS] YOLOv11 training completed")
                return True
            else:
                print(f"[ERROR] YOLOv11 training failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"[ERROR] YOLOv11 training error: {e}")
            return False
    
    def train_yolov8(self):
        """Train YOLOv8 model"""
        print("\n" + "="*60)
        print("[TRAIN] Starting YOLOv8 Training")
        print("="*60)
        
        try:
            from ultralytics import YOLO
            
            # Load YOLOv8 pose model
            model = YOLO(str(self.models['yolov8']['model_path']))
            
            # Train with multi-person detection support
            results = model.train(
                data=str(self.dataset_config),
                epochs=self.models['yolov8']['epochs'],
                batch=self.models['yolov8']['batch_size'],
                imgsz=640,
                project=str(self.yolo_dir / "runs" / "pose-train"),
                name="sleepy_v8_multiperson",
                save_period=10,
                device='cpu',  # Use CPU for compatibility
                workers=0,
                patience=15
            )
            
            print("[SUCCESS] YOLOv8 training completed")
            return True
            
        except Exception as e:
            print(f"[ERROR] YOLOv8 training error: {e}")
            return False
    
    def train_yolov5(self):
        """Train YOLOv5 model"""
        print("\n" + "="*60)
        print("[TRAIN] Starting YOLOv5 Training")
        print("="*60)
        
        try:
            # Check if YOLOv5 directory exists
            yolov5_dir = self.base_dir / "yolov5"
            if not yolov5_dir.exists():
                print("[ERROR] YOLOv5 directory not found")
                return False
            
            # Convert dataset to YOLOv5 format if needed
            yolov5_dataset = self.yolo_dir / "datasets" / "sleepy_pose_yolov5" / "dataset.yaml"
            
            if yolov5_dataset.exists():
                # Change to YOLOv5 directory and train
                original_cwd = Path.cwd()
                os.chdir(yolov5_dir)
                
                result = subprocess.run([
                    "python", "train.py",
                    "--data", str(yolov5_dataset),
                    "--weights", str(self.models['yolov5']['model_path']),
                    "--epochs", str(self.models['yolov5']['epochs']),
                    "--batch-size", str(self.models['yolov5']['batch_size']),
                    "--imgsz", "640",
                    "--project", str(self.yolo_dir / "runs" / "pose-train"),
                    "--name", "sleepy_v5_multiperson",
                    "--device", "cpu"
                ], capture_output=True, text=True, timeout=3600)
                
                os.chdir(original_cwd)
                
                if result.returncode == 0:
                    print("[SUCCESS] YOLOv5 training completed")
                    return True
                else:
                    print(f"[ERROR] YOLOv5 training failed:")
                    print(result.stderr)
                    return False
            else:
                print("[ERROR] YOLOv5 dataset config not found")
                return False
                
        except Exception as e:
            print(f"[ERROR] YOLOv5 training error: {e}")
            return False
    
    def train_all_models(self):
        """Train all YOLO models sequentially"""
        print("="*60)
        print("MULTI-MODEL TRAINING FOR MULTI-PERSON SLEEPY DETECTION")
        print("="*60)
        
        if not self.check_dataset():
            print("[ERROR] Dataset verification failed")
            return
        
        results = {}
        start_time = time.time()
        
        # Train YOLOv11 first (most recent)
        print(f"\n[START] Training YOLOv11...")
        results['yolov11'] = self.train_yolov11()
        
        # Train YOLOv8
        print(f"\n[START] Training YOLOv8...")
        results['yolov8'] = self.train_yolov8()
        
        # Train YOLOv5 
        print(f"\n[START] Training YOLOv5...")
        results['yolov5'] = self.train_yolov5()
        
        # Summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("MULTI-MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"[TIME] Total training time: {elapsed_time/60:.1f} minutes")
        print(f"[RESULTS] Training results:")
        
        successful = 0
        for model, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  - {model.upper()}: {status}")
            if success:
                successful += 1
        
        print(f"\n[SUMMARY] {successful}/{len(results)} models trained successfully")
        
        if successful > 0:
            print(f"\n[NEXT] Next steps:")
            print(f"  1. Test models: python yolo-sleepy-allinone-final/tools/benchmark_pose_models.py")
            print(f"  2. Update app for multi-person detection")
            print(f"  3. Test with real-time detection")

def main():
    import os
    
    trainer = MultiModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()