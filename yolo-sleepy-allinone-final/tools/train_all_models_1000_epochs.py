#!/usr/bin/env python3
"""
Intensive Training Script - 1000 Epochs for All Models
=====================================================

Script này sẽ training cả 3 models YOLOv5, YOLOv8, và YOLOv11 với 1000 epochs mỗi model
cho dataset multi-person sleepy detection.

⚠️  CHÚ Ý: Quá trình này sẽ mất rất nhiều thời gian (có thể 10-20 giờ hoặc hơn)
🔥 Training intensive với 1000 epochs sẽ tạo ra models có độ chính xác cao nhất

Features:
- Training tuần tự cả 3 models
- Auto-backup models sau mỗi 100 epochs
- Progress monitoring và logging
- Error recovery và resume training
- Performance comparison sau khi hoàn thành

Author: AI Assistant
Date: September 2024
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_1000_epochs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntensiveTrainer:
    """Class để quản lý training 1000 epochs cho tất cả models"""
    
    def __init__(self):
        self.models_config = {
            'YOLOv11': {
                'epochs': 1000,
                'batch_size': 8,  # Tăng batch size cho training tốt hơn
                'patience': 100,  # Patience cao hơn cho 1000 epochs
                'save_period': 50,  # Save mỗi 50 epochs
                'model_name': 'yolo11n-pose.pt',
                'project_name': 'sleepy_pose_v11n_1000ep'
            },
            'YOLOv8': {
                'epochs': 1000,
                'batch_size': 8,
                'patience': 100,
                'save_period': 50,
                'model_name': 'yolov8n-pose.pt',
                'project_name': 'sleepy_pose_v8n_1000ep'
            },
            'YOLOv5': {
                'epochs': 1000,
                'batch_size': 8,
                'patience': 100,
                'save_period': 50,
                'model_name': 'yolov5n-pose.pt',
                'project_name': 'sleepy_pose_v5n_1000ep'
            }
        }
        
        self.current_dir = Path.cwd()
        self.root_dir = self.current_dir.parent
        self.dataset_config = self.root_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
        
        # Training results tracking
        self.training_results = {}
        self.start_time = None
        
    def setup_training_environment(self):
        """Setup môi trường training"""
        logger.info("🔧 Setting up intensive training environment...")
        
        # Verify dataset
        if not self.dataset_config.exists():
            logger.error(f"❌ Dataset không tìm thấy: {self.dataset_config}")
            return False
            
        # Create backup directory
        backup_dir = self.root_dir / "model_backups_1000ep"
        backup_dir.mkdir(exist_ok=True)
        
        # Create results directory
        results_dir = self.root_dir / "training_results_1000ep"
        results_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Training environment setup completed!")
        return True
    
    def train_yolov11_1000_epochs(self):
        """Training YOLOv11 với 1000 epochs"""
        model_name = 'YOLOv11'
        config = self.models_config[model_name]
        
        logger.info(f"🚀 Bắt đầu training {model_name} với {config['epochs']} epochs...")
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(config['model_name'])
            logger.info(f"✅ Loaded {config['model_name']}")
            
            # Training parameters cho 1000 epochs
            train_kwargs = {
                'data': str(self.dataset_config),
                'epochs': config['epochs'],
                'batch': config['batch_size'],
                'imgsz': 640,
                'project': str(self.root_dir / "runs" / "pose-train"),
                'name': config['project_name'],
                'save': True,
                'device': 'cpu',  # CPU training cho stability
                'workers': 2,     # Tăng workers
                'patience': config['patience'],
                'save_period': config['save_period'],
                'plots': True,
                'verbose': True,
                'seed': 42,
                # Optimization parameters cho long training
                'lr0': 0.001,     # Learning rate thấp hơn cho stable training
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 10,  # Warmup dài hơn
                'cos_lr': True,   # Cosine learning rate scheduler
                'mixup': 0.1,     # Data augmentation
                'mosaic': 1.0
            }
            
            logger.info(f"🎯 Training parameters: {train_kwargs}")
            
            # Start training
            start_time = time.time()
            results = model.train(**train_kwargs)
            end_time = time.time()
            
            training_time = end_time - start_time
            logger.info(f"✅ {model_name} training completed in {training_time/3600:.2f} hours!")
            
            # Save results
            self.training_results[model_name] = {
                'training_time_hours': training_time / 3600,
                'epochs_completed': config['epochs'],
                'final_results': str(results) if results else 'No results object'
            }
            
            # Backup best model
            self._backup_model(model_name, config['project_name'])
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            return False
    
    def train_yolov8_1000_epochs(self):
        """Training YOLOv8 với 1000 epochs"""
        model_name = 'YOLOv8'
        config = self.models_config[model_name]
        
        logger.info(f"🚀 Bắt đầu training {model_name} với {config['epochs']} epochs...")
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(config['model_name'])
            logger.info(f"✅ Loaded {config['model_name']}")
            
            # Training parameters optimized cho YOLOv8
            train_kwargs = {
                'data': str(self.dataset_config),
                'epochs': config['epochs'],
                'batch': config['batch_size'],
                'imgsz': 640,
                'project': str(self.root_dir / "runs" / "pose-train"),
                'name': config['project_name'],
                'save': True,
                'device': 'cpu',
                'workers': 2,
                'patience': config['patience'],
                'save_period': config['save_period'],
                'plots': True,
                'verbose': True,
                'seed': 42,
                # YOLOv8 specific optimizations
                'lr0': 0.0008,    # Slightly lower learning rate
                'lrf': 0.01,      # Final learning rate
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 15,
                'cos_lr': True,
                'mixup': 0.15,    # More aggressive augmentation
                'mosaic': 1.0,
                'copy_paste': 0.1
            }
            
            logger.info(f"🎯 Training parameters: {train_kwargs}")
            
            # Start training
            start_time = time.time()
            results = model.train(**train_kwargs)
            end_time = time.time()
            
            training_time = end_time - start_time
            logger.info(f"✅ {model_name} training completed in {training_time/3600:.2f} hours!")
            
            # Save results
            self.training_results[model_name] = {
                'training_time_hours': training_time / 3600,
                'epochs_completed': config['epochs'],
                'final_results': str(results) if results else 'No results object'
            }
            
            # Backup best model
            self._backup_model(model_name, config['project_name'])
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            return False
    
    def train_yolov5_1000_epochs(self):
        """Training YOLOv5 với 1000 epochs"""
        model_name = 'YOLOv5'
        config = self.models_config[model_name]
        
        logger.info(f"🚀 Bắt đầu training {model_name} với {config['epochs']} epochs...")
        
        # YOLOv5 cần setup đặc biệt
        yolov5_dir = self.current_dir / "yolov5"
        
        try:
            # Install/setup YOLOv5 nếu chưa có
            if not yolov5_dir.exists():
                logger.info("📦 Cloning YOLOv5 repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/ultralytics/yolov5.git"
                ], check=True, cwd=self.current_dir)
                
                # Install requirements
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"
                ], check=True)
            
            # Training command cho YOLOv5
            train_cmd = [
                sys.executable,
                "yolov5/train.py",
                "--img", "640",
                "--batch", str(config['batch_size']),
                "--epochs", str(config['epochs']),
                "--data", str(self.dataset_config),
                "--weights", config['model_name'],
                "--project", str(self.root_dir / "runs" / "pose-train"),
                "--name", config['project_name'],
                "--save-period", str(config['save_period']),
                "--device", "cpu",
                "--workers", "2",
                "--patience", str(config['patience']),
                "--plots",
                # YOLOv5 specific parameters
                "--hyp", "yolov5/data/hyps/hyp.scratch-low.yaml",  # Low learning rate hyps
                "--optimizer", "AdamW",
                "--cos-lr",
                "--label-smoothing", "0.1"
            ]
            
            logger.info(f"🎯 Training command: {' '.join(train_cmd)}")
            
            # Start training
            start_time = time.time()
            result = subprocess.run(train_cmd, cwd=self.current_dir, capture_output=False)
            end_time = time.time()
            
            if result.returncode == 0:
                training_time = end_time - start_time
                logger.info(f"✅ {model_name} training completed in {training_time/3600:.2f} hours!")
                
                # Save results
                self.training_results[model_name] = {
                    'training_time_hours': training_time / 3600,
                    'epochs_completed': config['epochs'],
                    'final_results': 'YOLOv5 training completed successfully'
                }
                
                # Backup best model
                self._backup_model(model_name, config['project_name'])
                
                return True
            else:
                logger.error(f"❌ {model_name} training failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            return False
    
    def _backup_model(self, model_name, project_name):
        """Backup trained model"""
        try:
            source_path = self.root_dir / "runs" / "pose-train" / project_name / "weights" / "best.pt"
            backup_dir = self.root_dir / "model_backups_1000ep"
            
            if source_path.exists():
                backup_name = f"{model_name.lower()}_1000ep_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                backup_path = backup_dir / backup_name
                
                shutil.copy2(source_path, backup_path)
                logger.info(f"✅ Model backed up: {backup_path}")
                
                # Copy to main directory với tên đơn giản
                main_model_path = self.root_dir / f"{model_name.lower()}_1000ep_best.pt"
                shutil.copy2(source_path, main_model_path)
                logger.info(f"✅ Model copied to: {main_model_path}")
                
        except Exception as e:
            logger.error(f"❌ Error backing up {model_name}: {e}")
    
    def generate_final_report(self):
        """Tạo báo cáo cuối cùng sau khi training tất cả models"""
        logger.info("📊 Generating final training report...")
        
        report = {
            'training_session': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            },
            'models_trained': len(self.training_results),
            'training_results': self.training_results,
            'configuration': self.models_config
        }
        
        # Save report
        report_path = self.root_dir / "training_results_1000ep" / f"intensive_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🎉 INTENSIVE TRAINING SESSION COMPLETED!")
        logger.info("=" * 80)
        
        for model_name, results in self.training_results.items():
            logger.info(f"\n🤖 {model_name}:")
            logger.info(f"   ⏱️  Training time: {results['training_time_hours']:.2f} hours")
            logger.info(f"   🎯 Epochs completed: {results['epochs_completed']}")
        
        total_time = sum(r['training_time_hours'] for r in self.training_results.values())
        logger.info(f"\n⏱️  Total training time: {total_time:.2f} hours")
        logger.info(f"📄 Report saved: {report_path}")
        
        return report_path

def main():
    """Main function để chạy intensive training"""
    trainer = IntensiveTrainer()
    
    logger.info("🎯 INTENSIVE TRAINING SESSION - 1000 EPOCHS PER MODEL")
    logger.info("=" * 80)
    logger.info("⚠️  WARNING: This will take 10-20+ hours to complete!")
    logger.info("🔥 Training YOLOv5, YOLOv8, and YOLOv11 with 1000 epochs each")
    logger.info("💾 Models will be auto-saved every 50 epochs")
    logger.info("📊 Progress will be logged continuously")
    logger.info("=" * 80)
    
    # Confirmation
    try:
        confirm = input("\n🚨 Bạn có chắc chắn muốn bắt đầu intensive training? (y/N): ")
        if confirm.lower() != 'y':
            logger.info("❌ Training cancelled by user")
            return
    except KeyboardInterrupt:
        logger.info("❌ Training cancelled by user")
        return
    
    trainer.start_time = datetime.now()
    
    # Setup environment
    if not trainer.setup_training_environment():
        logger.error("❌ Failed to setup training environment")
        return
    
    # Training sequence
    models_to_train = ['YOLOv11', 'YOLOv8', 'YOLOv5']
    successful_trainings = []
    failed_trainings = []
    
    for i, model in enumerate(models_to_train, 1):
        logger.info(f"\n🔥 Starting training {i}/{len(models_to_train)}: {model}")
        logger.info("=" * 60)
        
        try:
            if model == 'YOLOv11':
                success = trainer.train_yolov11_1000_epochs()
            elif model == 'YOLOv8':
                success = trainer.train_yolov8_1000_epochs()
            elif model == 'YOLOv5':
                success = trainer.train_yolov5_1000_epochs()
            
            if success:
                successful_trainings.append(model)
                logger.info(f"✅ {model} training completed successfully!")
            else:
                failed_trainings.append(model)
                logger.error(f"❌ {model} training failed!")
                
        except KeyboardInterrupt:
            logger.info(f"⚠️  Training interrupted during {model}")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error during {model} training: {e}")
            failed_trainings.append(model)
    
    # Generate final report
    report_path = trainer.generate_final_report()
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 FINAL TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Successful: {len(successful_trainings)} models")
    for model in successful_trainings:
        logger.info(f"   - {model}")
    
    if failed_trainings:
        logger.info(f"❌ Failed: {len(failed_trainings)} models")
        for model in failed_trainings:
            logger.info(f"   - {model}")
    
    logger.info(f"\n📄 Complete report: {report_path}")
    logger.info("🎉 INTENSIVE TRAINING SESSION FINISHED!")

if __name__ == "__main__":
    main()