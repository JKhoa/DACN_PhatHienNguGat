#!/usr/bin/env python3
"""
Training Script cho YOLOv5 và YOLOv8 với 100 epochs
Tối ưu hóa thời gian training cho models còn lại
"""

import os
import sys
import json
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class FastTrainer:
    """Class để training nhanh YOLOv5 và YOLOv8 với 100 epochs"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.setup_logging()
        self.setup_directories()
        
        # Cấu hình training cho 100 epochs
        self.models_config = {
            'YOLOv8': {
                'model_name': 'yolov8n-pose.pt',
                'project_name': 'sleepy_pose_v8n_100ep',
                'epochs': 100,
                'batch': 8,
                'imgsz': 640,
                'device': 'cpu',
                'workers': 2,
                'patience': 30,  # Giảm patience cho training nhanh
                'save_period': 25,  # Lưu mỗi 25 epochs
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 10,  # Giảm warmup cho training nhanh
                'cos_lr': True,
                'mixup': 0.1,  # Giảm augmentation cho training nhanh
                'mosaic': 0.8,
                'copy_paste': 0.05
            },
            'YOLOv5': {
                'model_name': 'yolov5nu.pt',
                'project_name': 'sleepy_pose_v5n_100ep',
                'epochs': 100,
                'batch': 8,
                'imgsz': 640,
                'device': 'cpu',
                'workers': 2,
                'patience': 30,
                'save_period': 25,
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 10,
                'cos_lr': True,
                'mixup': 0.1,
                'mosaic': 0.8,
                'copy_paste': 0.05
            }
        }
        
        self.dataset_path = self.base_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
        self.training_results = {}
        
    def setup_logging(self):
        """Thiết lập logging cho training session"""
        log_file = self.base_dir / "tools" / "training_100_epochs.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Tạo directories cần thiết"""
        dirs_to_create = [
            "runs/pose-train",
            "model_backups_100ep",
            "training_results_100ep"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
    def train_yolov8_100_epochs(self):
        """Training YOLOv8 với 100 epochs"""
        try:
            model_name = "YOLOv8"
            config = self.models_config[model_name]
            
            self.logger.info(f"🚀 Bắt đầu training {model_name} với {config['epochs']} epochs...")
            
            # Kiểm tra dataset trước khi training
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            # Load model
            model = YOLO(config['model_name'])
            self.logger.info(f"✅ Loaded {config['model_name']}")
            
            # Tối ưu parameters cho training nhanh
            train_kwargs = {
                'data': str(self.dataset_path),
                'epochs': config['epochs'],
                'batch': config['batch'],
                'imgsz': config['imgsz'],
                'project': str(self.base_dir / "runs" / "pose-train"),
                'name': config['project_name'],
                'save': True,
                'device': config['device'],
                'workers': config['workers'],
                'patience': config['patience'],
                'save_period': config['save_period'],
                'plots': True,
                'verbose': True,
                'seed': 42,
                'lr0': config['lr0'],
                'lrf': config['lrf'],
                'momentum': config['momentum'],
                'weight_decay': config['weight_decay'],
                'warmup_epochs': config['warmup_epochs'],
                'cos_lr': config['cos_lr'],
                'mixup': config['mixup'],
                'mosaic': config['mosaic'],
                'copy_paste': config['copy_paste']
            }
            
            self.logger.info(f"🎯 Training parameters: {train_kwargs}")
            
            # Bắt đầu training
            start_time = time.time()
            self.logger.info("⏳ Starting model training...")
            results = model.train(**train_kwargs)
            training_time = time.time() - start_time
            
            # Lưu kết quả
            self.training_results[model_name] = {
                'status': 'completed',
                'training_time_hours': training_time / 3600,
                'epochs_completed': config['epochs'],
                'model_path': str(self.base_dir / "runs" / "pose-train" / config['project_name'] / "weights" / "best.pt")
            }
            
            self.logger.info(f"✅ {model_name} training completed in {training_time/3600:.2f} hours!")
            
            # Backup model
            self._backup_model(model_name, config['project_name'])
            
            return True
            
        except Exception as e:
            model_name = "YOLOv8"  # Đảm bảo model_name được định nghĩa trong exception
            self.logger.error(f"❌ Error training {model_name}: {str(e)}")
            self.training_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def train_yolov5_100_epochs(self):
        """Training YOLOv5 với 100 epochs"""
        try:
            model_name = "YOLOv5"
            config = self.models_config[model_name]
            
            self.logger.info(f"🚀 Bắt đầu training {model_name} với {config['epochs']} epochs...")
            
            # Kiểm tra dataset trước khi training
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            # Load model
            model = YOLO(config['model_name'])
            self.logger.info(f"✅ Loaded {config['model_name']}")
            
            # Tối ưu parameters cho training nhanh
            train_kwargs = {
                'data': str(self.dataset_path),
                'epochs': config['epochs'],
                'batch': config['batch'],
                'imgsz': config['imgsz'],
                'project': str(self.base_dir / "runs" / "pose-train"),
                'name': config['project_name'],
                'save': True,
                'device': config['device'],
                'workers': config['workers'],
                'patience': config['patience'],
                'save_period': config['save_period'],
                'plots': True,
                'verbose': True,
                'seed': 42,
                'lr0': config['lr0'],
                'lrf': config['lrf'],
                'momentum': config['momentum'],
                'weight_decay': config['weight_decay'],
                'warmup_epochs': config['warmup_epochs'],
                'cos_lr': config['cos_lr'],
                'mixup': config['mixup'],
                'mosaic': config['mosaic'],
                'copy_paste': config['copy_paste']
            }
            
            self.logger.info(f"🎯 Training parameters: {train_kwargs}")
            
            # Bắt đầu training
            start_time = time.time()
            results = model.train(**train_kwargs)
            training_time = time.time() - start_time
            
            # Lưu kết quả
            self.training_results[model_name] = {
                'status': 'completed',
                'training_time_hours': training_time / 3600,
                'epochs_completed': config['epochs'],
                'model_path': str(self.base_dir / "runs" / "pose-train" / config['project_name'] / "weights" / "best.pt")
            }
            
            self.logger.info(f"✅ {model_name} training completed in {training_time/3600:.2f} hours!")
            
            # Backup model
            self._backup_model(model_name, config['project_name'])
            
            return True
            
        except Exception as e:
            model_name = "YOLOv5"  # Đảm bảo model_name được định nghĩa trong exception
            self.logger.error(f"❌ Error training {model_name}: {str(e)}")
            self.training_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def _backup_model(self, model_name, project_name):
        """Backup trained model"""
        try:
            source_path = self.base_dir / "runs" / "pose-train" / project_name / "weights" / "best.pt"
            
            if not source_path.exists():
                self.logger.warning(f"⚠️ Model file not found: {source_path}")
                return
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_key = model_name.lower().replace('yolo', 'yolo')
            
            # Backup với timestamp
            backup_dir = self.base_dir / "model_backups_100ep"
            backup_path = backup_dir / f"{model_key}_100ep_best_{timestamp}.pt"
            shutil.copy2(source_path, backup_path)
            self.logger.info(f"✅ Model backed up: {backup_path}")
            
            # Copy to main directory với tên cố định
            main_model_path = self.base_dir / f"{model_key}_100ep_best.pt"
            shutil.copy2(source_path, main_model_path)
            self.logger.info(f"✅ Model copied to: {main_model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed for {model_name}: {str(e)}")
    
    def generate_final_report(self):
        """Tạo báo cáo tổng kết training"""
        try:
            self.logger.info("📊 Generating final training report...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.base_dir / "training_results_100ep" / f"fast_training_report_{timestamp}.json"
            
            report_data = {
                'training_session': {
                    'timestamp': timestamp,
                    'type': '100_epochs_fast_training',
                    'models_trained': list(self.training_results.keys())
                },
                'results': self.training_results,
                'summary': {
                    'total_models': len(self.training_results),
                    'successful_trainings': len([r for r in self.training_results.values() if r.get('status') == 'completed']),
                    'total_time_hours': sum([r.get('training_time_hours', 0) for r in self.training_results.values() if 'training_time_hours' in r])
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("================================================================================")
            self.logger.info("🎉 FAST TRAINING SESSION COMPLETED!")
            self.logger.info("================================================================================")
            
            # In kết quả summary
            for model_name, results in self.training_results.items():
                if results.get('status') == 'completed':
                    self.logger.info(f"\n🤖 {model_name}:")
                    self.logger.info(f"   ⏱️  Training time: {results['training_time_hours']:.2f} hours")
                    self.logger.info(f"   🎯 Epochs completed: {results['epochs_completed']}")
            
            total_time = sum([r.get('training_time_hours', 0) for r in self.training_results.values() if 'training_time_hours' in r])
            self.logger.info(f"\n⏱️  Total training time: {total_time:.2f} hours")
            self.logger.info(f"📄 Report saved: {report_path}")
            self.logger.info("================================================================================")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {str(e)}")
            return None

def main():
    """Hàm chính chạy fast training cho YOLOv5 và YOLOv8"""
    trainer = FastTrainer()
    
    models_to_train = ['YOLOv8', 'YOLOv5']
    successful_trainings = []
    
    trainer.logger.info("🚀 FAST TRAINING SESSION STARTED!")
    trainer.logger.info("================================================================================")
    trainer.logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.logger.info(f"🎯 Models to train: {models_to_train}")
    trainer.logger.info(f"⚡ Epochs per model: 100 (fast training)")
    trainer.logger.info("================================================================================")
    
    for i, model in enumerate(models_to_train, 1):
        try:
            trainer.logger.info(f"\n🔥 Starting training {i}/{len(models_to_train)}: {model}")
            trainer.logger.info("============================================================")
            
            if model == 'YOLOv8':
                success = trainer.train_yolov8_100_epochs()
            elif model == 'YOLOv5':
                success = trainer.train_yolov5_100_epochs()
            else:
                trainer.logger.error(f"❌ Unknown model: {model}")
                continue
                
            if success:
                successful_trainings.append(model)
                trainer.logger.info(f"✅ {model} training completed successfully!")
            else:
                trainer.logger.error(f"❌ {model} training failed!")
                
        except KeyboardInterrupt:
            trainer.logger.info(f"⚠️  Training interrupted during {model}")
            break
        except Exception as e:
            trainer.logger.error(f"❌ Unexpected error during {model}: {str(e)}")
    
    # Tạo báo cáo cuối
    report_path = trainer.generate_final_report()
    
    trainer.logger.info("📋 FINAL TRAINING SUMMARY")
    trainer.logger.info("================================================================================")
    trainer.logger.info(f"✅ Successful: {len(successful_trainings)} models")
    for model in successful_trainings:
        trainer.logger.info(f"    - {model}")
    
    if len(successful_trainings) < len(models_to_train):
        failed_models = [m for m in models_to_train if m not in successful_trainings]
        trainer.logger.info(f"❌ Failed: {len(failed_models)} models")
        for model in failed_models:
            trainer.logger.info(f"    - {model}")
    
    trainer.logger.info(f"\n📄 Complete report: {report_path}")
    trainer.logger.info("🎉 FAST TRAINING SESSION FINISHED!")

if __name__ == "__main__":
    main()