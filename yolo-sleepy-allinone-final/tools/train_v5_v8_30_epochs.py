#!/usr/bin/env python3
"""
Training Script siêu nhanh cho YOLOv5 và YOLOv8 với 30 epochs
Tối ưu hóa tối đa thời gian training
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

class SuperFastTrainer:
    """Class để training siêu nhanh YOLOv5 và YOLOv8 với 30 epochs"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.setup_logging()
        self.setup_directories()
        
        # Cấu hình training siêu nhanh với 30 epochs
        self.models_config = {
            'YOLOv8': {
                'model_name': 'yolov8n-pose.pt',
                'project_name': 'sleepy_pose_v8n_30ep',
                'epochs': 30,
                'batch': 8,
                'imgsz': 640,
                'device': 'cpu',
                'workers': 2,
                'patience': 10,  # Patience rất ngắn cho training nhanh
                'save_period': 10,  # Lưu mỗi 10 epochs
                'lr0': 0.002,  # Learning rate cao hơn để converge nhanh
                'lrf': 0.05,   # Final lr cao hơn
                'momentum': 0.9,
                'weight_decay': 0.0001,  # Giảm weight decay
                'warmup_epochs': 3,  # Warmup rất ngắn
                'cos_lr': True,
                'mixup': 0.05,  # Giảm augmentation tối đa
                'mosaic': 0.5,
                'copy_paste': 0.0,  # Tắt copy_paste
                'flipud': 0.0,  # Tắt flip vertical
                'degrees': 0.0,  # Tắt rotation
                'translate': 0.05,  # Giảm translation
                'scale': 0.25,  # Giảm scaling
                'shear': 0.0,  # Tắt shearing
                'perspective': 0.0,  # Tắt perspective
                'hsv_h': 0.01,  # Giảm color augmentation
                'hsv_s': 0.3,
                'hsv_v': 0.2
            },
            'YOLOv5': {
                'model_name': 'yolov5nu.pt',
                'project_name': 'sleepy_pose_v5n_30ep',
                'epochs': 30,
                'batch': 8,
                'imgsz': 640,
                'device': 'cpu',
                'workers': 2,
                'patience': 10,
                'save_period': 10,
                'lr0': 0.002,
                'lrf': 0.05,
                'momentum': 0.9,
                'weight_decay': 0.0001,
                'warmup_epochs': 3,
                'cos_lr': True,
                'mixup': 0.05,
                'mosaic': 0.5,
                'copy_paste': 0.0,
                'flipud': 0.0,
                'degrees': 0.0,
                'translate': 0.05,
                'scale': 0.25,
                'shear': 0.0,
                'perspective': 0.0,
                'hsv_h': 0.01,
                'hsv_s': 0.3,
                'hsv_v': 0.2
            }
        }
        
        self.dataset_path = self.base_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
        self.training_results = {}
        
    def setup_logging(self):
        """Thiết lập logging cho training session"""
        log_file = self.base_dir / "tools" / "training_30_epochs.log"
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
            "model_backups_30ep",
            "training_results_30ep"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
    def train_yolov8_30_epochs(self):
        """Training YOLOv8 với 30 epochs siêu nhanh"""
        try:
            model_name = "YOLOv8"
            config = self.models_config[model_name]
            
            self.logger.info(f"🚀 Bắt đầu SUPER FAST training {model_name} với {config['epochs']} epochs...")
            
            # Kiểm tra dataset trước khi training
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            # Load model
            model = YOLO(config['model_name'])
            self.logger.info(f"✅ Loaded {config['model_name']}")
            
            # Tối ưu parameters cho training siêu nhanh
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
                'copy_paste': config['copy_paste'],
                'flipud': config['flipud'],
                'degrees': config['degrees'],
                'translate': config['translate'],
                'scale': config['scale'],
                'shear': config['shear'],
                'perspective': config['perspective'],
                'hsv_h': config['hsv_h'],
                'hsv_s': config['hsv_s'],
                'hsv_v': config['hsv_v']
            }
            
            self.logger.info(f"⚡ SUPER FAST training parameters: {train_kwargs}")
            
            # Bắt đầu training
            start_time = time.time()
            self.logger.info("⏳ Starting SUPER FAST model training...")
            results = model.train(**train_kwargs)
            training_time = time.time() - start_time
            
            # Lưu kết quả
            self.training_results[model_name] = {
                'status': 'completed',
                'training_time_hours': training_time / 3600,
                'training_time_minutes': training_time / 60,
                'epochs_completed': config['epochs'],
                'model_path': str(self.base_dir / "runs" / "pose-train" / config['project_name'] / "weights" / "best.pt")
            }
            
            self.logger.info(f"✅ {model_name} SUPER FAST training completed in {training_time/60:.1f} minutes!")
            
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
    
    def train_yolov5_30_epochs(self):
        """Training YOLOv5 với 30 epochs siêu nhanh"""
        try:
            model_name = "YOLOv5"
            config = self.models_config[model_name]
            
            self.logger.info(f"🚀 Bắt đầu SUPER FAST training {model_name} với {config['epochs']} epochs...")
            
            # Kiểm tra dataset trước khi training
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
            # Load model
            model = YOLO(config['model_name'])
            self.logger.info(f"✅ Loaded {config['model_name']}")
            
            # Tối ưu parameters cho training siêu nhanh
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
                'copy_paste': config['copy_paste'],
                'flipud': config['flipud'],
                'degrees': config['degrees'],
                'translate': config['translate'],
                'scale': config['scale'],
                'shear': config['shear'],
                'perspective': config['perspective'],
                'hsv_h': config['hsv_h'],
                'hsv_s': config['hsv_s'],
                'hsv_v': config['hsv_v']
            }
            
            self.logger.info(f"⚡ SUPER FAST training parameters: {train_kwargs}")
            
            # Bắt đầu training
            start_time = time.time()
            self.logger.info("⏳ Starting SUPER FAST model training...")
            results = model.train(**train_kwargs)
            training_time = time.time() - start_time
            
            # Lưu kết quả
            self.training_results[model_name] = {
                'status': 'completed',
                'training_time_hours': training_time / 3600,
                'training_time_minutes': training_time / 60,
                'epochs_completed': config['epochs'],
                'model_path': str(self.base_dir / "runs" / "pose-train" / config['project_name'] / "weights" / "best.pt")
            }
            
            self.logger.info(f"✅ {model_name} SUPER FAST training completed in {training_time/60:.1f} minutes!")
            
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
            backup_dir = self.base_dir / "model_backups_30ep"
            backup_path = backup_dir / f"{model_key}_30ep_best_{timestamp}.pt"
            shutil.copy2(source_path, backup_path)
            self.logger.info(f"✅ Model backed up: {backup_path}")
            
            # Copy to main directory với tên cố định
            main_model_path = self.base_dir / f"{model_key}_30ep_best.pt"
            shutil.copy2(source_path, main_model_path)
            self.logger.info(f"✅ Model copied to: {main_model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed for {model_name}: {str(e)}")
    
    def generate_final_report(self):
        """Tạo báo cáo tổng kết training siêu nhanh"""
        try:
            self.logger.info("📊 Generating SUPER FAST training report...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.base_dir / "training_results_30ep" / f"super_fast_training_report_{timestamp}.json"
            
            report_data = {
                'training_session': {
                    'timestamp': timestamp,
                    'type': '30_epochs_super_fast_training',
                    'models_trained': list(self.training_results.keys())
                },
                'results': self.training_results,
                'summary': {
                    'total_models': len(self.training_results),
                    'successful_trainings': len([r for r in self.training_results.values() if r.get('status') == 'completed']),
                    'total_time_hours': sum([r.get('training_time_hours', 0) for r in self.training_results.values() if 'training_time_hours' in r]),
                    'total_time_minutes': sum([r.get('training_time_minutes', 0) for r in self.training_results.values() if 'training_time_minutes' in r])
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("================================================================================")
            self.logger.info("⚡ SUPER FAST TRAINING SESSION COMPLETED!")
            self.logger.info("================================================================================")
            
            # In kết quả summary
            for model_name, results in self.training_results.items():
                if results.get('status') == 'completed':
                    self.logger.info(f"\n⚡ {model_name}:")
                    self.logger.info(f"   ⏱️  Training time: {results['training_time_minutes']:.1f} minutes ({results['training_time_hours']:.2f} hours)")
                    self.logger.info(f"   🎯 Epochs completed: {results['epochs_completed']}")
            
            total_minutes = sum([r.get('training_time_minutes', 0) for r in self.training_results.values() if 'training_time_minutes' in r])
            total_hours = sum([r.get('training_time_hours', 0) for r in self.training_results.values() if 'training_time_hours' in r])
            self.logger.info(f"\n⏱️  Total training time: {total_minutes:.1f} minutes ({total_hours:.2f} hours)")
            self.logger.info(f"📄 Report saved: {report_path}")
            self.logger.info("================================================================================")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {str(e)}")
            return None

def main():
    """Hàm chính chạy SUPER FAST training cho YOLOv5 và YOLOv8"""
    trainer = SuperFastTrainer()
    
    models_to_train = ['YOLOv8', 'YOLOv5']
    successful_trainings = []
    
    trainer.logger.info("⚡ SUPER FAST TRAINING SESSION STARTED!")
    trainer.logger.info("================================================================================")
    trainer.logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.logger.info(f"🎯 Models to train: {models_to_train}")
    trainer.logger.info(f"⚡ Epochs per model: 30 (SUPER FAST training)")
    trainer.logger.info(f"🏃 Expected total time: ~30-60 minutes")
    trainer.logger.info("================================================================================")
    
    for i, model in enumerate(models_to_train, 1):
        try:
            trainer.logger.info(f"\n🔥 Starting SUPER FAST training {i}/{len(models_to_train)}: {model}")
            trainer.logger.info("============================================================")
            
            if model == 'YOLOv8':
                success = trainer.train_yolov8_30_epochs()
            elif model == 'YOLOv5':
                success = trainer.train_yolov5_30_epochs()
            else:
                trainer.logger.error(f"❌ Unknown model: {model}")
                continue
                
            if success:
                successful_trainings.append(model)
                trainer.logger.info(f"✅ {model} SUPER FAST training completed successfully!")
            else:
                trainer.logger.error(f"❌ {model} training failed!")
                
        except KeyboardInterrupt:
            trainer.logger.info(f"⚠️  Training interrupted during {model}")
            break
        except Exception as e:
            trainer.logger.error(f"❌ Unexpected error during {model}: {str(e)}")
    
    # Tạo báo cáo cuối
    report_path = trainer.generate_final_report()
    
    trainer.logger.info("📋 FINAL SUPER FAST TRAINING SUMMARY")
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
    trainer.logger.info("⚡ SUPER FAST TRAINING SESSION FINISHED!")

if __name__ == "__main__":
    main()