#!/usr/bin/env python3
"""
Training Script tối ưu cho YOLOv5 với 50 epochs
Cân bằng hoàn hảo giữa thời gian training và chất lượng model
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

class OptimalYOLOv5Trainer:
    """Class để training YOLOv5 với 50 epochs tối ưu"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.setup_logging()
        self.setup_directories()
        
        # Cấu hình tối ưu cho YOLOv5 với 50 epochs
        self.yolov5_config = {
            'model_name': 'yolov5nu.pt',
            'project_name': 'sleepy_pose_v5n_50ep_optimal',
            'epochs': 50,
            'batch': 8,
            'imgsz': 640,
            'device': 'cpu',
            'workers': 2,
            
            # Tối ưu cho 50 epochs - cân bằng training time và quality
            'patience': 20,  # Patience phù hợp với 50 epochs
            'save_period': 10,  # Lưu mỗi 10 epochs
            'lr0': 0.01,  # Learning rate cao hơn cho training nhanh
            'lrf': 0.1,  # Final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,  # Warmup ngắn cho 50 epochs
            'cos_lr': True,
            
            # Augmentation cân bằng - không quá mạnh để tiết kiệm thời gian
            'mixup': 0.15,
            'mosaic': 1.0,  # Mosaic mạnh để tăng đa dạng data
            'copy_paste': 0.1,
            'flipud': 0.0,  # Không flip up-down cho pose
            'fliplr': 0.5,  # Flip left-right OK cho pose
            'degrees': 10.0,  # Rotation nhẹ
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0001,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            
            # Optimizer tối ưu
            'optimizer': 'AdamW',
            'close_mosaic': 10,  # Tắt mosaic 10 epochs cuối
            
            # Validation
            'val': True,
            'plots': True,
            'save': True,
            'verbose': True,
            'seed': 42
        }
        
        self.dataset_path = self.base_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
        self.training_results = {}
        
    def setup_logging(self):
        """Thiết lập logging cho training session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.base_dir / "tools" / f"yolov5_50ep_training_{timestamp}.log"
        
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
            "model_backups_50ep",
            "training_results_50ep",
            "yolov5_training_logs"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
    def validate_setup(self):
        """Kiểm tra môi trường training"""
        self.logger.info("🔍 Validating training setup...")
        
        # Kiểm tra dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"❌ Dataset not found: {self.dataset_path}")
        self.logger.info(f"✅ Dataset found: {self.dataset_path}")
        
        # Kiểm tra model base
        model_path = self.yolov5_config['model_name']
        if not Path(model_path).exists() and not model_path.endswith('.pt'):
            # Thử tìm trong thư mục base
            base_model = self.base_dir / model_path
            if base_model.exists():
                self.yolov5_config['model_name'] = str(base_model)
                self.logger.info(f"✅ Found base model: {base_model}")
            else:
                self.logger.info(f"🔄 Will download model: {model_path}")
        
        # Kiểm tra GPU/CPU
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self.yolov5_config['device'] = 'cuda'
            self.yolov5_config['batch'] = 16  # Tăng batch size nếu có GPU
            self.logger.info(f"🚀 GPU detected! Using CUDA with batch size {self.yolov5_config['batch']}")
        else:
            self.logger.info(f"💻 Using CPU with batch size {self.yolov5_config['batch']}")
        
        self.logger.info("✅ Setup validation completed!")
        
    def train_yolov5_optimal(self):
        """Training YOLOv5 với cấu hình tối ưu 50 epochs"""
        try:
            model_name = "YOLOv5"
            config = self.yolov5_config
            
            self.logger.info("🚀 STARTING OPTIMAL YOLOv5 TRAINING")
            self.logger.info("=" * 80)
            self.logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"🎯 Model: {config['model_name']}")
            self.logger.info(f"⚡ Epochs: {config['epochs']} (OPTIMAL)")
            self.logger.info(f"🔧 Device: {config['device']}")
            self.logger.info(f"📦 Batch size: {config['batch']}")
            self.logger.info("=" * 80)
            
            # Validate setup trước khi training
            self.validate_setup()
            
            # Load model
            self.logger.info(f"🔄 Loading model: {config['model_name']}")
            model = YOLO(config['model_name'])
            self.logger.info("✅ Model loaded successfully!")
            
            # Tạo training arguments
            train_kwargs = {
                'data': str(self.dataset_path),
                'epochs': config['epochs'],
                'batch': config['batch'],
                'imgsz': config['imgsz'],
                'project': str(self.base_dir / "runs" / "pose-train"),
                'name': config['project_name'],
                'device': config['device'],
                'workers': config['workers'],
                'patience': config['patience'],
                'save_period': config['save_period'],
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
                'fliplr': config['fliplr'],
                'degrees': config['degrees'],
                'translate': config['translate'],
                'scale': config['scale'],
                'shear': config['shear'],
                'perspective': config['perspective'],
                'hsv_h': config['hsv_h'],
                'hsv_s': config['hsv_s'],
                'hsv_v': config['hsv_v'],
                'optimizer': config['optimizer'],
                'close_mosaic': config['close_mosaic'],
                'val': config['val'],
                'plots': config['plots'],
                'save': config['save'],
                'verbose': config['verbose'],
                'seed': config['seed'],
                'exist_ok': True
            }
            
            # Log training configuration
            self.logger.info("🎯 TRAINING CONFIGURATION:")
            self.logger.info("-" * 60)
            for key, value in train_kwargs.items():
                if key not in ['data', 'project']:
                    self.logger.info(f"   {key}: {value}")
            self.logger.info("-" * 60)
            
            # Bắt đầu training
            self.logger.info("⏳ STARTING TRAINING PROCESS...")
            start_time = time.time()
            
            results = model.train(**train_kwargs)
            
            training_time = time.time() - start_time
            
            # Phân tích kết quả training
            project_dir = self.base_dir / "runs" / "pose-train" / config['project_name']
            best_model_path = project_dir / "weights" / "best.pt"
            last_model_path = project_dir / "weights" / "last.pt"
            
            # Đọc metrics từ results.csv nếu có
            results_csv = project_dir / "results.csv"
            final_metrics = {}
            if results_csv.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(results_csv)
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        final_metrics = {
                            'final_epoch': int(last_row.get('epoch', config['epochs'])),
                            'box_loss': float(last_row.get('train/box_loss', 0)),
                            'pose_loss': float(last_row.get('train/pose_loss', 0)),
                            'obj_loss': float(last_row.get('train/obj_loss', 0)),
                            'val_box_loss': float(last_row.get('val/box_loss', 0)),
                            'val_pose_loss': float(last_row.get('val/pose_loss', 0)),
                            'val_obj_loss': float(last_row.get('val/obj_loss', 0)),
                            'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                            'mAP50_95': float(last_row.get('metrics/mAP50-95(B)', 0))
                        }
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not parse results.csv: {e}")
            
            # Lưu kết quả training
            self.training_results[model_name] = {
                'status': 'completed',
                'training_time_hours': round(training_time / 3600, 2),
                'training_time_minutes': round(training_time / 60, 2),
                'epochs_completed': config['epochs'],
                'best_model_path': str(best_model_path) if best_model_path.exists() else None,
                'last_model_path': str(last_model_path) if last_model_path.exists() else None,
                'project_directory': str(project_dir),
                'final_metrics': final_metrics,
                'config_used': config
            }
            
            # Success logging
            self.logger.info("=" * 80)
            self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️  Training time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
            self.logger.info(f"🎯 Epochs completed: {config['epochs']}")
            self.logger.info(f"📁 Project directory: {project_dir}")
            
            if final_metrics:
                self.logger.info("📊 FINAL METRICS:")
                self.logger.info(f"   📦 Box Loss: {final_metrics.get('box_loss', 'N/A')}")
                self.logger.info(f"   🤸 Pose Loss: {final_metrics.get('pose_loss', 'N/A')}")
                self.logger.info(f"   🎯 Object Loss: {final_metrics.get('obj_loss', 'N/A')}")
                self.logger.info(f"   📈 mAP@50: {final_metrics.get('mAP50', 'N/A')}")
                self.logger.info(f"   📈 mAP@50-95: {final_metrics.get('mAP50_95', 'N/A')}")
            
            if best_model_path.exists():
                model_size = best_model_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"💾 Best model: {best_model_path} ({model_size:.1f}MB)")
            
            # Backup models
            self._backup_trained_models(config['project_name'])
            
            self.logger.info("=" * 80)
            return True
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("❌ TRAINING FAILED!")
            self.logger.error("=" * 80)
            self.logger.error(f"💥 Error: {str(e)}")
            
            self.training_results["YOLOv5"] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def _backup_trained_models(self, project_name):
        """Backup trained models với timestamp"""
        try:
            self.logger.info("💾 Creating model backups...")
            
            project_dir = self.base_dir / "runs" / "pose-train" / project_name
            weights_dir = project_dir / "weights"
            
            if not weights_dir.exists():
                self.logger.warning("⚠️ No weights directory found for backup")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.base_dir / "model_backups_50ep"
            
            # Backup best.pt
            best_path = weights_dir / "best.pt"
            if best_path.exists():
                backup_best = backup_dir / f"yolov5_50ep_best_{timestamp}.pt"
                shutil.copy2(best_path, backup_best)
                
                # Copy to main directory
                main_best = self.base_dir / "yolov5_50ep_best.pt"
                shutil.copy2(best_path, main_best)
                
                model_size = best_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"✅ Best model backed up: {backup_best} ({model_size:.1f}MB)")
                self.logger.info(f"✅ Best model copied to: {main_best}")
            
            # Backup last.pt
            last_path = weights_dir / "last.pt"
            if last_path.exists():
                backup_last = backup_dir / f"yolov5_50ep_last_{timestamp}.pt"
                shutil.copy2(last_path, backup_last)
                self.logger.info(f"✅ Last model backed up: {backup_last}")
            
            # Backup training results
            results_csv = project_dir / "results.csv"
            if results_csv.exists():
                backup_results = backup_dir / f"yolov5_50ep_results_{timestamp}.csv"
                shutil.copy2(results_csv, backup_results)
                self.logger.info(f"✅ Training results backed up: {backup_results}")
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {str(e)}")
    
    def generate_training_report(self):
        """Tạo báo cáo chi tiết sau training"""
        try:
            self.logger.info("📊 Generating comprehensive training report...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = self.base_dir / "training_results_50ep"
            report_path = report_dir / f"yolov5_50ep_training_report_{timestamp}.json"
            
            # Thêm system info
            import platform
            import torch
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
            
            report_data = {
                'training_session': {
                    'timestamp': timestamp,
                    'type': 'yolov5_50_epochs_optimal',
                    'start_time': datetime.now().isoformat(),
                    'dataset': str(self.dataset_path),
                    'system_info': system_info
                },
                'training_results': self.training_results,
                'model_comparison': {
                    'yolov5_50ep': self.training_results.get('YOLOv5', {}),
                    'training_efficiency': {
                        'epochs_per_hour': 50 / self.training_results.get('YOLOv5', {}).get('training_time_hours', 1),
                        'minutes_per_epoch': self.training_results.get('YOLOv5', {}).get('training_time_minutes', 0) / 50
                    }
                }
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Generate summary log
            self.logger.info("\n📋 TRAINING SESSION SUMMARY")
            self.logger.info("=" * 80)
            
            if 'YOLOv5' in self.training_results and self.training_results['YOLOv5']['status'] == 'completed':
                results = self.training_results['YOLOv5']
                self.logger.info(f"🎯 Model: YOLOv5 (50 epochs OPTIMAL)")
                self.logger.info(f"⏱️  Training time: {results['training_time_hours']:.2f} hours")
                self.logger.info(f"📈 Training efficiency: {50/results['training_time_hours']:.1f} epochs/hour")
                self.logger.info(f"💾 Best model: {results.get('best_model_path', 'N/A')}")
                
                if results.get('final_metrics'):
                    metrics = results['final_metrics']
                    self.logger.info("📊 Final Performance:")
                    self.logger.info(f"   📦 Box Loss: {metrics.get('box_loss', 'N/A'):.4f}")
                    self.logger.info(f"   🤸 Pose Loss: {metrics.get('pose_loss', 'N/A'):.4f}")
                    self.logger.info(f"   🎯 mAP@50: {metrics.get('mAP50', 'N/A'):.4f}")
            
            self.logger.info(f"\n📄 Complete report saved: {report_path}")
            self.logger.info("🎉 TRAINING SESSION COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {str(e)}")
            return None

def main():
    """Hàm chính để chạy YOLOv5 optimal training"""
    trainer = OptimalYOLOv5Trainer()
    
    trainer.logger.info("🚀 YOLOv5 OPTIMAL TRAINING SESSION STARTED!")
    trainer.logger.info("=" * 80)
    trainer.logger.info(f"📅 Session date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.logger.info(f"🎯 Target: YOLOv5 with 50 epochs (OPTIMAL BALANCE)")
    trainer.logger.info(f"⚖️  Strategy: Balance between training time and model quality")
    trainer.logger.info(f"📊 Expected: High-quality model in reasonable time")
    trainer.logger.info("=" * 80)
    
    try:
        # Chạy training
        success = trainer.train_yolov5_optimal()
        
        if success:
            # Tạo báo cáo
            report_path = trainer.generate_training_report()
            
            trainer.logger.info("🎉 SUCCESS! YOLOv5 training completed optimally!")
            trainer.logger.info("✅ Model ready for comparison with YOLOv8 and YOLOv11")
            
        else:
            trainer.logger.error("❌ Training failed! Check logs for details.")
            
    except KeyboardInterrupt:
        trainer.logger.warning("⚠️ Training interrupted by user!")
    except Exception as e:
        trainer.logger.error(f"❌ Unexpected error: {str(e)}")
    
    trainer.logger.info("👋 YOLOv5 TRAINING SESSION ENDED!")

if __name__ == "__main__":
    main()