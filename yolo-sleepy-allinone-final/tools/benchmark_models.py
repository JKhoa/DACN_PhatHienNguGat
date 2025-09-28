#!/usr/bin/env python3
"""
Benchmark Script để so sánh performance của 3 YOLO models
YOLOv11 (1000ep), YOLOv8 (59ep), YOLOv5 (base)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

class ModelBenchmark:
    """Class để benchmark và so sánh performance của các models"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.setup_logging()
        
        # Cấu hình models để test
        self.models_config = {
            'YOLOv11_1000ep': {
                'model_path': self.base_dir / 'yolov11_1000ep_best.pt',
                'description': 'YOLOv11 trained with 1000 epochs',
                'training_epochs': 1000,
                'expected_performance': 'Excellent'
            },
            'YOLOv8_59ep': {
                'model_path': self.base_dir / 'runs' / 'pose-train' / 'sleepy_pose_v8n_100ep2' / 'weights' / 'best.pt',
                'description': 'YOLOv8 trained with 59 epochs', 
                'training_epochs': 59,
                'expected_performance': 'Good'
            },
            'YOLOv5_base': {
                'model_path': self.base_dir / 'yolov5nu.pt',
                'description': 'YOLOv5 base pre-trained model',
                'training_epochs': 0,
                'expected_performance': 'Basic'
            }
        }
        
        self.dataset_path = self.base_dir / "datasets" / "sleepy_pose" / "sleepy.yaml"
        self.results = {}
        
    def setup_logging(self):
        """Thiết lập logging cho benchmark session"""
        log_file = self.base_dir / "tools" / "model_benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_models_availability(self):
        """Kiểm tra tính khả dụng của các models"""
        self.logger.info("🔍 Checking models availability...")
        
        available_models = {}
        for model_name, config in self.models_config.items():
            model_path = config['model_path']
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                available_models[model_name] = {
                    'path': str(model_path),
                    'size_mb': round(file_size, 1),
                    'description': config['description']
                }
                self.logger.info(f"   ✅ {model_name}: {file_size:.1f}MB - {config['description']}")
            else:
                self.logger.warning(f"   ❌ {model_name}: Model not found at {model_path}")
        
        return available_models
    
    def validate_model(self, model_name, model_path):
        """Validate model trên validation dataset"""
        try:
            self.logger.info(f"🧪 Validating {model_name}...")
            
            # Load model
            model = YOLO(str(model_path))
            
            # Chạy validation
            start_time = time.time()
            results = model.val(
                data=str(self.dataset_path),
                split='val',
                imgsz=640,
                batch=8,
                device='cpu',
                verbose=False,
                plots=False
            )
            validation_time = time.time() - start_time
            
            # Lấy metrics
            metrics = {
                'validation_time_seconds': round(validation_time, 2),
                'box_map50': round(results.box.map50, 4) if hasattr(results.box, 'map50') else 0,
                'box_map50_95': round(results.box.map, 4) if hasattr(results.box, 'map') else 0,
                'pose_map50': round(results.pose.map50, 4) if hasattr(results.pose, 'map50') else 0,
                'pose_map50_95': round(results.pose.map, 4) if hasattr(results.pose, 'map') else 0,
                'box_precision': round(results.box.mp, 4) if hasattr(results.box, 'mp') else 0,
                'box_recall': round(results.box.mr, 4) if hasattr(results.box, 'mr') else 0,
                'pose_precision': round(results.pose.mp, 4) if hasattr(results.pose, 'mp') else 0,
                'pose_recall': round(results.pose.mr, 4) if hasattr(results.pose, 'mr') else 0
            }
            
            self.logger.info(f"   ✅ {model_name} validation completed in {validation_time:.1f}s")
            self.logger.info(f"      📊 Box mAP@50: {metrics['box_map50']:.3f}")
            self.logger.info(f"      📊 Pose mAP@50: {metrics['pose_map50']:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"   ❌ Error validating {model_name}: {str(e)}")
            return None
    
    def benchmark_inference_speed(self, model_name, model_path):
        """Benchmark tốc độ inference của model"""
        try:
            self.logger.info(f"⚡ Benchmarking inference speed for {model_name}...")
            
            # Load model
            model = YOLO(str(model_path))
            
            # Test trên một số ảnh validation
            val_images_dir = self.base_dir / "datasets" / "sleepy_pose" / "val" / "images"
            test_images = list(val_images_dir.glob("*.jpg"))[:10]  # Test 10 ảnh đầu
            
            if not test_images:
                self.logger.warning(f"   ⚠️ No test images found in {val_images_dir}")
                return None
            
            # Warm up
            for _ in range(3):
                model.predict(str(test_images[0]), verbose=False)
            
            # Benchmark inference
            start_time = time.time()
            for img_path in test_images:
                results = model.predict(str(img_path), verbose=False)
            total_time = time.time() - start_time
            
            avg_inference_time = total_time / len(test_images)
            fps = 1 / avg_inference_time
            
            speed_metrics = {
                'total_inference_time': round(total_time, 3),
                'avg_inference_time_ms': round(avg_inference_time * 1000, 2),
                'fps': round(fps, 1),
                'images_tested': len(test_images)
            }
            
            self.logger.info(f"   ⚡ {model_name} inference speed:")
            self.logger.info(f"      🏃 Average: {speed_metrics['avg_inference_time_ms']:.2f}ms per image")
            self.logger.info(f"      📈 FPS: {speed_metrics['fps']:.1f}")
            
            return speed_metrics
            
        except Exception as e:
            self.logger.error(f"   ❌ Error benchmarking {model_name}: {str(e)}")
            return None
    
    def run_comprehensive_benchmark(self):
        """Chạy benchmark tổng hợp cho tất cả models"""
        self.logger.info("🚀 STARTING COMPREHENSIVE MODEL BENCHMARK")
        self.logger.info("=" * 80)
        
        # Kiểm tra models có sẵn
        available_models = self.check_models_availability()
        
        if not available_models:
            self.logger.error("❌ No models available for benchmarking!")
            return
        
        # Benchmark từng model
        for model_name, model_info in available_models.items():
            self.logger.info(f"\n🔥 Benchmarking {model_name}")
            self.logger.info("-" * 60)
            
            config = self.models_config[model_name]
            model_path = Path(model_info['path'])
            
            # Validation metrics
            validation_results = self.validate_model(model_name, model_path)
            
            # Inference speed
            speed_results = self.benchmark_inference_speed(model_name, model_path)
            
            # Combine results
            self.results[model_name] = {
                'model_info': {
                    'description': config['description'],
                    'training_epochs': config['training_epochs'],
                    'file_size_mb': model_info['size_mb'],
                    'model_path': model_info['path']
                },
                'validation_metrics': validation_results,
                'speed_metrics': speed_results,
                'benchmark_timestamp': datetime.now().isoformat()
            }
        
        # Tạo báo cáo so sánh
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Tạo báo cáo so sánh chi tiết"""
        try:
            self.logger.info("\n📊 GENERATING COMPARISON REPORT")
            self.logger.info("=" * 80)
            
            # Lưu raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = self.base_dir / "benchmark_results"
            report_dir.mkdir(exist_ok=True)
            
            report_path = report_dir / f"model_benchmark_{timestamp}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            # Tạo comparison table
            self.logger.info("\n🏆 MODEL PERFORMANCE COMPARISON")
            self.logger.info("=" * 80)
            
            # Header
            self.logger.info(f"{'Model':<20} {'Epochs':<8} {'Size(MB)':<10} {'Box mAP@50':<12} {'Pose mAP@50':<12} {'FPS':<8}")
            self.logger.info("-" * 80)
            
            # Sắp xếp theo Box mAP@50 (descending)
            sorted_models = sorted(
                self.results.items(),
                key=lambda x: x[1]['validation_metrics']['box_map50'] if x[1]['validation_metrics'] else 0,
                reverse=True
            )
            
            for i, (model_name, data) in enumerate(sorted_models):
                epochs = data['model_info']['training_epochs']
                size = data['model_info']['file_size_mb']
                
                if data['validation_metrics']:
                    box_map = data['validation_metrics']['box_map50']
                    pose_map = data['validation_metrics']['pose_map50']
                else:
                    box_map = pose_map = 0
                
                if data['speed_metrics']:
                    fps = data['speed_metrics']['fps']
                else:
                    fps = 0
                
                # Thêm emoji ranking
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "📊"
                
                self.logger.info(f"{rank_emoji} {model_name:<18} {epochs:<8} {size:<10} {box_map:<12.3f} {pose_map:<12.3f} {fps:<8.1f}")
            
            # Recommendations
            self.logger.info("\n💡 RECOMMENDATIONS")
            self.logger.info("=" * 80)
            
            best_accuracy = max(self.results.items(), key=lambda x: x[1]['validation_metrics']['box_map50'] if x[1]['validation_metrics'] else 0)
            best_speed = max(self.results.items(), key=lambda x: x[1]['speed_metrics']['fps'] if x[1]['speed_metrics'] else 0)
            smallest_size = min(self.results.items(), key=lambda x: x[1]['model_info']['file_size_mb'])
            
            self.logger.info(f"🏆 Best Accuracy: {best_accuracy[0]} (Box mAP@50: {best_accuracy[1]['validation_metrics']['box_map50']:.3f})")
            self.logger.info(f"⚡ Best Speed: {best_speed[0]} ({best_speed[1]['speed_metrics']['fps']:.1f} FPS)")
            self.logger.info(f"💾 Smallest Size: {smallest_size[0]} ({smallest_size[1]['model_info']['file_size_mb']:.1f}MB)")
            
            self.logger.info(f"\n🎯 FOR PRODUCTION:")
            production_model = best_accuracy[0]
            self.logger.info(f"   Recommended: {production_model}")
            self.logger.info(f"   Reason: Highest accuracy with good balance of size and speed")
            
            self.logger.info(f"\n📄 Complete benchmark report saved: {report_path}")
            self.logger.info("🎉 BENCHMARK COMPLETED!")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {str(e)}")
            return None

def main():
    """Hàm chính chạy benchmark"""
    benchmark = ModelBenchmark()
    
    benchmark.logger.info("🎯 MODEL BENCHMARK SESSION STARTED")
    benchmark.logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    benchmark.logger.info(f"🎯 Models to benchmark: YOLOv11 (1000ep), YOLOv8 (59ep), YOLOv5 (base)")
    
    # Chạy benchmark tổng hợp
    benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    main()