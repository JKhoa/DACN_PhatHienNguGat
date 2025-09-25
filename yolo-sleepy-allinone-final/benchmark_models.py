#!/usr/bin/env python3
"""
Benchmark script để so sánh hiệu suất YOLOv5 vs v8 vs v11
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
import psutil
import os
from pathlib import Path

class YOLOBenchmark:
    def __init__(self):
        self.results = {}
        
    def create_test_images(self, num_images=10):
        """Tạo tập ảnh test"""
        print(f"📸 Tạo {num_images} ảnh test...")
        images = []
        
        for i in range(num_images):
            # Tạo ảnh 640x640 với noise và hình người
            img = np.random.randint(50, 150, (640, 640, 3), dtype=np.uint8)
            
            # Thêm hình người đơn giản ở vị trí ngẫu nhiên
            x_offset = np.random.randint(100, 400)
            y_offset = np.random.randint(50, 200)
            
            # Đầu
            cv2.circle(img, (x_offset, y_offset), 25, (255, 200, 200), -1)
            
            # Thân
            cv2.rectangle(img, (x_offset-20, y_offset+25), (x_offset+20, y_offset+150), (200, 255, 200), -1)
            
            # Tay
            cv2.rectangle(img, (x_offset-40, y_offset+40), (x_offset-20, y_offset+100), (200, 200, 255), -1)
            cv2.rectangle(img, (x_offset+20, y_offset+40), (x_offset+40, y_offset+100), (200, 200, 255), -1)
            
            images.append(img)
        
        return images
    
    def benchmark_model(self, model_name, model_path, test_images):
        """Benchmark một model cụ thể"""
        print(f"\n🧪 Benchmarking {model_name}...")
        
        try:
            # Load model
            start_load = time.time()
            model = YOLO(model_path)
            load_time = time.time() - start_load
            
            print(f"⏱️  Load time: {load_time:.3f}s")
            
            # Warm up
            warmup_img = test_images[0]
            model(warmup_img, verbose=False)
            
            # Benchmark inference
            inference_times = []
            total_detections = 0
            
            # Memory usage trước khi test
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            for img in test_images:
                start_inf = time.time()
                results = model(img, verbose=False)
                inf_time = time.time() - start_inf
                
                inference_times.append(inf_time)
                
                # Đếm số detections
                if results and len(results) > 0:
                    r = results[0]
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        total_detections += len(r.boxes)
            
            # Memory usage sau test
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_usage = mem_after - mem_before
            
            # Tính toán metrics
            avg_inference = np.mean(inference_times)
            fps = 1.0 / avg_inference
            total_time = sum(inference_times)
            
            results = {
                'model_name': model_name,
                'load_time': load_time,
                'avg_inference_time': avg_inference,
                'fps': fps,
                'total_time': total_time,
                'total_detections': total_detections,
                'memory_usage_mb': mem_usage,
                'min_inference': min(inference_times),
                'max_inference': max(inference_times)
            }
            
            print(f"📊 Results for {model_name}:")
            print(f"   Load time: {load_time:.3f}s")
            print(f"   Avg inference: {avg_inference:.3f}s")
            print(f"   FPS: {fps:.1f}")
            print(f"   Total detections: {total_detections}")
            print(f"   Memory usage: {mem_usage:.1f} MB")
            
            return results
            
        except Exception as e:
            print(f"❌ Lỗi với {model_name}: {e}")
            return None
    
    def run_benchmark(self):
        """Chạy benchmark cho tất cả models"""
        print("🚀 YOLO Model Benchmark")
        print("=" * 50)
        
        # Tạo test images
        test_images = self.create_test_images(20)
        
        # Định nghĩa models để test
        models_to_test = [
            ("YOLOv5n-pose", "yolov5nu.pt"),
            ("YOLOv8n-pose", "yolov8n-pose.pt"),  
            ("YOLOv11n-pose", "yolo11n-pose.pt"),
        ]
        
        all_results = []
        
        for model_name, model_path in models_to_test:
            result = self.benchmark_model(model_name, model_path, test_images)
            if result:
                all_results.append(result)
        
        # So sánh kết quả
        self.compare_results(all_results)
    
    def compare_results(self, results):
        """So sánh kết quả các models"""
        print("\n📈 COMPARISON RESULTS")
        print("=" * 60)
        
        if not results:
            print("❌ Không có kết quả để so sánh")
            return
        
        # Tìm model tốt nhất cho từng metric
        best_fps = max(results, key=lambda x: x['fps'])
        best_load = min(results, key=lambda x: x['load_time'])
        best_memory = min(results, key=lambda x: x['memory_usage_mb'])
        
        # In bảng so sánh
        print(f"{'Model':<15} {'FPS':<8} {'Load(s)':<8} {'Memory(MB)':<10} {'Detections':<12}")
        print("-" * 60)
        
        for result in results:
            name = result['model_name']
            fps = result['fps']
            load_time = result['load_time']
            memory = result['memory_usage_mb']
            detections = result['total_detections']
            
            print(f"{name:<15} {fps:<8.1f} {load_time:<8.3f} {memory:<10.1f} {detections:<12}")
        
        print("\n🏆 WINNERS:")
        print(f"   Fastest FPS: {best_fps['model_name']} ({best_fps['fps']:.1f} FPS)")
        print(f"   Fastest Load: {best_load['model_name']} ({best_load['load_time']:.3f}s)")
        print(f"   Least Memory: {best_memory['model_name']} ({best_memory['memory_usage_mb']:.1f} MB)")
        
        # Recommendation
        print("\n💡 RECOMMENDATIONS:")
        if best_fps['model_name'] == best_memory['model_name']:
            print(f"   🥇 Overall Winner: {best_fps['model_name']} (Best FPS + Memory)")
        else:
            print(f"   ⚡ For Speed: {best_fps['model_name']}")
            print(f"   💾 For Memory: {best_memory['model_name']}")


def main():
    """Main function"""
    benchmark = YOLOBenchmark()
    benchmark.run_benchmark()
    
    print("\n✨ Benchmark completed!")
    print("💡 Bạn có thể chọn model phù hợp dựa trên kết quả trên")


if __name__ == "__main__":
    main()