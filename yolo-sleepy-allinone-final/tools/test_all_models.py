#!/usr/bin/env python3
"""
Multi-Model Testing and Comparison Script
========================================

This script tests and compares YOLOv5, YOLOv8, and YOLOv11 models for 
multi-person sleepy detection with enhanced display features.

Features:
- Test all trained models (YOLOv5, v8, v11)
- Benchmark performance metrics
- Compare detection accuracy
- Test enhanced display features
- Generate comparison report

Usage:
    python test_all_models.py [--video path/to/video] [--image path/to/image] [--webcam]

Author: AI Assistant  
Date: December 2024
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json

class ModelTester:
    """Test and compare YOLO models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load all available trained models"""
        print("🔄 Loading trained models...")
        
        model_configs = [
            {
                'name': 'YOLOv11n',
                'path': '../yolo11n-pose.pt',
                'version': 'v11'
            },
            {
                'name': 'YOLOv8n-Custom',
                'path': '../yolo8n-pose-sleepy.pt', 
                'version': 'v8'
            },
            {
                'name': 'YOLOv5n-Custom',
                'path': '../yolo5n-pose-sleepy.pt',
                'version': 'v5'
            }
        ]
        
        try:
            from ultralytics import YOLO
            
            for config in model_configs:
                model_path = Path(config['path'])
                if model_path.exists():
                    print(f"✅ Loading {config['name']}...")
                    try:
                        model = YOLO(str(model_path))
                        self.models[config['name']] = {
                            'model': model,
                            'path': config['path'],
                            'version': config['version']
                        }
                        print(f"   Model loaded successfully!")
                    except Exception as e:
                        print(f"   ❌ Failed to load: {e}")
                else:
                    print(f"⚠️  Model not found: {config['path']}")
                    
        except ImportError:
            print("❌ Ultralytics not found. Please install: pip install ultralytics")
            return False
            
        print(f"\n✅ Loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def test_image(self, image_path):
        """Test all models on a single image"""
        print(f"\n🖼️  Testing image: {image_path}")
        
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
            
        results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\n🔄 Testing {model_name}...")
            
            try:
                start_time = time.time()
                
                # Run inference
                predictions = model_info['model'](image, verbose=False)
                
                inference_time = time.time() - start_time
                
                # Process results
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    
                    # Count detections
                    num_persons = len(pred.boxes) if pred.boxes is not None else 0
                    
                    # Get confidence scores
                    confidences = pred.boxes.conf.cpu().numpy() if pred.boxes is not None else []
                    avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0
                    
                    results[model_name] = {
                        'inference_time': inference_time * 1000,  # ms
                        'num_persons': num_persons,
                        'avg_confidence': float(avg_confidence),
                        'max_confidence': float(np.max(confidences)) if len(confidences) > 0 else 0,
                        'min_confidence': float(np.min(confidences)) if len(confidences) > 0 else 0
                    }
                    
                    print(f"   ✅ Detected {num_persons} persons")
                    print(f"   ⏱️  Inference: {inference_time*1000:.2f}ms")
                    print(f"   📊 Avg confidence: {avg_confidence:.3f}")
                else:
                    results[model_name] = {
                        'inference_time': inference_time * 1000,
                        'num_persons': 0,
                        'avg_confidence': 0,
                        'max_confidence': 0,
                        'min_confidence': 0
                    }
                    print(f"   ⚠️  No detections")
                    
            except Exception as e:
                print(f"   ❌ Error testing {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'inference_time': 0,
                    'num_persons': 0,
                    'avg_confidence': 0,
                    'max_confidence': 0,
                    'min_confidence': 0
                }
        
        return results
    
    def test_enhanced_display(self):
        """Test enhanced display features"""
        print("\n🎨 Testing Enhanced Display Features...")
        
        try:
            # Import enhanced display module
            sys.path.append('..')
            from enhanced_display import draw_enhanced_multi_person_display, draw_person_id_circles
            
            # Create test data
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_persons = [
                {
                    'id': 1,
                    'bbox': (100, 100, 200, 300),
                    'pose_data': np.random.rand(17, 3),
                    'sleepy_state': 'awake',
                    'confidence': 0.85,
                    'sleep_duration': 0
                },
                {
                    'id': 2, 
                    'bbox': (300, 100, 400, 300),
                    'pose_data': np.random.rand(17, 3),
                    'sleepy_state': 'sleepy',
                    'confidence': 0.72,
                    'sleep_duration': 120
                }
            ]
            
            # Test enhanced display
            display_frame = draw_enhanced_multi_person_display(
                dummy_frame.copy(), dummy_persons, 0, time.time(), 300, time.time()
            )
            
            # Test person circles
            circle_frame = draw_person_id_circles(
                dummy_frame.copy(), dummy_persons
            )
            
            print("✅ Enhanced display functions working correctly")
            return True
            
        except ImportError as e:
            print(f"⚠️  Enhanced display module not available: {e}")
            return False
        except Exception as e:
            print(f"❌ Error testing enhanced display: {e}")
            return False
    
    def generate_report(self, test_results):
        """Generate comprehensive test report"""
        print("\n📊 Generating Comparison Report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_tested': len(self.models),
            'model_details': {},
            'test_results': test_results,
            'summary': {}
        }
        
        # Model details
        for name, info in self.models.items():
            report['model_details'][name] = {
                'version': info['version'],
                'path': info['path']
            }
        
        # Performance summary
        if test_results:
            for model_name in self.models.keys():
                if model_name in test_results and 'error' not in test_results[model_name]:
                    result = test_results[model_name]
                    report['summary'][model_name] = {
                        'avg_inference_time': result.get('inference_time', 0),
                        'detection_capability': result.get('num_persons', 0) > 0,
                        'confidence_score': result.get('avg_confidence', 0)
                    }
        
        # Save report
        report_path = f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📋 Model Comparison Summary")
        print("=" * 50)
        
        for model_name, info in self.models.items():
            print(f"\n🤖 {model_name} ({info['version']})")
            if model_name in test_results:
                result = test_results[model_name]
                if 'error' not in result:
                    print(f"   ⏱️  Inference: {result['inference_time']:.2f}ms")
                    print(f"   👥 Detections: {result['num_persons']}")
                    print(f"   📊 Confidence: {result['avg_confidence']:.3f}")
                else:
                    print(f"   ❌ Error: {result['error']}")
            else:
                print("   ⚠️  No test results")
        
        print(f"\n✅ Report saved: {report_path}")
        return report_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Model Testing and Comparison')
    parser.add_argument('--image', type=str, help='Test image path')
    parser.add_argument('--video', type=str, help='Test video path')
    parser.add_argument('--webcam', action='store_true', help='Test with webcam')
    parser.add_argument('--report-only', action='store_true', help='Generate report only')
    
    args = parser.parse_args()
    
    print("🎯 Multi-Model Testing and Comparison")
    print("=" * 50)
    
    tester = ModelTester()
    
    # Load models
    if not tester.load_models():
        print("❌ No models loaded. Exiting.")
        return
    
    # Test enhanced display
    tester.test_enhanced_display()
    
    # Run tests
    test_results = None
    
    if not args.report_only:
        if args.image:
            test_results = tester.test_image(args.image)
        elif args.video:
            print("🎥 Video testing not implemented yet")
        elif args.webcam:
            print("📹 Webcam testing not implemented yet")
        else:
            # Use default test image
            test_images = [
                '../../data_raw/cap_000000.jpg',
                '../../data_raw/cap_000001.jpg',
                '../../data_raw/cap_000010.jpg'
            ]
            
            for img_path in test_images:
                if Path(img_path).exists():
                    test_results = tester.test_image(img_path)
                    break
            
            if test_results is None:
                print("⚠️  No test images found. Skipping image tests.")
    
    # Generate report
    report_path = tester.generate_report(test_results or {})
    
    print(f"\n🎉 Testing completed!")
    print(f"📄 Report: {report_path}")

if __name__ == "__main__":
    main()