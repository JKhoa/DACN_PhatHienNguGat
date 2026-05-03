"""
Check trained model information (classes, keypoints, architecture)
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ultralytics import YOLO
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'sleepy_pose_v11n_full_best.pt')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    # Model names/classes
    if hasattr(model, 'names'):
        print(f"\n🏷️  Classes: {model.names}")
        print(f"   Total classes: {len(model.names)}")
    
    # Model task
    if hasattr(model, 'task'):
        print(f"\n🎯 Task: {model.task}")
    
    # Model info
    print(f"\n📊 Model architecture:")
    print(f"   Type: {type(model.model).__name__}")
    
    # Try to get keypoint info for pose models
    if hasattr(model.model, 'yaml'):
        yaml_info = model.model.yaml
        if 'kpt_shape' in yaml_info:
            print(f"   Keypoint shape: {yaml_info['kpt_shape']}")
    
    # Model size
    try:
        import torch
        params = sum(p.numel() for p in model.model.parameters())
        print(f"   Parameters: {params:,}")
    except:
        pass
    
    print("\n" + "="*60)
    print("✅ Model loaded successfully!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
