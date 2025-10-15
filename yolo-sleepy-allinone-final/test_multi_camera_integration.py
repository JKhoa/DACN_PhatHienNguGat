#!/usr/bin/env python3
"""
Quick Test: Verify Multi-Camera Integration
Run this to check if everything is installed correctly
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    tests = []
    
    # Core dependencies
    try:
        import cv2
        tests.append(("✅", "OpenCV", cv2.__version__))
    except ImportError:
        tests.append(("❌", "OpenCV", "NOT INSTALLED"))
    
    try:
        import numpy
        tests.append(("✅", "NumPy", numpy.__version__))
    except ImportError:
        tests.append(("❌", "NumPy", "NOT INSTALLED"))
    
    try:
        from ultralytics import YOLO
        tests.append(("✅", "Ultralytics YOLO", "OK"))
    except ImportError:
        tests.append(("❌", "Ultralytics YOLO", "NOT INSTALLED"))
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5 import QtCore
        tests.append(("✅", "PyQt5", QtCore.PYQT_VERSION_STR))
    except ImportError:
        tests.append(("❌", "PyQt5", "NOT INSTALLED"))
    
    try:
        import yaml
        tests.append(("✅", "PyYAML", yaml.__version__))
    except ImportError:
        tests.append(("❌", "PyYAML", "NOT INSTALLED"))
    
    # Project files
    try:
        from camera_core import CameraConfig, CameraStream
        tests.append(("✅", "camera_core.py", "OK"))
    except ImportError as e:
        tests.append(("❌", "camera_core.py", f"ERROR: {e}"))
    
    try:
        from multi_camera_gui import MultiCameraWidget
        tests.append(("✅", "multi_camera_gui.py", "OK"))
    except ImportError as e:
        tests.append(("❌", "multi_camera_gui.py", f"ERROR: {e}"))
    
    # Print results
    print("\n" + "="*60)
    print("IMPORT TEST RESULTS")
    print("="*60)
    for status, name, version in tests:
        print(f"{status} {name:25} {version}")
    print("="*60)
    
    # Check if all passed
    failed = [t for t in tests if t[0] == "❌"]
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed!")
        print("\nTo fix, run:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All tests passed!")
        return True


def test_files():
    """Test if required files exist"""
    print("\n🔍 Testing files...")
    
    required_files = [
        "camera_core.py",
        "multi_camera_gui.py",
        "gui_app.py",
        "requirements.txt",
        "cameras.sample.yaml"
    ]
    
    optional_files = [
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "cameras.yaml"
    ]
    
    print("\n" + "="*60)
    print("FILE CHECK")
    print("="*60)
    
    all_good = True
    
    print("\nRequired files:")
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} - MISSING!")
            all_good = False
    
    print("\nOptional files:")
    for f in optional_files:
        if os.path.exists(f):
            print(f"  ✅ {f}")
        else:
            print(f"  ⚠️  {f} - Not found (will download if needed)")
    
    print("="*60)
    
    return all_good


def test_gui():
    """Test if GUI can be imported"""
    print("\n🔍 Testing GUI integration...")
    
    try:
        import gui_app
        
        # Check if multi-camera is in gui_app
        source = open("gui_app.py", "r", encoding="utf-8").read()
        
        checks = [
            ("multi_camera_gui import", "from multi_camera_gui import" in source),
            ("MultiCameraWidget", "MultiCameraWidget" in source),
            ("Multi-Camera tab", "Multi-Camera" in source),
            ("HAS_MULTI_CAMERA", "HAS_MULTI_CAMERA" in source)
        ]
        
        print("\n" + "="*60)
        print("GUI INTEGRATION CHECK")
        print("="*60)
        
        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {name}")
        
        print("="*60)
        
        if all(c[1] for c in checks):
            print("\n✅ GUI integration verified!")
            return True
        else:
            print("\n❌ GUI integration incomplete!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error checking GUI: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 MULTI-CAMERA INTEGRATION TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test files
    results.append(("Files", test_files()))
    
    # Test GUI integration
    results.append(("GUI Integration", test_gui()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20} {status}")
    
    print("="*60)
    
    if all(r[1] for r in results):
        print("\n🎉 All tests passed! Ready to use multi-camera!")
        print("\nTo start:")
        print("  python gui_app.py")
        print("  → Click '📹 Multi-Camera' tab")
        print("  → Click '➕ Add Camera'")
        print("  → Click '▶️ Start All'")
        print("\nOr run demo:")
        print("  python demo_multi_camera_gui.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
