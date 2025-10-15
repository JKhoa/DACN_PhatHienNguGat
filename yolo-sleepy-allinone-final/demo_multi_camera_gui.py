#!/usr/bin/env python3
"""
Quick Demo: Multi-Camera GUI Test
Test the multi-camera feature with mock cameras
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from multi_camera_gui import MultiCameraWidget
    from camera_core import CameraConfig, CameraStream
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("pip install PyQt5 opencv-python ultralytics pyyaml")
    sys.exit(1)


class DemoWindow(QMainWindow):
    """Demo window for multi-camera widget"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Demo - Quick Test")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create multi-camera widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Check for YOLO model
        model_paths = [
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
            "yolov11n-pose.pt",
            "best.pt"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("⚠️  No YOLO model found!")
            print("Looking for: yolo11n-pose.pt, yolo11s-pose.pt, or best.pt")
            print("Downloading default model...")
            from ultralytics import YOLO
            model = YOLO("yolo11n-pose.pt")  # This will download if not exists
            model_path = "yolo11n-pose.pt"
        
        print(f"✅ Using model: {model_path}")
        
        # Create widget
        self.multi_cam = MultiCameraWidget(model_path=model_path, parent=self)
        layout.addWidget(self.multi_cam)
        
        # Add demo cameras
        self._add_demo_cameras()
        
        print("\n" + "="*60)
        print("🎬 MULTI-CAMERA DEMO")
        print("="*60)
        print("\n📹 Demo cameras added:")
        print("   1. Webcam 0 (if available)")
        print("   2. Webcam 1 (if available)")
        print("\n🎮 Controls:")
        print("   • Click '▶️ Start All' to begin")
        print("   • Switch between 'Grid View' and 'Single View'")
        print("   • Click '➕ Add Camera' to add IP cameras")
        print("   • Click '💾 Save Config' to save setup")
        print("\n💡 Tips:")
        print("   • If no webcam detected, add IP camera manually")
        print("   • Test IP camera with 'Test Connection' before adding")
        print("   • Use Grid View to monitor multiple cameras")
        print("   • Use Single View to focus on one camera")
        print("\n" + "="*60 + "\n")
    
    def _add_demo_cameras(self):
        """Add demo cameras for testing"""
        # Try to add 2 webcams
        demo_configs = [
            CameraConfig(
                name="Webcam 0 - Default",
                type="webcam",
                source=0,
                enabled=True
            ),
            CameraConfig(
                name="Webcam 1 - Secondary",
                type="webcam",
                source=1,
                enabled=True
            )
        ]
        
        for config in demo_configs:
            stream = CameraStream(config=config)
            self.multi_cam.streams.append(stream)
        
        self.multi_cam.update_camera_list()
        
        print(f"\n✅ Added {len(demo_configs)} demo cameras")
        print("   (Note: Only available webcams will work)")
    
    def closeEvent(self, event):
        """Clean shutdown"""
        print("\n🛑 Stopping all cameras...")
        self.multi_cam.stop_all()
        event.accept()
        print("✅ Demo closed cleanly")


def main():
    """Run demo"""
    print("🚀 Starting Multi-Camera Demo...")
    
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    # Create and show window
    window = DemoWindow()
    window.show()
    
    # Run
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
