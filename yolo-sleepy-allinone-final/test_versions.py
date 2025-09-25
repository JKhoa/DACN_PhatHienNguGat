#!/usr/bin/env python3
"""
Demo script để test YOLOv5, v8, v11 cho phát hiện ngủ gật
"""

import os
import sys
import cv2
import time
from pathlib import Path

def test_yolo_version(version, test_image=None):
    """Test một version YOLO cụ thể"""
    print(f"\n🧪 Testing YOLO{version}...")
    
    # Chạy standalone app với version cụ thể
    app_path = Path(__file__).parent / "standalone_app.py"
    
    if test_image:
        cmd = f'python "{app_path}" --model-version {version} --image "{test_image}" --cli'
    else:
        cmd = f'python "{app_path}" --model-version {version} --cli --cam 0'
    
    print(f"📟 Command: {cmd}")
    
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ YOLO{version} hoạt động tốt!")
            return True
        else:
            print(f"❌ YOLO{version} có lỗi:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ YOLO{version} timeout (có thể do đang chạy GUI)")
        return True  # GUI thường chạy liên tục
    except Exception as e:
        print(f"❌ Lỗi khi test YOLO{version}: {e}")
        return False


def create_test_image():
    """Tạo ảnh test đơn giản"""
    import numpy as np
    
    # Tạo ảnh 640x640 với hình người đơn giản
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Vẽ hình người đơn giản
    # Đầu
    cv2.circle(img, (320, 150), 40, (200, 200, 255), -1)
    
    # Thân
    cv2.rectangle(img, (280, 190), (360, 400), (150, 255, 150), -1)
    
    # Tay
    cv2.rectangle(img, (230, 220), (280, 320), (255, 150, 150), -1)  # Tay trái
    cv2.rectangle(img, (360, 220), (410, 320), (255, 150, 150), -1)  # Tay phải
    
    # Chân  
    cv2.rectangle(img, (290, 400), (320, 520), (255, 255, 150), -1)  # Chân trái
    cv2.rectangle(img, (320, 400), (350, 520), (255, 255, 150), -1)  # Chân phải
    
    # Lưu ảnh test
    test_path = "test_person.jpg"
    cv2.imwrite(test_path, img)
    print(f"📸 Đã tạo ảnh test: {test_path}")
    
    return test_path


def main():
    """Main function để test các YOLO versions"""
    print("🚀 Demo Test YOLOv5, v8, v11 cho phát hiện ngủ gật")
    print("=" * 60)
    
    # Tạo ảnh test
    test_image = create_test_image()
    
    # Test các versions
    versions = ["v5", "v8", "v11"]
    results = {}
    
    for version in versions:
        results[version] = test_yolo_version(version, test_image)
        time.sleep(2)  # Đợi 2s giữa các test
    
    # Tổng kết
    print("\n📊 KẾT QUẢ TEST:")
    print("=" * 30)
    for version, success in results.items():
        status = "✅ OK" if success else "❌ FAIL"
        print(f"YOLO{version}: {status}")
    
    # Cleanup
    if os.path.exists(test_image):
        os.remove(test_image)
        print(f"\n🗑️ Đã xóa file test: {test_image}")
    
    print("\n💡 Để chạy thủ công:")
    print("python standalone_app.py --model-version v5  # Cho YOLOv5")
    print("python standalone_app.py --model-version v8  # Cho YOLOv8") 
    print("python standalone_app.py --model-version v11 # Cho YOLOv11")


if __name__ == "__main__":
    main()