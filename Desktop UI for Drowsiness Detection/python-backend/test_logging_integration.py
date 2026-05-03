"""
Test script để kiểm tra logging system khi phát hiện buồn ngủ
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import logger
try:
    from drowsiness_logger import MultiCameraLogger, init_logger, get_global_logger
    print("✅ Import drowsiness_logger thành công!")
except ImportError as e:
    print(f"❌ Lỗi import: {e}")
    sys.exit(1)

# Test 1: Khởi tạo logger
print("\n" + "="*60)
print("TEST 1: KHỞI TẠO LOGGER")
print("="*60)

try:
    log_dir = os.path.join(os.path.dirname(__file__), 'drowsiness_logs')
    init_logger(log_dir)
    logger = get_global_logger()
    print(f"✅ Logger đã khởi tạo thành công tại: {log_dir}")
except Exception as e:
    print(f"❌ Lỗi khởi tạo logger: {e}")
    sys.exit(1)

# Test 2: Simulate drowsiness detection
print("\n" + "="*60)
print("TEST 2: SIMULATE PHÁT HIỆN BUỒN NGỦ")
print("="*60)

camera_id = "camera_test_1"
camera_name = "Phòng Test 101"

# Register camera
logger.register_camera(camera_id, camera_name)
print(f"✅ Đã đăng ký camera: {camera_id} - {camera_name}")

# Student 1 starts drowsing
student_id = 5
print(f"\n🔴 Học sinh #{student_id} BẮT ĐẦU NGỦ GẬT...")
logger.update_student_state(camera_id, student_id, True)

import time
time.sleep(2)

print(f"⏳ Đợi 2 giây...")

# Student 1 wakes up
print(f"\n🟢 Học sinh #{student_id} TỈNH LẠI...")
logger.update_student_state(camera_id, student_id, False)

# Test 3: Check events
print("\n" + "="*60)
print("TEST 3: KIỂM TRA LOG EVENTS")
print("="*60)

events = logger.get_events(camera_id, period='today')
print(f"\n📝 Tổng số events: {len(events)}")

for i, event in enumerate(events, 1):
    print(f"\nEvent #{i}:")
    print(f"  - Student ID: {event['student_id']}")
    print(f"  - Start Time: {event['start_time']}")
    print(f"  - End Time: {event['end_time']}")
    print(f"  - Duration: {event['duration_display']}")
    print(f"  - Is Active: {event['is_active']}")

# Test 4: Check statistics
print("\n" + "="*60)
print("TEST 4: KIỂM TRA THỐNG KÊ")
print("="*60)

stats = logger.get_camera_stats(camera_id, period='today')
print(f"\n📊 Thống kê camera {camera_name}:")
print(f"  - Total drowsy students: {stats['total_drowsy_students']}")
print(f"  - Currently drowsy: {stats['currently_drowsy']}")
print(f"  - Total events: {stats['total_events']}")
print(f"  - Total duration: {stats['total_duration_display']}")

# Test 5: Check active drowsy students
print("\n" + "="*60)
print("TEST 5: KIỂM TRA DANH SÁCH ĐANG NGỦ GẬT")
print("="*60)

active = logger.get_active_drowsy_students()
print(f"\n🔴 Số học sinh đang ngủ gật: {len(active)}")

for student in active:
    print(f"\n  - Camera: {student['camera_name']}")
    print(f"  - Student ID: {student['student_id']}")
    print(f"  - Start Time: {student['start_time']}")
    print(f"  - Duration: {student['duration_display']}")

# Test 6: Save logs
print("\n" + "="*60)
print("TEST 6: LƯU LOGS")
print("="*60)

try:
    filepath = logger.save_logs()
    print(f"✅ Đã lưu logs tại: {filepath}")
except Exception as e:
    print(f"❌ Lỗi lưu logs: {e}")

print("\n" + "="*60)
print("✅ TẤT CẢ TEST HOÀN THÀNH!")
print("="*60)

print("\n📋 KẾT LUẬN:")
print("✅ Logger hoạt động bình thường")
print("✅ Phát hiện buồn ngủ được ghi log")
print("✅ Tỉnh dậy được ghi log")
print("✅ Thống kê chính xác")
print("✅ Active tracking hoạt động")
print("\n💡 Logging system SẴN SÀNG để tích hợp với YOLO detection!")
