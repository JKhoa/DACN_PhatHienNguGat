"""
Test phát hiện ngủ gật với 30 học sinh đồng thời
Kiểm tra performance và độ chính xác của tracker
"""

import sys
import os
import time
import cv2
import numpy as np

# Add backend to path
backend_dir = os.path.join(os.path.dirname(__file__), 'Desktop UI for Drowsiness Detection', 'python-backend')
sys.path.insert(0, backend_dir)

try:
    from yolo_detector import initialize_detector, detect_frame, DetectionResult
    from server_with_tracking_backup import EnhancedTracker
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"❌ Không thể import YOLO detector: {e}")
    YOLO_AVAILABLE = False
    sys.exit(1)


def create_synthetic_classroom(width=1920, height=1080, num_students=30):
    """Tạo ảnh lớp học giả lập với nhiều người"""
    # Tạo background màu trắng (bảng)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Vẽ grid các vị trí học sinh (6 hàng x 5 cột)
    rows = 6
    cols = 5
    
    # Kích thước mỗi ô
    cell_width = width // (cols + 1)
    cell_height = height // (rows + 1)
    
    positions = []
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= num_students:
                break
            # Vị trí trung tâm mỗi ô
            x = cell_width * (col + 1)
            y = cell_height * (row + 1)
            positions.append((x, y))
    
    # Vẽ người đơn giản (hình oval đại diện)
    for idx, (cx, cy) in enumerate(positions[:num_students]):
        # Body (oval lớn)
        body_w = 60
        body_h = 100
        cv2.ellipse(frame, (cx, cy), (body_w//2, body_h//2), 0, 0, 360, (100, 100, 200), -1)
        
        # Head (circle nhỏ hơn)
        head_r = 25
        head_y = cy - body_h//2 - head_r
        cv2.circle(frame, (cx, head_y), head_r, (200, 150, 100), -1)
        
        # ID text
        cv2.putText(frame, f"{idx+1}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Simulate drowsy students (20% ngủ gật)
        if idx % 5 == 0:
            # Vẽ head nghiêng (drowsy)
            cv2.ellipse(frame, (cx+5, head_y+5), (head_r, head_r), 30, 0, 360, (150, 100, 100), -1)
    
    return frame


def test_yolo_detection_30_students():
    """Test YOLO detection với 30 học sinh"""
    print("\n" + "="*60)
    print("🧪 TEST PHÁT HIỆN NGỦ GẬT - 30 HỌC SINH")
    print("="*60 + "\n")
    
    # 1. Khởi tạo detector
    print("📋 Bước 1: Khởi tạo YOLO detector...")
    try:
        model_path = os.path.join(backend_dir, 'yolo11n-pose.pt')
        if not os.path.exists(model_path):
            model_path = 'yolo11n-pose.pt'
        initialize_detector(model_path)
        print("   ✅ YOLO detector khởi tạo thành công")
    except Exception as e:
        print(f"   ❌ Lỗi khởi tạo: {e}")
        return
    
    # 2. Tạo ảnh test với 30 học sinh
    print("\n📋 Bước 2: Tạo ảnh test classroom...")
    frame = create_synthetic_classroom(num_students=30)
    print(f"   ✅ Đã tạo ảnh {frame.shape[1]}x{frame.shape[0]} với 30 học sinh")
    
    # Lưu ảnh để debug
    test_img_path = os.path.join(os.path.dirname(__file__), 'test_30_students.jpg')
    cv2.imwrite(test_img_path, frame)
    print(f"   💾 Đã lưu ảnh test: {test_img_path}")
    
    # 3. Chạy detection
    print("\n📋 Bước 3: Chạy YOLO detection...")
    start_time = time.time()
    result = detect_frame(frame)
    detection_time = time.time() - start_time
    
    print(f"   ⏱️  Thời gian xử lý: {detection_time*1000:.1f}ms")
    print(f"   👥 Số người phát hiện: {len(result.persons)}")
    print(f"   📊 FPS: {result.fps:.1f}")
    
    # 4. Kiểm tra tracker
    print("\n📋 Bước 4: Test Enhanced Tracker...")
    tracker = EnhancedTracker(iou_thr=0.35, max_age=25)
    
    # Simulate 5 frames liên tiếp
    total_track_time = 0
    for frame_idx in range(5):
        # Thêm noise nhẹ vào positions
        noisy_frame = frame.copy()
        if frame_idx > 0:
            noise = np.random.randint(-5, 5, size=frame.shape, dtype=np.int16)
            noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Detect
        track_start = time.time()
        result = detect_frame(noisy_frame)
        tracked_persons = tracker.update(result.persons)
        track_time = time.time() - track_start
        total_track_time += track_time
        
        print(f"   Frame {frame_idx+1}: {len(tracked_persons)} người tracked | {track_time*1000:.1f}ms")
    
    avg_track_time = total_track_time / 5
    print(f"\n   📈 Thời gian tracking trung bình: {avg_track_time*1000:.1f}ms")
    print(f"   📈 FPS ước tính (với tracking): {1.0/avg_track_time:.1f}")
    
    # 5. Phân tích kết quả
    print("\n📋 Bước 5: Phân tích kết quả...")
    
    # Đếm trạng thái
    drowsy_count = sum(1 for p in result.persons if p.drowsiness_state in ['drowsy', 'sleeping'])
    awake_count = sum(1 for p in result.persons if p.drowsiness_state == 'awake')
    
    print(f"   👤 Tỉnh táo: {awake_count}")
    print(f"   😴 Buồn ngủ/Ngủ gật: {drowsy_count}")
    
    # Confidence trung bình
    if result.persons:
        avg_conf = sum(p.confidence for p in result.persons) / len(result.persons)
        print(f"   🎯 Confidence trung bình: {avg_conf:.2f}")
    
    # 6. Đánh giá performance
    print("\n📋 Bước 6: Đánh giá performance...")
    
    target_fps = 15  # Mục tiêu tối thiểu
    current_fps = 1.0 / avg_track_time
    
    if current_fps >= target_fps:
        print(f"   ✅ PASS: {current_fps:.1f} FPS >= {target_fps} FPS (mục tiêu)")
    else:
        print(f"   ⚠️  CẢNH BÁO: {current_fps:.1f} FPS < {target_fps} FPS (mục tiêu)")
    
    # Memory estimate
    tracker_size = len(tracker.tracks)
    print(f"   💾 Số tracks đang quản lý: {tracker_size}")
    
    # 7. Kiểm tra độ ổn định tracking
    print("\n📋 Bước 7: Test độ ổn định tracking (20 frames)...")
    track_ids_history = []
    
    for frame_idx in range(20):
        noisy_frame = frame.copy()
        if frame_idx > 0:
            noise = np.random.randint(-3, 3, size=frame.shape, dtype=np.int16)
            noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        result = detect_frame(noisy_frame)
        tracked_persons = tracker.update(result.persons)
        
        # Lưu track IDs
        track_ids = sorted([p.track_id for p in tracked_persons if p.track_id])
        track_ids_history.append(track_ids)
    
    # Phân tích độ ổn định
    if len(track_ids_history) > 1:
        # Số lượng IDs ổn định qua các frames
        first_ids = set(track_ids_history[0])
        last_ids = set(track_ids_history[-1])
        
        stability = len(first_ids & last_ids) / max(len(first_ids), 1) * 100
        print(f"   📊 Độ ổn định tracking: {stability:.1f}%")
        print(f"   🔢 IDs đầu tiên: {len(first_ids)}, IDs cuối: {len(last_ids)}")
        
        if stability >= 80:
            print(f"   ✅ Tracking ổn định (>= 80%)")
        else:
            print(f"   ⚠️  Tracking chưa ổn định (< 80%)")
    
    # 8. Kết luận
    print("\n" + "="*60)
    print("📊 KẾT LUẬN")
    print("="*60)
    
    can_detect = len(result.persons) > 0
    is_realtime = current_fps >= target_fps
    is_stable = stability >= 80 if len(track_ids_history) > 1 else False
    
    print(f"\n✓ Khả năng phát hiện: {'✅ CÓ' if can_detect else '❌ KHÔNG'}")
    print(f"✓ Đạt realtime (≥15 FPS): {'✅ CÓ' if is_realtime else '❌ KHÔNG'}")
    print(f"✓ Tracking ổn định (≥80%): {'✅ CÓ' if is_stable else '❌ KHÔNG'}")
    
    print(f"\n📈 Số liệu chi tiết:")
    print(f"   - Số người phát hiện: {len(result.persons)}/30")
    print(f"   - FPS thực tế: {current_fps:.1f}")
    print(f"   - Thời gian xử lý/frame: {avg_track_time*1000:.1f}ms")
    print(f"   - Độ ổn định tracking: {stability:.1f}%" if len(track_ids_history) > 1 else "   - Độ ổn định: N/A")
    
    if can_detect and is_realtime and is_stable:
        print(f"\n🎉 KẾT LUẬN: Hệ thống CÓ THỂ phát hiện 30 học sinh realtime!")
    elif can_detect and is_realtime:
        print(f"\n⚠️  KẾT LUẬN: Hệ thống CÓ THỂ phát hiện nhưng tracking chưa ổn định")
    elif can_detect:
        print(f"\n⚠️  KẾT LUẬN: Hệ thống CÓ THỂ phát hiện nhưng chưa đạt realtime")
    else:
        print(f"\n❌ KẾT LUẬN: Hệ thống CHƯA ĐỦ MẠNH cho 30 học sinh")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'detected': len(result.persons),
        'fps': current_fps,
        'stability': stability if len(track_ids_history) > 1 else 0,
        'can_detect': can_detect,
        'is_realtime': is_realtime,
        'is_stable': is_stable
    }


def test_with_real_classroom_image():
    """Test với ảnh lớp học thật nếu có"""
    print("\n🔍 Tìm kiếm ảnh lớp học thật trong data_raw...")
    
    data_raw = os.path.join(os.path.dirname(__file__), 'data_raw')
    if not os.path.exists(data_raw):
        print("   ⚠️  Không tìm thấy thư mục data_raw")
        return None
    
    # Tìm ảnh có nhiều người
    classroom_images = []
    for fname in os.listdir(data_raw):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(data_raw, fname)
            classroom_images.append(fpath)
    
    if not classroom_images:
        print("   ⚠️  Không tìm thấy ảnh trong data_raw")
        return None
    
    print(f"   ✅ Tìm thấy {len(classroom_images)} ảnh")
    
    # Test với ảnh đầu tiên
    test_img = classroom_images[0]
    print(f"\n📸 Test với ảnh: {os.path.basename(test_img)}")
    
    frame = cv2.imread(test_img)
    if frame is None:
        print("   ❌ Không đọc được ảnh")
        return None
    
    print(f"   ✅ Đã load ảnh {frame.shape[1]}x{frame.shape[0]}")
    
    # Detect
    start_time = time.time()
    result = detect_frame(frame)
    detection_time = time.time() - start_time
    
    print(f"   ⏱️  Thời gian: {detection_time*1000:.1f}ms")
    print(f"   👥 Phát hiện: {len(result.persons)} người")
    print(f"   📊 FPS: {result.fps:.1f}")
    
    return result


if __name__ == '__main__':
    print("\n🚀 BẮT ĐẦU TEST PHÁT HIỆN 30 HỌC SINH\n")
    
    # Test 1: Synthetic classroom
    results = test_yolo_detection_30_students()
    
    # Test 2: Real image (nếu có)
    print("\n" + "="*60)
    test_with_real_classroom_image()
    
    print("\n✅ HOÀN TẤT TEST!\n")
