"""
Test script for Drowsiness Logging System
Demonstrates all features:
- Multi-camera logging
- Time-based statistics (today, week, month)
- Detailed event logs
- Active drowsy student tracking
"""

import time
import json
from datetime import datetime, timedelta
from drowsiness_logger import MultiCameraLogger

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    # Initialize logger
    logger = MultiCameraLogger(log_dir="test_logs")
    
    # Register cameras (phòng học)
    logger.register_camera("camera_1", "Phòng 101 - Toán Cao Cấp")
    logger.register_camera("camera_2", "Phòng 102 - Lập Trình Python")
    logger.register_camera("camera_3", "Phòng 103 - Cơ Sở Dữ Liệu")
    
    # Simulate drowsiness events
    print_section("SIMULATING DROWSINESS EVENTS")
    
    # Phòng 101: Student 5 starts drowsy
    print("⏰ 09:00 - Phòng 101: Học sinh #5 bắt đầu ngủ gật")
    logger.update_student_state("camera_1", 5, True)
    time.sleep(2)
    
    # Phòng 102: Student 10 và 12 start drowsy
    print("⏰ 09:02 - Phòng 102: Học sinh #10 và #12 bắt đầu ngủ gật")
    logger.update_student_state("camera_2", 10, True)
    logger.update_student_state("camera_2", 12, True)
    time.sleep(1)
    
    # Phòng 101: Student 5 wakes up, Student 8 starts drowsy
    print("⏰ 09:03 - Phòng 101: Học sinh #5 tỉnh lại, Học sinh #8 bắt đầu ngủ gật")
    logger.update_student_state("camera_1", 5, False)
    logger.update_student_state("camera_1", 8, True)
    time.sleep(1.5)
    
    # Phòng 102: Student 10 wakes up
    print("⏰ 09:04 - Phòng 102: Học sinh #10 tỉnh lại")
    logger.update_student_state("camera_2", 10, False)
    time.sleep(1)
    
    # Phòng 103: Student 15, 16, 17 start drowsy
    print("⏰ 09:05 - Phòng 103: Học sinh #15, #16, #17 bắt đầu ngủ gật")
    logger.update_student_state("camera_3", 15, True)
    logger.update_student_state("camera_3", 16, True)
    logger.update_student_state("camera_3", 17, True)
    time.sleep(2)
    
    # Phòng 101: Student 8 wakes up
    print("⏰ 09:07 - Phòng 101: Học sinh #8 tỉnh lại")
    logger.update_student_state("camera_1", 8, False)
    time.sleep(1)
    
    # Phòng 103: Student 16 wakes up
    print("⏰ 09:08 - Phòng 103: Học sinh #16 tỉnh lại")
    logger.update_student_state("camera_3", 16, False)
    
    # =====================================================
    # DISPLAY STATISTICS
    # =====================================================
    
    # 1. Active drowsy students
    print_section("ACTIVE DROWSY STUDENTS (ĐANG NGỦ GẬT)")
    active = logger.get_active_drowsy_all_cameras()
    print(json.dumps(active, indent=2, ensure_ascii=False))
    print(f"\n📊 Tổng số học sinh đang ngủ gật: {sum(len(students) for students in active.values())}")
    
    # 2. Today's summary
    print_section("THỐNG KÊ TỔNG HỢP HÔM NAY")
    summary = logger.get_summary_stats('today')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 3. Camera-specific stats
    print_section("THỐNG KÊ CHI TIẾT TỪNG PHÒNG")
    for camera_id in ["camera_1", "camera_2", "camera_3"]:
        stats = logger.get_camera_stats(camera_id, 'today')
        print(f"\n🏫 {stats['camera_name']}:")
        print(f"   • Tổng học sinh ngủ gật: {stats['total_drowsy_students']}")
        print(f"   • Đang ngủ gật: {stats['currently_drowsy']}")
        print(f"   • Số sự kiện: {stats['total_events']}")
        print(f"   • Tổng thời gian: {stats['total_duration_display']}")
        if stats['most_drowsy_student']:
            print(f"   • Học sinh ngủ gật nhiều nhất: #{stats['most_drowsy_student']} ({stats['most_drowsy_duration']:.1f}s)")
    
    # 4. Detailed event logs
    print_section("LOG CHI TIẾT CÁC SỰ KIỆN NGỦ GẬT")
    for camera_id in ["camera_1", "camera_2", "camera_3"]:
        events = logger.get_camera_events(camera_id, 'today')
        if events:
            print(f"\n🏫 {logger.cameras[camera_id].camera_name}:")
            for event in events:
                status = "🟢 Đã tỉnh" if not event['is_active'] else "🔴 Đang ngủ"
                print(f"   {status} Học sinh #{event['student_id']}: "
                      f"{event['start_time']} → {event['end_time']} "
                      f"({event['duration_display']})")
    
    # 5. Week and month statistics
    print_section("THỐNG KÊ TUẦN NÀY")
    week_summary = logger.get_summary_stats('week')
    print(f"📅 Thời gian: {week_summary['period_start'][:10]} → {week_summary['period_end'][:10]}")
    print(f"📊 Tổng số phòng: {week_summary['total_cameras']}")
    print(f"👥 Tổng học sinh ngủ gật: {week_summary['total_drowsy_students_unique']}")
    print(f"📝 Tổng sự kiện: {week_summary['total_events']}")
    print(f"⏱️  Tổng thời gian: {week_summary['total_duration_display']}")
    
    print_section("THỐNG KÊ THÁNG NÀY")
    month_summary = logger.get_summary_stats('month')
    print(f"📅 Thời gian: {month_summary['period_start'][:10]} → {month_summary['period_end'][:10]}")
    print(f"📊 Tổng số phòng: {month_summary['total_cameras']}")
    print(f"👥 Tổng học sinh ngủ gật: {month_summary['total_drowsy_students_unique']}")
    print(f"📝 Tổng sự kiện: {month_summary['total_events']}")
    print(f"⏱️  Tổng thời gian: {month_summary['total_duration_display']}")
    
    # 6. Custom date range example
    print_section("THỐNG KÊ KHOẢNG THỜI GIAN TÙY CHỈNH")
    today = datetime.now().strftime('%Y-%m-%d')
    custom_period = f"{today}_{today}"  # Today only
    custom_summary = logger.get_summary_stats(custom_period)
    print(f"📅 Khoảng thời gian: {custom_period}")
    print(f"👥 Tổng học sinh ngủ gật: {custom_summary['total_drowsy_students_unique']}")
    print(f"📝 Tổng sự kiện: {custom_summary['total_events']}")
    
    # 7. Save to file
    print_section("LƯU LOG VÀO FILE")
    logger.save_to_file("drowsiness_report.json")
    print("✅ Đã lưu báo cáo vào: drowsiness_report.json")
    
    # 8. API Response Examples
    print_section("MẪU API RESPONSES")
    
    print("\n📡 GET /api/logs/cameras")
    print(json.dumps({
        'success': True,
        'cameras': [
            {'camera_id': c['camera_id'], 'camera_name': c['camera_name']} 
            for c in logger.get_all_cameras_stats('today')
        ]
    }, indent=2, ensure_ascii=False))
    
    print("\n📡 GET /api/logs/stats/camera_1?period=today")
    stats_api = logger.get_camera_stats("camera_1", 'today')
    print(json.dumps({'success': True, 'stats': stats_api}, indent=2, ensure_ascii=False))
    
    print("\n📡 GET /api/logs/active")
    print(json.dumps({
        'success': True,
        'active_drowsy': active,
        'total_active': sum(len(students) for students in active.values())
    }, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("  ✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
