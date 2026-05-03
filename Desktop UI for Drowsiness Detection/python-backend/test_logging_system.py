#!/usr/bin/env python3
"""
Test script để kiểm tra hệ thống logging ngủ gật
Kiểm tra xem học sinh có được đánh số thứ tự khi ngủ gật hay không
"""

import requests
import json
import time
from datetime import datetime

# Base URL của backend
BASE_URL = "http://127.0.0.1:5000"

def test_logging_endpoints():
    """Test các API endpoints của logging system"""
    print("🔥 TESTING DROWSINESS LOGGING SYSTEM")
    print("=" * 50)
    
    # Test 1: Kiểm tra active drowsy students
    print("\n📋 1. Testing Active Drowsy Students...")
    try:
        response = requests.get(f"{BASE_URL}/api/drowsiness/active")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data['success']}")
            active_students = data.get('active_drowsy_students', {})
            print(f"👥 Active drowsy students across all cameras: {len(active_students)}")
            for camera_id, students in active_students.items():
                print(f"   📹 Camera {camera_id}: {len(students)} students drowsy")
                for student in students:
                    print(f"      - Student #{student['student_id']}: {student['duration_display']} drowsy")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Test 2: Kiểm tra summary stats
    print("\n📊 2. Testing Summary Statistics...")
    try:
        response = requests.get(f"{BASE_URL}/api/drowsiness/summary?period=today")
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print(f"✅ Success: {data['success']}")
            print(f"📈 Today's Summary:")
            print(f"   - Total cameras: {summary.get('total_cameras', 0)}")
            print(f"   - Students with drowsiness: {summary.get('total_drowsy_students_unique', 0)}")
            print(f"   - Total events: {summary.get('total_events', 0)}")
            print(f"   - Total drowsy time: {summary.get('total_duration_display', '0s')}")
            print(f"   - Currently drowsy: {summary.get('currently_drowsy_all_cameras', 0)}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Test 3: Kiểm tra camera-specific stats
    print("\n📹 3. Testing Camera-Specific Statistics...")
    camera_id = "default_camera"  # Default camera from webcam
    try:
        response = requests.get(f"{BASE_URL}/api/drowsiness/stats/{camera_id}?period=today")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            print(f"✅ Success for camera {camera_id}")
            print(f"📊 Camera Stats:")
            print(f"   - Camera name: {stats.get('camera_name', 'Unknown')}")
            print(f"   - Total drowsy students: {stats.get('total_drowsy_students', 0)}")
            print(f"   - Currently drowsy: {stats.get('currently_drowsy', 0)}")
            print(f"   - Total events: {stats.get('total_events', 0)}")
            print(f"   - Total duration: {stats.get('total_duration_display', '0s')}")
            if stats.get('most_drowsy_student'):
                print(f"   - Most drowsy student: #{stats.get('most_drowsy_student')} ({stats.get('most_drowsy_duration', 0)}s)")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Test 4: Kiểm tra detailed events
    print("\n📝 4. Testing Detailed Events Log...")
    try:
        response = requests.get(f"{BASE_URL}/api/drowsiness/events/{camera_id}?period=today")
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"✅ Success: Found {len(events)} events today")
            
            if events:
                print("🕒 Recent drowsiness events (newest first):")
                for i, event in enumerate(events[:5]):  # Show only first 5
                    status = "🔴 ĐANG NGỦ" if event['is_active'] else "✅ ĐÃ TỈNH"
                    print(f"   {i+1}. Student #{event['student_id']}")
                    print(f"      Start: {event['start_time']}")
                    print(f"      End: {event['end_time']}")
                    print(f"      Duration: {event['duration_display']}")
                    print(f"      Status: {status}")
                    print()
            else:
                print("💡 No drowsiness events recorded today")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")

def monitor_realtime():
    """Monitor real-time updates (would use WebSocket in real app)"""
    print("\n🔄 MONITORING REAL-TIME UPDATES")
    print("=" * 50)
    print("Polling active students every 5 seconds... (Press Ctrl+C to stop)")
    
    try:
        while True:
            response = requests.get(f"{BASE_URL}/api/drowsiness/active")
            if response.status_code == 200:
                data = response.json()
                active_students = data.get('active_drowsy_students', {})
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{timestamp}] Active Drowsy Students:")
                
                if active_students:
                    total_drowsy = sum(len(students) for students in active_students.values())
                    print(f"   📊 Total: {total_drowsy} students across {len(active_students)} cameras")
                    
                    for camera_id, students in active_students.items():
                        print(f"   📹 Camera {camera_id}:")
                        for i, student in enumerate(students, 1):
                            print(f"      {i}. Student #{student['student_id']} - {student['duration_display']} (Started: {student['start_time'][-8:]})")
                else:
                    print("   ✅ All students are awake!")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")

def main():
    print("🎯 DROWSINESS LOGGING SYSTEM TESTER")
    print("Testing backend at:", BASE_URL)
    print("Make sure the backend is running!")
    
    # Test endpoints first
    test_logging_endpoints()
    
    # Ask user if they want to monitor real-time
    try:
        choice = input("\n🤔 Do you want to monitor real-time updates? (y/n): ").lower()
        if choice == 'y':
            monitor_realtime()
    except KeyboardInterrupt:
        pass
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()