import requests
import json

BASE_URL = "http://127.0.0.1:5001"

print("🧪 TESTING LOGGING SYSTEM ON PORT 5001")
print("=" * 50)

# Test 1: Basic connectivity
print("\n1. Testing basic connectivity...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"✅ Server response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: API test endpoint
print("\n2. Testing API endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/test")
    print(f"✅ API response: {response.json()}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Create test logging data
print("\n3. Creating test logging data...")
try:
    response = requests.get(f"{BASE_URL}/api/drowsiness/test-log")
    data = response.json()
    if data['success']:
        print("✅ Test data created successfully!")
        print(f"📊 Active students: {data.get('active_students', {})}")
        print(f"📈 Summary: {data.get('summary', {}).get('total_events', 0)} events")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown')}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Get active drowsy students
print("\n4. Testing active drowsy students...")
try:
    response = requests.get(f"{BASE_URL}/api/drowsiness/active")
    data = response.json()
    if data['success']:
        print("✅ Active drowsy students API working!")
        active = data.get('active_drowsy_students', {})
        for camera_id, students in active.items():
            print(f"📹 Camera {camera_id}: {len(students)} students")
            for student in students:
                print(f"   - Student #{student['student_id']}: {student['duration_display']}")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown')}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Get summary statistics
print("\n5. Testing summary statistics...")
try:
    response = requests.get(f"{BASE_URL}/api/drowsiness/summary?period=today")
    data = response.json()
    if data['success']:
        print("✅ Summary statistics API working!")
        summary = data.get('summary', {})
        print(f"📊 Total Events: {summary.get('total_events', 0)}")
        print(f"👥 Total Students: {summary.get('total_drowsy_students_unique', 0)}")
        print(f"⏰ Total Duration: {summary.get('total_duration_display', '0s')}")
        print(f"🔴 Currently Drowsy: {summary.get('currently_drowsy_all_cameras', 0)}")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown')}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 6: Get session summary with unique IDs
print("\n📊 6. Testing Session Summary (Unique Student IDs)...")
try:
    response = requests.get(f"{BASE_URL}/api/drowsiness/session-summary")
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print("✅ Session summary API working!")
            session = data.get('session_summary', {})
            print(f"📈 Session Summary:")
            print(f"   - Total unique students: {session.get('total_unique_students_all_cameras', 0)}")
            print(f"   - Student IDs: {session.get('all_student_ids', [])}")
            print(f"   - Total events: {session.get('total_events_all_cameras', 0)}")
            print(f"   - Currently active: {session.get('total_active_events_all_cameras', 0)}")
            
            # Show camera-specific summaries
            cameras = session.get('camera_summaries', [])
            for cam in cameras:
                print(f"   📹 {cam.get('camera_name', 'Unknown')}: {cam.get('total_unique_students', 0)} students")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown')}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🎉 Logging system test completed!")