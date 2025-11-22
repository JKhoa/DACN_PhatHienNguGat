#!/usr/bin/env python3
"""
Camera Connection Test Script
Test if cameras are available and working
"""

import cv2
import sys

def test_camera_availability():
    """Test available cameras on the system"""
    print("🔍 Testing camera availability...")
    
    available_cameras = []
    
    # Test cameras from index 0 to 5
    for i in range(6):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Camera {i}: Available ({w}x{h})")
                available_cameras.append({
                    'index': i,
                    'width': w,
                    'height': h,
                    'name': f'Camera {i}'
                })
            else:
                print(f"  ❌ Camera {i}: Can't read frame")
        else:
            print(f"  ❌ Camera {i}: Not accessible")
        
        cap.release()
    
    return available_cameras

def test_camera_stream(camera_index=0):
    """Test camera streaming for 5 seconds"""
    print(f"\n🎥 Testing camera {camera_index} stream...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return False
    
    frame_count = 0
    success_count = 0
    
    print("Recording for 5 seconds... Press 'q' to quit early")
    
    for i in range(150):  # ~5 seconds at 30fps
        ret, frame = cap.read()
        frame_count += 1
        
        if ret:
            success_count += 1
            cv2.imshow(f'Camera {camera_index} Test', frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"⚠️  Frame {frame_count}: Failed to read")
    
    cap.release()
    cv2.destroyAllWindows()
    
    success_rate = (success_count / frame_count) * 100
    print(f"📊 Stream test results:")
    print(f"   Total frames: {frame_count}")
    print(f"   Successful frames: {success_count}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    return success_rate > 80

def test_camera_for_backend(camera_index=0):
    """Test camera compatibility with backend detection"""
    print(f"\n🔬 Testing camera {camera_index} for backend compatibility...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return False, "Cannot open camera"
    
    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Camera settings: {width}x{height} @ {fps}fps")
    
    # Test frame reading
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Cannot read frames"
    
    if frame is None:
        return False, "Frame is None"
    
    if frame.shape[0] < 240 or frame.shape[1] < 320:
        return False, f"Resolution too small: {frame.shape[1]}x{frame.shape[0]}"
    
    print(f"   ✅ Compatible: {frame.shape[1]}x{frame.shape[0]} resolution")
    return True, "OK"

def add_camera_to_backend(camera_index=0):
    """Add camera to backend via API"""
    import requests
    
    print(f"\n📡 Adding camera {camera_index} to backend...")
    
    try:
        # Test if backend is running
        response = requests.get("http://127.0.0.1:5000/api/cameras", timeout=5)
        if response.status_code != 200:
            return False, "Backend not responding"
        
        # Add webcam camera using correct endpoint
        camera_data = {
            "id": f"webcam_{camera_index}",
            "name": f"Webcam {camera_index}",
            "type": "webcam",
            "source": str(camera_index),
            "room_id": "room_1"
        }
        
        response = requests.post("http://127.0.0.1:5000/api/camera/add", 
                               json=camera_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"   ✅ Camera added successfully")
                return True, "Camera added"
            else:
                return False, f"API error: {result.get('error', 'Unknown')}"
        else:
            return False, f"HTTP error: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Network error: {e}"

def start_camera_in_backend(camera_id):
    """Start camera in backend"""
    import requests
    
    print(f"\n▶️  Starting camera {camera_id} in backend...")
    
    try:
        response = requests.post(f"http://127.0.0.1:5000/api/camera/{camera_id}/start", 
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"   ✅ Camera started successfully")
                return True, "Camera started"
            else:
                return False, f"Start error: {result.get('error', 'Unknown')}"
        else:
            return False, f"HTTP error: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Network error: {e}"

if __name__ == "__main__":
    print("=" * 50)
    print("   CAMERA CONNECTION DIAGNOSTIC")
    print("=" * 50)
    
    # Step 1: Check available cameras
    cameras = test_camera_availability()
    
    if not cameras:
        print("\n❌ No cameras found!")
        print("Possible solutions:")
        print("1. Check if webcam is connected")
        print("2. Check camera permissions")
        print("3. Close other applications using camera")
        print("4. Restart computer")
        sys.exit(1)
    
    print(f"\n✅ Found {len(cameras)} camera(s):")
    for cam in cameras:
        print(f"   - Camera {cam['index']}: {cam['width']}x{cam['height']}")
    
    # Step 2: Test first available camera stream
    first_camera = cameras[0]['index']
    
    print(f"\n🎬 Testing camera {first_camera} stream...")
    if test_camera_stream(first_camera):
        print("✅ Camera stream test passed")
    else:
        print("❌ Camera stream test failed")
        sys.exit(1)
    
    # Step 3: Test backend compatibility
    compatible, msg = test_camera_for_backend(first_camera)
    if compatible:
        print("✅ Camera is compatible with backend")
    else:
        print(f"❌ Camera compatibility issue: {msg}")
        sys.exit(1)
    
    # Step 4: Add to backend
    added, msg = add_camera_to_backend(first_camera)
    if added:
        print("✅ Camera added to backend")
    else:
        print(f"❌ Failed to add camera: {msg}")
        sys.exit(1)
    
    # Step 5: Start camera
    started, msg = start_camera_in_backend(f"webcam_{first_camera}")
    if started:
        print("✅ Camera started in backend")
    else:
        print(f"❌ Failed to start camera: {msg}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("   🎉 CAMERA CONNECTION SUCCESS!")
    print("=" * 50)
    print(f"Camera webcam_{first_camera} is now available in the app")
    print("You can now use the Desktop UI to view the camera feed")
    print("=" * 50)