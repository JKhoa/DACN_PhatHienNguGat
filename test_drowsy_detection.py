"""
Test script to verify drowsiness detection and red tracking box display
Sends frames to backend and checks if drowsy state is detected
"""
import base64
import time
import socketio
import json
from pathlib import Path
import cv2
import numpy as np

# Create WebSocket client
sio = socketio.Client()

test_results = {
    'frames_sent': 0,
    'detections': [],
    'drowsy_detected': False,
    'awake_detected': False
}

@sio.event(namespace='/ws/detect')
def connect():
    print('✅ [WebSocket] Connected to /ws/detect')

@sio.on('hello', namespace='/ws/detect')
def on_hello(data):
    print(f'👋 [WebSocket] Hello: {data}')

@sio.on('result', namespace='/ws/detect')
def on_result(data):
    test_results['frames_sent'] += 1
    success = data.get('success', False)
    persons = data.get('persons', [])
    fps = data.get('fps', 0)
    
    print(f'\n📊 [Result #{test_results["frames_sent"]}]')
    print(f'   Success: {success}')
    print(f'   Persons detected: {len(persons)}')
    print(f'   FPS: {fps:.2f}')
    
    # Check drowsiness states
    for i, person in enumerate(persons):
        state = person.get('drowsiness_state', 'unknown')
        score = person.get('drowsiness_score', 0)
        track_id = person.get('track_id', person.get('id', 'N/A'))
        
        print(f'   Person {i+1} (ID: {track_id}):')
        print(f'      State: {state}')
        print(f'      Score: {score:.2f}')
        
        if state == 'awake':
            test_results['awake_detected'] = True
        elif state in ['drowsy', 'sleeping']:
            test_results['drowsy_detected'] = True
            print(f'      🔴 DROWSY/SLEEPING DETECTED!')
        
        test_results['detections'].append({
            'person_id': track_id,
            'state': state,
            'score': score,
            'timestamp': time.time()
        })

@sio.event(namespace='/ws/detect')
def disconnect():
    print('❌ [WebSocket] Disconnected')

def create_drowsy_pose_frame():
    """Create a synthetic frame simulating a person with head down (drowsy pose)"""
    # Create a blank frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Draw a simple representation of a person with head down
    # Body (rectangle)
    cv2.rectangle(frame, (250, 200), (390, 400), (100, 100, 150), -1)
    
    # Head tilted down (ellipse lower than shoulders)
    cv2.ellipse(frame, (320, 250), (40, 50), 45, 0, 360, (150, 120, 100), -1)
    
    # Add text
    cv2.putText(frame, 'DROWSY POSE TEST', (200, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def create_normal_pose_frame():
    """Create a synthetic frame simulating a person sitting upright (normal pose)"""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    
    # Body
    cv2.rectangle(frame, (250, 220), (390, 400), (100, 100, 150), -1)
    
    # Head upright (ellipse above shoulders)
    cv2.ellipse(frame, (320, 180), (40, 50), 0, 0, 360, (150, 120, 100), -1)
    
    # Add text
    cv2.putText(frame, 'NORMAL POSE TEST', (200, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 data URL"""
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{jpg_as_text}'

def send_test_frames():
    """Send test frames to backend and analyze results"""
    print('\n' + '='*60)
    print('🔬 DROWSINESS DETECTION TEST')
    print('='*60)
    
    # Connect to backend
    url = 'http://127.0.0.1:5000'
    print(f'\n📡 Connecting to {url}...')
    
    try:
        sio.connect(url, namespaces=['/ws/detect'])
    except Exception as e:
        print(f'❌ Connection failed: {e}')
        print('\n⚠️  Make sure backend is running:')
        print('    python "Desktop UI for Drowsiness Detection\\python-backend\\server_with_tracking_backup.py"')
        return
    
    # Wait for connection
    time.sleep(1)
    
    # Test 1: Send normal pose frame
    print('\n\n📤 Test 1: Sending NORMAL pose frame...')
    normal_frame = create_normal_pose_frame()
    normal_b64 = frame_to_base64(normal_frame)
    sio.emit('frame', {'frame': normal_b64, 'camera_id': 'test_webcam'}, namespace='/ws/detect')
    time.sleep(2)
    
    # Test 2: Send drowsy pose frame
    print('\n📤 Test 2: Sending DROWSY pose frame...')
    drowsy_frame = create_drowsy_pose_frame()
    drowsy_b64 = frame_to_base64(drowsy_frame)
    sio.emit('frame', {'frame': drowsy_b64, 'camera_id': 'test_webcam'}, namespace='/ws/detect')
    time.sleep(2)
    
    # Send a few more to build up temporal history
    print('\n📤 Test 3-5: Sending more DROWSY frames to build temporal history...')
    for i in range(3):
        sio.emit('frame', {'frame': drowsy_b64, 'camera_id': 'test_webcam'}, namespace='/ws/detect')
        time.sleep(0.5)
    
    # Wait for all results
    time.sleep(2)
    
    # Print summary
    print('\n' + '='*60)
    print('📋 TEST RESULTS SUMMARY')
    print('='*60)
    print(f'Frames sent: {test_results["frames_sent"]}')
    print(f'Awake state detected: {"✅ YES" if test_results["awake_detected"] else "❌ NO"}')
    print(f'Drowsy/Sleeping state detected: {"✅ YES" if test_results["drowsy_detected"] else "❌ NO"}')
    
    print('\n📝 Detection History:')
    for i, det in enumerate(test_results['detections'], 1):
        state_icon = '🔴' if det['state'] in ['drowsy', 'sleeping'] else '🟢'
        print(f'   {i}. {state_icon} Person {det["person_id"]}: {det["state"]} (score: {det["score"]:.2f})')
    
    # Final verdict
    print('\n' + '='*60)
    if test_results['drowsy_detected']:
        print('✅ SUCCESS: Drowsiness detection is working!')
        print('   → Backend can detect drowsy/sleeping states')
        print('   → WebSocket transmits state correctly')
        print('   → Frontend should display RED tracking box when drowsy')
    else:
        print('⚠️  WARNING: No drowsy state detected')
        print('   This could be due to:')
        print('   1. Temporal smoothing needs more frames (try >15 frames)')
        print('   2. Synthetic test frames may not trigger pose classifier')
        print('   3. Model requires real webcam footage with actual keypoints')
        print('\n   💡 Recommendation: Test with REAL WEBCAM and head-down pose')
    print('='*60)
    
    # Disconnect
    try:
        sio.disconnect()
    except:
        pass

if __name__ == '__main__':
    send_test_frames()
