#!/usr/bin/env python3
"""
Test script to check camera access directly
"""
import cv2
import time

def test_camera_access():
    print("Testing camera access...")
    
    # Test different camera indices
    for i in range(5):
        print(f"\nTesting camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"[OK] Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] Camera {i} can read frames: {frame.shape}")
                
                # Try to encode to JPEG
                success, buf = cv2.imencode('.jpg', frame)
                if success:
                    print(f"[OK] Camera {i} can encode to JPEG: {len(buf)} bytes")
                else:
                    print(f"[ERROR] Camera {i} failed to encode to JPEG")
            else:
                print(f"[ERROR] Camera {i} cannot read frames")
            
            cap.release()
        else:
            print(f"[ERROR] Camera {i} failed to open")
    
    print("\nTesting with different backends...")
    
    # Test with different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any")
    ]
    
    for backend, name in backends:
        print(f"\nTesting with {name} backend...")
        cap = cv2.VideoCapture(0, backend)
        
        if cap.isOpened():
            print(f"[OK] {name} backend works")
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[OK] {name} can read frames: {frame.shape}")
            else:
                print(f"[ERROR] {name} cannot read frames")
            cap.release()
        else:
            print(f"[ERROR] {name} backend failed")

if __name__ == "__main__":
    test_camera_access()


