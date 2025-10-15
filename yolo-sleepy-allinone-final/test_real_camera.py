#!/usr/bin/env python3
"""
Test Real Camera Connection
Script để test kết nối với camera thật trước khi thêm vào hệ thống
"""

import cv2
import sys

def test_webcam(camera_id=0):
    """Test webcam connection"""
    print(f"\n🎥 Testing Webcam {camera_id}...")
    print("="*60)
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Cannot open webcam {camera_id}")
        print("\nTips:")
        print("  • Try other IDs: 0, 1, 2")
        print("  • Check if camera is used by another app")
        print("  • Check camera permissions")
        return False
    
    # Test read
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"❌ Cannot read from webcam {camera_id}")
        cap.release()
        return False
    
    # Get info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✅ Webcam {camera_id} connected!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frame shape: {frame.shape}")
    
    # Show preview
    print("\n📺 Showing preview... Press 'q' to close")
    
    cv2.namedWindow(f"Webcam {camera_id} Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add info to frame
        cv2.putText(frame, f"Webcam {camera_id} Test - Press Q to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(f"Webcam {camera_id} Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test completed. Total frames: {frame_count}")
    return True


def test_ip_camera(ip, port=554, username="admin", password="", brand="generic"):
    """Test IP camera connection"""
    print(f"\n🎥 Testing IP Camera...")
    print("="*60)
    print(f"IP: {ip}:{port}")
    print(f"Username: {username}")
    print(f"Brand: {brand}")
    print("="*60)
    
    # Generate RTSP URL based on brand
    rtsp_paths = {
        "imou": "/cam/realmonitor?channel=1&subtype=0",
        "hikvision": "/Streaming/Channels/101",
        "dahua": "/cam/realmonitor?channel=1&subtype=0",
        "tapo": "/stream1",
        "tplink": "/stream1",
        "xiaomi": "/live/ch00_0",
        "mijia": "/live/ch00_0",
        "reolink": "/h264Preview_01_main",
        "foscam": "/videoMain",
        "axis": "/axis-media/media.amp?videocodec=h264",
        "generic": "/stream1",
        "onvif": "/onvif1",
    }
    
    path = rtsp_paths.get(brand.lower(), "/stream1")
    
    if username and password:
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
        display_url = f"rtsp://{username}:****@{ip}:{port}{path}"
    else:
        rtsp_url = f"rtsp://{ip}:{port}{path}"
        display_url = rtsp_url
    
    print(f"\n🔗 RTSP URL: {display_url}")
    print(f"⏳ Connecting... (this may take 10-30 seconds)")
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"\n❌ Cannot connect to camera")
        print("\nPossible issues:")
        print("  • Wrong IP address")
        print("  • Wrong username/password")
        print("  • Wrong port (default is 554)")
        print("  • Camera and computer not on same network")
        print("  • RTSP not enabled on camera")
        print("  • Firewall blocking connection")
        print(f"  • Wrong brand (try 'generic' or other brands)")
        return False
    
    print(f"✅ Connected! Reading frames...")
    
    # Test read
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"❌ Cannot read frames from camera")
        cap.release()
        return False
    
    # Get info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n✅ IP Camera connected successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frame shape: {frame.shape}")
    
    # Show preview
    print(f"\n📺 Showing preview... Press 'q' to close")
    
    cv2.namedWindow("IP Camera Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame read failed, reconnecting...")
            break
        
        # Add info to frame
        cv2.putText(frame, f"IP Camera Test - Press Q to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"IP: {ip}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("IP Camera Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test completed. Total frames: {frame_count}")
    return True


def interactive_test():
    """Interactive camera testing"""
    print("\n" + "="*60)
    print("🎥 CAMERA CONNECTION TEST")
    print("="*60)
    
    print("\nSelect camera type:")
    print("1. Webcam (USB/Laptop Camera)")
    print("2. IP Camera (Network Camera)")
    print("3. Test all webcams (0-2)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Test webcam
        camera_id = input("Enter webcam ID (0-2, default 0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        test_webcam(camera_id)
        
    elif choice == "2":
        # Test IP camera
        print("\n" + "="*60)
        print("IP CAMERA CONFIGURATION")
        print("="*60)
        
        ip = input("IP Address (e.g., 192.168.1.100): ").strip()
        if not ip:
            print("❌ IP address required!")
            return
        
        port = input("Port (default 554): ").strip()
        port = int(port) if port else 554
        
        username = input("Username (default 'admin'): ").strip()
        username = username if username else "admin"
        
        password = input("Password: ").strip()
        
        print("\nSelect brand:")
        brands = ["imou", "hikvision", "dahua", "tapo", "xiaomi", "reolink", "generic"]
        for i, brand in enumerate(brands, 1):
            print(f"{i}. {brand.title()}")
        
        brand_choice = input(f"Enter brand number (1-{len(brands)}, default generic): ").strip()
        if brand_choice and brand_choice.isdigit():
            brand_idx = int(brand_choice) - 1
            if 0 <= brand_idx < len(brands):
                brand = brands[brand_idx]
            else:
                brand = "generic"
        else:
            brand = "generic"
        
        test_ip_camera(ip, port, username, password, brand)
        
    elif choice == "3":
        # Test all webcams
        print("\n🔍 Scanning for webcams...")
        for i in range(3):
            print(f"\n{'='*60}")
            print(f"Testing webcam {i}...")
            print('='*60)
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Webcam {i} is available")
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"   Resolution: {width}x{height}")
                else:
                    print(f"⚠️  Webcam {i} opened but cannot read frames")
                cap.release()
            else:
                print(f"❌ Webcam {i} not available")
        
        # Ask which one to test
        test_id = input("\nEnter webcam ID to test (0-2): ").strip()
        if test_id and test_id.isdigit():
            test_webcam(int(test_id))
    
    else:
        print("❌ Invalid choice!")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "--webcam":
            camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            test_webcam(camera_id)
        elif sys.argv[1] == "--ip":
            if len(sys.argv) < 3:
                print("Usage: python test_real_camera.py --ip <ip> [port] [username] [password] [brand]")
                return
            ip = sys.argv[2]
            port = int(sys.argv[3]) if len(sys.argv) > 3 else 554
            username = sys.argv[4] if len(sys.argv) > 4 else "admin"
            password = sys.argv[5] if len(sys.argv) > 5 else ""
            brand = sys.argv[6] if len(sys.argv) > 6 else "generic"
            test_ip_camera(ip, port, username, password, brand)
        else:
            print("Usage:")
            print("  python test_real_camera.py --webcam [id]")
            print("  python test_real_camera.py --ip <ip> [port] [username] [password] [brand]")
    else:
        # Interactive mode
        interactive_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
