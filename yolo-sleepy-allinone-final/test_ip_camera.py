#!/usr/bin/env python3
"""
Test script for IP Camera connection
Kiểm tra kết nối IP Camera trước khi sử dụng trong ứng dụng chính
"""

import cv2
import time
import argparse

def test_camera_connection(rtsp_url, timeout=10):
    """Test IP camera connection"""
    print(f"🔗 Testing camera connection: {rtsp_url}")
    print(f"⏱️  Timeout: {timeout}s")
    
    start_time = time.time()
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
    except:
        pass
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Cannot read frame from camera")
        cap.release()
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    elapsed = time.time() - start_time
    
    print("✅ Camera connection successful!")
    print(f"📏 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps:.1f}")
    print(f"⏱️  Connection time: {elapsed:.2f}s")
    
    # Show frame for 3 seconds
    cv2.imshow("Camera Test", frame)
    print("📺 Displaying frame for 3 seconds... (Press any key to close)")
    cv2.waitKey(3000)
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def generate_rtsp_url(ip, port, username, password, brand, quality):
    """Generate RTSP URL for different camera brands"""
    
    rtsp_paths = {
        # IMOU & Dahua cameras
        "imou": {
            "main": "/cam/realmonitor?channel=1&subtype=0",
            "sub": "/cam/realmonitor?channel=1&subtype=1"
        },
        "dahua": {
            "main": "/cam/realmonitor?channel=1&subtype=0",
            "sub": "/cam/realmonitor?channel=1&subtype=1"
        },
        
        # Hikvision cameras
        "hikvision": {
            "main": "/Streaming/Channels/101",
            "sub": "/Streaming/Channels/102"
        },
        
        # TP-Link Tapo cameras
        "tplink": {
            "main": "/stream1",
            "sub": "/stream2"
        },
        "tapo": {
            "main": "/stream1",
            "sub": "/stream2"
        },
        
        # Xiaomi cameras
        "xiaomi": {
            "main": "/live/ch00_0",
            "sub": "/live/ch00_1"
        },
        "mijia": {
            "main": "/live/ch00_0",
            "sub": "/live/ch00_1"
        },
        
        # Reolink cameras
        "reolink": {
            "main": "/h264Preview_01_main",
            "sub": "/h264Preview_01_sub"
        },
        
        # Foscam cameras
        "foscam": {
            "main": "/videoMain",
            "sub": "/videoSub"
        },
        
        # Axis cameras
        "axis": {
            "main": "/axis-media/media.amp?videocodec=h264",
            "sub": "/axis-media/media.amp?videocodec=h264&resolution=320x240"
        },
        
        # Bosch cameras
        "bosch": {
            "main": "/rtsp_tunnel?h264&unicast&line=1",
            "sub": "/rtsp_tunnel?h264&unicast&line=2"
        },
        
        # Sony cameras
        "sony": {
            "main": "/media/video1",
            "sub": "/media/video2"
        },
        
        # Panasonic cameras
        "panasonic": {
            "main": "/MediaInput/stream_1",
            "sub": "/MediaInput/stream_2"
        },
        
        # Vivotek cameras
        "vivotek": {
            "main": "/live.sdp",
            "sub": "/live2.sdp"
        },
        
        # D-Link cameras
        "dlink": {
            "main": "/play1.sdp",
            "sub": "/play2.sdp"
        },
        
        # Netgear Arlo cameras
        "netgear": {
            "main": "/rtspstream/video",
            "sub": "/rtspstream/video2"
        },
        "arlo": {
            "main": "/rtspstream/video",
            "sub": "/rtspstream/video2"
        },
        
        # Generic/Unknown cameras
        "generic": {
            "main": "/stream1",
            "sub": "/stream2"
        },
        
        # ONVIF compatible cameras
        "onvif": {
            "main": "/onvif1",
            "sub": "/onvif2"
        },
        
        # Standard MJPEG cameras
        "standard": {
            "main": "/video.mjpg",
            "sub": "/video2.mjpg"
        }
    }
    
    brand_lower = brand.lower()
    if brand_lower in rtsp_paths and quality in rtsp_paths[brand_lower]:
        path = rtsp_paths[brand_lower][quality]
    else:
        # Fallback to generic
        path = rtsp_paths["generic"].get(quality, rtsp_paths["generic"]["main"])
    
    if username and password:
        return f"rtsp://{username}:{password}@{ip}:{port}{path}"
    else:
        return f"rtsp://{ip}:{port}{path}"

def main():
    parser = argparse.ArgumentParser(description="Test IP Camera Connection")
    parser.add_argument("--ip", default="192.168.1.100", help="Camera IP address")
    parser.add_argument("--port", type=int, default=554, help="Camera port")
    parser.add_argument("--username", default="admin", help="Camera username")
    parser.add_argument("--password", default="", help="Camera password")
    parser.add_argument("--brand", choices=["imou", "hikvision", "dahua", "tplink", "tapo", 
                                           "xiaomi", "mijia", "reolink", "foscam", "axis", 
                                           "bosch", "sony", "panasonic", "vivotek", "dlink", 
                                           "netgear", "arlo", "generic", "onvif", "standard"], 
                       default="imou", help="Camera brand")
    parser.add_argument("--quality", choices=["main", "sub"], default="main", 
                       help="Stream quality")
    parser.add_argument("--timeout", type=int, default=10, help="Connection timeout")
    parser.add_argument("--rtsp-url", help="Direct RTSP URL (overrides other options)")
    
    args = parser.parse_args()
    
    print("🎥 IP Camera Connection Test")
    print("=" * 40)
    
    if args.rtsp_url:
        rtsp_url = args.rtsp_url
        print(f"📡 Using direct RTSP URL: {rtsp_url}")
    else:
        rtsp_url = generate_rtsp_url(
            args.ip, args.port, args.username, args.password, 
            args.brand, args.quality
        )
        
        print(f"📍 Camera Info:")
        print(f"   IP: {args.ip}:{args.port}")
        print(f"   Brand: {args.brand}")
        print(f"   Quality: {args.quality}")
        print(f"   Username: {args.username}")
        print(f"   Password: {'***' if args.password else '(none)'}")
        print(f"📡 Generated RTSP URL: {rtsp_url}")
    
    print()
    
    # Test connection
    success = test_camera_connection(rtsp_url, args.timeout)
    
    if success:
        print("\n🎉 Camera test completed successfully!")
        print("💡 You can now use this camera with:")
        print(f"   python standalone_app.py --ip-camera --ip {args.ip} \\")
        print(f"     --username {args.username} --password {args.password} \\")
        print(f"     --camera-brand {args.brand} --stream-quality {args.quality}")
    else:
        print("\n❌ Camera test failed!")
        print("🔧 Troubleshooting tips:")
        print("   1. Check IP address and network connection")
        print("   2. Verify username and password")
        print("   3. Try different port (8554, 80, 8080)")
        print("   4. Test with VLC: Media → Open Network Stream")
        print(f"   5. Try URL: {rtsp_url}")

if __name__ == "__main__":
    main()