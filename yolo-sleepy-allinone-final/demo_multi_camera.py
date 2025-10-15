#!/usr/bin/env python3
"""
Demo script để test nhiều camera IP cùng lúc
Multi-Camera IP Testing Demo
"""

import cv2
import time
import threading
from typing import List, Dict, Any

# Cấu hình camera mẫu cho các thương hiệu khác nhau
CAMERA_CONFIGS = [
    {
        "name": "IMOU Ranger",
        "ip": "192.168.1.100",
        "username": "admin",
        "password": "123456",
        "brand": "imou",
        "quality": "main"
    },
    {
        "name": "Hikvision DS-2CD",
        "ip": "192.168.1.101", 
        "username": "admin",
        "password": "hikpass",
        "brand": "hikvision",
        "quality": "main"
    },
    {
        "name": "TP-Link Tapo C200",
        "ip": "192.168.1.102",
        "username": "admin", 
        "password": "tapopass",
        "brand": "tapo",
        "quality": "main"
    },
    {
        "name": "Xiaomi Mi Home",
        "ip": "192.168.1.103",
        "username": "admin",
        "password": "xiaomipass", 
        "brand": "xiaomi",
        "quality": "main"
    },
    {
        "name": "Reolink RLC-410",
        "ip": "192.168.1.104",
        "username": "admin",
        "password": "reopass",
        "brand": "reolink", 
        "quality": "main"
    }
]

def generate_rtsp_url(config: Dict[str, Any]) -> str:
    """Generate RTSP URL for camera config"""
    
    rtsp_paths = {
        "imou": "/cam/realmonitor?channel=1&subtype=0",
        "hikvision": "/Streaming/Channels/101", 
        "tapo": "/stream1",
        "xiaomi": "/live/ch00_0",
        "reolink": "/h264Preview_01_main",
        "foscam": "/videoMain",
        "axis": "/axis-media/media.amp?videocodec=h264",
        "bosch": "/rtsp_tunnel?h264&unicast&line=1",
        "sony": "/media/video1",
        "panasonic": "/MediaInput/stream_1",
        "vivotek": "/live.sdp",
        "dlink": "/play1.sdp",
        "arlo": "/rtspstream/video",
        "generic": "/stream1"
    }
    
    path = rtsp_paths.get(config["brand"], "/stream1")
    
    return f"rtsp://{config['username']}:{config['password']}@{config['ip']}:554{path}"

def test_single_camera(config: Dict[str, Any]) -> Dict[str, Any]:
    """Test single camera connection"""
    
    result = {
        "name": config["name"],
        "brand": config["brand"],
        "ip": config["ip"],
        "success": False,
        "error": "",
        "resolution": "",
        "fps": 0,
        "connection_time": 0
    }
    
    rtsp_url = generate_rtsp_url(config)
    
    print(f"🔗 Testing {config['name']} ({config['brand']})...")
    
    start_time = time.time()
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                result["success"] = True
                result["resolution"] = f"{width}x{height}"
                result["fps"] = fps
                result["connection_time"] = time.time() - start_time
                
                print(f"✅ {config['name']}: {width}x{height} @ {fps:.1f}fps")
            else:
                result["error"] = "Cannot read frame"
                print(f"❌ {config['name']}: Cannot read frame")
        else:
            result["error"] = "Cannot open connection"
            print(f"❌ {config['name']}: Cannot open connection")
            
        cap.release()
        
    except Exception as e:
        result["error"] = str(e)
        print(f"❌ {config['name']}: {str(e)}")
    
    result["connection_time"] = time.time() - start_time
    return result

def test_cameras_parallel(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Test multiple cameras in parallel"""
    
    results = []
    threads = []
    
    def worker(config, results_list):
        result = test_single_camera(config)
        results_list.append(result)
    
    # Start all threads
    for config in configs:
        thread = threading.Thread(target=worker, args=(config, results))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return results

def display_results(results: List[Dict[str, Any]]):
    """Display test results in a nice format"""
    
    print("\n" + "="*80)
    print("🎥 CAMERA IP TEST RESULTS")
    print("="*80)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    print()
    
    if successful:
        print("🟢 SUCCESSFUL CAMERAS:")
        print("-" * 50)
        for result in successful:
            print(f"📹 {result['name']} ({result['brand']})")
            print(f"   IP: {result['ip']}")
            print(f"   Resolution: {result['resolution']}")
            print(f"   FPS: {result['fps']:.1f}")
            print(f"   Connection time: {result['connection_time']:.2f}s")
            print()
    
    if failed:
        print("🔴 FAILED CAMERAS:")
        print("-" * 50)
        for result in failed:
            print(f"📹 {result['name']} ({result['brand']})")
            print(f"   IP: {result['ip']}")
            print(f"   Error: {result['error']}")
            print(f"   Connection time: {result['connection_time']:.2f}s")
            print()
    
    print("💡 USAGE EXAMPLES:")
    print("-" * 50)
    for result in successful:
        print(f"# {result['name']}")
        print(f"python standalone_app.py --ip-camera --ip {result['ip']} \\")
        print(f"  --username admin --password [your_password] --camera-brand {result['brand']}")
        print()

def main():
    print("🎬 Multi-Camera IP Test Demo")
    print("Testing multiple camera brands simultaneously...")
    print(f"📊 Testing {len(CAMERA_CONFIGS)} camera configurations")
    print()
    
    # Update IP addresses for your local network
    print("🔧 CONFIGURATION:")
    print("Make sure to update IP addresses in CAMERA_CONFIGS for your network")
    print("Current test IPs: 192.168.1.100-104")
    print()
    
    input("Press Enter to start testing (or Ctrl+C to exit)...")
    
    start_time = time.time()
    
    # Test cameras in parallel for speed
    results = test_cameras_parallel(CAMERA_CONFIGS)
    
    total_time = time.time() - start_time
    
    # Display results
    display_results(results)
    
    print(f"⏱️  Total test time: {total_time:.2f}s")
    print("\n🎯 Integration Examples:")
    print("For successful cameras, you can now use them with the main app!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")