#!/usr/bin/env python3
"""
Camera IP Finder
Tool đơn giản để tìm IP camera trong mạng local
"""

import socket
import subprocess
import re
import platform
from typing import List, Tuple


def get_local_ip() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "192.168.1.1"


def get_network_range(ip: str) -> str:
    """Get network range from IP"""
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def scan_network_arp() -> List[Tuple[str, str]]:
    """
    Scan network using ARP
    Returns list of (IP, MAC) tuples
    """
    print("\n🔍 Scanning network with ARP...")
    
    devices = []
    
    try:
        # Get ARP table
        if platform.system() == "Windows":
            result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
        else:
            result = subprocess.run(['arp', '-n'], capture_output=True, text=True)
        
        output = result.stdout
        
        # Parse ARP output
        # Windows: 192.168.1.100    00-11-22-33-44-55     dynamic
        # Linux: 192.168.1.100 ether 00:11:22:33:44:55 C eth0
        
        if platform.system() == "Windows":
            pattern = r'(\d+\.\d+\.\d+\.\d+)\s+([\da-fA-F]{2}-[\da-fA-F]{2}-[\da-fA-F]{2}-[\da-fA-F]{2}-[\da-fA-F]{2}-[\da-fA-F]{2})'
        else:
            pattern = r'(\d+\.\d+\.\d+\.\d+)\s+\w+\s+([\da-fA-F]{2}:[\da-fA-F]{2}:[\da-fA-F]{2}:[\da-fA-F]{2}:[\da-fA-F]{2}:[\da-fA-F]{2})'
        
        matches = re.findall(pattern, output)
        
        for ip, mac in matches:
            devices.append((ip, mac))
        
        print(f"✅ Found {len(devices)} devices in ARP table")
        
    except Exception as e:
        print(f"❌ ARP scan error: {e}")
    
    return devices


def identify_camera_by_mac(mac: str) -> str:
    """Identify camera brand by MAC address (OUI)"""
    # Common camera manufacturer MAC prefixes
    mac_prefixes = {
        "00-12-12": "IMOU/Dahua",
        "00-18-82": "Hikvision",
        "BC-46-99": "TP-Link",
        "34-CE-00": "TP-Link Tapo",
        "78-11-DC": "Xiaomi",
        "34-3D-C4": "Xiaomi",
        "50-8A-06": "Xiaomi",
        "00-0F-E2": "Reolink",
        "EC-71-DB": "Reolink",
        "C0-56-27": "EZVIZ",
    }
    
    # Normalize MAC (remove : and -)
    mac_upper = mac.upper().replace(":", "-")
    prefix = mac_upper[:8]
    
    return mac_prefixes.get(prefix, "Unknown")


def test_rtsp_port(ip: str, port: int = 554, timeout: float = 1.0) -> bool:
    """Test if RTSP port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


def test_http_port(ip: str, port: int = 80, timeout: float = 1.0) -> bool:
    """Test if HTTP port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False


def find_cameras():
    """Find cameras in local network"""
    print("\n" + "="*70)
    print("🎥 CAMERA IP FINDER")
    print("="*70)
    
    # Get local network
    local_ip = get_local_ip()
    print(f"\n📡 Your IP: {local_ip}")
    print(f"📡 Network: {get_network_range(local_ip)}")
    
    # Scan network
    devices = scan_network_arp()
    
    if not devices:
        print("\n❌ No devices found!")
        print("\nTips:")
        print("  • Make sure cameras are powered on")
        print("  • Make sure cameras are connected to WiFi")
        print("  • Try pinging devices first to populate ARP table")
        print("  • Check your router's DHCP client list")
        return
    
    # Identify potential cameras
    print("\n" + "="*70)
    print("POTENTIAL CAMERAS FOUND")
    print("="*70)
    
    cameras_found = []
    
    for ip, mac in devices:
        brand = identify_camera_by_mac(mac)
        
        # Test RTSP port
        has_rtsp = test_rtsp_port(ip, 554, timeout=0.5)
        
        # Test HTTP port
        has_http = test_http_port(ip, 80, timeout=0.5)
        
        # If has RTSP or known camera brand, likely a camera
        is_likely_camera = (has_rtsp or brand != "Unknown")
        
        if is_likely_camera:
            cameras_found.append((ip, mac, brand, has_rtsp, has_http))
            
            status_rtsp = "✅" if has_rtsp else "❌"
            status_http = "✅" if has_http else "❌"
            
            print(f"\n📷 {ip}")
            print(f"   MAC: {mac}")
            print(f"   Brand: {brand}")
            print(f"   RTSP (554): {status_rtsp}")
            print(f"   HTTP (80): {status_http}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if cameras_found:
        print(f"\n✅ Found {len(cameras_found)} potential camera(s)")
        
        print("\n📝 Next Steps:")
        print("\n1. Test camera connection:")
        for ip, mac, brand, has_rtsp, has_http in cameras_found:
            brand_name = brand.split('/')[0].lower() if '/' in brand else brand.lower()
            print(f"   python test_real_camera.py --ip {ip} 554 admin <password> {brand_name}")
        
        print("\n2. Or use GUI:")
        print("   python gui_app.py")
        print("   → Tab '📹 Multi-Camera'")
        print("   → Add Camera")
        for ip, mac, brand, has_rtsp, has_http in cameras_found:
            print(f"   → IP: {ip}, Brand: {brand}")
        
        print("\n3. Default credentials to try:")
        print("   • Username: admin")
        print("   • Password: admin, 12345, <empty>, or password set in camera app")
        
    else:
        print("\n⚠️  No obvious cameras found")
        print("\nAll devices found:")
        for ip, mac in devices:
            print(f"   {ip:15} {mac:17} {identify_camera_by_mac(mac)}")
        
        print("\nTips:")
        print("  • Cameras may not respond to port scans")
        print("  • Check router's connected devices list")
        print("  • Use camera app to find IP address")
        print("  • Try manual IP input if you know the range")


def manual_test():
    """Manual test specific IP"""
    print("\n" + "="*70)
    print("MANUAL CAMERA TEST")
    print("="*70)
    
    ip = input("\nEnter camera IP address: ").strip()
    
    if not ip:
        print("❌ IP required!")
        return
    
    print(f"\n🔍 Testing {ip}...")
    
    # Test ping
    print(f"\n1️⃣  Testing ping...")
    if platform.system() == "Windows":
        result = subprocess.run(['ping', '-n', '1', '-w', '1000', ip], 
                              capture_output=True, text=True)
    else:
        result = subprocess.run(['ping', '-c', '1', '-W', '1', ip], 
                              capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ Ping successful")
    else:
        print(f"   ❌ Ping failed - device may be offline or blocking ICMP")
    
    # Test RTSP port
    print(f"\n2️⃣  Testing RTSP port (554)...")
    if test_rtsp_port(ip, 554, timeout=2.0):
        print(f"   ✅ RTSP port is open")
    else:
        print(f"   ❌ RTSP port is closed or filtered")
    
    # Test HTTP port
    print(f"\n3️⃣  Testing HTTP port (80)...")
    if test_http_port(ip, 80, timeout=2.0):
        print(f"   ✅ HTTP port is open")
    else:
        print(f"   ❌ HTTP port is closed")
    
    # Test other common ports
    print(f"\n4️⃣  Testing other common ports...")
    common_ports = {
        8000: "HTTP Alt",
        8080: "HTTP Alt",
        443: "HTTPS",
        37777: "Dahua",
        9000: "Camera Web"
    }
    
    for port, name in common_ports.items():
        if test_http_port(ip, port, timeout=1.0):
            print(f"   ✅ Port {port} ({name}) is open")
    
    # Suggest next steps
    print(f"\n📝 Next Steps:")
    print(f"\n1. Try with VLC Player:")
    print(f"   Media → Open Network Stream")
    print(f"   Try these URLs:")
    print(f"     rtsp://admin:password@{ip}:554/stream1")
    print(f"     rtsp://admin:password@{ip}:554/cam/realmonitor?channel=1&subtype=0")
    print(f"     http://{ip}")
    
    print(f"\n2. Test with our tool:")
    print(f"   python test_real_camera.py --ip {ip}")


def main():
    """Main menu"""
    print("\n" + "="*70)
    print("🎥 CAMERA IP FINDER - MENU")
    print("="*70)
    
    print("\n1. Auto scan network for cameras")
    print("2. Manual test specific IP")
    print("3. Show network info")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        find_cameras()
    elif choice == "2":
        manual_test()
    elif choice == "3":
        local_ip = get_local_ip()
        print(f"\n📡 Your IP: {local_ip}")
        print(f"📡 Network: {get_network_range(local_ip)}")
        print(f"📡 Platform: {platform.system()}")
    elif choice == "4":
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
