#!/usr/bin/env python3
"""
WiFi Camera Connection Module
Kết nối camera qua WiFi như app điện thoại (QR Code, Device ID, Cloud API)
"""

import cv2
import requests
import json
import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class WiFiCameraConfig:
    """Configuration for WiFi camera connection"""
    device_id: str  # Device ID hoặc Serial Number
    brand: str  # imou, tapo, xiaomi, ezviz, v.v.
    cloud_username: str = ""  # Username đăng nhập app
    cloud_password: str = ""  # Password đăng nhập app
    app_id: str = ""  # App ID (nếu có)
    app_secret: str = ""  # App Secret (nếu có)


class IMOUCloudConnector:
    """
    IMOU Cloud API Connector
    Kết nối camera IMOU qua Cloud API như app IMOU Life
    """
    
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_url = "https://openapi.easy4ip.com:443/openapi"
        self.access_token = None
        
    def get_access_token(self) -> Optional[str]:
        """Get access token from IMOU Cloud"""
        try:
            # IMOU sử dụng OAuth 2.0
            url = f"{self.api_url}/getAccessToken"
            
            params = {
                "appId": self.app_id,
                "appSecret": self.app_secret,
                "system": "windows",
                "ver": "1.0"
            }
            
            response = requests.post(url, json=params)
            data = response.json()
            
            if data.get("code") == "0":
                self.access_token = data["result"]["accessToken"]
                return self.access_token
            else:
                print(f"❌ IMOU Auth failed: {data.get('msg')}")
                return None
                
        except Exception as e:
            print(f"❌ IMOU Auth error: {e}")
            return None
    
    def get_device_list(self) -> list:
        """Get list of devices linked to account"""
        if not self.access_token:
            self.get_access_token()
        
        try:
            url = f"{self.api_url}/deviceBaseList"
            params = {
                "token": self.access_token
            }
            
            response = requests.post(url, json=params)
            data = response.json()
            
            if data.get("code") == "0":
                return data["result"]["deviceList"]
            else:
                return []
                
        except Exception as e:
            print(f"❌ Get device list error: {e}")
            return []
    
    def get_live_stream_url(self, device_id: str, channel: int = 0) -> Optional[str]:
        """Get RTSP live stream URL for device"""
        if not self.access_token:
            self.get_access_token()
        
        try:
            url = f"{self.api_url}/liveAddress"
            params = {
                "token": self.access_token,
                "deviceId": device_id,
                "channelId": str(channel),
                "streamId": 0  # 0=HD, 1=SD
            }
            
            response = requests.post(url, json=params)
            data = response.json()
            
            if data.get("code") == "0":
                return data["result"]["hls"]  # Hoặc rtsp, rtmp
            else:
                print(f"❌ Get stream URL failed: {data.get('msg')}")
                return None
                
        except Exception as e:
            print(f"❌ Get stream URL error: {e}")
            return None


class TapoCloudConnector:
    """
    TP-Link Tapo Cloud Connector
    Kết nối camera Tapo qua Cloud như app Tapo
    """
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.token = None
        self.api_url = "https://wap.tplinkcloud.com"
    
    def login(self) -> bool:
        """Login to Tapo Cloud"""
        try:
            url = f"{self.api_url}/"
            
            # Tapo sử dụng MD5 hash cho password
            password_hash = hashlib.md5(self.password.encode()).hexdigest()
            
            payload = {
                "method": "login",
                "params": {
                    "appType": "Tapo_Android",
                    "cloudUserName": self.username,
                    "cloudPassword": password_hash,
                    "terminalUUID": "88-00-00-00-00-00"
                }
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if data.get("error_code") == 0:
                self.token = data["result"]["token"]
                return True
            else:
                print(f"❌ Tapo login failed: {data.get('msg')}")
                return False
                
        except Exception as e:
            print(f"❌ Tapo login error: {e}")
            return False
    
    def get_device_list(self) -> list:
        """Get Tapo devices"""
        if not self.token:
            self.login()
        
        try:
            url = f"{self.api_url}/?token={self.token}"
            payload = {
                "method": "getDeviceList"
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if data.get("error_code") == 0:
                return data["result"]["deviceList"]
            else:
                return []
                
        except Exception as e:
            print(f"❌ Get Tapo devices error: {e}")
            return []


class XiaomiCloudConnector:
    """
    Xiaomi/Mijia Cloud Connector
    Kết nối camera Xiaomi qua Mi Home Cloud
    """
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.token = None
        self.user_id = None
        self.api_url = "https://api.io.mi.com/app"
    
    def login(self) -> bool:
        """Login to Xiaomi Cloud"""
        try:
            # Xiaomi cloud authentication is complex
            # Simplified version here
            print("⚠️  Xiaomi Cloud API requires complex OAuth flow")
            print("   Consider using local IP connection instead")
            return False
        except Exception as e:
            print(f"❌ Xiaomi login error: {e}")
            return False


class WiFiCameraManager:
    """
    Manager for WiFi camera connections
    Quản lý kết nối camera WiFi qua Cloud API
    """
    
    def __init__(self):
        self.connectors = {}
    
    def connect_imou_device(self, device_id: str, app_id: str, app_secret: str) -> Optional[str]:
        """
        Kết nối IMOU camera qua Device ID
        
        Args:
            device_id: Device ID của camera (ví dụ: A12345678)
            app_id: IMOU App ID
            app_secret: IMOU App Secret
            
        Returns:
            RTSP URL nếu thành công, None nếu thất bại
        """
        print(f"\n🔌 Connecting to IMOU camera: {device_id}")
        
        connector = IMOUCloudConnector(app_id, app_secret)
        
        # Get access token
        if not connector.get_access_token():
            return None
        
        # Get stream URL
        stream_url = connector.get_live_stream_url(device_id)
        
        if stream_url:
            print(f"✅ IMOU camera connected!")
            print(f"   Stream URL: {stream_url}")
            return stream_url
        else:
            return None
    
    def connect_tapo_device(self, device_id: str, username: str, password: str) -> Optional[str]:
        """
        Kết nối Tapo camera qua Device ID
        
        Args:
            device_id: Device ID của camera
            username: Tapo account username
            password: Tapo account password
            
        Returns:
            Stream URL nếu thành công
        """
        print(f"\n🔌 Connecting to Tapo camera: {device_id}")
        
        connector = TapoCloudConnector(username, password)
        
        if not connector.login():
            return None
        
        # Get devices
        devices = connector.get_device_list()
        
        # Find device by ID
        for device in devices:
            if device.get("deviceId") == device_id:
                # Tapo typically uses local IP + RTSP
                device_ip = device.get("deviceIp")
                if device_ip:
                    rtsp_url = f"rtsp://{username}:{password}@{device_ip}:554/stream1"
                    print(f"✅ Tapo camera found!")
                    print(f"   IP: {device_ip}")
                    print(f"   Stream URL: rtsp://{username}:****@{device_ip}:554/stream1")
                    return rtsp_url
        
        print(f"❌ Tapo camera {device_id} not found in account")
        return None
    
    def scan_qr_code(self) -> Optional[Dict[str, str]]:
        """
        Quét QR code từ camera để lấy thông tin kết nối
        
        Returns:
            Dictionary chứa thông tin camera
        """
        print("\n📷 Starting QR Code scanner...")
        print("   Point your webcam at the camera's QR code")
        print("   Press 'q' to quit")
        
        try:
            import cv2
            from pyzbar import pyzbar  # pip install pyzbar
            
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Decode QR codes
                qr_codes = pyzbar.decode(frame)
                
                for qr in qr_codes:
                    qr_data = qr.data.decode('utf-8')
                    print(f"\n✅ QR Code detected!")
                    print(f"   Data: {qr_data}")
                    
                    # Parse QR data (format varies by brand)
                    camera_info = self._parse_qr_data(qr_data)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    return camera_info
                
                # Draw rectangles around QR codes
                for qr in qr_codes:
                    points = qr.polygon
                    if len(points) == 4:
                        pts = [(p.x, p.y) for p in points]
                        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 3)
                
                cv2.putText(frame, "Scan QR Code - Press Q to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("QR Code Scanner", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except ImportError:
            print("❌ pyzbar not installed!")
            print("   Install: pip install pyzbar")
            print("   Note: May also need to install ZBar:")
            print("     Windows: Download from http://zbar.sourceforge.net/")
        except Exception as e:
            print(f"❌ QR scan error: {e}")
        
        return None
    
    def _parse_qr_data(self, qr_data: str) -> Dict[str, str]:
        """Parse QR code data to extract camera info"""
        try:
            # Try JSON format
            return json.loads(qr_data)
        except:
            # Try URL format
            if "://" in qr_data:
                # Extract info from URL
                parts = qr_data.split("?")
                if len(parts) > 1:
                    params = {}
                    for param in parts[1].split("&"):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            params[key] = value
                    return params
            
            # Return raw data
            return {"data": qr_data}


def demo_wifi_connection():
    """Demo WiFi camera connection"""
    print("\n" + "="*60)
    print("📡 WIFI CAMERA CONNECTION DEMO")
    print("="*60)
    
    print("\nSelect connection method:")
    print("1. IMOU Camera (Device ID + App credentials)")
    print("2. Tapo Camera (Device ID + Account)")
    print("3. Scan QR Code")
    print("4. Manual Cloud API setup")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    manager = WiFiCameraManager()
    
    if choice == "1":
        # IMOU
        print("\n" + "="*60)
        print("IMOU CAMERA SETUP")
        print("="*60)
        print("\n⚠️  You need:")
        print("   1. Device ID (from camera label or IMOU Life app)")
        print("   2. IMOU Developer App ID & Secret")
        print("   3. Register at: https://open.imou.com/")
        
        device_id = input("\nDevice ID: ").strip()
        app_id = input("App ID: ").strip()
        app_secret = input("App Secret: ").strip()
        
        stream_url = manager.connect_imou_device(device_id, app_id, app_secret)
        
        if stream_url:
            print(f"\n✅ Success! You can now use this URL:")
            print(f"   {stream_url}")
    
    elif choice == "2":
        # Tapo
        print("\n" + "="*60)
        print("TAPO CAMERA SETUP")
        print("="*60)
        
        device_id = input("\nDevice ID (from Tapo app): ").strip()
        username = input("Tapo account email: ").strip()
        password = input("Tapo account password: ").strip()
        
        stream_url = manager.connect_tapo_device(device_id, username, password)
        
        if stream_url:
            print(f"\n✅ Success! You can now use this URL:")
            print(f"   {stream_url}")
    
    elif choice == "3":
        # QR Code
        camera_info = manager.scan_qr_code()
        if camera_info:
            print("\n✅ Camera info extracted:")
            for key, value in camera_info.items():
                print(f"   {key}: {value}")
    
    elif choice == "4":
        # Manual
        print("\n" + "="*60)
        print("MANUAL CLOUD API SETUP GUIDE")
        print("="*60)
        print("\nFor different brands:")
        print("\n📱 IMOU:")
        print("   1. Register: https://open.imou.com/")
        print("   2. Create app to get App ID & Secret")
        print("   3. Use Device ID from camera")
        
        print("\n📱 Tapo:")
        print("   1. Use your Tapo app account")
        print("   2. Get Device ID from app")
        print("   3. Cloud API may be limited")
        
        print("\n📱 Xiaomi:")
        print("   1. Use Mi Home app")
        print("   2. Enable developer mode")
        print("   3. Get device token")
        
        print("\n💡 Alternative: Use Local IP Connection")
        print("   Most WiFi cameras also support RTSP on local network")
        print("   Check camera IP in router or app")
        print("   Use: python test_real_camera.py --ip <camera_ip>")


if __name__ == "__main__":
    demo_wifi_connection()
