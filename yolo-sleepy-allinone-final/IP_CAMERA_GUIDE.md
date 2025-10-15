# 📹 Hướng Dẫn Kết Nối IP Camera

## 🎯 Tổng Quan
Hệ thống hỗ trợ kết nối với các camera IP phổ biến trong trường học và gia đình như IMOU Ranger, Hikvision, Dahua qua giao thức RTSP.

## 🔧 Cài Đặt Camera IP

### 1. **Chuẩn Bị Camera**
- Đảm bảo camera đã kết nối WiFi/Ethernet
- Biết địa chỉ IP của camera (kiểm tra trong router hoặc app của camera)
- Có username/password để truy cập camera

### 2. **Tìm Địa Chỉ IP Camera**

#### Cách 1: Qua App Mobile
- **IMOU**: Mở app IMOU Life → Device Settings → Device Info
- **Hikvision**: Mở app Hik-Connect → Device Settings → Network
- **Dahua**: Mở app DMSS → Device Settings → Network Info

#### Cách 2: Qua Router
- Đăng nhập vào router (thường 192.168.1.1)
- Vào phần "Connected Devices" để xem IP của camera

#### Cách 3: Scan Network
```bash
# Windows
arp -a | findstr "192.168"

# Hoặc dùng công cụ IP Scanner
```

## 🚀 Sử Dụng IP Camera

### **Lệnh Cơ Bản**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.100 --username admin --password 123456
```

### **Các Tham Số IP Camera**

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `--ip-camera` | - | Kích hoạt chế độ IP camera |
| `--ip` | 192.168.1.100 | Địa chỉ IP của camera |
| `--port` | 554 | Port RTSP (thường là 554) |
| `--username` | admin | Tên đăng nhập camera |
| `--password` | - | Mật khẩu camera |
| `--camera-brand` | imou | Loại camera (imou/hikvision/dahua/generic) |
| `--stream-quality` | main | Chất lượng stream (main/sub) |
| `--rtsp-path` | auto | Đường dẫn RTSP tùy chỉnh |
| `--connection-timeout` | 10 | Timeout kết nối (giây) |

## 📱 Cấu Hình Theo Từng Loại Camera

### **IMOU Ranger Series**
```bash
# Stream chính (HD)
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --username admin --password 123456 \
  --camera-brand imou --stream-quality main

# Stream phụ (SD - tiết kiệm băng thông)
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --username admin --password 123456 \
  --camera-brand imou --stream-quality sub
```

**RTSP URLs được tạo:**
- Main stream: `rtsp://admin:123456@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0`
- Sub stream: `rtsp://admin:123456@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1`

### **Hikvision**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.101 \
  --username admin --password 12345 \
  --camera-brand hikvision --stream-quality main
```

**RTSP URLs:**
- Main stream: `rtsp://admin:12345@192.168.1.101:554/Streaming/Channels/101`
- Sub stream: `rtsp://admin:12345@192.168.1.101:554/Streaming/Channels/102`

### **Dahua**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.102 \
  --username admin --password abcd1234 \
  --camera-brand dahua --stream-quality main
```

**RTSP URLs:**
- Main stream: `rtsp://admin:abcd1234@192.168.1.102:554/cam/realmonitor?channel=1&subtype=0`
- Sub stream: `rtsp://admin:abcd1234@192.168.1.102:554/cam/realmonitor?channel=1&subtype=1`

### **Generic/Other Brands**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.103 \
  --username user --password pass123 \
  --camera-brand generic --rtsp-path "/your/custom/path"
```

## 🔍 Troubleshooting

### **Lỗi Thường Gặp**

#### **1. Không kết nối được camera**
```
❌ Không thể kết nối đến IP camera
```

**Giải pháp:**
- Kiểm tra IP address: `ping 192.168.1.100`
- Đảm bảo camera và máy tính cùng mạng
- Kiểm tra username/password
- Thử port khác: `--port 8554` hoặc `--port 80`

#### **2. Stream bị giật lag**
**Giải pháp:**
- Dùng sub stream: `--stream-quality sub`
- Giảm resolution: `--res 640x480`
- Kiểm tra băng thông mạng

#### **3. Lỗi authentication**
```
❌ Lỗi xác thực
```

**Giải pháp:**
- Kiểm tra username/password trong app camera
- Thử username mặc định: `admin`, `user`, `root`
- Reset camera về cài đặt gốc

#### **4. RTSP path không đúng**
**Giải pháp:**
- Thử brand khác: `--camera-brand generic`
- Dùng custom path: `--rtsp-path "/stream"`
- Tham khảo manual camera

### **Debug Mode**
```bash
# Hiển thị chi tiết kết nối
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --username admin --password 123456 --yolo-verbose
```

## 📋 Kiểm Tra Camera Trước Khi Sử Dụng

### **Test với VLC Media Player**
1. Mở VLC → Media → Open Network Stream
2. Nhập URL: `rtsp://admin:123456@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0`
3. Nếu thấy video → Camera hoạt động bình thường

### **Test với FFmpeg**
```bash
ffplay rtsp://admin:123456@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### **Test với OpenCV**
```python
import cv2

rtsp_url = "rtsp://admin:123456@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("✅ Camera hoạt động bình thường")
        cv2.imshow("Test", frame)
        cv2.waitKey(0)
    else:
        print("❌ Không đọc được frame")
else:
    print("❌ Không mở được camera")

cap.release()
cv2.destroyAllWindows()
```

## 🏫 Cấu Hình Cho Trường Học

### **Setup Classroom Monitoring**
```bash
# Camera góc lớp học
python standalone_app.py --ip-camera \
  --ip 192.168.1.100 --username teacher --password school123 \
  --camera-brand imou --stream-quality main \
  --max-people 30 --conf 0.4

# Camera hành lang
python standalone_app.py --ip-camera \
  --ip 192.168.1.101 --username admin --password hall456 \
  --camera-brand hikvision --stream-quality sub \
  --max-people 50 --conf 0.3
```

### **Multi-Camera Script**
```python
# multi_camera_monitor.py
import subprocess
import threading

cameras = [
    {"ip": "192.168.1.100", "name": "Classroom_A", "brand": "imou"},
    {"ip": "192.168.1.101", "name": "Classroom_B", "brand": "hikvision"},
    {"ip": "192.168.1.102", "name": "Library", "brand": "dahua"}
]

def monitor_camera(camera):
    cmd = f"""python standalone_app.py --ip-camera 
    --ip {camera['ip']} --username admin --password 123456 
    --camera-brand {camera['brand']} --cli"""
    subprocess.run(cmd.split())

for camera in cameras:
    thread = threading.Thread(target=monitor_camera, args=(camera,))
    thread.start()
```

## 🏠 Cấu Hình Cho Gia Đình

### **Home Security Setup**
```bash
# Camera phòng khách
python standalone_app.py --ip-camera \
  --ip 192.168.1.50 --username family --password home789 \
  --camera-brand imou --stream-quality main \
  --max-people 5

# Camera phòng ngủ trẻ em
python standalone_app.py --ip-camera \
  --ip 192.168.1.51 --username parent --password safe123 \
  --camera-brand imou --stream-quality sub \
  --max-people 2 --conf 0.6
```

## 🔧 Tối Ưu Hiệu Năng

### **Cho Mạng Chậm**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --stream-quality sub --res 640x480 --conf 0.7
```

### **Cho Xử Lý Nhiều Camera**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --imgsz 640 --max-people 10 --conf 0.5
```

### **Tiết Kiệm CPU**
```bash
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --model-version v5 --imgsz 416 --conf 0.6
```

## 📞 Hỗ Trợ Camera Brands

| Brand | Tested | Main Stream | Sub Stream | Notes |
|-------|--------|-------------|------------|--------|
| **IMOU** | ✅ | `/cam/realmonitor?channel=1&subtype=0` | `/cam/realmonitor?channel=1&subtype=1` | Ranger, Bullet series |
| **Hikvision** | ✅ | `/Streaming/Channels/101` | `/Streaming/Channels/102` | DS-2CD series |
| **Dahua** | ✅ | `/cam/realmonitor?channel=1&subtype=0` | `/cam/realmonitor?channel=1&subtype=1` | IPC-HFW series |
| **TP-Link Tapo** | 🔄 | `/stream1` | `/stream2` | Testing |
| **Xiaomi** | 🔄 | `/live` | `/live_sub` | Testing |

## 🎯 Best Practices

1. **Luôn test kết nối trước với VLC**
2. **Dùng sub stream cho monitoring lâu dài**
3. **Cấu hình firewall cho port 554**
4. **Đặt IP tĩnh cho camera**
5. **Backup cấu hình camera thường xuyên**
6. **Sử dụng password mạnh**
7. **Update firmware camera định kỳ**

## 📈 Performance Tips

- **Main stream**: Chất lượng cao, tiêu tốn CPU/bandwidth nhiều
- **Sub stream**: Chất lượng thấp, tiết kiệm tài nguyên
- **Confidence threshold**: Tăng lên 0.6-0.7 để giảm false positive
- **Max people**: Giới hạn số người detect để tăng tốc
- **Image size**: Giảm từ 960 xuống 640 hoặc 416 để tăng FPS