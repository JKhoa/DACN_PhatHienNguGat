# 🎥 **KIỂM TRA CÁC CÁCH KẾT NỐI CAMERA**

## ✅ **Tình trạng hiện tại:**

### **📹 Camera Types được hỗ trợ:**

#### **1. IP Camera:**
- ✅ **Hikvision**: `rtsp://user:pass@ip:port/Streaming/Channels/101`
- ✅ **Dahua**: `rtsp://user:pass@ip:port/cam/realmonitor?channel=1&subtype=0`
- ✅ **Ezviz**: `rtsp://user:pass@ip:port/h264/ch1/main/av_stream`
- ✅ **KBVision**: `rtsp://user:pass@ip:port/stream0`
- ✅ **Generic**: `rtsp://user:pass@ip:port/stream`

#### **2. Webcam:**
- ✅ **USB Webcam**: Device ID 0, 1, 2...
- ✅ **Direct capture**: cv2.VideoCapture(deviceId)

### **🔧 Camera Configuration:**

#### **IP Camera Fields:**
```typescript
{
  name: string,           // Tên camera
  brand: string,          // Hikvision, Dahua, Ezviz, KBVision
  ip: string,            // IP address (VD: 192.168.1.100)
  port: number,          // Port (mặc định: 554)
  username: string,      // Username (mặc định: admin)
  password: string,      // Password (mặc định: admin123)
  streamQuality: 'main' | 'sub'  // Main stream hoặc sub stream
}
```

#### **Webcam Fields:**
```typescript
{
  name: string,          // Tên camera
  deviceId: number,      // Device ID (0, 1, 2...)
}
```

### **🎯 Camera Dialog Features:**

#### **✅ Đã có:**
- ✅ **Camera Type Selection**: IP Camera / Webcam
- ✅ **Brand Selection**: Hikvision, Dahua, Ezviz, KBVision
- ✅ **IP Configuration**: IP, Port, Username, Password
- ✅ **Stream Quality**: Main / Sub stream
- ✅ **Device ID**: For webcam selection
- ✅ **Test Connection**: Kiểm tra kết nối
- ✅ **RTSP URL Generation**: Tự động tạo URL
- ✅ **Configuration**: YOLO model settings

#### **🔧 Configuration Options:**
- ✅ **Model**: yolo11n-pose.pt
- ✅ **Confidence**: 0.5
- ✅ **Strategy**: YOLO
- ✅ **Show FPS**: true
- ✅ **Show Overlay**: true
- ✅ **Max Queue Size**: 2

### **📱 UI Components:**

#### **✅ CameraDialog:**
- ✅ **Form fields**: Name, Type, Brand, IP, Port, Username, Password
- ✅ **Stream Quality**: Main/Sub selection
- ✅ **Device ID**: For webcam
- ✅ **Test Button**: Test connection
- ✅ **Save Button**: Save configuration
- ✅ **Cancel Button**: Close dialog

#### **✅ Toolbar:**
- ✅ **Add Button**: Mở CameraDialog
- ✅ **Delete Button**: Xóa camera đã chọn
- ✅ **Start All**: Khởi động tất cả camera
- ✅ **Stop All**: Dừng tất cả camera

#### **✅ CameraSidebar:**
- ✅ **Camera List**: Hiển thị danh sách camera
- ✅ **Status Indicators**: Online/Offline/Reconnecting
- ✅ **Search**: Tìm kiếm camera
- ✅ **Selection**: Chọn camera

#### **✅ CameraGrid:**
- ✅ **Grid Layout**: 1x1, 2x2, 3x3, 4x4
- ✅ **Video Feeds**: Hiển thị video streams
- ✅ **Overlays**: Student detection overlays
- ✅ **Performance**: FPS, Latency, Confidence

### **🚀 Backend Integration:**

#### **✅ Python Backend:**
- ✅ **CameraManager**: Quản lý camera
- ✅ **DrowsinessDetector**: YOLO detection
- ✅ **RTSP Support**: OpenCV VideoCapture
- ✅ **Webcam Support**: Device ID selection
- ✅ **Real-time Processing**: Frame-by-frame detection

#### **✅ Flask API:**
- ✅ **GET /api/cameras**: Lấy danh sách camera
- ✅ **POST /api/camera/add**: Thêm camera mới
- ✅ **POST /api/camera/{id}/start**: Khởi động camera
- ✅ **POST /api/camera/{id}/stop**: Dừng camera
- ✅ **DELETE /api/camera/{id}/remove**: Xóa camera
- ✅ **GET /api/system/stats**: Thống kê hệ thống

#### **✅ Electron Integration:**
- ✅ **IPC Handlers**: Giao tiếp với Python backend
- ✅ **Process Management**: Khởi động/dừng Python process
- ✅ **Error Handling**: Xử lý lỗi kết nối

### **📊 Test Cases:**

#### **✅ IP Camera Tests:**
1. **Hikvision Camera**:
   - IP: 192.168.1.100
   - Port: 554
   - Username: admin
   - Password: admin123
   - Expected URL: `rtsp://admin:admin123@192.168.1.100:554/Streaming/Channels/101`

2. **Dahua Camera**:
   - IP: 192.168.1.101
   - Port: 554
   - Username: admin
   - Password: admin123
   - Expected URL: `rtsp://admin:admin123@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0`

3. **Ezviz Camera**:
   - IP: 192.168.1.102
   - Port: 554
   - Username: admin
   - Password: admin123
   - Expected URL: `rtsp://admin:admin123@192.168.1.102:554/h264/ch1/main/av_stream`

#### **✅ Webcam Tests:**
1. **Device 0**: Default webcam
2. **Device 1**: Secondary webcam
3. **Device 2**: External webcam

### **🎉 Kết quả:**

#### **✅ Đã hoàn thành:**
- ✅ **IP Camera Support**: Hikvision, Dahua, Ezviz, KBVision
- ✅ **Webcam Support**: USB webcam với device ID
- ✅ **RTSP URL Generation**: Tự động tạo URL theo brand
- ✅ **Connection Testing**: Test kết nối trước khi lưu
- ✅ **Configuration Dialog**: Form đầy đủ các trường
- ✅ **Backend Integration**: Python + Flask + Electron
- ✅ **Real-time Detection**: YOLO model integration
- ✅ **Error Handling**: Xử lý lỗi kết nối
- ✅ **UI Components**: Dialog, Toolbar, Sidebar, Grid

#### **🔧 Cách sử dụng:**
1. **Nhấn "Thêm"** trong Toolbar
2. **Chọn loại camera**: IP Camera hoặc Webcam
3. **Nhập thông tin**:
   - IP Camera: Brand, IP, Port, Username, Password
   - Webcam: Device ID
4. **Nhấn "Test"** để kiểm tra kết nối
5. **Nhấn "Save"** để lưu camera
6. **Nhấn "Start"** để khởi động detection

---
**🎯 Tất cả các cách kết nối camera đã được kiểm tra và hoạt động tốt!**

