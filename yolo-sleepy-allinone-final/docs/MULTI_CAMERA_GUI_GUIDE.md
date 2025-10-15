# 📹 Multi-Camera GUI - Quick Start Guide

## 🎯 Overview

Multi-Camera Tab trong GUI cho phép bạn giám sát không giới hạn số lượng camera cùng lúc với YOLO detection, tất cả trong một giao diện đồ họa thân thiện.

## 🚀 Quick Start (3 bước)

### Bước 1: Mở GUI
```bash
python gui_app.py
```

### Bước 2: Chuyển sang Tab Multi-Camera
- Click vào tab **"📹 Multi-Camera"** ở bên phải

### Bước 3: Thêm Camera
1. Click **"➕ Add Camera"**
2. Nhập thông tin camera
3. Click **"▶️ Start All"**

## 📸 Thêm Camera

### Webcam
1. Chọn Type: **Webcam**
2. Nhập tên: "Laptop Camera"
3. Webcam ID: 0 (hoặc 1, 2...)
4. Check "Enabled"
5. Click **OK**

### IP Camera
1. Chọn Type: **IP**
2. Nhập tên: "Phòng học 101"
3. Chọn Brand: IMOU, Hikvision, Dahua, Tapo, v.v.
4. Nhập IP Address: `192.168.1.100`
5. Port: `554` (mặc định RTSP)
6. Username: `admin`
7. Password: `password123`
8. Stream Quality: **Main** (chất lượng cao) hoặc **Sub** (băng thông thấp)
9. Click **"Test Connection"** để kiểm tra
10. Click **OK**

## 🎮 Controls

### Button Controls
- **➕ Add Camera** - Thêm camera mới
- **✏️ Edit** - Chỉnh sửa camera đã chọn
- **🗑️ Remove** - Xóa camera đã chọn
- **📁 Load Config** - Load từ file YAML
- **💾 Save Config** - Lưu ra file YAML
- **▶️ Start All** - Bắt đầu tất cả camera
- **⏹️ Stop All** - Dừng tất cả camera

### Display Modes
- **Grid View** - Xem tất cả camera trong lưới (mosaic)
- **Single View** - Xem từng camera toàn màn hình (click camera trong list để chọn)

## 📋 Camera List

Camera list hiển thị:
- 🟢 **Connected** - Camera đang hoạt động
- 🟡 **Connecting** - Đang kết nối
- ⚪ **Disconnected** - Chưa kết nối
- 🔴 **Error** - Lỗi kết nối

## 💾 Save/Load Configuration

### Save Config
1. Thêm tất cả camera cần giám sát
2. Click **"💾 Save Config"**
3. Chọn vị trí và tên file (ví dụ: `classroom_cameras.yaml`)
4. File được lưu

### Load Config
1. Click **"📁 Load Config"**
2. Chọn file YAML đã lưu trước đó
3. Tất cả camera được load tự động

## 📊 Thông tin hiển thị

### Grid View
Mỗi ô camera hiển thị:
- Tên camera
- FPS hiện tại
- Detection boxes (nếu phát hiện người ngủ gật)

### Single View
Hiển thị đầy đủ:
- Tên camera
- FPS
- Số lượng detection

### Stats Panel (dưới camera list)
- **Total cameras** - Tổng số camera
- **Active cameras** - Số camera đang hoạt động

## 🎯 Use Cases

### 1. Giám sát phòng học (4 camera)
```
Camera 1: Phòng 101 - Hàng đầu
Camera 2: Phòng 101 - Hàng giữa
Camera 3: Phòng 102 - Toàn cảnh
Camera 4: Phòng 103 - Góc học
```
**Display:** Grid View để xem tất cả

### 2. Giám sát lái xe (2 camera)
```
Camera 1: Laptop webcam - Khuôn mặt
Camera 2: Phone IP camera - Góc rộng
```
**Display:** Single View để xem kỹ

### 3. Văn phòng (6 camera)
```
6 IP cameras ở các vị trí khác nhau
```
**Display:** Grid View, switch sang Single để xem chi tiết

## ⚙️ Performance Tips

### Giảm băng thông
- Chọn **Sub stream** thay vì Main stream
- Giảm số camera hiển thị đồng thời

### Tăng FPS
- Đóng các ứng dụng khác
- Sử dụng model nhẹ hơn (YOLOv11n thay vì YOLOv11m)

### Optimize cho nhiều camera
- Grid View hiệu quả hơn Single View
- Disable camera không cần thiết

## 🔧 Troubleshooting

### Camera không kết nối
1. ✅ Check IP address đúng chưa
2. ✅ Check username/password
3. ✅ Check port (thường là 554)
4. ✅ Camera và máy tính cùng mạng
5. ✅ Firewall không block RTSP
6. ✅ Click "Test Connection" để kiểm tra

### Camera status "Error"
- Đọc error message trong list
- Stop và Start lại camera
- Remove và Add lại với config mới

### FPS thấp
- Giảm số camera
- Chọn Sub stream
- Đóng các tab khác
- Tắt các app nặng khác

### Detection không chính xác
- Trong tab Settings, điều chỉnh Confidence threshold
- Switch sang model khác (YOLOv11 chính xác hơn YOLOv5)

## 🎨 Best Practices

1. **Test trước khi thêm nhiều camera**
   - Add 1 camera và test
   - Sau đó mới add thêm

2. **Đặt tên camera rõ ràng**
   - "Phòng 101 - Hàng đầu" thay vì "Camera 1"

3. **Save config thường xuyên**
   - Sau khi setup xong, save ngay

4. **Use Grid View cho monitoring**
   - Single View cho investigation chi tiết

5. **Disable camera không dùng**
   - Uncheck "Enabled" thay vì remove

## 📝 Example Configuration

File `classroom_cameras.yaml`:
```yaml
cameras:
  - name: "Phòng 101 - Góc trước"
    type: ip
    brand: hikvision
    ip: 192.168.1.100
    port: 554
    username: admin
    password: pass123
    stream_quality: main
    enabled: true
    
  - name: "Phòng 102 - Toàn cảnh"
    type: ip
    brand: dahua
    ip: 192.168.1.101
    port: 554
    username: admin
    password: pass456
    stream_quality: sub
    enabled: true
    
  - name: "Laptop Camera"
    type: webcam
    source: 0
    enabled: true
```

Load file này bằng **"📁 Load Config"**

## 🎓 Tips & Tricks

- **Keyboard**: Click vào camera trong list rồi dùng mũi tên lên/xuống để chuyển
- **Quick test**: Add webcam trước để test system
- **Network**: Dùng sub stream khi mạng chậm
- **Scale**: System đã test với 10+ cameras

## 📞 Support

Nếu gặp vấn đề:
1. Check docs/MULTI_CAMERA_GUIDE.md (chi tiết hơn)
2. Check docs/CAMERA_SUPPORT_EXTENDED.md (danh sách camera brands)
3. Test với standalone: `python multi_camera_app.py`

---

✅ **Ready to monitor!** Click **"➕ Add Camera"** để bắt đầu!
