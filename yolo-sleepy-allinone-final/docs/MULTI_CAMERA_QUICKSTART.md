# 🚀 QUICK START - Multi-Camera Monitoring

## 3 Bước Để Bắt Đầu

### **Bước 1: Setup Configuration (2 phút)**

```bash
# Sao chép file mẫu
copy cameras.sample.yaml cameras.yaml

# Mở và chỉnh sửa cameras.yaml
notepad cameras.yaml
```

**Ví dụ cấu hình đơn giản:**
```yaml
cameras:
  # Webcam laptop
  - name: "My Webcam"
    type: webcam
    source: 0
    enabled: true
  
  # Camera IP IMOU
  - name: "Office Camera"
    type: ip
    brand: imou
    ip: "192.168.1.100"
    username: "admin"
    password: "123456"
    enabled: true
```

### **Bước 2: Run Application (1 lệnh)**

```bash
python multi_camera_app.py --config cameras.yaml
```

### **Bước 3: Xem và Điều Khiển**

**Keyboard controls:**
- `Q` - Thoát
- `G` - Grid view (xem tất cả)
- `S` - Single view (xem từng camera)
- `N` / `P` - Next/Previous camera

---

## 🎯 Use Cases

### **📚 Trường Học (4 phòng học)**
```bash
# 1. Setup cameras.yaml với 4 camera
# 2. Run:
python multi_camera_app.py --config cameras.yaml --view grid

# Result: 2x2 grid hiển thị 4 phòng học
```

### **🚗 Công ty Vận Tải (10 xe)**
```bash
# 1. Setup cameras.yaml với 10 camera
# 2. Run với optimization:
python multi_camera_app.py --config cameras.yaml --stride 2 --max-fps 15

# Result: 3x4 grid với performance tối ưu
```

### **🏭 Nhà Máy (6 khu vực)**
```bash
# 1. Setup cameras.yaml với 6 camera Hikvision/Dahua
# 2. Run:
python multi_camera_app.py --config cameras.yaml --width 2560 --height 1440

# Result: 2x3 grid màn hình lớn
```

### **💻 Server Monitoring (CLI mode)**
```bash
# Chạy không cần GUI trên server
python multi_camera_app.py --config cameras.yaml --mode cli

# Result: Real-time stats in terminal
```

---

## ⚙️ Performance Tips

### **Nhiều Camera (>10)**
```bash
--stride 3 --max-fps 10
# Process every 3rd frame, 10 FPS per camera
```

### **Chất Lượng Cao**
```bash
--stride 1 --max-fps 30
# Every frame, 30 FPS
```

### **Mạng Chậm**
```yaml
# Trong cameras.yaml:
stream_quality: sub  # Thay vì main
```

---

## 🔧 Troubleshooting 1-Minute Fixes

### **Camera không kết nối?**
```bash
# Test camera riêng lẻ:
python test_ip_camera.py --ip 192.168.1.100 --username admin --password 123456 --brand imou
```

### **CPU quá cao?**
```bash
# Giảm tải:
python multi_camera_app.py --config cameras.yaml --stride 3 --max-fps 10
```

### **Không thấy gì trên màn hình?**
```bash
# Check cameras.yaml - đảm bảo enabled: true
# Check IP address - phải đúng
# Check network - ping camera trước
```

---

## 📊 Grid Layouts

| Cameras | Layout | Command |
|---------|--------|---------|
| 1-2 | 1x2 | Auto |
| 3-4 | 2x2 | Auto |
| 5-6 | 2x3 | Auto |
| 7-9 | 3x3 | Auto |
| 10-12 | 3x4 | Auto |
| 13-16 | 4x4 | Auto |
| 20+ | Dynamic | Auto |

---

## 💡 Common Configurations

### **Mix Webcam + IP Camera**
```yaml
cameras:
  - name: "Laptop Webcam"
    type: webcam
    source: 0
    enabled: true
  
  - name: "IP Camera 1"
    type: ip
    brand: imou
    ip: "192.168.1.100"
    username: "admin"
    password: "pass1"
    enabled: true
  
  - name: "IP Camera 2"
    type: ip
    brand: hikvision
    ip: "192.168.1.101"
    username: "admin"
    password: "pass2"
    enabled: true
```

### **Multiple Brands**
```yaml
cameras:
  - name: "IMOU Office"
    type: ip
    brand: imou
    ip: "192.168.1.100"
    enabled: true
  
  - name: "Hikvision Entrance"
    type: ip
    brand: hikvision
    ip: "192.168.1.101"
    enabled: true
  
  - name: "Tapo Living Room"
    type: ip
    brand: tapo
    ip: "192.168.1.102"
    enabled: true
  
  - name: "Xiaomi Bedroom"
    type: ip
    brand: xiaomi
    ip: "192.168.1.103"
    enabled: true
```

---

## 🎓 Next Steps

1. **Đọc full guide**: [docs/MULTI_CAMERA_GUIDE.md](MULTI_CAMERA_GUIDE.md)
2. **Camera brands**: [CAMERA_SUPPORT_EXTENDED.md](CAMERA_SUPPORT_EXTENDED.md)
3. **Advanced config**: Xem `cameras.sample.yaml`

---

**🎉 Bây giờ bạn đã sẵn sàng giám sát nhiều camera!**

Questions? Check [MULTI_CAMERA_GUIDE.md](MULTI_CAMERA_GUIDE.md) for detailed help.