# 📹 Hướng Dẫn Giám Sát Đa Camera

> 🎯 **Hỗ trợ không giới hạn số lượng camera** với YOLO detection real-time

## 🌟 Tính Năng Chính

### ✅ **Unlimited Cameras**
- Không giới hạn số lượng camera
- Hỗ trợ cả webcam và IP camera
- 15+ thương hiệu IP camera

### 🖥️ **Multiple Display Modes**
- **Grid View**: Mosaic tất cả camera
- **Single View**: Xem từng camera riêng lẻ
- **CLI Mode**: Monitoring không GUI (server mode)

### ⚡ **Performance Optimized**
- Multi-threading cho mỗi camera
- Dynamic frame processing (stride)
- FPS limiting per camera
- Automatic reconnection

### 🎨 **Smart Layout**
- Dynamic grid calculation (2x2, 3x3, 4x4, ...)
- Auto-resize cameras to fit screen
- Real-time detection overlay
- Status indicators per camera

---

## 🚀 Quick Start

### **1. Setup Configuration**

Sao chép file mẫu:
```bash
cd yolo-sleepy-allinone-final
copy cameras.sample.yaml cameras.yaml
```

Chỉnh sửa `cameras.yaml`:
```yaml
cameras:
  - name: "Office Camera"
    type: ip
    brand: imou
    ip: "192.168.1.100"
    username: "admin"
    password: "123456"
    enabled: true
  
  - name: "Entrance"
    type: ip
    brand: hikvision
    ip: "192.168.1.101"
    username: "admin"
    password: "abcd"
    enabled: true
```

### **2. Run Multi-Camera App**

**Basic Usage:**
```bash
python multi_camera_app.py --config cameras.yaml
```

**With Options:**
```bash
# Grid view với 4 camera
python multi_camera_app.py --config cameras.yaml --view grid

# Single view mode
python multi_camera_app.py --config cameras.yaml --view single

# CLI mode (no GUI, for servers)
python multi_camera_app.py --config cameras.yaml --mode cli

# Performance tuning (process every 2nd frame)
python multi_camera_app.py --config cameras.yaml --stride 2 --max-fps 15
```

---

## 🎮 Controls & Keyboard Shortcuts

### **Grid View Mode**
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit application |
| `G` | Switch to Grid view |
| `S` | Switch to Single view |
| `H` | Switch to HUD mode |

### **Single View Mode**
| Key | Action |
|-----|--------|
| `N` | Next camera |
| `P` | Previous camera |
| `G` | Back to Grid view |

---

## 📊 Configuration Options

### **Command Line Arguments**

```bash
python multi_camera_app.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | yolov11_1000ep_best.pt | YOLO model path |
| `--conf` | 0.5 | Confidence threshold (0.0-1.0) |
| `--stride` | 1 | Process every N frames |
| `--max-fps` | 30 | Maximum FPS per camera |
| `--config` | cameras.yaml | Camera config file |
| `--add-webcam` | - | Add default webcam |
| `--mode` | gui | Display mode (gui/cli) |
| `--width` | 1920 | Display width |
| `--height` | 1080 | Display height |
| `--view` | grid | Initial view (grid/single) |

### **Camera Configuration (YAML)**

```yaml
cameras:
  - name: "Camera Name"        # Display name
    type: webcam or ip         # Camera type
    source: 0                  # For webcam: camera index
    brand: imou               # For IP: camera brand
    ip: "192.168.1.100"       # For IP: camera IP
    port: 554                 # For IP: RTSP port
    username: "admin"         # For IP: username
    password: "password"      # For IP: password
    stream_quality: main      # main or sub
    enabled: true             # Enable/disable camera
```

---

## 💡 Usage Examples

### **Example 1: School Monitoring (4 Classrooms)**

**cameras.yaml:**
```yaml
cameras:
  - name: "Classroom 1"
    type: ip
    brand: imou
    ip: "192.168.1.101"
    username: "admin"
    password: "school123"
    enabled: true
  
  - name: "Classroom 2"
    type: ip
    brand: imou
    ip: "192.168.1.102"
    username: "admin"
    password: "school123"
    enabled: true
  
  - name: "Classroom 3"
    type: ip
    brand: hikvision
    ip: "192.168.1.103"
    username: "admin"
    password: "school123"
    enabled: true
  
  - name: "Library"
    type: ip
    brand: hikvision
    ip: "192.168.1.104"
    username: "admin"
    password: "school123"
    enabled: true
```

**Run:**
```bash
python multi_camera_app.py --config cameras.yaml --view grid --stride 2
```

### **Example 2: Transport Company (10+ Vehicles)**

**cameras.yaml:**
```yaml
cameras:
  - name: "Bus 01"
    type: ip
    brand: tapo
    ip: "192.168.1.201"
    stream_quality: sub  # Use sub for bandwidth
    enabled: true
  
  - name: "Bus 02"
    type: ip
    brand: tapo
    ip: "192.168.1.202"
    stream_quality: sub
    enabled: true
  
  # ... add more buses
  
  - name: "Bus 10"
    type: ip
    brand: xiaomi
    ip: "192.168.1.210"
    stream_quality: sub
    enabled: true
```

**Run:**
```bash
# Use lower FPS and stride for many cameras
python multi_camera_app.py --config cameras.yaml --max-fps 10 --stride 3
```

### **Example 3: Factory Safety (6 Zones)**

**cameras.yaml:**
```yaml
cameras:
  - name: "Production Line A"
    type: ip
    brand: hikvision
    ip: "192.168.10.101"
    enabled: true
  
  - name: "Production Line B"
    type: ip
    brand: hikvision
    ip: "192.168.10.102"
    enabled: true
  
  - name: "Assembly Area"
    type: ip
    brand: dahua
    ip: "192.168.10.103"
    enabled: true
  
  - name: "Quality Control"
    type: ip
    brand: axis
    ip: "192.168.10.104"
    enabled: true
  
  - name: "Shipping"
    type: ip
    brand: dahua
    ip: "192.168.10.105"
    enabled: true
  
  - name: "Break Room"
    type: ip
    brand: imou
    ip: "192.168.10.106"
    enabled: true
```

**Run:**
```bash
python multi_camera_app.py --config cameras.yaml --width 2560 --height 1440
```

### **Example 4: Server Monitoring (CLI Mode)**

Chạy trên server không có màn hình:
```bash
python multi_camera_app.py --config cameras.yaml --mode cli
```

Output:
```
================================================================================
MULTI-CAMERA MONITORING SYSTEM
================================================================================

🟢 Camera 1: Office Camera
   Status: connected
   FPS: 28.5
   Detections: 2
   Frames: 1245

🟢 Camera 2: Entrance
   Status: connected
   FPS: 29.1
   Detections: 0
   Frames: 1267

🟡 Camera 3: Parking
   Status: connecting

================================================================================
Active Cameras: 2/3 | Total FPS: 57.6
================================================================================
```

---

## ⚙️ Performance Tuning

### **For Many Cameras (10+)**

```bash
# Reduce processing load
python multi_camera_app.py --config cameras.yaml \
  --stride 3 \          # Process every 3rd frame
  --max-fps 10 \        # Limit to 10 FPS per camera
  --conf 0.6            # Higher confidence = less processing
```

### **For High Accuracy**

```bash
# Maximum quality
python multi_camera_app.py --config cameras.yaml \
  --stride 1 \          # Process every frame
  --max-fps 30 \        # Full 30 FPS
  --conf 0.4            # Lower confidence = more detections
```

### **For Remote Cameras (Slow Network)**

```yaml
# In cameras.yaml, use 'sub' stream quality
cameras:
  - name: "Remote Camera"
    stream_quality: sub  # Lower bandwidth
```

---

## 📐 Grid Layout Examples

### **Automatic Layout Calculation**

| Cameras | Grid Layout | Example |
|---------|-------------|---------|
| 1 | 1x1 | Single fullscreen |
| 2 | 1x2 | Side by side |
| 3-4 | 2x2 | 2 rows, 2 columns |
| 5-6 | 2x3 | 2 rows, 3 columns |
| 7-9 | 3x3 | 3 rows, 3 columns |
| 10-12 | 3x4 | 3 rows, 4 columns |
| 13-16 | 4x4 | 4 rows, 4 columns |
| 17+ | Dynamic | Auto-calculated |

---

## 🔧 Troubleshooting

### **Problem: High CPU Usage**

**Solutions:**
1. Increase stride: `--stride 3`
2. Reduce FPS: `--max-fps 10`
3. Use 'sub' stream quality
4. Reduce number of active cameras

### **Problem: Cameras Not Connecting**

**Check:**
1. IP address correct
2. Username/password correct
3. Camera is online (ping test)
4. RTSP port open (usually 554)
5. Camera brand setting correct

**Test individual camera:**
```bash
python test_ip_camera.py --ip 192.168.1.100 --username admin --password 123456 --brand imou
```

### **Problem: Low FPS**

**Solutions:**
1. Use faster YOLO model (YOLOv5 instead of v11)
2. Increase stride
3. Reduce camera resolution
4. Use 'sub' stream quality
5. Upgrade hardware (GPU recommended)

### **Problem: Camera Disconnects**

**App handles this automatically:**
- Auto-reconnection every 5 seconds
- Status indicator shows connection state
- Error messages displayed in GUI/CLI

---

## 🎯 Best Practices

### **Network Setup**
1. ✅ Use wired ethernet for IP cameras (not WiFi)
2. ✅ All cameras on same subnet for simplicity
3. ✅ Static IP addresses for cameras
4. ✅ Sufficient network bandwidth (5-10 Mbps per camera)

### **Performance**
1. ✅ Start with 2-4 cameras, add gradually
2. ✅ Use 'sub' stream for >6 cameras
3. ✅ Match stride/FPS to your needs
4. ✅ Monitor CPU/RAM usage

### **Security**
1. ✅ Change default camera passwords
2. ✅ Use separate network for cameras (VLAN)
3. ✅ Keep credentials in secure `cameras.yaml`
4. ✅ Don't commit `cameras.yaml` to git

### **Reliability**
1. ✅ Enable only needed cameras
2. ✅ Test each camera individually first
3. ✅ Use quality network switches
4. ✅ Plan for power backup (UPS)

---

## 📊 System Requirements

### **Minimum (2-4 cameras)**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8 GB
- Network: 100 Mbps
- OS: Windows 10, Ubuntu 20.04, macOS 11+

### **Recommended (10+ cameras)**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16 GB
- GPU: NVIDIA GTX 1660 or better
- Network: 1 Gbps
- Storage: SSD for better performance

### **Enterprise (20+ cameras)**
- CPU: Intel i9 / AMD Ryzen 9
- RAM: 32 GB
- GPU: NVIDIA RTX 3060 or better
- Network: 2.5 Gbps or 10 Gbps
- Storage: NVMe SSD

---

## 🔗 Related Files

- **Main App**: `multi_camera_app.py`
- **Config Sample**: `cameras.sample.yaml`
- **Your Config**: `cameras.yaml` (create from sample)
- **Single Camera**: `standalone_app.py`
- **Camera Test**: `test_ip_camera.py`

---

## 📞 Support

### **Common Issues:**
- Camera not connecting → Check IP, credentials, brand
- High CPU → Increase stride, reduce FPS
- Low quality → Use 'main' stream, reduce stride
- Network issues → Check bandwidth, use ethernet

### **More Help:**
- 📖 Main README: [../README.md](../README.md)
- 📷 Camera Support: [CAMERA_SUPPORT_EXTENDED.md](CAMERA_SUPPORT_EXTENDED.md)
- 🐛 Report Issues: [GitHub Issues](https://github.com/JKhoa/DACN_PhatHienNguGat/issues)

---

**🎉 Bây giờ bạn có thể giám sát không giới hạn camera với YOLO detection!**