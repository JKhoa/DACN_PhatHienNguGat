# 📡 Hướng Dẫn Kết Nối Camera qua WiFi

## 🎯 Tổng Quan

Có **3 cách** kết nối camera WiFi vào hệ thống:

1. **Cloud API** (như app điện thoại) - Phức tạp, cần developer account
2. **Local IP + RTSP** (đơn giản nhất) - ✅ **Khuyến nghị**
3. **QR Code** - Cần thư viện đặc biệt

---

## ✅ CÁCH 1: Kết Nối Local IP (KHUYẾN NGHỊ)

### Tại sao nên dùng cách này?
- ✅ Đơn giản nhất
- ✅ Không cần Cloud API
- ✅ Nhanh hơn (không qua internet)
- ✅ Ổn định hơn
- ✅ Bảo mật hơn (local network)

### Các Bước:

#### Bước 1: Tìm IP Camera

**Cách 1: Dùng App Camera**
- Mở app (IMOU Life, Tapo, Mi Home, v.v.)
- Vào Settings camera → Device Info
- Tìm **IP Address** (ví dụ: 192.168.1.100)

**Cách 2: Vào Router**
- Đăng nhập router (192.168.1.1 hoặc 192.168.0.1)
- Xem danh sách thiết bị kết nối
- Tìm camera theo tên/MAC address

**Cách 3: Dùng Tool Quét**
- Windows: Download **Advanced IP Scanner**
- Quét mạng → Tìm camera

**Cách 4: Command Line**
```bash
# Windows PowerShell
arp -a

# Tìm IP có tên liên quan camera
```

#### Bước 2: Test Kết Nối

```bash
python test_real_camera.py --ip 192.168.1.100
```

Hoặc interactive:
```bash
python test_real_camera.py
# Chọn option 2 (IP Camera)
# Nhập IP, username, password
```

#### Bước 3: Thêm vào GUI

1. Mở GUI: `python gui_app.py`
2. Tab **"📹 Multi-Camera"**
3. Click **"➕ Add Camera"**
4. Chọn **Loại: ip**
5. Chọn **Brand**: IMOU, Tapo, Xiaomi, v.v.
6. Nhập **IP Address**: 192.168.1.100
7. **Port**: 554
8. **Username**: admin
9. **Password**: password của camera
10. Click **"Test Connection"**
11. Nếu OK → Click **OK**
12. Click **"▶️ Start All"**

---

## 🌐 CÁCH 2: Kết Nối Cloud API (Nâng Cao)

### 📱 IMOU Camera (IMOU Ranger, Cruiser, v.v.)

#### Yêu Cầu:
- ✅ IMOU Developer Account
- ✅ App ID & App Secret
- ✅ Device ID (từ camera)

#### Các Bước:

**1. Đăng Ký Developer Account**
- Truy cập: https://open.imou.com/
- Đăng ký tài khoản
- Tạo application mới
- Lấy **App ID** và **App Secret**

**2. Lấy Device ID**
- Mở app **IMOU Life**
- Chọn camera
- Settings → Device Info
- Copy **Device ID** (ví dụ: A12345678)

**3. Kết Nối**
```bash
python wifi_camera_connector.py
# Chọn option 1 (IMOU)
# Nhập Device ID, App ID, App Secret
```

**4. Sử Dụng Stream URL**
- Script sẽ trả về stream URL
- Dùng URL này như RTSP URL bình thường

#### Code Example:
```python
from wifi_camera_connector import WiFiCameraManager

manager = WiFiCameraManager()
stream_url = manager.connect_imou_device(
    device_id="A12345678",
    app_id="your_app_id",
    app_secret="your_app_secret"
)

if stream_url:
    print(f"Stream URL: {stream_url}")
    # Use with OpenCV
    import cv2
    cap = cv2.VideoCapture(stream_url)
```

---

### 📱 TP-Link Tapo Camera

#### Yêu Cầu:
- ✅ Tapo app account (email + password)
- ✅ Device ID từ app

#### Các Bước:

**1. Lấy Device ID**
- Mở app **Tapo**
- Chọn camera
- Settings → Device Info
- Copy **Device ID**

**2. Kết Nối**
```bash
python wifi_camera_connector.py
# Chọn option 2 (Tapo)
# Nhập Device ID, email, password
```

**Note**: Tapo Cloud API có giới hạn, nên dùng **Local IP** tốt hơn!

---

### 📱 Xiaomi/Mijia Camera

#### Khó khăn:
- ❌ Xiaomi Cloud API rất phức tạp
- ❌ Cần OAuth 2.0 với nhiều bước
- ❌ Token expires thường xuyên

#### Khuyến nghị:
✅ Dùng **Local IP** thay vì Cloud API

---

## 📷 CÁCH 3: Quét QR Code

### Yêu Cầu:
```bash
pip install pyzbar
# Windows: Download ZBar từ http://zbar.sourceforge.net/
```

### Sử Dụng:

```bash
python wifi_camera_connector.py
# Chọn option 3 (Scan QR Code)
# Hướng webcam vào QR code trên camera
```

### QR Code Chứa Gì?
- Device ID
- Serial Number
- Initial WiFi credentials
- Setup token

### Giới Hạn:
- ⚠️ QR code thường chỉ để setup ban đầu
- ⚠️ Không phải lúc nào cũng có stream URL
- ⚠️ Vẫn cần thêm bước xác thực

---

## 🎯 SO SÁNH CÁC PHƯƠNG PHÁP

| Phương Pháp | Độ Khó | Tốc Độ | Ổn Định | Khuyến Nghị |
|-------------|--------|--------|---------|-------------|
| **Local IP + RTSP** | ⭐ Dễ | ⚡ Nhanh | ✅ Cao | ✅ **BEST** |
| **Cloud API** | ⭐⭐⭐ Khó | 🐢 Chậm | ⚠️ Trung bình | ❌ Không nên |
| **QR Code** | ⭐⭐ Trung bình | ⚡ Nhanh | ✅ Cao | ⚠️ Tùy trường hợp |

---

## 📖 HƯỚNG DẪN CHI TIẾT TỪNG BRAND

### 🎥 IMOU Camera

**Thông tin RTSP:**
- Port: 554
- Path: `/cam/realmonitor?channel=1&subtype=0`
- Username: admin (mặc định)
- Password: Mật khẩu đặt trong app

**Cách tìm IP:**
1. App IMOU Life → Camera
2. Settings → Device Info → IP Address

**RTSP URL:**
```
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

**Test:**
```bash
python test_real_camera.py --ip 192.168.1.100 554 admin your_password imou
```

---

### 🎥 TP-Link Tapo Camera

**Thông tin RTSP:**
- Port: 554
- Path: `/stream1` (HD) hoặc `/stream2` (SD)
- Username: admin hoặc account email
- Password: Mật khẩu camera (trong app)

**Cách tìm IP:**
1. App Tapo → Camera
2. Settings → Camera Info → IP Address

**RTSP URL:**
```
rtsp://admin:password@192.168.1.101:554/stream1
```

**Test:**
```bash
python test_real_camera.py --ip 192.168.1.101 554 admin your_password tapo
```

---

### 🎥 Xiaomi/Mijia Camera

**Thông tin RTSP:**
- Port: 554
- Path: `/live/ch00_0` (HD) hoặc `/live/ch00_1` (SD)
- Username: admin
- Password: Tìm trong Mi Home app

**Lưu ý**: Một số model Xiaomi cần **enable RTSP** trong app!

**RTSP URL:**
```
rtsp://admin:password@192.168.1.102:554/live/ch00_0
```

---

### 🎥 Hikvision Camera

**Thông tin RTSP:**
- Port: 554
- Path: `/Streaming/Channels/101` (HD) hoặc `/Streaming/Channels/102` (SD)
- Username: admin (mặc định)
- Password: Mật khẩu admin

**RTSP URL:**
```
rtsp://admin:password@192.168.1.103:554/Streaming/Channels/101
```

---

### 🎥 Dahua Camera

**Thông tin RTSP:**
- Port: 554
- Path: `/cam/realmonitor?channel=1&subtype=0`
- Username: admin
- Password: Mật khẩu admin

**RTSP URL:**
```
rtsp://admin:password@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0
```

---

## 🔧 Troubleshooting

### ❌ "Cannot connect to camera"

**Check:**
1. ✅ Camera và máy tính cùng mạng WiFi
2. ✅ IP address đúng
3. ✅ Port 554 không bị firewall block
4. ✅ Username/password đúng
5. ✅ RTSP enabled trên camera

**Test ping:**
```bash
ping 192.168.1.100
```

**Test RTSP với VLC:**
1. Mở VLC Media Player
2. Media → Open Network Stream
3. Paste RTSP URL
4. Play → Nếu thấy video = OK

---

### ❌ "Connection timeout"

**Giải pháp:**
1. Check firewall: Tắt tạm để test
2. Router settings: Forward port 554
3. Camera settings: Enable RTSP stream
4. Try different path:
   - `/stream1`
   - `/live`
   - `/h264`

---

### ❌ "Authentication failed"

**Giải pháp:**
1. Check username/password trong app
2. Try `admin`/`admin`
3. Try `admin`/`<empty password>`
4. Reset camera về factory settings

---

## 💡 TIPS & BEST PRACTICES

### ✅ Tip 1: Dùng Static IP
- Vào router
- Đặt IP tĩnh cho camera
- Camera sẽ luôn có cùng IP

### ✅ Tip 2: Sub Stream cho Nhiều Camera
- Main stream: HD (tốn băng thông)
- Sub stream: SD (tiết kiệm)
- Dùng sub khi monitor nhiều camera

### ✅ Tip 3: Test từng camera trước
```bash
# Test từng camera một
python test_real_camera.py --ip 192.168.1.100
python test_real_camera.py --ip 192.168.1.101
```

### ✅ Tip 4: Lưu Config
Sau khi test OK, save config:
```yaml
cameras:
  - name: "Phòng 101"
    type: ip
    brand: imou
    ip: 192.168.1.100
    username: admin
    password: your_password
    enabled: true
```

### ✅ Tip 5: Network Performance
- Camera và PC cùng WiFi 5GHz tốt hơn 2.4GHz
- Dùng LAN cable nếu có thể
- Router quality matters!

---

## 📝 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  QUICK WIFI CAMERA CONNECTION                   │
├─────────────────────────────────────────────────┤
│  1. Find IP:                                    │
│     • Check app → Device Info → IP              │
│     • Check router → Connected devices          │
│                                                  │
│  2. Test:                                       │
│     python test_real_camera.py --ip <ip>       │
│                                                  │
│  3. Add to GUI:                                 │
│     python gui_app.py                           │
│     → Tab "📹 Multi-Camera"                    │
│     → Add Camera → Type: IP                     │
│     → Enter IP, username, password              │
│     → Test Connection → OK                      │
│                                                  │
│  4. Start:                                      │
│     Click "▶️ Start All"                       │
└─────────────────────────────────────────────────┘
```

---

## 🆘 Need Help?

1. **Test camera trước:**
   ```bash
   python test_real_camera.py
   ```

2. **Test với VLC Player:**
   - Open Network Stream
   - Paste RTSP URL
   - Check xem có video không

3. **Check docs:**
   - Camera manual
   - Brand-specific RTSP paths
   - Port forwarding guide

4. **Alternative:**
   - Dùng camera app share screen
   - Dùng OBS Virtual Camera
   - Dùng IP Webcam app (Android)

---

✅ **Khuyến nghị cuối: Dùng Local IP + RTSP là đơn giản và hiệu quả nhất!**
