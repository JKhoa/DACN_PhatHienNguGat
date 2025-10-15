# 📷 Hỗ Trợ Camera IP Mở Rộng - 15+ Thương Hiệu

> 🎯 **Hệ thống hỗ trợ đa dạng camera IP** từ gia đình đến doanh nghiệp

## 🌟 Danh Sách Camera Được Hỗ Trợ Hoàn Toàn

### 🏠 **Camera Gia Đình & Văn Phòng Nhỏ**

#### **IMOU (Dahua Ecosystem)**
- **Models**: Ranger 2, Ranger Pro, Cruiser, Cell Pro
- **Code**: `imou` hoặc `dahua`
- **RTSP Path**: `/cam/realmonitor?channel=1&subtype=0`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.100 --username admin --password 123456 --camera-brand imou
```

#### **TP-Link Tapo Series**
- **Models**: Tapo C200, C210, C310, C320WS
- **Code**: `tplink` hoặc `tapo`
- **RTSP Path**: `/stream1` (main), `/stream2` (sub)
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.101 --username admin --password yourpass --camera-brand tapo
```

#### **Xiaomi Mi Home Security**
- **Models**: Mi Home Security Camera 360°, Dafang, Xiaofang
- **Code**: `xiaomi` hoặc `mijia`
- **RTSP Path**: `/live/ch00_0` (main), `/live/ch00_1` (sub)
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.102 --username admin --password xiaomipass --camera-brand xiaomi
```

#### **Reolink Series**
- **Models**: RLC-410, RLC-520, Argus 2, Argus 3 Pro
- **Code**: `reolink`
- **RTSP Path**: `/h264Preview_01_main`, `/h264Preview_01_sub`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.103 --username admin --password reopass --camera-brand reolink
```

#### **Foscam Series**
- **Models**: FI9821P, R2M, R4M, E1
- **Code**: `foscam`
- **RTSP Path**: `/videoMain`, `/videoSub`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.104 --username admin --password foscampass --camera-brand foscam
```

### 🏢 **Camera Chuyên Nghiệp & Doanh Nghiệp**

#### **Hikvision DS Series**
- **Models**: DS-2CD2085, DS-2CD2383, DS-2CD2T47, DS-2CD2H85
- **Code**: `hikvision`
- **RTSP Path**: `/Streaming/Channels/101`, `/Streaming/Channels/102`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.105 --username admin --password hikpass --camera-brand hikvision
```

#### **Dahua Professional**
- **Models**: IPC-HDW4831EM, IPC-HFW4831E, IPC-HDBW4831E
- **Code**: `dahua`
- **RTSP Path**: `/cam/realmonitor?channel=1&subtype=0`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.106 --username admin --password dahuapass --camera-brand dahua
```

#### **Axis Professional**
- **Models**: P3367, M3045, P1435, M2025
- **Code**: `axis`
- **RTSP Path**: `/axis-media/media.amp?videocodec=h264`
- **Port mặc định**: 554
- **Username mặc định**: root

```bash
python standalone_app.py --ip-camera --ip 192.168.1.107 --username root --password axispass --camera-brand axis
```

#### **Bosch Security**
- **Models**: DINION IP 4000, FLEXIDOME IP 5000
- **Code**: `bosch`
- **RTSP Path**: `/rtsp_tunnel?h264&unicast&line=1`
- **Port mặc định**: 554
- **Username mặc định**: service

```bash
python standalone_app.py --ip-camera --ip 192.168.1.108 --username service --password boschpass --camera-brand bosch
```

#### **Sony Professional**
- **Models**: SNC-CH135, SNC-DH135, SNC-EB630
- **Code**: `sony`
- **RTSP Path**: `/media/video1`, `/media/video2`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.109 --username admin --password sonypass --camera-brand sony
```

#### **Panasonic i-PRO**
- **Models**: WV-SFV130, WV-SFN130, WV-S1111
- **Code**: `panasonic`
- **RTSP Path**: `/MediaInput/stream_1`, `/MediaInput/stream_2`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.110 --username admin --password panapass --camera-brand panasonic
```

#### **Vivotek Professional**
- **Models**: IB9365-HT, FD9369-HV, IP9165-HP
- **Code**: `vivotek`
- **RTSP Path**: `/live.sdp`, `/live2.sdp`
- **Port mặc định**: 554
- **Username mặc định**: root

```bash
python standalone_app.py --ip-camera --ip 192.168.1.111 --username root --password vivopass --camera-brand vivotek
```

### 🌐 **Camera Khác & Generic**

#### **D-Link DCS Series**
- **Models**: DCS-2630L, DCS-8300LH, DCS-8526LH
- **Code**: `dlink`
- **RTSP Path**: `/play1.sdp`, `/play2.sdp`
- **Port mặc định**: 554
- **Username mặc định**: admin

```bash
python standalone_app.py --ip-camera --ip 192.168.1.112 --username admin --password dlinkpass --camera-brand dlink
```

#### **Netgear Arlo (qua Base Station)**
- **Models**: Arlo Pro, Arlo Pro 2, Arlo Ultra
- **Code**: `netgear` hoặc `arlo`
- **RTSP Path**: `/rtspstream/video`
- **Port mặc định**: 554
- **Lưu ý**: Cần Arlo Base Station hỗ trợ RTSP

```bash
python standalone_app.py --ip-camera --ip 192.168.1.113 --username admin --password arlopass --camera-brand arlo
```

#### **ONVIF Compatible Cameras**
- **Models**: Bất kỳ camera hỗ trợ ONVIF
- **Code**: `onvif`
- **RTSP Path**: `/onvif1`, `/onvif2`
- **Port mặc định**: 554

```bash
python standalone_app.py --ip-camera --ip 192.168.1.114 --username admin --password onvifpass --camera-brand onvif
```

#### **Generic RTSP Cameras**
- **Models**: Camera không thuộc thương hiệu trên
- **Code**: `generic`
- **RTSP Path**: `/stream1`, `/stream2`
- **Port mặc định**: 554

```bash
python standalone_app.py --ip-camera --ip 192.168.1.115 --username admin --password genericpass --camera-brand generic
```

#### **Standard MJPEG Cameras**
- **Models**: Camera cũ sử dụng MJPEG
- **Code**: `standard`
- **RTSP Path**: `/video.mjpg`, `/video2.mjpg`
- **Port mặc định**: 554

```bash
python standalone_app.py --ip-camera --ip 192.168.1.116 --username admin --password mjpegpass --camera-brand standard
```

## 🔧 Sử Dụng Custom RTSP Path

Nếu camera của bạn không thuộc các thương hiệu trên, bạn có thể sử dụng custom RTSP path:

```bash
python standalone_app.py --ip-camera --ip 192.168.1.100 --username admin --password yourpass --rtsp-path "/your/custom/path"
```

## 🚀 Test Kết Nối Trước Khi Sử Dụng

Luôn test kết nối camera trước:

```bash
python test_ip_camera.py --ip 192.168.1.100 --username admin --password yourpass --camera-brand your_brand
```

## 📋 Bảng Tóm Tắt Nhanh

| Brand | Code | Default Port | Default User | Common Models |
|-------|------|--------------|--------------|---------------|
| IMOU | `imou` | 554 | admin | Ranger, Cruiser |
| Hikvision | `hikvision` | 554 | admin | DS-2CD series |
| Dahua | `dahua` | 554 | admin | IPC series |
| TP-Link | `tplink` | 554 | admin | Tapo C200/C210 |
| Xiaomi | `xiaomi` | 554 | admin | Mi Home Security |
| Reolink | `reolink` | 554 | admin | RLC series |
| Foscam | `foscam` | 554 | admin | FI/R series |
| Axis | `axis` | 554 | root | P/M series |
| Bosch | `bosch` | 554 | service | DINION/FLEXIDOME |
| Sony | `sony` | 554 | admin | SNC series |
| Panasonic | `panasonic` | 554 | admin | WV series |
| Vivotek | `vivotek` | 554 | root | IP series |
| D-Link | `dlink` | 554 | admin | DCS series |
| Netgear | `arlo` | 554 | admin | Arlo series |
| Generic | `generic` | 554 | admin | Any RTSP camera |

## 🎯 Khuyến Nghị Sử Dụng

### 🏠 **Cho Gia Đình**
- **Tốt nhất**: IMOU Ranger (dễ cài đặt, ổn định)
- **Thay thế**: TP-Link Tapo, Xiaomi Mi Home
- **Ngân sách thấp**: Foscam, Generic RTSP

### 🏢 **Cho Trường Học/Doanh Nghiệp**
- **Chuyên nghiệp**: Hikvision, Dahua
- **Cao cấp**: Axis, Bosch, Sony
- **Cân bằng**: Vivotek, Panasonic

### ⚙️ **Lưu Ý Kỹ Thuật**
- Port 554 là chuẩn RTSP, một số camera có thể dùng port khác
- Username/password mặc định nên được thay đổi vì lý do bảo mật
- Stream quality "main" cho chất lượng cao, "sub" cho tiết kiệm băng thông
- Timeout 10 giây phù hợp cho hầu hết trường hợp

## 🆘 Troubleshooting

### **Không Kết Nối Được**
1. Kiểm tra IP camera có đúng không
2. Thử ping IP camera: `ping 192.168.1.100`
3. Kiểm tra username/password
4. Thử port khác: 80, 8080, 8554
5. Thử brand khác: `generic`, `onvif`

### **Kết Nối Chậm**
1. Sử dụng stream quality "sub"
2. Tăng timeout: `--connection-timeout 30`
3. Kiểm tra băng thông mạng

### **Hình Ảnh Không Ổn Định**
1. Kiểm tra kết nối WiFi camera
2. Thử ethernet thay vì WiFi
3. Restart camera

---

**🎉 Với 15+ thương hiệu được hỗ trợ, hệ thống có thể kết nối với hầu hết camera IP trên thị trường!**