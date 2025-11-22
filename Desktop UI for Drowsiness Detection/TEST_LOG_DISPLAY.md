# 🧪 Hướng Dẫn Kiểm Tra Log Hiển Thị

## ✅ Đã Hoàn Thành

### 1. **Backend Update**
- ✅ Thêm code register camera với logger khi camera được khởi tạo
- ✅ File: `python-backend/server_with_tracking_backup.py` (dòng 235-242)
- ✅ Code mới:
```python
# Register camera with drowsiness logger
if LOGGER_AVAILABLE:
    logger = get_global_logger()
    camera_name = cam_id.split('/')[-1] if '/' in cam_id else cam_id
    logger.register_camera(cam_id, camera_name)
    app.logger.info(f"[{self.cam_id}] Camera registered with drowsiness logger as '{camera_name}'")
```

### 2. **App Status**
- ✅ Python backend running: `http://127.0.0.1:5000`
- ✅ Electron app running: Desktop UI loaded
- ✅ Logger initialized successfully
- ✅ YOLO detector initialized

---

## 📋 Các Bước Test Log Display

### Bước 1: Kiểm Tra Backend API
```powershell
# Test cameras endpoint (nên trả về empty array ban đầu)
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/cameras" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Kết quả mong đợi ban đầu:**
```json
{"success": true, "cameras": [], "total": 0}
```

### Bước 2: Thêm Camera trong UI
1. Mở Electron app (đang chạy)
2. Click tab **📹 Camera**
3. Click nút **+ Thêm** để add camera
4. Chọn:
   - Webcam: `Webcam 0` (hoặc số khác)
   - Hoặc IP Camera: Nhập URL
5. Click **Thêm Camera**
6. Click **Start** trên camera vừa thêm

**Backend console sẽ hiển thị:**
```
[2025-11-10 13:XX:XX] INFO: [cam_id] EnhancedCameraWorker initialized with URL: 0, detection: True
[2025-11-10 13:XX:XX] INFO: [cam_id] Camera registered with drowsiness logger as 'Webcam 0'
[2025-11-10 13:XX:XX] INFO: [cam_id] Starting enhanced camera worker thread...
```

### Bước 3: Kiểm Tra Camera Registered
```powershell
# Test lại cameras endpoint (giờ nên có camera)
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/cameras" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Kết quả mong đợi sau khi start camera:**
```json
{
  "success": true,
  "cameras": [
    {
      "camera_id": "1/Webcam 0",
      "camera_name": "Webcam 0",
      "active_drowsy_count": 0
    }
  ],
  "total": 1
}
```

### Bước 4: Phát Hiện Ngủ Gật
1. Đảm bảo camera đang hoạt động (thấy video stream)
2. **Giả vờ ngủ gật** trước camera:
   - Gục đầu xuống (~30-45 độ)
   - Nhắm mắt
   - Giữ tư thế 2-3 giây

**Backend console sẽ hiển thị:**
```
[Webcam 0] Học sinh #X BẮT ĐẦU ngủ gật lúc 13:XX:XX
```

**Trên UI:**
- Detection box hiển thị: `#X BUỒN NGỦ`
- Màu đỏ/cam cảnh báo

### Bước 5: Kiểm Tra API Active Students
```powershell
# Test active drowsy students endpoint
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/active" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content
```

**Kết quả mong đợi khi có ngủ gật:**
```json
{
  "success": true,
  "active_drowsy_students": [
    {
      "camera_id": "1/Webcam 0",
      "camera_name": "Webcam 0",
      "student_id": 5,
      "start_time": "2025-11-10T13:XX:XX",
      "current_duration_seconds": 15,
      "current_duration_display": "0m 15s"
    }
  ],
  "total": 1
}
```

### Bước 6: Kiểm Tra LogPanel
1. Trong Electron app, click tab **📊 Dashboard** hoặc mở **LogPanel** (nếu có tab riêng)
2. Đợi 3-5 giây (auto-refresh)

**Kết quả mong đợi:**
- ✅ LogPanel hiển thị log entry mới:
  ```
  🔴 [13:XX:XX] Webcam 0
  Học sinh #5 đang ngủ gật (0m 15s)
  ```
- ✅ Toast notification xuất hiện:
  ```
  🟠 Webcam 0: Phát hiện buồn ngủ!
  ```
- ✅ Statistics cards cập nhật:
  - Hôm nay: 1 học sinh
  - Tuần này: 1 học sinh
  - Tháng này: 1 học sinh

### Bước 7: Kiểm Tra Dashboard Detail
1. Click tab **📊 Dashboard**
2. Click vào camera card "Webcam 0"
3. Panel chi tiết xuất hiện

**Kết quả mong đợi:**
- ✅ Hiển thị 4 thẻ stats:
  - Tổng sự kiện: 1
  - Số học sinh: 1
  - Tổng thời gian: Xm Ys
  - TB/sự kiện: Xm Ys
- ✅ Hiển thị "Ngủ gật lâu nhất: Xm Ys"
- ✅ Hiển thị "HS ngủ gật nhiều nhất: #5"
- ✅ Biểu đồ phân bố theo giờ có 1 bar (giờ hiện tại)

### Bước 8: Tỉnh Lại
1. Ngẩng đầu lên
2. Mở mắt
3. Giữ tư thế 2-3 giây

**Backend console sẽ hiển thị:**
```
[Webcam 0] Học sinh #X TỈNH LẠI lúc 13:XX:XX (Ngủ gật: 0m 25s)
```

**Kết quả:**
- ✅ Detection box hiển thị: `#X BÌNH THƯỜNG`
- ✅ API `/api/logs/active` giờ trả về empty array
- ✅ LogPanel vẫn giữ log cũ (lịch sử)
- ✅ Dashboard stats vẫn hiển thị 1 sự kiện

---

## 🔍 Debug Checklist

### Nếu không thấy logs:

#### 1. Kiểm tra Backend Console
```
Tìm dòng:
✅ Camera registered with drowsiness logger as 'Webcam 0'
✅ [Webcam 0] Học sinh #X BẮT ĐẦU ngủ gật
```

Nếu không thấy → Camera chưa được register → Check camera start code

#### 2. Kiểm tra API Cameras
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/cameras" -Method GET -UseBasicParsing
```

Nếu trả về empty array → Camera chưa start → Click Start button

#### 3. Kiểm tra Browser Console (F12)
```
Tìm dòng:
[App] Active drowsy students: [...]
```

Nếu không thấy → Frontend polling có vấn đề → Reload app

#### 4. Kiểm tra LogPanel Fetch
```
F12 → Network tab → Filter: "logs"
Xem có requests đến:
- /api/logs/summary
- /api/logs/events/<camera_id>
- /api/logs/active
```

Nếu không có requests → LogPanel chưa mount → Check component

#### 5. Kiểm tra YOLO Detection
```
Backend console:
[cam_id] Tracked X persons (Y drowsy, Z alert)
```

Nếu không thấy → Detection không chạy → Check enable detection

---

## 📊 Flow Hoàn Chỉnh

```
1. User adds camera
   ↓
2. EnhancedCameraWorker.__init__()
   ↓
3. logger.register_camera(cam_id, name)
   ✅ Camera registered in logger.cameras
   ↓
4. Camera starts, YOLO detects drowsiness
   ↓
5. logger.update_student_state(cam_id, student_id, True)
   ✅ Event saved in logger
   ↓
6. API /api/logs/active returns active students
   ↓
7. Frontend App.tsx polls every 3s
   ↓
8. Creates LogEvent and adds to logs state
   ↓
9. LogPanel displays in UI
   ✅ User sees log!
   ↓
10. Toast notification appears
    ✅ User gets alert!
```

---

## ✅ Success Criteria

Sau khi test, bạn nên thấy:

1. ✅ Backend console log camera registration
2. ✅ Backend console log drowsy detection
3. ✅ API `/api/logs/cameras` returns camera list
4. ✅ API `/api/logs/active` returns active students
5. ✅ LogPanel displays log entries
6. ✅ Toast notifications appear
7. ✅ Dashboard shows statistics
8. ✅ Dashboard detail panel shows metrics
9. ✅ Charts tab shows graphs (if implemented)
10. ✅ Export PDF/Excel works

---

## 🎯 Quick Test Script

```powershell
# Test toàn bộ API flow
Write-Host "Testing API endpoints..." -ForegroundColor Cyan

# 1. Test cameras (empty initially)
Write-Host "`n1. GET /api/logs/cameras" -ForegroundColor Yellow
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/cameras" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json

# 2. Test summary
Write-Host "`n2. GET /api/logs/summary?period=today" -ForegroundColor Yellow
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/summary?period=today" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json

# 3. Test active (empty initially)
Write-Host "`n3. GET /api/logs/active" -ForegroundColor Yellow
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/logs/active" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json

Write-Host "`nNow add a camera in UI and test again!" -ForegroundColor Green
```

---

## 📝 Expected Output

### Khi KHÔNG có camera:
```json
{
  "success": true,
  "cameras": [],
  "total": 0
}
```

### Khi CÓ camera NHƯNG CHƯA có ngủ gật:
```json
{
  "success": true,
  "cameras": [
    {
      "camera_id": "1/Webcam 0",
      "camera_name": "Webcam 0",
      "active_drowsy_count": 0
    }
  ],
  "total": 1
}
```

### Khi CÓ học sinh ĐANG ngủ gật:
```json
{
  "success": true,
  "active_drowsy_students": [
    {
      "camera_id": "1/Webcam 0",
      "camera_name": "Webcam 0",
      "student_id": 5,
      "start_time": "2025-11-10T13:XX:XX",
      "current_duration_seconds": 15,
      "current_duration_display": "0m 15s"
    }
  ],
  "total": 1
}
```

---

**Hệ thống đã sẵn sàng để test!** 🚀

**Bắt đầu từ Bước 1 và làm theo từng bước.** ✅
