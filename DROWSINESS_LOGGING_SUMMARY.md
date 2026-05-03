# 📊 TÓM TẮT HỆ THỐNG LOGGING NGỦ GẬT MULTI-CAMERA

## 🎯 YÊU CẦU ĐÃ THỰC HIỆN

✅ **Hiển thị từng camera khác nhau** (mỗi camera = một phòng học)
✅ **Log chi tiết** chỉ hiển thị khi học sinh ngủ gật (thời gian bắt đầu → thức dậy)
✅ **Thống kê tổng học sinh ngủ gật** trong từng phòng
✅ **Khoảng thời gian trong ngày** có thể chỉnh sửa (custom date range)
✅ **Tổng học sinh ngủ gật trong tuần**
✅ **Tổng học sinh ngủ gật trong tháng**

---

## 📁 CÁC FILE ĐÃ TẠO

### 1. `drowsiness_logger.py` (600+ dòng)
**Core logging system** với các class chính:

- **`DrowsinessEvent`**: Một sự kiện ngủ gật (start_time, end_time, duration)
- **`CameraLogger`**: Logger cho một camera/phòng học
- **`MultiCameraLogger`**: Quản lý nhiều camera đồng thời

**Tính năng:**
- Tự động ghi log khi học sinh bắt đầu ngủ gật
- Tự động tính thời lượng khi học sinh tỉnh lại
- Thống kê theo nhiều khoảng thời gian (today, week, month, custom)
- Auto-save mỗi 5 phút
- Thread-safe với `threading.Lock()`

### 2. `server_with_tracking_backup.py` (Updated)
**Tích hợp logging vào detection flow:**

- Import `drowsiness_logger` module
- Khởi tạo global logger khi server starts
- Tự động ghi log trong `_update_states_and_logs()`
- Thêm 7 API endpoints mới:
  - `GET /api/logs/cameras` - Danh sách camera
  - `GET /api/logs/stats/<camera_id>` - Thống kê một camera
  - `GET /api/logs/stats` - Thống kê tất cả camera
  - `GET /api/logs/summary` - Thống kê tổng hợp
  - `GET /api/logs/events/<camera_id>` - Log chi tiết sự kiện
  - `GET /api/logs/active` - Học sinh đang ngủ gật (real-time)
  - `POST /api/logs/save` - Lưu logs ra file

### 3. `test_drowsiness_logger.py` (180+ dòng)
**Test script đầy đủ** minh họa tất cả tính năng:

- Simulate drowsiness events từ 3 camera
- Test statistics (today, week, month)
- Test detailed event logs
- Test active drowsy tracking
- Test API response format

### 4. `DROWSINESS_LOGGING_GUIDE.md` (600+ dòng)
**Documentation chi tiết** với:

- API documentation đầy đủ
- Use cases và examples
- Frontend integration guide
- Troubleshooting
- Configuration options

---

## 🚀 CÁCH SỬ DỤNG

### Backend Auto-Integration

Logging được tích hợp tự động trong detection flow:

```python
# Trong EnhancedCameraWorker._update_states_and_logs()
if eff_state in ('Ngủ gật', 'Gục xuống bàn'):
    # Học sinh BẮT ĐẦU ngủ gật
    if LOGGER_AVAILABLE:
        logger = get_global_logger()
        logger.update_student_state(self.cam_id, track_id, True)

elif eff_state == 'Thức dậy':
    # Học sinh TỈNH LẠI
    if LOGGER_AVAILABLE:
        logger = get_global_logger()
        logger.update_student_state(self.cam_id, track_id, False)
```

### Test Logging System

```bash
cd "Desktop UI for Drowsiness Detection/python-backend"
python test_drowsiness_logger.py
```

**Output mẫu:**
```
============================================================
  SIMULATING DROWSINESS EVENTS
============================================================

⏰ 09:00 - Phòng 101: Học sinh #5 bắt đầu ngủ gật
[Phòng 101 - Toán Cao Cấp] Học sinh #5 BẮT ĐẦU ngủ gật lúc 09:00:00

⏰ 09:03 - Phòng 101: Học sinh #5 tỉnh lại
[Phòng 101 - Toán Cao Cấp] Học sinh #5 TỈNH LẠI lúc 09:03:00 (Ngủ gật: 0m 3s)

============================================================
  THỐNG KÊ TỔNG HỢP HÔM NAY
============================================================

{
  "period": "today",
  "total_cameras": 3,
  "total_drowsy_students_unique": 7,
  "total_events": 10,
  "total_duration_seconds": 15.5,
  "currently_drowsy_all_cameras": 3
}
```

---

## 📡 API ENDPOINTS EXAMPLES

### 1. Lấy Log Chi Tiết Một Phòng (Theo Yêu Cầu ✅)

**Chỉ hiển thị khi học sinh ngủ gật (thời gian bắt đầu → thức dậy)**

```http
GET /api/logs/events/camera_1?period=today
```

**Response:**
```json
{
  "success": true,
  "camera_id": "camera_1",
  "period": "today",
  "events": [
    {
      "student_id": 5,
      "start_time": "2025-11-10 09:15:30",    // ⏰ BẮT ĐẦU
      "end_time": "2025-11-10 09:18:45",      // ⏰ THỨC DẬY
      "duration_seconds": 195.0,
      "duration_display": "3m 15s",           // 📊 THỜI LƯỢNG
      "is_active": false                       // ✅ ĐÃ KẾT THÚC
    },
    {
      "student_id": 8,
      "start_time": "2025-11-10 10:30:00",    // ⏰ BẮT ĐẦU
      "end_time": "Đang ngủ",                 // 🔴 ĐANG NGỦ
      "duration_seconds": 0.0,
      "duration_display": "Đang ngủ",
      "is_active": true                        // 🔴 ĐANG DIỄN RA
    }
  ],
  "total_events": 2
}
```

### 2. Thống Kê Tổng Học Sinh Ngủ Gật Trong Phòng (Theo Yêu Cầu ✅)

```http
GET /api/logs/stats/camera_1?period=today
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "camera_id": "camera_1",
    "camera_name": "Phòng 101 - Toán",
    "total_drowsy_students": 5,        // 📊 TỔNG HỌC SINH NGỦ GẬT
    "currently_drowsy": 2,              // 🔴 ĐANG NGỦ GẬT
    "total_events": 8,                  // 📝 SỐ SỰ KIỆN
    "total_duration_seconds": 345.67,   // ⏱️ TỔNG THỜI GIAN
    "total_duration_display": "5m 45s"
  }
}
```

### 3. Khoảng Thời Gian Trong Ngày (Custom Range - Theo Yêu Cầu ✅)

```http
GET /api/logs/events/camera_1?period=2025-11-10_2025-11-10
```

Hoặc khoảng thời gian tuỳ chỉnh:
```http
GET /api/logs/events/camera_1?period=2025-11-01_2025-11-15
```

### 4. Tổng Học Sinh Ngủ Gật Trong Tuần (Theo Yêu Cầu ✅)

```http
GET /api/logs/summary?period=week
```

**Response:**
```json
{
  "success": true,
  "summary": {
    "period": "week",
    "period_start": "2025-11-04T00:00:00",  // 📅 Thứ 2 tuần này
    "period_end": "2025-11-10T23:59:59",
    "total_drowsy_students_unique": 25,     // 📊 TỔNG HỌC SINH (TUẦN)
    "total_events": 87,
    "total_duration_seconds": 4567.89,
    "total_duration_display": "1h 16m"
  }
}
```

### 5. Tổng Học Sinh Ngủ Gật Trong Tháng (Theo Yêu Cầu ✅)

```http
GET /api/logs/summary?period=month
```

**Response:**
```json
{
  "success": true,
  "summary": {
    "period": "month",
    "period_start": "2025-11-01T00:00:00",  // 📅 Đầu tháng
    "period_end": "2025-11-10T23:59:59",
    "total_drowsy_students_unique": 156,    // 📊 TỔNG HỌC SINH (THÁNG)
    "total_events": 453,
    "total_duration_seconds": 23456.78,
    "total_duration_display": "6h 30m"
  }
}
```

### 6. Hiển thị Từng Camera (Theo Yêu Cầu ✅)

```http
GET /api/logs/cameras
```

**Response:**
```json
{
  "success": true,
  "cameras": [
    {
      "camera_id": "camera_1",
      "camera_name": "Phòng 101 - Toán",
      "active_drowsy_count": 2
    },
    {
      "camera_id": "camera_2",
      "camera_name": "Phòng 102 - Văn",
      "active_drowsy_count": 1
    },
    {
      "camera_id": "camera_3",
      "camera_name": "Phòng 103 - Anh",
      "active_drowsy_count": 0
    }
  ],
  "total": 3
}
```

---

## 💡 USE CASE: DASHBOARD GIÁM SÁT

### UI Component Mẫu

```jsx
function DrowsinessMonitor() {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState(null);
  const [period, setPeriod] = useState('today');

  // 1. Load danh sách camera
  useEffect(() => {
    fetch('/api/logs/cameras')
      .then(r => r.json())
      .then(data => setCameras(data.cameras));
  }, []);

  // 2. Load log chi tiết khi chọn camera
  useEffect(() => {
    if (selectedCamera) {
      fetch(`/api/logs/events/${selectedCamera}?period=${period}`)
        .then(r => r.json())
        .then(data => setEvents(data.events));
        
      fetch(`/api/logs/stats/${selectedCamera}?period=${period}`)
        .then(r => r.json())
        .then(data => setStats(data.stats));
    }
  }, [selectedCamera, period]);

  return (
    <div>
      {/* Chọn phòng */}
      <select onChange={e => setSelectedCamera(e.target.value)}>
        {cameras.map(cam => (
          <option key={cam.camera_id} value={cam.camera_id}>
            {cam.camera_name} ({cam.active_drowsy_count} đang ngủ)
          </option>
        ))}
      </select>

      {/* Chọn khoảng thời gian */}
      <select value={period} onChange={e => setPeriod(e.target.value)}>
        <option value="today">Hôm nay</option>
        <option value="week">Tuần này</option>
        <option value="month">Tháng này</option>
      </select>

      {/* Thống kê phòng */}
      {stats && (
        <div className="stats-card">
          <h3>{stats.camera_name}</h3>
          <p>Tổng học sinh ngủ gật: {stats.total_drowsy_students}</p>
          <p>Đang ngủ gật: {stats.currently_drowsy}</p>
          <p>Số sự kiện: {stats.total_events}</p>
          <p>Tổng thời gian: {stats.total_duration_display}</p>
        </div>
      )}

      {/* Log chi tiết */}
      <div className="events-list">
        <h3>Log Chi Tiết</h3>
        {events.map((event, idx) => (
          <div key={idx} className={event.is_active ? 'active' : 'completed'}>
            <span className={event.is_active ? '🔴' : '🟢'}>
              Học sinh #{event.student_id}
            </span>
            <div>
              <strong>Bắt đầu:</strong> {event.start_time}
            </div>
            <div>
              <strong>Kết thúc:</strong> {event.end_time}
            </div>
            <div>
              <strong>Thời lượng:</strong> {event.duration_display}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## 🎨 CONSOLE OUTPUT EXAMPLE

Khi chạy server, bạn sẽ thấy logs như:

```
[2025-11-10 10:15:30,123] INFO: Initializing Drowsiness Logger...
✅ Drowsiness Logger initialized successfully (log_dir: drowsiness_logs)

[2025-11-10 10:15:45,456] INFO: Initializing YOLO detector...
✅ YOLO detector initialized successfully

[2025-11-10 10:16:00,789] INFO: Starting Flask+SocketIO server...

# Khi phát hiện ngủ gật:
[Phòng 101 - Toán] Học sinh #5 BẮT ĐẦU ngủ gật lúc 10:16:15

# Khi tỉnh lại:
[Phòng 101 - Toán] Học sinh #5 TỈNH LẠI lúc 10:18:30 (Ngủ gật: 2m 15s)

# Auto-save:
💾 Auto-saved logs to drowsiness_logs/autosave_20251110_101830.json
```

---

## 📊 DATA FLOW

```
┌─────────────────┐
│  YOLO Detection │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ EnhancedCameraWorker    │
│ _update_states_and_logs │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ MultiCameraLogger       │
│ update_student_state()  │
└────────┬────────────────┘
         │
         ├─► DrowsinessEvent (start_time, end_time)
         │
         ├─► CameraLogger (per camera)
         │
         └─► Auto-save every 5 minutes
         
         
API Endpoints ◄────────┐
         │             │
         ▼             │
┌─────────────────┐    │
│  Frontend UI    │────┘
│  - Dashboard    │
│  - Statistics   │
│  - Alerts       │
└─────────────────┘
```

---

## ✅ CHECKLIST YÊU CẦU

| Yêu Cầu | Trạng Thái | API Endpoint |
|---------|------------|--------------|
| ✅ Hiển thị từng camera (phòng) | **HOÀN THÀNH** | `GET /api/logs/cameras` |
| ✅ Log chi tiết (bắt đầu → thức dậy) | **HOÀN THÀNH** | `GET /api/logs/events/<camera_id>?period=today` |
| ✅ Thống kê tổng học sinh ngủ gật trong phòng | **HOÀN THÀNH** | `GET /api/logs/stats/<camera_id>?period=today` |
| ✅ Khoảng thời gian trong ngày (chỉnh sửa) | **HOÀN THÀNH** | `?period=2025-11-10_2025-11-10` |
| ✅ Tổng học sinh ngủ gật trong tuần | **HOÀN THÀNH** | `GET /api/logs/summary?period=week` |
| ✅ Tổng học sinh ngủ gật trong tháng | **HOÀN THÀNH** | `GET /api/logs/summary?period=month` |

---

## 🔧 TÍNH NĂNG BỔ SUNG

### 1. Real-time Active Tracking
```http
GET /api/logs/active
```
Theo dõi học sinh **đang ngủ gật** theo thời gian thực

### 2. Auto-save Every 5 Minutes
Tự động lưu logs để tránh mất dữ liệu

### 3. Thread-safe Implementation
An toàn cho multi-threading với `threading.Lock()`

### 4. Cache Optimization
Cache statistics 5 giây để giảm tải CPU

### 5. Human-readable Duration Format
- `"3s"` - dưới 1 phút
- `"2m 15s"` - dưới 1 giờ  
- `"1h 30m"` - trên 1 giờ

---

## 🚀 NEXT STEPS (Tương Lai)

1. **Frontend Dashboard UI**
   - Real-time monitoring
   - Charts và graphs
   - Alert notifications

2. **Export Reports**
   - PDF reports
   - Excel spreadsheets
   - Email summaries

3. **Advanced Analytics**
   - Xu hướng theo giờ trong ngày
   - So sánh giữa các phòng
   - Dự đoán học sinh có nguy cơ cao

4. **Alerts System**
   - Email alerts khi quá nhiều học sinh ngủ gật
   - SMS notifications cho giáo viên
   - Âm thanh cảnh báo trong phòng

5. **Student Profiles**
   - Lịch sử ngủ gật của từng học sinh
   - Recommendations cho can thiệp
   - Tư vấn cải thiện

---

## 📝 KẾT LUẬN

✅ **Hệ thống logging ngủ gật đã HOÀN THÀNH đầy đủ tất cả yêu cầu:**

1. ✅ Hiển thị từng camera (phòng học)
2. ✅ Log chi tiết (thời gian bắt đầu → thức dậy)
3. ✅ Thống kê tổng học sinh ngủ gật trong phòng
4. ✅ Khoảng thời gian tùy chỉnh
5. ✅ Thống kê tuần
6. ✅ Thống kê tháng

**Backend:** ✅ Hoàn thành (4 files)
**API Endpoints:** ✅ 7 endpoints đầy đủ
**Documentation:** ✅ Chi tiết với examples
**Testing:** ✅ Test script sẵn sàng

**🎉 SẴN SÀNG ĐỂ TÍCH HỢP VÀO FRONTEND!**
