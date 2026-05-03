# 📊 Hệ Thống Logging Ngủ Gật Multi-Camera

## 🎯 Tính Năng

Hệ thống logging chi tiết cho phát hiện ngủ gật với các tính năng:

### ✅ Đa Camera/Đa Phòng
- Quản lý nhiều camera đồng thời
- Mỗi camera = một phòng học
- Tự động đăng ký camera khi phát hiện

### ✅ Log Chi Tiết
- Ghi nhận **thời điểm bắt đầu** ngủ gật
- Ghi nhận **thời điểm tỉnh lại**
- Tính toán **thời lượng** ngủ gật tự động
- Theo dõi **từng học sinh** riêng biệt (track_id)

### ✅ Thống Kê Đa Chiều
- **Hôm nay** (`period=today`)
- **Tuần này** (`period=week`)
- **Tháng này** (`period=month`)
- **Khoảng tùy chỉnh** (`period=2025-11-01_2025-11-10`)

### ✅ Theo Dõi Real-time
- Danh sách học sinh **đang ngủ gật**
- Thời lượng **đang diễn ra**
- Cập nhật **theo từng frame**

---

## 🚀 API Endpoints

### 1️⃣ Lấy Danh Sách Camera

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
    }
  ],
  "total": 3
}
```

---

### 2️⃣ Thống Kê Một Camera

```http
GET /api/logs/stats/{camera_id}?period=today
```

**Parameters:**
- `camera_id`: ID của camera (vd: `camera_1`)
- `period`: `today` | `week` | `month` | `YYYY-MM-DD_YYYY-MM-DD`

**Response:**
```json
{
  "success": true,
  "stats": {
    "camera_id": "camera_1",
    "camera_name": "Phòng 101 - Toán",
    "period_start": "2025-11-10T00:00:00",
    "period_end": "2025-11-10T23:59:59",
    "total_drowsy_students": 5,
    "currently_drowsy": 2,
    "total_events": 8,
    "total_duration_seconds": 345.67,
    "total_duration_display": "5m 45s",
    "most_drowsy_student": 5,
    "most_drowsy_duration": 125.5
  }
}
```

---

### 3️⃣ Thống Kê Tất Cả Camera

```http
GET /api/logs/stats?period=today
```

**Response:**
```json
{
  "success": true,
  "stats": [
    { "camera_id": "camera_1", ... },
    { "camera_id": "camera_2", ... }
  ],
  "period": "today"
}
```

---

### 4️⃣ Thống Kê Tổng Hợp

```http
GET /api/logs/summary?period=today
```

**Response:**
```json
{
  "success": true,
  "summary": {
    "period": "today",
    "total_cameras": 3,
    "total_drowsy_students_unique": 15,
    "total_events": 25,
    "total_duration_seconds": 1234.56,
    "total_duration_display": "20m 34s",
    "currently_drowsy_all_cameras": 5,
    "cameras": [ ... ]
  }
}
```

---

### 5️⃣ Log Chi Tiết Sự Kiện

```http
GET /api/logs/events/{camera_id}?period=today
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
      "start_time": "2025-11-10 09:15:30",
      "end_time": "2025-11-10 09:18:45",
      "duration_seconds": 195.0,
      "duration_display": "3m 15s",
      "is_active": false
    },
    {
      "student_id": 8,
      "start_time": "2025-11-10 10:30:00",
      "end_time": "Đang ngủ",
      "duration_seconds": 0.0,
      "duration_display": "Đang ngủ",
      "is_active": true
    }
  ],
  "total_events": 2
}
```

**Ý nghĩa:**
- `is_active: true` → Học sinh **đang ngủ gật**
- `is_active: false` → Đã tỉnh lại, có thời lượng cụ thể

---

### 6️⃣ Học Sinh Đang Ngủ Gật (Real-time)

```http
GET /api/logs/active
```

**Response:**
```json
{
  "success": true,
  "active_drowsy": {
    "camera_1": [
      {
        "student_id": 5,
        "start_time": "2025-11-10T10:30:00",
        "duration_seconds": 125.5,
        "duration_display": "2m 5s"
      }
    ],
    "camera_3": [
      {
        "student_id": 15,
        "start_time": "2025-11-10T10:28:30",
        "duration_seconds": 245.0,
        "duration_display": "4m 5s"
      }
    ]
  },
  "total_active": 2,
  "cameras_with_drowsy": 2
}
```

---

### 7️⃣ Lưu Log Ra File

```http
POST /api/logs/save
Content-Type: application/json

{
  "filepath": "logs/drowsiness_2025-11-10.json"  // Optional
}
```

**Response:**
```json
{
  "success": true,
  "message": "Logs saved successfully"
}
```

---

## 💡 Use Cases

### UC1: Dashboard Giám Sát Real-time

```javascript
// Lấy danh sách học sinh đang ngủ gật (mỗi 5s)
setInterval(async () => {
  const response = await fetch('/api/logs/active');
  const data = await response.json();
  
  if (data.total_active > 0) {
    console.log(`⚠️ ${data.total_active} học sinh đang ngủ gật!`);
    // Hiển thị cảnh báo, phát âm thanh, v.v.
  }
}, 5000);
```

### UC2: Báo Cáo Cuối Ngày

```javascript
// Lấy thống kê hôm nay khi kết thúc buổi học
const response = await fetch('/api/logs/summary?period=today');
const summary = await response.json();

console.log(`📊 BÁO CÁO CUỐI NGÀY ${new Date().toLocaleDateString()}`);
console.log(`Tổng học sinh ngủ gật: ${summary.summary.total_drowsy_students_unique}`);
console.log(`Tổng thời gian: ${summary.summary.total_duration_display}`);
```

### UC3: Log Chi Tiết Từng Phòng

```javascript
// Xem chi tiết phòng 101 trong tuần này
const response = await fetch('/api/logs/events/camera_1?period=week');
const events = await response.json();

console.log(`📋 LOG PHÒNG 101 - TUẦN NÀY`);
events.events.forEach(event => {
  console.log(`Học sinh #${event.student_id}: ${event.start_time} → ${event.end_time} (${event.duration_display})`);
});
```

### UC4: Thống Kê Tháng

```javascript
// Báo cáo tháng cho tất cả phòng
const response = await fetch('/api/logs/summary?period=month');
const summary = await response.json();

summary.summary.cameras.forEach(cam => {
  console.log(`🏫 ${cam.camera_name}:`);
  console.log(`   - Học sinh ngủ gật: ${cam.total_drowsy_students}`);
  console.log(`   - Tổng thời gian: ${cam.total_duration_display}`);
});
```

### UC5: Tìm Học Sinh Ngủ Gật Nhiều Nhất

```javascript
// Phân tích xu hướng
const response = await fetch('/api/logs/stats/camera_1?period=week');
const stats = await response.json();

if (stats.stats.most_drowsy_student) {
  console.log(`⚠️ Học sinh #${stats.stats.most_drowsy_student} ngủ gật nhiều nhất:`);
  console.log(`   Tổng thời gian: ${stats.stats.most_drowsy_duration}s`);
  // Đề xuất can thiệp, tư vấn, v.v.
}
```

---

## 🔧 Tích Hợp Backend

### Automatic Logging

Hệ thống tự động ghi log khi phát hiện thay đổi trạng thái:

```python
# Trong yolo_detector.py hoặc camera worker
def update_detection_result(self, result):
    for person in result.persons:
        track_id = person.track_id
        is_drowsy = person.drowsiness_state in ['ngugat', 'gucxuongban', 'drowsy']
        
        # Tự động ghi log
        from drowsiness_logger import get_global_logger
        logger = get_global_logger()
        logger.update_student_state(camera_id, track_id, is_drowsy)
```

### Manual Logging

```python
from drowsiness_logger import get_global_logger

logger = get_global_logger()

# Đăng ký camera
logger.register_camera("camera_1", "Phòng 101 - Toán")

# Ghi log học sinh ngủ gật
logger.update_student_state("camera_1", student_id=5, is_drowsy=True)

# Ghi log học sinh tỉnh lại
logger.update_student_state("camera_1", student_id=5, is_drowsy=False)

# Lấy thống kê
stats = logger.get_camera_stats("camera_1", period='today')
```

---

## 📁 File Structure

```
python-backend/
├── drowsiness_logger.py          # Core logging system
├── server_with_tracking_backup.py # Tích hợp API endpoints
├── test_drowsiness_logger.py      # Test script
├── DROWSINESS_LOGGING_GUIDE.md    # Documentation
└── drowsiness_logs/               # Log files (auto-created)
    ├── autosave_20251110_100530.json
    ├── autosave_20251110_101030.json
    └── drowsiness_log_20251110_153045.json
```

---

## 🧪 Testing

Chạy test script:

```bash
cd "Desktop UI for Drowsiness Detection/python-backend"
python test_drowsiness_logger.py
```

Output mẫu:
```
============================================================
  SIMULATING DROWSINESS EVENTS
============================================================

⏰ 09:00 - Phòng 101: Học sinh #5 bắt đầu ngủ gật
[Phòng 101 - Toán Cao Cấp] Học sinh #5 BẮT ĐẦU ngủ gật lúc 09:00:00

⏰ 09:03 - Phòng 101: Học sinh #5 tỉnh lại
[Phòng 101 - Toán Cao Cấp] Học sinh #5 TỈNH LẠI lúc 09:03:00 (Ngủ gật: 0m 2s)

============================================================
  THỐNG KÊ TỔNG HỢP HÔM NAY
============================================================

{
  "period": "today",
  "total_cameras": 3,
  "total_drowsy_students_unique": 7,
  "total_events": 10,
  "total_duration_seconds": 15.5,
  "total_duration_display": "15s",
  "currently_drowsy_all_cameras": 3
}
```

---

## 📊 Auto-save Feature

Hệ thống tự động lưu log mỗi **5 phút**:

```
💾 Auto-saved logs to drowsiness_logs/autosave_20251110_100530.json
💾 Auto-saved logs to drowsiness_logs/autosave_20251110_101030.json
```

---

## 🎨 Frontend Integration Example

### React Component

```tsx
import { useState, useEffect } from 'react';

function DrowsinessMonitor() {
  const [activeDrowsy, setActiveDrowsy] = useState({});
  const [stats, setStats] = useState(null);

  // Real-time active drowsy students
  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch('/api/logs/active');
      const data = await res.json();
      setActiveDrowsy(data.active_drowsy);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Daily statistics
  useEffect(() => {
    const fetchStats = async () => {
      const res = await fetch('/api/logs/summary?period=today');
      const data = await res.json();
      setStats(data.summary);
    };
    fetchStats();
  }, []);

  return (
    <div>
      <h2>Học Sinh Đang Ngủ Gật</h2>
      {Object.entries(activeDrowsy).map(([cameraId, students]) => (
        <div key={cameraId}>
          <h3>{cameraId}</h3>
          {students.map(s => (
            <div key={s.student_id}>
              Học sinh #{s.student_id}: {s.duration_display}
            </div>
          ))}
        </div>
      ))}

      {stats && (
        <div>
          <h2>Thống Kê Hôm Nay</h2>
          <p>Tổng học sinh: {stats.total_drowsy_students_unique}</p>
          <p>Tổng thời gian: {stats.total_duration_display}</p>
        </div>
      )}
    </div>
  );
}
```

---

## ⚙️ Configuration

### Autosave Interval

```python
# Trong drowsiness_logger.py
self.autosave_interval = 300  # 5 minutes (default)

# Hoặc khi khởi tạo
logger = MultiCameraLogger(log_dir="custom_logs")
logger.autosave_interval = 600  # 10 minutes
```

### Cache Settings

```python
# Statistics cache validity
if (datetime.now() - self._cache_timestamp).total_seconds() < 5:
    # Use cached stats (5 seconds validity)
```

---

## 🐛 Troubleshooting

### Logger Not Available

```json
{
  "success": false,
  "error": "Logger not available"
}
```

**Solution:** Kiểm tra import trong `server_with_tracking_backup.py`:
```python
from drowsiness_logger import get_global_logger, init_logger
```

### Camera Not Found

```json
{
  "success": false,
  "error": "Camera camera_1 not found"
}
```

**Solution:** Camera tự động đăng ký khi có detection. Hoặc đăng ký thủ công:
```python
logger.register_camera("camera_1", "Phòng 101")
```

---

## 📝 Notes

- ✅ Thread-safe với `threading.Lock()`
- ✅ Tự động lưu mỗi 5 phút
- ✅ Cache thống kê 5 giây để tối ưu
- ✅ Hỗ trợ JSON serialization
- ✅ Format thời gian human-readable
- ✅ Theo dõi học sinh biến mất (cleanup)

---

## 🚀 Next Steps

1. **Frontend UI**: Tạo dashboard hiển thị thống kê real-time
2. **Alerts**: Cảnh báo khi quá nhiều học sinh ngủ gật
3. **Reports**: Xuất báo cáo PDF/Excel
4. **Analytics**: Phân tích xu hướng theo giờ, ngày trong tuần
5. **Notifications**: Email/SMS cho giáo viên khi phát hiện ngủ gật

---

## ✅ Checklist Tính Năng

- [x] Multi-camera logging
- [x] Chi tiết từng sự kiện (start/end time, duration)
- [x] Thống kê theo ngày/tuần/tháng
- [x] Khoảng thời gian tùy chỉnh
- [x] Real-time active tracking
- [x] Auto-save mỗi 5 phút
- [x] Thread-safe implementation
- [x] API endpoints đầy đủ
- [x] Test script
- [x] Documentation

---

**🎉 Hệ thống logging ngủ gật đã sẵn sàng sử dụng!**
