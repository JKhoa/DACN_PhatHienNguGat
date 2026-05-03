# 📊 Tính năng Dashboard nâng cấp

## ✨ Các cải tiến mới

### 1. **Thẻ Camera Chi Tiết Hơn**
Mỗi thẻ camera trong grid giờ hiển thị:
- ✅ **Số sự kiện ngủ gật** (tổng số lần phát hiện)
- ✅ **Số học sinh unique** đã ngủ gật
- ✅ **Thời gian sự kiện cuối cùng** (lần cuối phát hiện)
- ✅ **Trạng thái real-time** (Bình thường/Cảnh báo/Cần chú ý)
- ✅ **Icon visual** cho từng metric

```
┌─────────────────────────────────┐
│ 🎥 Phòng 101         [2 ngủ gật]│
│                                 │
│ ⚠️ Cảnh báo                     │
│ 📊 15 sự kiện    👥 8 HS        │
│ 🕐 Lần cuối: 14:25:30           │
└─────────────────────────────────┘
```

### 2. **Chọn Camera để Xem Chi Tiết**
Click vào bất kỳ camera nào để mở **Panel Chi Tiết**:

#### **Panel Chi Tiết Hiển Thị:**

**A. 4 Thẻ Thống Kê Chính:**
1. **Tổng sự kiện** - Số lần phát hiện ngủ gật
2. **Số học sinh** - Số học sinh unique đã ngủ gật
3. **Tổng thời gian** - Tổng thời gian ngủ gật (h m s)
4. **TB/sự kiện** - Thời gian trung bình mỗi lần ngủ gật

**B. Các Chỉ Số Quan Trọng:**
- 🔴 **Ngủ gật lâu nhất** - Thời gian ngủ gật dài nhất được ghi nhận
- 👥 **HS ngủ gật nhiều nhất** - Học sinh nào hay ngủ gật nhất (student ID)
- ✅ **TB mỗi sự kiện** - Thời gian trung bình

**C. Biểu Đồ Phân Bố Theo Giờ:**
- Hiển thị số sự kiện ngủ gật theo từng giờ trong ngày
- Bar chart trực quan với màu xanh
- Giúp xác định **khung giờ nào học sinh hay ngủ gật nhất**

```
┌─────────────────────────────────────────────┐
│ 📊 Chi tiết: Phòng 101              [X]     │
├─────────────────────────────────────────────┤
│ [15] Tổng sự kiện  [8] Số HS  [45m] Tổng   │
│                                             │
│ Các chỉ số quan trọng:                      │
│ 🔴 Ngủ gật lâu nhất: 5m 30s                │
│ 👥 HS ngủ gật nhiều nhất: #12              │
│                                             │
│ Phân bố theo giờ:                           │
│ 08:00 ████████████ 12 sự kiện              │
│ 09:00 ██████ 6 sự kiện                     │
│ 10:00 ████████ 8 sự kiện                   │
│ 11:00 ████ 4 sự kiện                       │
│ 14:00 ██████████████ 15 sự kiện           │
│ 15:00 ████████ 8 sự kiện                   │
└─────────────────────────────────────────────┘
```

### 3. **Lọc Theo Khoảng Thời Gian**
Dropdown menu để chọn:
- 📅 **Hôm nay** - Dữ liệu ngày hiện tại
- 📅 **Tuần này** - 7 ngày gần nhất
- 📅 **Tháng này** - 30 ngày gần nhất

**Tất cả thống kê đều tự động cập nhật** khi thay đổi khoảng thời gian!

### 4. **Visual Indicators (Màu Sắc)**

#### **Thẻ Camera:**
- 🟢 **Xanh lá** (Bình thường) - 0 học sinh ngủ gật
- 🟡 **Vàng** (Cảnh báo) - 1-2 học sinh ngủ gật
- 🔴 **Đỏ** (Nguy hiểm) - 3+ học sinh ngủ gật

#### **Camera Được Chọn:**
- 🔵 **Ring xanh dương** - Hiển thị camera đang xem chi tiết
- ▶️ **Icon mũi tên** - Chỉ ra camera active

### 5. **Auto-Refresh**
- Tự động làm mới **mỗi 5 giây**
- Đảm bảo dữ liệu luôn real-time
- Không cần reload trang thủ công

## 🎯 Cách Sử Dụng

### Bước 1: Mở Dashboard
```
Click tab [📊 Dashboard] ở thanh navigation
```

### Bước 2: Xem Tổng Quan
Quan sát 4 thẻ thống kê trên cùng:
- Tổng số phòng đang giám sát
- Tổng học sinh ngủ gật (unique)
- Số học sinh đang ngủ gật (real-time)
- Tổng số sự kiện

### Bước 3: Chọn Khoảng Thời Gian
```
Dropdown menu [Hôm nay ▼] → Chọn:
- Hôm nay
- Tuần này  
- Tháng này
```

### Bước 4: Xem Chi Tiết Camera
```
1. Click vào bất kỳ camera nào trong grid
2. Panel chi tiết xuất hiện phía dưới
3. Xem các metric chi tiết:
   - Số sự kiện, học sinh, thời gian
   - Ngủ gật lâu nhất
   - Học sinh hay ngủ gật nhất
   - Biểu đồ phân bố theo giờ
```

### Bước 5: Đóng Panel Chi Tiết
```
Click nút [X] ở góc phải panel chi tiết
Hoặc click vào camera khác để xem camera đó
```

### Bước 6: Export Báo Cáo (Nếu Cần)
```
Click nút [📥 PDF] hoặc [📥 Excel] ở góc phải
```

## 📋 Ví Dụ Thực Tế

### Tình huống 1: Kiểm tra phòng nào ngủ gật nhiều
```
1. Mở Dashboard
2. Quan sát grid camera
3. Tìm thẻ màu đỏ (3+ ngủ gật) hoặc vàng (1-2 ngủ gật)
4. Click vào thẻ đó để xem chi tiết
5. Xem biểu đồ "Phân bố theo giờ" để biết giờ nào hay ngủ nhất
```

### Tình huống 2: Xem học sinh nào hay ngủ gật
```
1. Click vào camera bất kỳ
2. Xem phần "HS ngủ gật nhiều nhất: #X"
3. Ghi nhận student ID #X
4. Có thể theo dõi thêm ở LogPanel
```

### Tình huống 3: Phân tích xu hướng theo thời gian
```
1. Chọn "Tuần này" ở dropdown
2. Click từng camera
3. So sánh số sự kiện giữa các camera
4. Xác định phòng nào cần chú ý nhiều hơn
```

### Tình huống 4: Tìm khung giờ nguy hiểm
```
1. Click vào camera có nhiều sự kiện
2. Xem biểu đồ "Phân bố theo giờ"
3. Tìm khung giờ có bar dài nhất
4. Đó là giờ cần tăng cường giám sát
```

## 🔍 Chi Tiết Kỹ Thuật

### API Endpoints Sử Dụng:
```javascript
// 1. Danh sách cameras
GET http://localhost:5000/api/logs/cameras
→ Returns: {cameras: [{camera_id, camera_name, active_drowsy_count, ...}]}

// 2. Thống kê tổng hợp
GET http://localhost:5000/api/logs/summary?period=today|week|month
→ Returns: {summary: {total_events, unique_students, ...}}

// 3. Học sinh đang ngủ gật
GET http://localhost:5000/api/logs/active
→ Returns: {active_drowsy_students: [{camera_id, student_id, duration, ...}]}

// 4. Chi tiết events của camera
GET http://localhost:5000/api/logs/events/:camera_id?period=today|week|month
→ Returns: {events: [{start_time, duration, student_id, ...}]}
```

### Tính Toán Thống Kê:
```typescript
// Unique students
const uniqueStudents = new Set(events.map(e => e.student_id)).size;

// Average duration
const avgDuration = totalDuration / events.length;

// Longest event
const longest = events.reduce((max, e) => 
  e.duration_seconds > max.duration_seconds ? e : max
);

// Events by hour
const eventsByHour = {};
events.forEach(e => {
  const hour = new Date(e.start_time).getHours();
  eventsByHour[hour] = (eventsByHour[hour] || 0) + 1;
});

// Most frequent student
const studentCounts = {};
events.forEach(e => {
  studentCounts[e.student_id] = (studentCounts[e.student_id] || 0) + 1;
});
const mostFrequent = Object.entries(studentCounts)
  .reduce((max, [id, count]) => count > max[1] ? [id, count] : max);
```

## 🎨 UI/UX Improvements

### 1. Color Coding
- **Red/Orange** - Urgent attention needed
- **Yellow** - Warning, monitor closely
- **Green** - Normal, all good
- **Blue** - Selected camera

### 2. Visual Hierarchy
```
1. Header (Dashboard title + controls)
2. Summary Cards (4 key metrics)
3. Camera Grid (overview of all cameras)
4. Active Students (real-time alerts)
5. Camera Detail Panel (deep dive stats)
```

### 3. Interactive Elements
- ✅ Clickable camera cards
- ✅ Hover effects for better UX
- ✅ Selected state indication
- ✅ Close button for detail panel
- ✅ Dropdown for time period

### 4. Responsive Design
- 📱 Mobile: 1 column
- 💻 Tablet: 2 columns
- 🖥️ Desktop: 3-4 columns

## 🚀 Performance

- **Auto-refresh**: 5 seconds
- **Data caching**: In React state
- **API calls**: Throttled per camera
- **Chart rendering**: Max 10 hours shown
- **Build size**: ~963 KB (optimized)

## 📊 Data Flow

```
Backend Logger
    ↓
API Endpoints
    ↓
Dashboard Fetch (5s interval)
    ↓
React State Update
    ↓
UI Re-render
    ↓
User Sees Live Data
```

## ✨ Tính Năng Nổi Bật

1. ✅ **Real-time monitoring** - Cập nhật tự động mỗi 5 giây
2. ✅ **Camera selection** - Click để xem chi tiết bất kỳ camera nào
3. ✅ **Time filtering** - Lọc theo hôm nay/tuần/tháng
4. ✅ **Visual indicators** - Màu sắc trực quan cho severity
5. ✅ **Detailed stats** - 10+ metrics cho mỗi camera
6. ✅ **Hour distribution** - Biểu đồ phân bố theo giờ
7. ✅ **Export capability** - Xuất PDF/Excel
8. ✅ **Responsive layout** - Tối ưu mọi kích thước màn hình

**Dashboard đã sẵn sàng!** 🎉
Mở app và test ngay thôi!
