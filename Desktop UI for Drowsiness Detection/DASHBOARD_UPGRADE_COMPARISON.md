# 📊 Dashboard - So Sánh Trước & Sau

## ⚙️ Các Thay Đổi Chính

### 1️⃣ **Thẻ Camera Grid**

#### ❌ TRƯỚC:
```tsx
<Card className="cursor-pointer">
  <CardHeader>
    <CardTitle>Phòng 101</CardTitle>
    <Badge>2 ngủ gật</Badge>
  </CardHeader>
  <CardContent>
    <Activity /> Cảnh báo
  </CardContent>
</Card>
```
**Hiển thị:**
- Tên camera
- Số học sinh ngủ gật hiện tại
- Trạng thái (Bình thường/Cảnh báo/Nguy hiểm)

---

#### ✅ SAU:
```tsx
<Card 
  onClick={() => setSelectedCamera(camera.camera_id)}
  className={`cursor-pointer ${isSelected ? 'ring-2 ring-blue-500' : ''}`}
>
  <CardHeader>
    <CardTitle>
      Phòng 101 
      {isSelected && <ChevronRight />}
    </CardTitle>
    <Badge>2 ngủ gật</Badge>
  </CardHeader>
  <CardContent>
    <Activity /> Cảnh báo
    
    {/* MỚI: Thêm metrics */}
    <div className="grid grid-cols-2 gap-2">
      <div><BarChart3 /> 15 sự kiện</div>
      <div><Users /> 8 HS</div>
    </div>
    
    {/* MỚI: Thời gian sự kiện cuối */}
    <div><Clock /> Lần cuối: 14:25:30</div>
  </CardContent>
</Card>
```
**Hiển thị:**
- ✅ Tên camera + icon mũi tên nếu được chọn
- ✅ Số học sinh ngủ gật hiện tại
- ✅ Trạng thái (Bình thường/Cảnh báo/Nguy hiểm)
- ✅ **[MỚI]** Tổng số sự kiện ngủ gật
- ✅ **[MỚI]** Số học sinh unique đã ngủ gật
- ✅ **[MỚI]** Thời gian sự kiện cuối cùng
- ✅ **[MỚI]** Clickable để xem chi tiết
- ✅ **[MỚI]** Ring xanh khi được chọn

---

### 2️⃣ **Panel Chi Tiết Camera**

#### ❌ TRƯỚC:
```
Không có panel chi tiết!
Click vào camera không làm gì cả.
```

---

#### ✅ SAU:
```tsx
{selectedCamera && cameraDetail && (
  <Card className="border-blue-300 bg-blue-50">
    <CardHeader>
      <CardTitle>📊 Chi tiết: {cameraDetail.camera_name}</CardTitle>
      <Button onClick={() => setSelectedCamera(null)}>
        <X /> {/* Nút đóng */}
      </Button>
    </CardHeader>
    
    <CardContent>
      {/* 4 Thẻ thống kê chính */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardTitle>Tổng sự kiện</CardTitle>
          <div className="text-2xl">{cameraDetail.total_events}</div>
          <BarChart3 />
        </Card>
        
        <Card>
          <CardTitle>Số học sinh</CardTitle>
          <div className="text-2xl">{cameraDetail.unique_students}</div>
          <Users />
        </Card>
        
        <Card>
          <CardTitle>Tổng thời gian</CardTitle>
          <div className="text-2xl">{cameraDetail.total_duration}</div>
          <Clock />
        </Card>
        
        <Card>
          <CardTitle>TB/sự kiện</CardTitle>
          <div className="text-2xl">{cameraDetail.avg_duration}</div>
          <TrendingUp />
        </Card>
      </div>
      
      {/* Các chỉ số quan trọng */}
      <Card>
        <CardTitle>Các chỉ số quan trọng</CardTitle>
        <div>🔴 Ngủ gật lâu nhất: {cameraDetail.longest_duration}</div>
        <div>👥 HS ngủ gật nhiều nhất: #{cameraDetail.most_frequent_student}</div>
        <div>✅ TB mỗi sự kiện: {cameraDetail.avg_duration}</div>
      </Card>
      
      {/* Biểu đồ phân bố theo giờ */}
      <Card>
        <CardTitle>Phân bố theo giờ</CardTitle>
        {Object.entries(cameraDetail.events_by_hour).map(([hour, count]) => (
          <div key={hour}>
            <span>{hour}</span>
            <span>{count} sự kiện</span>
            <div className="progress-bar" style={{width: `${percentage}%`}} />
          </div>
        ))}
      </Card>
    </CardContent>
  </Card>
)}
```

**Hiển thị:**
- ✅ **[MỚI]** Panel xuất hiện khi click camera
- ✅ **[MỚI]** 4 thẻ thống kê: Events, Students, Duration, Average
- ✅ **[MỚI]** Ngủ gật lâu nhất (longest event)
- ✅ **[MỚI]** Học sinh hay ngủ gật nhất (most frequent)
- ✅ **[MỚI]** Biểu đồ phân bố theo giờ (bar chart)
- ✅ **[MỚI]** Nút đóng panel (X button)

---

### 3️⃣ **State Management**

#### ❌ TRƯỚC:
```tsx
const [cameras, setCameras] = useState<CameraInfo[]>([]);
const [summary, setSummary] = useState<SummaryStats | null>(null);
const [activeStudents, setActiveStudents] = useState<ActiveStudent[]>([]);
const [period, setPeriod] = useState('today');
const [isLoading, setIsLoading] = useState(true);
```

---

#### ✅ SAU:
```tsx
const [cameras, setCameras] = useState<CameraInfo[]>([]);
const [summary, setSummary] = useState<SummaryStats | null>(null);
const [activeStudents, setActiveStudents] = useState<ActiveStudent[]>([]);
const [period, setPeriod] = useState('today');
const [isLoading, setIsLoading] = useState(true);

// MỚI: State cho camera selection
const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
const [cameraDetail, setCameraDetail] = useState<CameraDetailStats | null>(null);
```

---

### 4️⃣ **Data Fetching**

#### ❌ TRƯỚC:
```tsx
const fetchDashboardData = async () => {
  // Fetch cameras
  const camerasRes = await fetch('http://localhost:5000/api/logs/cameras');
  
  // Fetch summary
  const summaryRes = await fetch(`http://localhost:5000/api/logs/summary?period=${period}`);
  
  // Fetch active students
  const activeRes = await fetch('http://localhost:5000/api/logs/active');
};

useEffect(() => {
  fetchDashboardData();
  const interval = setInterval(fetchDashboardData, 5000);
  return () => clearInterval(interval);
}, [period]);
```

---

#### ✅ SAU:
```tsx
const fetchDashboardData = async () => {
  // Fetch cameras (same)
  const camerasRes = await fetch('http://localhost:5000/api/logs/cameras');
  
  // Fetch summary (same)
  const summaryRes = await fetch(`http://localhost:5000/api/logs/summary?period=${period}`);
  
  // Fetch active students (same)
  const activeRes = await fetch('http://localhost:5000/api/logs/active');
};

// MỚI: Fetch camera detail khi chọn camera
const fetchCameraDetail = async (cameraId: string) => {
  const response = await fetch(
    `http://localhost:5000/api/logs/events/${encodeURIComponent(cameraId)}?period=${period}`
  );
  const data = await response.json();
  
  if (data.success && data.events) {
    // Calculate detailed statistics
    const uniqueStudents = new Set(events.map(e => e.student_id)).size;
    const totalDuration = events.reduce((sum, e) => sum + e.duration_seconds, 0);
    const avgDuration = totalDuration / events.length;
    const longestEvent = events.reduce((max, e) => e.duration_seconds > max ? e : max);
    
    // Count events by hour
    const eventsByHour = {};
    events.forEach(e => {
      const hour = new Date(e.start_time).getHours();
      eventsByHour[hour] = (eventsByHour[hour] || 0) + 1;
    });
    
    // Find most frequent student
    const studentCounts = {};
    events.forEach(e => {
      studentCounts[e.student_id] = (studentCounts[e.student_id] || 0) + 1;
    });
    
    setCameraDetail({...});
  }
};

useEffect(() => {
  fetchDashboardData();
  const interval = setInterval(fetchDashboardData, 5000);
  return () => clearInterval(interval);
}, [period]);

// MỚI: Effect cho camera detail
useEffect(() => {
  if (selectedCamera) {
    fetchCameraDetail(selectedCamera);
  }
}, [selectedCamera, period]);
```

---

### 5️⃣ **Helper Functions**

#### ❌ TRƯỚC:
```
Không có helper functions!
```

---

#### ✅ SAU:
```tsx
// MỚI: Format duration helper
const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
};

// Sử dụng:
<div>{formatDuration(totalDuration)}</div>
<div>{formatDuration(avgDuration)}</div>
<div>{formatDuration(longestDuration)}</div>
```

---

## 📊 Metrics So Sánh

### Số lượng thông tin hiển thị

| Metric | Trước | Sau |
|--------|-------|-----|
| **Camera Card Info** | 3 items | 7 items |
| **Detail Stats** | 0 | 10+ metrics |
| **Visual Charts** | 0 | 1 bar chart |
| **Clickable Elements** | 0 | All cameras |
| **Interactive Panels** | 0 | 1 detail panel |

### API Calls

| Endpoint | Trước | Sau |
|----------|-------|-----|
| `/api/logs/cameras` | ✅ | ✅ |
| `/api/logs/summary` | ✅ | ✅ |
| `/api/logs/active` | ✅ | ✅ |
| `/api/logs/events/:id` | ❌ | ✅ **[MỚI]** |

### Code Size

| File | Trước | Sau | Tăng |
|------|-------|-----|------|
| `DashboardPanel.tsx` | ~300 lines | ~570 lines | +90% |
| Build bundle | 956 KB | 963 KB | +0.7% |

---

## 🎯 Tính Năng Mới Tóm Tắt

### 1. Camera Cards Enhancement
- ✅ Hiển thị tổng số sự kiện
- ✅ Hiển thị số học sinh unique
- ✅ Hiển thị thời gian sự kiện cuối
- ✅ Icon cho mỗi metric
- ✅ Visual indicator khi được chọn

### 2. Camera Detail Panel
- ✅ 4 thẻ thống kê chính
- ✅ Các chỉ số quan trọng (longest, most frequent, average)
- ✅ Biểu đồ phân bố theo giờ
- ✅ Nút đóng panel
- ✅ Auto-update khi thay đổi period

### 3. Interactive Features
- ✅ Click camera để xem chi tiết
- ✅ Visual feedback (ring, arrow icon)
- ✅ Close button
- ✅ Responsive layout

### 4. Data Processing
- ✅ Calculate unique students
- ✅ Calculate average duration
- ✅ Find longest event
- ✅ Count events by hour
- ✅ Find most frequent student

---

## 🚀 Cách Test

### Test 1: Camera Selection
```
1. Mở Dashboard
2. Click vào bất kỳ camera nào
3. ✅ Panel chi tiết xuất hiện phía dưới
4. ✅ Camera có ring xanh và icon mũi tên
5. Click camera khác
6. ✅ Panel cập nhật với data camera mới
```

### Test 2: Detail Statistics
```
1. Click camera bất kỳ
2. Kiểm tra 4 thẻ thống kê:
   ✅ Tổng sự kiện > 0
   ✅ Số học sinh > 0
   ✅ Tổng thời gian hiển thị (h m s)
   ✅ TB/sự kiện hiển thị
```

### Test 3: Hour Distribution Chart
```
1. Click camera có nhiều sự kiện
2. Scroll xuống biểu đồ "Phân bố theo giờ"
3. ✅ Thấy các giờ có bar khác nhau
4. ✅ Bar dài nhất = giờ có nhiều sự kiện nhất
5. ✅ Hiển thị "X sự kiện" cho mỗi giờ
```

### Test 4: Time Period Filter
```
1. Chọn camera A
2. Xem stats
3. Đổi từ "Hôm nay" → "Tuần này"
4. ✅ Stats tự động cập nhật
5. ✅ Biểu đồ thay đổi theo data mới
```

### Test 5: Close Panel
```
1. Click camera để mở panel
2. Click nút [X] ở góc phải panel
3. ✅ Panel đóng lại
4. ✅ Ring xanh biến mất
5. ✅ Icon mũi tên biến mất
```

---

## 📈 Kết Quả Mong Đợi

Sau khi nâng cấp, Dashboard giờ cung cấp:

1. **Tổng quan nhanh** - 4 summary cards
2. **Chi tiết từng camera** - Click để deep dive
3. **Phân tích xu hướng** - Biểu đồ theo giờ
4. **Xác định vấn đề** - Học sinh/giờ nguy hiểm
5. **Interactive UX** - Click, select, close
6. **Real-time data** - Auto-refresh 5s

**Dashboard đã trở thành công cụ phân tích mạnh mẽ!** 🎉
