# 🎉 HƯỚNG DẪN TÍCH HỢP CÁC TÍNH NĂNG MỚI

## ✅ ĐÃ TẠO CÁC FILE SAU:

### Backend (Python):
1. **`report_generator.py`** - Module tạo báo cáo PDF và Excel
2. **`server_with_tracking_backup.py`** (đã update) - Thêm 2 API endpoints:
   - `POST /api/logs/export/pdf` - Xuất báo cáo PDF
   - `POST /api/logs/export/excel` - Xuất báo cáo Excel

### Frontend (React/TypeScript):
1. **`DashboardPanel.tsx`** - Dashboard real-time monitoring
2. **`ChartsPanel.tsx`** - Biểu đồ thống kê (Line, Bar, Pie)
3. **`DateRangeSettingsPanel.tsx`** - Settings panel với date range picker

---

## 🔧 CÀI ĐẶT DEPENDENCIES

### 1. Backend - Python packages

```bash
cd "Desktop UI for Drowsiness Detection/python-backend"
pip install reportlab pandas openpyxl matplotlib
```

Hoặc:

```bash
pip install -r requirements.txt
```

### 2. Frontend - NPM packages

```bash
cd "Desktop UI for Drowsiness Detection"
npm install date-fns
```

---

## 📝 TÍCH HỢP VÀO APP.TSX

### Bước 1: Import các components mới

Thêm vào đầu file `src/App.tsx`:

```tsx
import { DashboardPanel } from './components/DashboardPanel';
import { ChartsPanel } from './components/ChartsPanel';
import { DateRangeSettingsPanel } from './components/DateRangeSettingsPanel';
```

### Bước 2: Thêm state cho tabs

Trong component `App`, thêm state:

```tsx
const [activeTab, setActiveTab] = useState<'cameras' | 'dashboard' | 'charts' | 'logs'>('cameras');
```

### Bước 3: Thêm navigation tabs

Thêm vào Toolbar hoặc tạo navigation mới:

```tsx
<div className="flex gap-2 border-b">
  <button 
    onClick={() => setActiveTab('cameras')}
    className={activeTab === 'cameras' ? 'active' : ''}
  >
    📹 Camera
  </button>
  
  <button 
    onClick={() => setActiveTab('dashboard')}
    className={activeTab === 'dashboard' ? 'active' : ''}
  >
    📊 Dashboard
  </button>
  
  <button 
    onClick={() => setActiveTab('charts')}
    className={activeTab === 'charts' ? 'active' : ''}
  >
    📈 Biểu đồ
  </button>
  
  <button 
    onClick={() => setActiveTab('logs')}
    className={activeTab === 'logs' ? 'active' : ''}
  >
    📝 Logs
  </button>
</div>
```

### Bước 4: Render theo tab

Replace nội dung chính:

```tsx
{activeTab === 'cameras' && (
  // Existing camera grid
  <CameraGrid ... />
)}

{activeTab === 'dashboard' && (
  <DashboardPanel />
)}

{activeTab === 'charts' && (
  <ChartsPanel />
)}

{activeTab === 'logs' && (
  <LogPanel />
)}
```

---

## 🎯 SỬ DỤNG CÁC COMPONENTS

### 1. Dashboard Panel

```tsx
// Sử dụng standalone
<DashboardPanel />
```

**Tính năng:**
- ✅ Hiển thị tổng quan (tổng số phòng, học sinh ngủ gật, đang ngủ gật, sự kiện)
- ✅ Grid view tất cả camera với màu cảnh báo (xanh/vàng/đỏ)
- ✅ Danh sách real-time học sinh đang ngủ gật
- ✅ Nút xuất báo cáo PDF/Excel
- ✅ Auto-refresh mỗi 5 giây

### 2. Charts Panel

```tsx
// Sử dụng standalone
<ChartsPanel />
```

**Tính năng:**
- ✅ Line chart: Xu hướng ngủ gật theo giờ trong ngày
- ✅ Bar chart: So sánh số lượng giữa các phòng
- ✅ Pie chart: Phân bố phần trăm theo phòng
- ✅ Chọn period: today/week/month

### 3. Date Range Settings Panel

```tsx
// Sử dụng cho một camera cụ thể
<DateRangeSettingsPanel 
  cameraId="camera_1" 
  cameraName="Phòng 101"
/>

// Sử dụng cho tất cả camera
<DateRangeSettingsPanel />
```

**Tính năng:**
- ✅ Date picker chọn ngày bắt đầu và kết thúc
- ✅ Quick presets (hôm nay, 7 ngày, 30 ngày, tháng này)
- ✅ Xem thống kê khoảng thời gian tùy chỉnh
- ✅ Xuất báo cáo PDF/Excel cho khoảng thời gian đã chọn
- ❌ Không có âm thanh (theo yêu cầu)

**Tích hợp vào CameraGrid:**

Thêm nút vào mỗi camera card:

```tsx
<div className="camera-card-actions">
  <DateRangeSettingsPanel 
    cameraId={camera.id}
    cameraName={camera.name}
  />
</div>
```

---

## 📡 API ENDPOINTS ĐÃ THÊM

### 1. Export PDF Report

```http
POST /api/logs/export/pdf
Content-Type: application/json

{
  "period": "today",           // "today" | "week" | "month" | "YYYY-MM-DD_YYYY-MM-DD"
  "camera_ids": ["camera_1"]   // Optional, null = all cameras
}

Response: PDF file download
```

### 2. Export Excel Report

```http
POST /api/logs/export/excel
Content-Type: application/json

{
  "period": "2025-11-01_2025-11-10",
  "camera_ids": null  // All cameras
}

Response: Excel file download (3 sheets: Tổng quan, Thống kê phòng, Chi tiết sự kiện)
```

---

## 🎨 MÃ MẪU TÍCH HỢP HOÀN CHỈNH

### App.tsx (Updated)

```tsx
import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { DashboardPanel } from './components/DashboardPanel';
import { ChartsPanel } from './components/ChartsPanel';
import { CameraGrid } from './components/CameraGrid';
import { LogPanel } from './components/LogPanel';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="h-screen flex flex-col">
      <Toolbar />
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
        <TabsList className="w-full justify-start">
          <TabsTrigger value="dashboard">📊 Dashboard</TabsTrigger>
          <TabsTrigger value="cameras">📹 Camera</TabsTrigger>
          <TabsTrigger value="charts">📈 Biểu đồ</TabsTrigger>
          <TabsTrigger value="logs">📝 Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="flex-1 overflow-auto">
          <DashboardPanel />
        </TabsContent>

        <TabsContent value="cameras" className="flex-1 overflow-auto">
          <CameraGrid cameras={cameras} />
        </TabsContent>

        <TabsContent value="charts" className="flex-1 overflow-auto">
          <ChartsPanel />
        </TabsContent>

        <TabsContent value="logs" className="flex-1 overflow-auto">
          <LogPanel />
        </TabsContent>
      </Tabs>

      <StatusBar />
    </div>
  );
}
```

### Thêm DateRangeSettings vào Camera Card

```tsx
// Trong CameraCard.tsx hoặc tương tự
import { DateRangeSettingsPanel } from './DateRangeSettingsPanel';

function CameraCard({ camera }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{camera.name}</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Camera feed */}
        <img src={camera.stream} alt={camera.name} />
        
        {/* Actions */}
        <div className="flex gap-2 mt-4">
          <Button>Start</Button>
          <Button variant="outline">Stop</Button>
          <DateRangeSettingsPanel 
            cameraId={camera.id}
            cameraName={camera.name}
          />
        </div>
      </CardContent>
    </Card>
  );
}
```

---

## 🐛 TROUBLESHOOTING

### Lỗi: "Module not found: date-fns"

```bash
npm install date-fns
```

### Lỗi: "No module named 'reportlab'"

```bash
pip install reportlab pandas openpyxl matplotlib
```

### Backend không tạo được PDF

Kiểm tra thư mục `reports/` đã tồn tại:

```python
# Trong report_generator.py, __init__ tự động tạo folder
# Nhưng nếu gặp lỗi quyền, tạo thủ công:
mkdir reports
```

### CORS error khi fetch

Đảm bảo Flask CORS đã được config đúng:

```python
# Trong server_with_tracking_backup.py
from flask_cors import CORS
CORS(app)
```

---

## 📊 KẾT QUẢ DEMO

### Dashboard Panel:
- 4 summary cards (Tổng phòng, Học sinh ngủ gật, Đang ngủ gật, Tổng sự kiện)
- Grid view 3x3 các phòng học với màu cảnh báo
- Danh sách real-time học sinh đang ngủ gật
- Auto-refresh mỗi 5 giây

### Charts Panel:
- Tab 1: Line chart xu hướng theo giờ
- Tab 2: Bar chart so sánh các phòng
- Tab 3: Pie chart phân bố + legend chi tiết

### Date Range Settings:
- Calendar picker cho ngày bắt đầu/kết thúc
- 4 quick presets
- Hiển thị thống kê ngay trong dialog
- Nút xuất PDF/Excel

---

## 🚀 NEXT STEPS

1. **Cài đặt dependencies:**
   ```bash
   # Backend
   cd "Desktop UI for Drowsiness Detection/python-backend"
   pip install -r requirements.txt

   # Frontend
   cd ..
   npm install date-fns
   ```

2. **Test backend API:**
   ```bash
   python server_with_tracking_backup.py
   ```

   Thử gọi API:
   ```bash
   curl -X POST http://localhost:5000/api/logs/export/pdf \
     -H "Content-Type: application/json" \
     -d '{"period":"today"}' \
     --output test_report.pdf
   ```

3. **Tích hợp vào App.tsx** (xem mã mẫu ở trên)

4. **Test frontend:**
   ```bash
   npm run dev
   # hoặc
   npm run electron-dev
   ```

5. **Verify:**
   - Dashboard hiển thị đúng số liệu
   - Charts render các biểu đồ
   - Date picker hoạt động
   - Export PDF/Excel thành công

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:

1. Check console log (F12)
2. Check backend terminal output
3. Verify API endpoints đang chạy: http://localhost:5000/api/logs/cameras
4. Kiểm tra file `reports/` folder có tạo được không

---

## ✅ CHECKLIST HOÀN TẤT

- [ ] Cài đặt Python dependencies
- [ ] Cài đặt NPM dependencies
- [ ] Test backend API export PDF/Excel
- [ ] Import components vào App.tsx
- [ ] Thêm tabs/navigation
- [ ] Test DashboardPanel
- [ ] Test ChartsPanel
- [ ] Test DateRangeSettingsPanel
- [ ] Tích hợp DateRange vào camera cards
- [ ] Verify export reports hoạt động
- [ ] Test real-time updates (5s refresh)

---

**🎉 HOÀN TẤT! Bạn đã có đầy đủ 4 tính năng:**

1. ✅ Settings Panel (Date Range Picker) - Không âm thanh
2. ✅ Export Reports (PDF + Excel)
3. ✅ Charts & Graphs (Line, Bar, Pie)
4. ✅ Dashboard Real-time Monitoring
