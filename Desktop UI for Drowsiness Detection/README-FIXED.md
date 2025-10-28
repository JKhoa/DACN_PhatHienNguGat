# Hệ thống Phát hiện Ngủ gật - Desktop App

## ✅ Đã sửa các lỗi:
- ❌ Lỗi Python không tìm thấy → ✅ Đã loại bỏ dependency Python
- ❌ Lỗi file dist/index.html không tồn tại → ✅ Đã sửa cấu hình Vite
- ❌ Lỗi Electron không khởi động → ✅ Đã cập nhật main.js

## 🚀 Cách khởi động ứng dụng:

### Phương pháp 1: Khởi động nhanh (Khuyến nghị)
```
Double-click: START-APP-SIMPLE.bat
```

### Phương pháp 2: Development mode
```
Double-click: START-DEV-MODE.bat
```

### Phương pháp 3: Thủ công
```bash
# Build ứng dụng
npm run build

# Chạy desktop app
npm run electron
```

## 📱 Tính năng ứng dụng:

### 🎥 Camera Phòng học
- **6 Camera**: Phòng học 1A, 1B, 2A, 2B, 3A, 3B
- **Video Feed**: Hiển thị trực tiếp từ camera (simulated)
- **Tracking**: Phát hiện và theo dõi học sinh

### 👁️ Phát hiện Ngủ gật
- **AI Detection**: Sử dụng thuật toán phát hiện tư thế
- **Real-time**: Phát hiện ngay lập tức
- **Visual Markers**: Đánh dấu học sinh buồn ngủ bằng màu đỏ

### 📊 Dashboard Thống kê
- **Tổng Camera**: 6 camera phòng học
- **Camera Hoạt động**: Số camera đang giám sát
- **Tổng Học sinh**: Số học sinh được phát hiện
- **Học sinh Buồn ngủ**: Số học sinh cần chú ý
- **Mức Cảnh báo**: Thấp/Trung bình/Cao

## 🎯 Cách sử dụng:

1. **Khởi động**: Chạy `START-APP-SIMPLE.bat`
2. **Bật Camera**: Nhấn nút "Bật" trên camera muốn giám sát
3. **Bật Tất cả**: Nhấn "Bật tất cả" để giám sát toàn bộ
4. **Theo dõi**: Quan sát các điểm đánh dấu học sinh
5. **Cảnh báo**: Chú ý các điểm màu đỏ (học sinh buồn ngủ)

## 🔧 Troubleshooting:

### Lỗi "Cannot find module"
```bash
npm install
```

### Lỗi build
```bash
npm run build
```

### Lỗi Electron
```bash
npm run electron
```

## 📁 Cấu trúc file:
```
Desktop UI for Drowsiness Detection/
├── START-APP-SIMPLE.bat      # Khởi động nhanh
├── START-DEV-MODE.bat        # Development mode
├── electron/main.js          # Electron main process
├── dist/                     # Built application
│   ├── index.html
│   └── assets/
├── src/
│   ├── components/
│   │   ├── ClassroomDashboard.tsx
│   │   └── RealCameraCard.tsx
│   └── App.tsx
└── package.json
```

## ✅ Trạng thái:
- ✅ Ứng dụng desktop hoạt động
- ✅ Giao diện Classroom Dashboard
- ✅ Camera simulation với tracking
- ✅ Phát hiện ngủ gật (simulated)
- ✅ Thống kê real-time
- ✅ Không cần Python backend

---
**Lưu ý**: Ứng dụng hiện tại sử dụng simulation data. Để tích hợp camera thực tế, cần kết nối với backend Python YOLO.

