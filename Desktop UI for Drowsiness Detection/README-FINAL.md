# Hệ thống Phát hiện Ngủ gật - Desktop App

## Mô tả
Ứng dụng Desktop chuyên dụng cho việc giám sát và phát hiện tình trạng ngủ gật của học sinh trong các phòng học, sử dụng công nghệ YOLO và Pose Detection.

## Tính năng chính

### 🎥 Giám sát Camera Phòng học
- Quản lý nhiều camera phòng học đồng thời
- Hiển thị trực tiếp video feed từ camera
- Tracking và phát hiện học sinh trong thời gian thực

### 👁️ Phát hiện Ngủ gật Thông minh
- Sử dụng AI để phát hiện tư thế ngủ gật
- Theo dõi và đánh dấu học sinh buồn ngủ
- Cảnh báo tự động khi phát hiện ngủ gật

### 📊 Dashboard Giám sát
- Tổng quan tình trạng tất cả phòng học
- Thống kê số lượng học sinh và tình trạng
- Mức độ cảnh báo theo thời gian thực

### 🖥️ Ứng dụng Desktop
- Giao diện desktop chuyên dụng
- Không cần trình duyệt web
- Tích hợp sâu với hệ điều hành

## Cách khởi động

### Phương pháp 1: Khởi động nhanh
1. Double-click vào file `START-DESKTOP-APP.bat`
2. Đợi ứng dụng desktop mở

### Phương pháp 2: Khởi động thủ công
```bash
# Cài đặt dependencies
npm install

# Build ứng dụng
npm run build

# Khởi động desktop app
npm run electron
```

### Phương pháp 3: Development mode
```bash
# Khởi động development server
npm run dev

# Trong terminal khác, khởi động Electron
npm run electron-dev
```

## Cấu trúc dự án
```
Desktop UI for Drowsiness Detection/
├── src/
│   ├── components/
│   │   ├── ClassroomDashboard.tsx    # Dashboard chính
│   │   ├── RealCameraCard.tsx       # Component camera thực tế
│   │   └── ui/                      # UI components
│   ├── App.tsx                      # App chính
│   └── types/                       # TypeScript types
├── electron/
│   └── main.js                      # Electron main process
├── build/                           # Built application
└── START-DESKTOP-APP.bat           # Script khởi động
```

## Tính năng Camera

### Camera Phòng học
- **Camera Phòng học 1A**: Giám sát phòng học tầng 1
- **Camera Phòng học 1B**: Giám sát phòng học tầng 1
- **Camera Phòng học 2A**: Giám sát phòng học tầng 2
- **Camera Phòng học 2B**: Giám sát phòng học tầng 2
- **Camera Phòng học 3A**: Giám sát phòng học tầng 3
- **Camera Phòng học 3B**: Giám sát phòng học tầng 3

### Tracking Học sinh
- Phát hiện tự động học sinh trong khung hình
- Theo dõi vị trí và chuyển động
- Đánh dấu học sinh buồn ngủ bằng màu đỏ
- Hiển thị tên và độ tin cậy phát hiện

## Yêu cầu hệ thống
- **OS**: Windows 10/11
- **RAM**: 8GB+ (khuyến nghị 16GB)
- **CPU**: Intel i5 hoặc tương đương
- **GPU**: NVIDIA GTX 1060+ (khuyến nghị)
- **Camera**: Webcam hoặc camera USB
- **Node.js**: 18+
- **Python**: 3.8+

## Troubleshooting

### Lỗi "Cannot find module"
```bash
npm install
```

### Lỗi Electron không khởi động
```bash
npm run build
npm run electron
```

### Lỗi Camera không hoạt động
1. Kiểm tra camera có được kết nối
2. Cấp quyền truy cập camera cho ứng dụng
3. Kiểm tra driver camera

### Lỗi Python backend
```bash
cd ../yolo-sleepy-allinone-final
pip install -r requirements.txt
python standalone_app.py
```

## Hướng dẫn sử dụng

### 1. Khởi động ứng dụng
- Chạy `START-DESKTOP-APP.bat`
- Đợi ứng dụng desktop mở

### 2. Bật Camera
- Nhấn nút "Bật" trên camera muốn giám sát
- Hoặc nhấn "Bật tất cả" để giám sát toàn bộ

### 3. Theo dõi
- Xem video feed trực tiếp từ camera
- Quan sát các điểm đánh dấu học sinh
- Chú ý các điểm màu đỏ (học sinh buồn ngủ)

### 4. Thống kê
- Xem tổng số học sinh được phát hiện
- Kiểm tra số lượng học sinh buồn ngủ
- Theo dõi mức độ cảnh báo

## Liên hệ và Hỗ trợ
Nếu gặp vấn đề, vui lòng:
1. Kiểm tra log trong terminal
2. Kiểm tra camera và kết nối
3. Tạo issue với thông tin chi tiết

---
**Lưu ý**: Ứng dụng này được thiết kế để giám sát phòng học và hỗ trợ giáo dục. Vui lòng tuân thủ các quy định về quyền riêng tư và bảo mật.

