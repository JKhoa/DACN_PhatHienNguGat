# ĐẶC TẢ GIAO DIỆN (UI SPECIFICATION) - AI STUDENT MONITORING V2.0

## 1. Cấu trúc Thư mục & Công nghệ
- **Framework:** React (Vite) + Tailwind CSS.
- **Thư viện UI:** Shadcn UI (Radix UI), Lucide Icons, Framer Motion (cho animation).
- **Biểu đồ:** Recharts.
- **Kết nối:** WebSocket (SocketIO) cho luồng nhận diện thời gian thực.

## 2. Các Module Chính & Yêu cầu Chi tiết

### A. Dashboard Tổng quan (Analytics)
- **Top Stats Cards:** 4 thẻ hiển thị chỉ số tổng quát:
  1. `Số học sinh hiện diện`: Lấy từ tổng `Person` nhận diện được trên tất cả camera.
  2. `Cảnh báo Ngủ gật`: Tổng số sự kiện `drowsy` trong phiên.
  3. `Sử dụng Điện thoại`: Tổng số sự kiện `dien_thoai` từ ensemble detector.
  4. `Chỉ số tập trung`: (Tổng thời gian tỉnh táo / Tổng thời gian học) * 100.
- **Main Charts:**
  - **Biểu đồ Cột (Hourly Violation):** Trục X là thời gian (giờ), Trục Y là số lần vi phạm. Dùng Stacked Bar để tách màu Đỏ (Điện thoại) và Vàng (Ngủ gật).
  - **Biểu đồ Tròn (Behavior Distribution):** Tỷ lệ Awake vs Drowsy vs Phone Usage.

### B. Live Monitoring (Multi-Camera Grid)
- **Grid Layout:** Hỗ trợ linh hoạt 1x1, 2x2, 3x3.
- **Focus Mode:** Khi click vào một CameraCard, nó sẽ bung to ra (dùng Framer Motion layout transition) và hiển thị thêm biểu đồ EAR Real-time bên dưới.
- **Camera Card UI:**
  - **Status Badge:** Hiển thị `Connected`, `FPS`, `Latency (ms)`.
  - **Drawing Logic (Đồng nhất):** 
    - Cả `ngu_gat` và `dien_thoai` đều sử dụng khung bo góc (Rounded corners) với nét vẽ dày 3px.
    - Label background trùng màu với khung, text trắng.
    - **Phone:** Màu Đỏ (#EF4444).
    - **Drowsy:** Màu Vàng Cam (#F59E0B).
  - **Alert Effect:** Khi phát hiện vi phạm, viền Card nhấp nháy (Pulse effect) theo màu tương ứng.

### C. Quản lý (CRUD)
- **Student Management:** 
  - Table hiển thị danh sách SV từ Database.
  - Form Thêm/Sửa (Modal) với các trường: MSSV, Tên, Lớp, Ảnh đại diện.
- **Camera Configuration:**
  - Danh sách nguồn (Webcam ID, RTSP URL, IP Camera).
  - Slider điều chỉnh `Sensitivity` (Ngưỡng EAR và YOLO Confidence).

### D. Nhật ký (History Logs)
- Hiển thị danh sách sự kiện theo thời gian thực (Real-time feed).
- Mỗi log entry có: Snapshot (nếu có), Loại vi phạm (Badge màu), Thời gian, Tên Camera.

## 3. Danh sách kiểm tra (Checklist)
- [ ] Giao diện đồng nhất giữa Phone và Drowsy detection.
- [ ] Biểu đồ hoạt động mượt mà với dữ liệu từ backend.
- [ ] Hỗ trợ chuyển đổi Dark/Light mode toàn diện.
- [ ] Hiệu ứng chuyển cảnh Camera chuyên nghiệp.
- [ ] Chức năng CRUD hoạt động qua Modal.
