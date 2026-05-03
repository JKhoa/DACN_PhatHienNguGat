# Phát hiện ngủ gật trong lớp học

Dự án bao gồm desktop UI hoàn chỉnh và CLI cho hệ thống phát hiện ngủ gật đa camera trong lớp học, sử dụng Ultralytics YOLO pose.

> **Bắt đầu nhanh nhất:** xem [QUICKSTART.md](./QUICKSTART.md) — hướng dẫn chi tiết cài đặt và chạy localhost (Windows) cho UI hiện tại (React/Vite + Python backend).

## Bắt đầu nhanh (CLI / PyQt5 GUI cũ)

1) Tạo và kích hoạt môi trường ảo (Windows PowerShell):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2) Cài đặt dependencies:

```powershell
pip install -r requirements.txt
```

3) Chạy GUI (khuyến nghị):

```powershell
python standalone_app.py --gui
```

- Mặc định model tự động resolve về `yolo11n-pose.pt` trong repo nếu có; nếu không sẽ tự download qua Ultralytics.
- Trong GUI bạn có thể chọn nguồn (webcam, RTSP/HTTP, video, ảnh), đổi model, ghi video đã annotate, và dùng tab Multi-Camera.

4) Ví dụ CLI:

```powershell
# Webcam
python standalone_app.py --cam 0 --res 1280x720

# IP camera (ví dụ IMOU/Dahua)
python standalone_app.py --ip-camera --ip 192.168.1.100 --username admin --password 123456 --camera-brand imou --stream-quality main

# File video
python standalone_app.py --video data_raw/cap_000000.jpg
```

## Cấu trúc dự án

- `standalone_app.py` — launcher gốc, dispatch tới app đầy đủ trong `yolo-sleepy-allinone-final/`.
- `yolo-sleepy-allinone-final/gui_app.py` — Desktop UI hoàn chỉnh với các tab và layout hiện đại.
- `yolo-sleepy-allinone-final/multi_camera_gui.py` — Quản lý đa camera (thêm/sửa/test IP/Webcam, start/stop tất cả, chế độ grid/single view).
- `yolo-sleepy-allinone-final/standalone_app.py` — CLI runner hỗ trợ video/webcam/IP camera.
- `yolo-sleepy-allinone-final/enhanced_display.py` — Overlay multi-person nâng cao.
- `yolo-sleepy-allinone-final/camera_core.py` — Tiện ích capture dùng chung, có hàng đợi frame chạy thread riêng để giảm độ trễ.

## Lưu ý

- Torch (PyTorch) không pin version trong `requirements.txt` vì phải khớp với GPU/CPU và nền tảng của bạn. Cài Torch trước nếu cần, theo hướng dẫn tại https://pytorch.org.
- Nếu thiếu PyQt5 hoặc chạy với `--cli`, app sẽ fallback về chế độ console.
- Với tab multi-camera, cấu hình có thể save/load dạng YAML; `PyYAML` đã có sẵn trong requirements gốc.

## Web UI (React/Vite + Python backend) — phiên bản hiện tại

Dự án có một web UI hiện đại, viết bằng React + Vite, **chạy localhost**, đặt tại:

- `Desktop UI for Drowsiness Detection/`

Đây là **giao diện chính đang được phát triển** — chỉ chạy ở browser localhost, không còn dùng Electron. Để chạy nhanh:

```powershell
# Cài Node.js nếu chưa có (https://nodejs.org)

# Từ thư mục gốc repo
cd "Desktop UI for Drowsiness Detection"
npm install
npm run dev
# Mở URL hiển thị (thường là http://localhost:3000/)
```

Hoặc chạy qua script tiện ích từ thư mục gốc:

```powershell
python start_desktop_ui.py
```

> **Hướng dẫn đầy đủ kèm troubleshooting:** [QUICKSTART.md](./QUICKSTART.md)

## Phát hiện realtime qua WebSocket (frontend stream video, backend trả kết quả)

Dự án đã tích hợp đường WebSocket độ trễ thấp cho phát hiện qua webcam:

- **Frontend:** stream webcam qua `getUserMedia`, capture frame, gửi qua Socket.IO WebSocket.
- **Backend (Flask + Socket.IO):** nhận frame, chạy model phát hiện ngủ gật (YOLO pose), trả ngay kết quả có cấu trúc (ID, bounding box đầu, trạng thái).
- **Frontend:** vẽ overlay kết quả (xanh = bình thường, đỏ = buồn ngủ, tím = gục bàn) trực tiếp lên video đang phát; video vẫn chạy local nên độ trễ rất thấp.

Cách chạy:

```powershell
# 1) Trong Python venv, khởi động backend kèm hỗ trợ WebSocket
python start_python_backend.py

# 2) Mở terminal khác, khởi động React UI
cd "Desktop UI for Drowsiness Detection"
npm install
npm run dev
```

Cách hoạt động:

- WebSocket namespace: `ws://127.0.0.1:5000/ws/detect`
- Frontend client: `src/lib/wsDetection.ts`
- Component webcam: `src/components/CameraCard.tsx` — stream frame qua WS và vẽ overlay từ kết quả phát hiện.
- IP camera: vẫn được backend quản lý bằng thread; UI poll `/api/camera/<id>/detection` để lấy kết quả và vẽ overlay. Có thể chuyển sang WebSocket cho từng camera sau nếu cần.

> **Mẹo:** Nếu chỉ muốn nhận kết quả phát hiện (không cần truyền ảnh đã annotate qua mạng), giữ video element ở local (webcam) và dùng kết quả WS để vẽ lên canvas overlay.
