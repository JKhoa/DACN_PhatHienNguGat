# QUICKSTART — Hướng dẫn chạy localhost (Windows)

App phát hiện ngủ gật + sử dụng điện thoại. Frontend React/TS (Vite) + Backend Python Flask/SocketIO + YOLO ensemble.

> **Folder làm việc:** mọi lệnh dưới đây chạy trong `Desktop UI for Drowsiness Detection/`.

---

## 1. Yêu cầu hệ thống

| Thành phần | Phiên bản khuyến nghị | Ghi chú |
|---|---|---|
| **OS** | Windows 10/11 | Đã test trên Windows 11 |
| **Python** | 3.10 hoặc 3.11 | KHÔNG dùng 3.12+ (ultralytics có thể lỗi) |
| **Node.js** | 18.x hoặc 20.x LTS | Tải từ https://nodejs.org |
| **Camera** | USB webcam (index 0) | Hoặc camera tích hợp laptop |
| **GPU (tùy chọn)** | NVIDIA + CUDA 11.8/12.x | Không bắt buộc; chạy CPU vẫn được, chậm hơn |
| **RAM** | ≥ 8 GB | YOLO ensemble cần ~2GB |
| **Ổ đĩa** | ~3 GB | Cho `node_modules` + môi trường Python |

Kiểm tra nhanh:
```powershell
python --version    # 3.10.x hoặc 3.11.x
node --version      # v18.x.x hoặc v20.x.x
npm --version
```

---

## 2. Clone repo

```powershell
git clone -b ui-new-dev https://github.com/JKhoa/DACN_PhatHienNguGat.git
cd DACN_PhatHienNguGat
cd "Desktop UI for Drowsiness Detection"
```

---

## 3. Cài đặt dependencies

### 3.1 Python backend

```powershell
# Tạo virtual env (khuyến nghị)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Nếu PowerShell chặn script: chạy 1 lần với quyền user
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install --upgrade pip
pip install -r python-backend\requirements.txt
```

**Cài torch theo phần cứng:**
- **CPU only** (đơn giản, chậm hơn): `requirements.txt` đã đủ.
- **GPU NVIDIA (CUDA 12.1)**: cài lại torch GPU sau bước trên:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

### 3.2 Frontend (Node)

```powershell
npm install
```

Lệnh này cài ~700 packages, mất 2-5 phút lần đầu.

---

## 4. Cấu hình ENV (tùy chọn)

Tạo file `python-backend\.env` từ template:

```powershell
Copy-Item python-backend\.env.example python-backend\.env
notepad python-backend\.env
```

Nội dung:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Chỉ cần** nếu dùng tính năng **AI chatbot** (Google Gemini). Nếu không cần, bỏ qua — phần phát hiện ngủ gật vẫn chạy bình thường.
>
> Lấy key miễn phí tại: https://aistudio.google.com/apikey

---

## 5. Kiểm tra weight models

Các file `.pt` phải có sẵn ở vị trí sau (đã đi kèm trong repo):

```
Desktop UI for Drowsiness Detection/
├── yolo11n-pose.pt
├── python-backend/
│   ├── yolo11n-pose.pt
│   └── models/
│       ├── yolo11n.pt           # Person detector
│       ├── drowsiness_cls.pt    # Drowsy/Non-Drowsy classifier
│       └── phone_det.pt         # Phone bbox detector
```

Nếu thiếu file nào, tải lại bằng:
```powershell
python python-backend\download_models.py
```

---

## 6. Chạy app (localhost)

```powershell
.\start-web.bat
```

Lệnh này sẽ:
1. Mở cửa sổ riêng chạy backend Python tại `http://127.0.0.1:5000`
2. Khởi động Vite dev server tại `http://localhost:3000` và tự động mở trình duyệt

Hoặc chạy thủ công 2 terminal:

**Terminal 1 (Backend):**
```powershell
.\.venv\Scripts\Activate.ps1
cd python-backend
python server.py
```

**Terminal 2 (Frontend):**
```powershell
npm run dev
```

Mở browser: http://localhost:3000

> **Lưu ý:** App đã được chuyển sang **chỉ chạy localhost**. Phiên bản Electron desktop trước đây đã được loại bỏ — nếu cần có thể quay lại git history (commit trước `5bf6c87`).

---

## 7. Kiểm tra hoạt động

1. Browser hiện UI dashboard, tab **Camera** hiển thị video stream từ webcam.
2. Bounding box vẽ quanh người trong khung hình.
3. Khi nhắm mắt / cúi đầu → label **"Ngủ gật"** (đỏ) + cảnh báo âm thanh.
4. Khi cầm điện thoại lên → label **"Điện thoại"** / **"Bấm điện thoại"** (vàng).

Test backend riêng:
```powershell
curl http://127.0.0.1:5000/api/v1/detect/health
# Kỳ vọng: {"status":"ok",...}
```

---

## 8. Troubleshooting

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Chưa activate venv. Chạy `.\.venv\Scripts\Activate.ps1` rồi cài lại requirements. |
| Backend báo `Camera not found` / index 0 lỗi | Webcam đang bị app khác chiếm (Zoom, Teams, OBS). Đóng app đó. Hoặc đổi `CAMERA_INDEX=1` trong `.env`. |
| `torch.cuda.is_available() == False` mà có GPU | Cài đúng phiên bản CUDA-torch ở mục 3.1. Kiểm tra `nvidia-smi`. |
| Vite báo port 3000 đang dùng | Đổi port trong `vite.config.ts` (`server.port`) hoặc kill process: `netstat -ano \| findstr :3000` rồi `taskkill /PID <pid> /F`. |
| `Failed to load model drowsiness_cls.pt` | File chưa tải đủ. Chạy `python python-backend\download_models.py`. |
| Frontend không kết nối WebSocket | Backend chưa chạy hoặc bị firewall chặn. Kiểm tra `http://127.0.0.1:5000/api/v1/detect/health`. |
| `npm install` fail trên Windows | Chạy PowerShell as Admin, hoặc xóa `node_modules` + `package-lock.json` rồi cài lại. |
| Detection chậm (>1s/frame) | Bình thường nếu chạy CPU. Bật GPU (mục 3.1) hoặc giảm độ phân giải video stream trong cài đặt. |

---

## 9. Cấu trúc dự án (tóm tắt)

```
Desktop UI for Drowsiness Detection/
├── src/                    # React/TS frontend
│   ├── App.tsx
│   ├── components/
│   ├── lib/
│   └── config/env.ts
├── python-backend/
│   ├── server.py           # Flask + SocketIO entry
│   ├── api_v1.py           # REST API blueprint
│   ├── detectors/          # YOLO ensemble logic
│   ├── models/             # Weights (.pt)
│   └── requirements.txt
├── electron/               # Electron wrapper
│   ├── main.js
│   └── preload.js
├── start-web.bat           # Chạy web localhost
├── start-desktop.bat       # Chạy Electron desktop
├── package.json            # Frontend deps
└── vite.config.ts
```

---

## 10. Tham khảo thêm

- `python-backend/README_detection_v1.md` — chi tiết REST API v1 (`/api/v1/detect/{health,image,video,realtime}`)
- `README.md` — tổng quan ngắn (đã có sẵn)
- Issue/feedback: https://github.com/JKhoa/DACN_PhatHienNguGat/issues
