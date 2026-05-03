
  # Desktop UI cho Phát hiện ngủ gật

  Web app phát hiện ngủ gật + sử dụng điện thoại — chạy localhost. Frontend React/TS (Vite) + Backend Python Flask/SocketIO + YOLO ensemble.

  > **Hướng dẫn đầy đủ (yêu cầu hệ thống, cài đặt, troubleshooting):** xem [`../QUICKSTART.md`](../QUICKSTART.md).

  ## Cài đặt

  ```
  npm install
  pip install -r python-backend/requirements.txt
  ```

  ## Chạy localhost

  ```
  start-web.bat
  ```

  Script này khởi động:
  1. Backend Python tại `http://127.0.0.1:5000` (cửa sổ riêng)
  2. Vite dev server tại `http://localhost:3000` và tự động mở trình duyệt

  Hoặc chạy thủ công 2 terminal:

  **Terminal 1 — Backend:**
  ```
  cd python-backend
  python server.py
  ```

  **Terminal 2 — Frontend:**
  ```
  npm run dev
  ```

  Sau đó mở `http://localhost:3000` trong browser.

  ## Cấu hình

  Frontend gọi trực tiếp `http://127.0.0.1:5000` qua `fetch` và `socket.io-client`. Có thể override URL backend bằng biến môi trường `VITE_BACKEND_URL` nếu cần (vd backend ở host khác).
