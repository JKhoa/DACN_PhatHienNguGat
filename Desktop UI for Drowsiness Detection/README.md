
  # Desktop UI cho Phát hiện ngủ gật

  Đây là code bundle cho Desktop UI Phát hiện ngủ gật. Bản thiết kế gốc trên Figma: https://www.figma.com/design/LSvpZ0P3ysfvfn4Ru2PieR/Desktop-UI-for-Drowsiness-Detection.

  > **Hướng dẫn đầy đủ (yêu cầu hệ thống, cài đặt, troubleshooting):** xem [`../QUICKSTART.md`](../QUICKSTART.md).

  ## Cài đặt

  ```
  npm install
  pip install -r python-backend/requirements.txt
  ```

  ## Chạy app — hai chế độ

  **Desktop (Electron)** — một cửa sổ duy nhất, Electron tự động khởi chạy backend Python:

  ```
  start-desktop.bat
  ```

  **Web (localhost)** — backend Python + Vite dev server hiển thị trên trình duyệt:

  ```
  start-web.bat
  ```

  Frontend tự động phát hiện đang chạy ở chế độ nào (dựa vào sự xuất hiện của `window.appApi`).
  Ở chế độ web, frontend giao tiếp trực tiếp với `http://127.0.0.1:5000` qua `fetch` và
  `socket.io-client`. Có thể override URL backend bằng biến môi trường `VITE_BACKEND_URL` nếu cần.
