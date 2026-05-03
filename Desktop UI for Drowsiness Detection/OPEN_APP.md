# Hướng dẫn mở ứng dụng Phát hiện Ngủ gật

## 🚀 Khởi động nhanh (< 2 phút)

### Yêu cầu
- **Node.js** 18+ (kiểm tra: `node -v`)
- **Python** 3.10+ (kiểm tra: `python --version`)
- **Electron** 38+ (cài qua npm, tự động)
- Thư mục dự án: `D:\Study\DoAnChuyenNganh\DACN_PhatHienNguGat\Desktop UI for Drowsiness Detection`

### Các bước

1. **Mở Terminal/PowerShell trong thư mục dự án**
   ```bash
   cd "D:\Study\DoAnChuyenNganh\DACN_PhatHienNguGat\Desktop UI for Drowsiness Detection"
   ```

2. **Chạy ứng dụng**
   ```bash
   npm start
   ```
   
   Ứng dụng sẽ:
   - Build React app (vite) → 5-6 giây
   - Khởi động Python backend → ~40 giây (lần đầu cài dependencies)
   - Mở cửa sổ Electron với DevTools (detach)
   - In ra log trong Terminal

3. **Chờ khoảng 60 giây** để Python backend sẵn sàng
   - Log: `⏳ Waiting for Python backend... (1-60)`
   - Khi ready: `✅ Python backend ready (attempt N)`
   - React app tự động load

### ✅ Kiểm tra khởi động thành công

Bạn sẽ thấy:
- [ ] Cửa sổ Electron mở ra
- [ ] Giao diện React với 4 tab: 📷 Camera | 📊 Dashboard | 📈 Biểu đồ | 💬 Chat AI
- [ ] DevTools console (bên cạnh) không có lỗi đỏ
- [ ] Terminal: `[wsBridge] IPC channels registered` + `Loading React app from:...`

---

## 📋 Troubleshooting

| Triệu chứng | Nguyên nhân | Cách fix |
|------------|-----------|---------|
| `npm: command not found` | Node.js chưa cài | Cài từ nodejs.org, restart Terminal |
| `python: command not found` | Python chưa cài | Cài từ python.org, restart Terminal |
| Window đen / không hiển thị React | Python đang load YOLO model | Chờ 40-60 giây |
| Terminal báo `Module not found` | Dependencies chưa cài | Chạy `npm install` rồi `npm start` |
| Cửa sổ Electron gặp lỗi | DevTools sẽ hiện error → copy + gửi cho dev |

---

## 🔧 Lệnh hữu ích

```bash
# Build React production (không chạy Electron)
npm run build

# Dev mode (Vite hot-reload)
npm run dev

# Chỉ chạy Electron (giả sử đã build xong)
npx electron .

# Xóa cache build + dependencies (nếu bị bug lạ)
rm -r node_modules dist
npm install
npm start
```

---

## 💡 Ghi chú

- **DevTools**: Đã bật mặc định (detach mode). Có thể tắt sau bằng cách comment dòng 33 trong `electron/main.js` trước khi bảo vệ.
- **Database**: Lưu tại `python-backend/drowsiness_logs/events.db` (SQLite)
- **Network**: App **không** kết nối internet, chỉ dùng `127.0.0.1:5000` qua IPC (an toàn)
- **Log file**: Terminal log có thể scroll lên xem full history

---

**Trở lại dự án:** Chỉ cần chạy `npm start` + chờ ~60s là sẵn sàng demo! 🎉
