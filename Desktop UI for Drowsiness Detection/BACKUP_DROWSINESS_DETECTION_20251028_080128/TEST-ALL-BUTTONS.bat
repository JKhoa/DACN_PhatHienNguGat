@echo off
echo ========================================
echo    TEST TẤT CẢ BUTTONS VÀ DROPDOWNS
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Building React app with All Functional Buttons...
call npm run build
if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)
echo ✅ React app built successfully

echo.
echo [2/3] Starting Python backend...
start /B python python-backend/server.py
timeout /t 3 /nobreak >nul

echo.
echo [3/3] Starting Desktop App for Button Functionality Test...
echo.
echo 🎯 Button & Dropdown Functionality Test:
echo.
echo 📱 Toolbar Buttons:
echo - Start All: Khởi động tất cả camera
echo - Stop All: Dừng tất cả camera
echo - Add: Thêm camera mới
echo - Delete: Xóa camera đã chọn
echo - Import: Import cấu hình (placeholder)
echo - Export: Export cấu hình camera
echo - Save Layout: Lưu bố cục hiện tại
echo - Restore Layout: Khôi phục bố cục đã lưu
echo - Toggle Overlay: Bật/tắt overlay
echo - Toggle Performance: Bật/tắt hiệu năng
echo - Toggle Logging: Bật/tắt logging
echo - Toggle Theme: Bật/tắt dark mode
echo - Settings: Mở cài đặt hệ thống
echo.
echo 📱 Camera Card Dropdown:
echo - Hiện/Ẩn Chi tiết Tracking: Toggle tracking details
echo - Pop Out: Mở camera trong window mới
echo - Cấu hình: Mở dialog cấu hình camera
echo - Toggle Overlay: Bật/tắt overlay cho camera
echo - Toggle Logging: Bật/tắt logging cho camera
echo - Chụp ảnh: Chụp ảnh từ camera
echo - Ghi video: Bắt đầu ghi video từ camera
echo.
echo 📱 Camera Grid Controls:
echo - Grid Size Selector: Thay đổi kích thước grid (1x1, 2x2, 3x3, 4x4)
echo.
echo 📱 Log Panel Controls:
echo - Search: Tìm kiếm trong logs
echo - Filter by Camera: Lọc theo camera
echo - Filter by Type: Lọc theo loại log
echo - Export Logs: Xuất logs ra CSV
echo.
echo 📱 Settings Dialog:
echo - Model & Detection Tab: Cấu hình model
echo - Hiệu năng Tab: Cấu hình hiệu năng
echo - Giao diện Tab: Cấu hình giao diện
echo - Cấu hình Tab: Cấu hình hệ thống
echo.
echo 🧪 Test Steps:
echo 1. Test tất cả Toolbar buttons
echo 2. Thêm camera và test dropdown menu
echo 3. Test Camera Grid controls
echo 4. Test Log Panel controls
echo 5. Test Settings Dialog
echo 6. Verify tất cả buttons đều có response
echo.

call npm run electron

echo.
echo 🛑 Shutting down...
taskkill /F /IM python.exe 2>nul
echo ✅ System stopped

pause

