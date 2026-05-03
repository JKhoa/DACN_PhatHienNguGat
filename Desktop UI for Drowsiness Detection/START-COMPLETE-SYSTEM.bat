@echo off
echo ========================================
echo    HỆ THỐNG PHÁT HIỆN NGỦ GẬT HOÀN CHỈNH
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Kiểm tra Python dependencies...
cd python-backend
python -c "import cv2, ultralytics, flask, flask_cors; print('✅ Python dependencies OK')" 2>nul
if errorlevel 1 (
    echo ❌ Installing Python dependencies...
    pip install opencv-python ultralytics flask flask-cors
) else (
    echo ✅ Python dependencies already installed
)
cd ..

echo.
echo [2/4] Building React app...
call npm run build
if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)
echo ✅ React app built successfully

echo.
echo [3/4] Starting Python backend...
start /B python python-backend/server.py
timeout /t 3 /nobreak >nul

echo.
echo [4/4] Starting Desktop App...
echo.
echo 🎯 Features:
echo - Empty camera slots ready for connection
echo - YOLO model integration for real-time detection
echo - Support for IP cameras (Hikvision, Dahua, Ezviz, KBVision)
echo - Support for USB webcams
echo - Real-time student tracking (10-50 students per camera)
echo - Drowsiness detection with confidence scores
echo - Event logging with timestamps
echo - System performance monitoring
echo.
echo 📱 How to use:
echo 1. Click "Thêm" to add new camera
echo 2. Configure IP camera or webcam settings
echo 3. Test connection before saving
echo 4. Click "Start All" to begin detection
echo 5. Monitor real-time results in grid and logs
echo.

call npm run electron

echo.
echo 🛑 Shutting down...
taskkill /F /IM python.exe 2>nul
echo ✅ System stopped

pause

