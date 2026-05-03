@echo off
echo ========================================
echo    TEST STUDENT TRACKING CHI TIẾT
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Building React app with Student Tracking Details...
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
echo [3/3] Starting Desktop App for Student Tracking Test...
echo.
echo 🎯 Student Tracking Features:
echo - Real-time student detection with YOLO
echo - Detailed student information display
echo - Position tracking (x, y coordinates)
echo - State classification (Normal, Sleepy, Head Down)
echo - Confidence scores for each detection
echo - Sleep duration tracking
echo - Bounding box information
echo - Last update timestamps
echo.
echo 📱 How to test:
echo 1. Click "Thêm" to add a camera
echo 2. Configure IP camera or webcam
echo 3. Click "Start" to begin detection
echo 4. Click "..." menu on camera card
echo 5. Select "Hiện Chi tiết Tracking"
echo 6. View detailed student information
echo.

call npm run electron

echo.
echo 🛑 Shutting down...
taskkill /F /IM python.exe 2>nul
echo ✅ System stopped

pause

