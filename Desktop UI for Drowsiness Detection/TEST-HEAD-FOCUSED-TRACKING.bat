@echo off
echo ========================================
echo    TEST HEAD-FOCUSED STUDENT TRACKING
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Building React app with Head-Focused Tracking...
call npm run build
if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)
echo ✅ React app built successfully

echo.
echo [2/3] Starting Python backend with Head-Focused Detection...
start /B python python-backend/server.py
timeout /t 3 /nobreak >nul

echo.
echo [3/3] Starting Desktop App for Head-Focused Tracking Test...
echo.
echo 🎯 Head-Focused Tracking Features:
echo - Focus on head region (top 40% of body)
echo - Smaller bounding boxes to avoid overlap
echo - Head-only keypoints (eyes only)
echo - Smaller student circles (radius 6px)
echo - Reduced grid spacing (30px instead of 50px)
echo - Head-focused student ID generation
echo - Separate headBbox and full body bbox
echo - Smaller confidence labels
echo - Reduced desk spacing for better head focus
echo.
echo 📱 How to test:
echo 1. Click "Thêm" to add a camera
echo 2. Configure IP camera or webcam
echo 3. Click "Start" to begin head-focused detection
echo 4. Observe smaller, non-overlapping bounding boxes
echo 5. Click "..." menu on camera card
echo 6. Select "Hiện Chi tiết Tracking"
echo 7. View head-focused bounding box information
echo.

call npm run electron

echo.
echo 🛑 Shutting down...
taskkill /F /IM python.exe 2>nul
echo ✅ System stopped

pause

