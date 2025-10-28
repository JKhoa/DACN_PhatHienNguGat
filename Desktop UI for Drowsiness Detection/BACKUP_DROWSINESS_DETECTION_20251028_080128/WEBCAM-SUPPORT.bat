@echo off
echo ========================================
echo    WEBCAM VIDEO FEED SUPPORT
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running processes...
taskkill /F /IM electron.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Verifying updated files...
if exist "dist\assets\index-BMqbLjSD.js" (
    echo ✅ Updated JavaScript file exists
) else (
    echo ❌ JavaScript file missing
    pause
    exit /b 1
)

if exist "src\components\CameraCardWithVideo.tsx" (
    echo ✅ New CameraCard with video support exists
) else (
    echo ❌ CameraCard with video support missing
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Desktop App with Webcam Support...
echo.
echo 🔧 New Features:
echo - Webcam shows real video feed instead of black screen
echo - IP cameras show connection status
echo - Head-focused tracking with smaller bounding boxes
echo - Real-time student detection overlay
echo - Better webcam backend compatibility
echo.
echo 📹 To test webcam:
echo 1. Click "Add Camera" button
echo 2. Select "Webcam" type
echo 3. Enter device ID (usually 0 for default camera)
echo 4. Click "Start" - should show live video feed
echo.

call npm run electron

echo.
echo 🛑 App closed
pause

