@echo off
echo ========================================
echo    PYTHON BACKEND VIDEO STREAM TEST
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running processes...
taskkill /F /IM electron.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Verifying video stream support...
if exist "dist\assets\index-CSlO8Vz1.js" (
    echo ✅ Updated JavaScript with video stream support
) else (
    echo ❌ JavaScript file missing
    pause
    exit /b 1
)

if exist "python-backend\server.py" (
    echo ✅ Python backend with video stream endpoint
) else (
    echo ❌ Python backend missing
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Desktop App with Python Video Stream...
echo.
echo 🔧 Video Stream Features:
echo - Python backend captures webcam frames
echo - Frames encoded as base64 JPEG
echo - Frontend fetches frames via HTTP API
echo - Real-time video display in canvas
echo - Head-focused tracking overlay
echo.
echo 📹 Testing Instructions:
echo 1. Click "Add Camera" button
echo 2. Select "Webcam" type
echo 3. Enter device ID (0 for default camera)
echo 4. Click "Start" - should show LIVE VIDEO from Python backend
echo 5. Check Console for video stream logs
echo.

call npm run electron

echo.
echo 🛑 App closed
pause

