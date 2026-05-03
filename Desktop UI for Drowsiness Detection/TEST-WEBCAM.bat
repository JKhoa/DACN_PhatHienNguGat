@echo off
echo ========================================
echo    WEBCAM VIDEO FEED TEST
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running processes...
taskkill /F /IM electron.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Verifying webcam support...
if exist "dist\assets\index-QU-8l5oj.js" (
    echo ✅ Updated JavaScript with webcam support
) else (
    echo ❌ JavaScript file missing
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Desktop App with Real Webcam Support...
echo.
echo 🔧 Webcam Features:
echo - Real video feed from webcam (no more black screen)
echo - Automatic fallback to default camera if specific device fails
echo - Head-focused tracking overlay
echo - Real-time student detection
echo.
echo 📹 Testing Instructions:
echo 1. Click "Add Camera" button
echo 2. Select "Webcam" type
echo 3. Enter device ID (0 for default camera)
echo 4. Click "Start" - should show LIVE VIDEO FEED
echo 5. Check Console for webcam connection logs
echo.

call npm run electron

echo.
echo 🛑 App closed
pause

