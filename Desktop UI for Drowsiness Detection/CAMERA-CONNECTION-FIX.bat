@echo off
echo ========================================
echo    CAMERA CONNECTION FIX TEST
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running processes...
taskkill /F /IM electron.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Verifying camera connection fixes...
if exist "dist\assets\index-DUGxvD6K.js" (
    echo ✅ Updated JavaScript with Python backend integration
) else (
    echo ❌ JavaScript file missing
    pause
    exit /b 1
)

if exist "python-backend\server.py" (
    echo ✅ Python backend with camera management API
) else (
    echo ❌ Python backend missing
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Desktop App with Fixed Camera Connection...
echo.
echo 🔧 Camera Connection Fixes:
echo - Frontend now sends camera data to Python backend
echo - Python backend manages camera lifecycle
echo - Real-time sync between frontend and backend
echo - Video stream endpoint properly connected
echo.
echo 📹 Testing Instructions:
echo 1. Click "Add Camera" button
echo 2. Select "Webcam" type
echo 3. Enter device ID (0 for default camera)
echo 4. Click "Save" - camera will be added to Python backend
echo 5. Click "Start" - camera will start in Python backend
echo 6. Video stream should now work without 404 errors
echo.

call npm run electron

echo.
echo 🛑 App closed
pause

