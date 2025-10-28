@echo off
echo ========================================
echo    RESTART APP - FIX WHITE SCREEN
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running processes...
taskkill /F /IM electron.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Rebuilding React app...
call npm run build
if errorlevel 1 (
    echo ❌ Build failed!
    pause
    exit /b 1
)
echo ✅ React app built successfully

echo.
echo [3/3] Starting Desktop App with fixes...
echo.
echo 🔧 Fixes Applied:
echo - Fixed asset paths to relative paths
echo - Added CSS to ensure root div has height
echo - DevTools will open for debugging
echo - Check Console tab for any JavaScript errors
echo.

call npm run electron

echo.
echo 🛑 App closed
pause

