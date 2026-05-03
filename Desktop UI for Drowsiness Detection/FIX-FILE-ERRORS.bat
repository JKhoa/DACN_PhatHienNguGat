@echo off
echo ========================================
echo    FIX FILE NOT FOUND ERRORS
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running processes...
taskkill /F /IM electron.exe 2>nul
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/3] Verifying asset files...
if exist "dist\assets\index-BKc35DHV.js" (
    echo ✅ JavaScript file exists
) else (
    echo ❌ JavaScript file missing
    pause
    exit /b 1
)

if exist "dist\assets\index-DCHVCEDf.css" (
    echo ✅ CSS file exists
) else (
    echo ❌ CSS file missing
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Desktop App with fixed paths...
echo.
echo 🔧 Fixes Applied:
echo - Fixed asset paths to relative paths (./assets/...)
echo - Added CSS to ensure root div has height
echo - DevTools will open for debugging
echo - Check Console tab - should show no file errors
echo.

call npm run electron

echo.
echo 🛑 App closed
pause

