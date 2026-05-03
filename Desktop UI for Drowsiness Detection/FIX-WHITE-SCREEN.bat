@echo off
echo ========================================
echo    FIX WHITE SCREEN ISSUE
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Checking dist files...
if exist "dist\index.html" (
    echo ✅ dist\index.html exists
) else (
    echo ❌ dist\index.html missing
    pause
    exit /b 1
)

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
echo [2/2] Starting Electron with DevTools...
echo.
echo 🔧 Debug Info:
echo - Using loadFile instead of loadURL
echo - DevTools will open automatically
echo - Check Console tab for errors
echo - Asset paths fixed to relative paths
echo.

call npm run electron

pause

