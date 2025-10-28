@echo off
echo Starting Desktop UI for Drowsiness Detection with Full Layout...
echo.

cd /d "%~dp0"

echo Building React app with full layout...
call npm run build

echo.
echo Starting Electron with complete UI layout...
echo Features:
echo - Toolbar with Start All, Stop All, Add, Delete buttons
echo - Camera Sidebar with Vietnamese text
echo - Camera Grid with 2x2 layout
echo - Event Log Panel with Vietnamese text
echo - Status Bar with FPS, CPU, GPU
echo.

call npm run electron

pause

