@echo off
REM Desktop mode: Electron runs Vite + spawns the Python backend itself.
cd /d "%~dp0"
echo [start-desktop] Launching Electron desktop app...
call npm run electron-dev
