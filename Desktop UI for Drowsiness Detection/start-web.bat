@echo off
REM Web (localhost) mode: Python backend + Vite dev server, browser opens automatically.
cd /d "%~dp0"

echo [start-web] Starting Python backend in a new window...
start "Drowsiness Backend" cmd /k "cd /d %~dp0python-backend && python server.py"

echo [start-web] Starting Vite dev server (browser will open at http://localhost:3000)...
call npm run dev
