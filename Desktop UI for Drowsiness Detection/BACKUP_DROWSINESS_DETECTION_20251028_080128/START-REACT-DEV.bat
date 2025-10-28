@echo off
echo Starting React Development Server...
echo.

cd /d "%~dp0"

echo Starting Vite development server...
start "Vite Dev Server" cmd /k "npm run dev"

echo.
echo Waiting 5 seconds for server to start...
timeout /t 5 /nobreak > nul

echo Starting Electron with development server...
echo The React app will load from http://localhost:3000
echo.

call npm run electron-dev

pause

