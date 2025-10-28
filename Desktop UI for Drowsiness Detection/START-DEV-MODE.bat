@echo off
echo Starting Desktop App in Development Mode...
echo.

cd /d "%~dp0"

echo Starting development server and Electron...
echo The app will open in desktop window...
echo.

call npm run electron-dev

pause

