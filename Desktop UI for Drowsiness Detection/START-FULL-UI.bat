@echo off
echo Starting Desktop UI for Drowsiness Detection with FULL UI...
echo.

cd /d "%~dp0"

echo Building React app with FULL UI from src(UI FULL)...
call npm run build

echo.
echo Starting Electron with complete FULL UI...
echo Features:
echo - Complete UI from src(UI FULL)
echo - Professional design and layout
echo - All components and functionality
echo - Desktop app (not website)
echo.

call npm run electron

pause

