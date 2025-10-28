@echo off
echo Starting Enhanced Desktop UI for Drowsiness Detection...
echo.

cd /d "%~dp0"

echo Building React app with enhanced UI...
call npm run build

echo.
echo Starting Electron with beautiful UI...
echo Features:
echo - Beautiful gradient background
echo - Empty camera slots ready for connection
echo - Modern card design with shadows
echo - Real-time stats and monitoring
echo.

call npm run electron

pause

