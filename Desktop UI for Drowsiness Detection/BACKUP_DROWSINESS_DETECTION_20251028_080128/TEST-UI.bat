@echo off
echo Testing Desktop UI for Drowsiness Detection...
echo.

cd /d "%~dp0"

echo Starting Electron with simple HTML...
echo The desktop application should open with UI now...
echo.

call npm run electron

pause

