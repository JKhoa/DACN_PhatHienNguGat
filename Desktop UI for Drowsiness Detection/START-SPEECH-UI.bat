@echo off
echo Starting Desktop UI for Drowsiness Detection with Speech UI...
echo.

cd /d "%~dp0"

echo Building React app with Speech UI...
call npm run build

echo.
echo Starting Electron with new UI...
echo The application should now show the Speech Detection UI style
echo.

call npm run electron

pause

