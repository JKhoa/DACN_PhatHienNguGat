@echo off
echo Starting Desktop App for Drowsiness Detection...
echo.

cd /d "%~dp0"

echo Installing dependencies...
call npm install

echo.
echo Building the application...
call npm run build

echo.
echo Starting Desktop App...
echo The desktop application will open shortly...
echo.

call npm run electron

pause

