@echo off
echo Installing Python dependencies for YOLO drowsiness detection...
echo.

cd /d "%~dp0"

echo Installing Python packages...
cd python-backend
pip install -r requirements.txt

echo.
echo Installing Flask for API server...
pip install flask flask-cors

echo.
echo Back to main directory...
cd ..

echo Building React app...
call npm run build

echo.
echo Starting Desktop App with YOLO Backend...
echo Features:
echo - Empty camera slots ready for connection
echo - YOLO model integration for real-time detection
echo - Support for IP cameras and webcams
echo - Real-time student tracking (10-50 students per camera)
echo - Drowsiness detection with confidence scores
echo.

call npm run electron

pause

