@echo off
echo ========================================
echo YOLO Drowsiness Detection Test
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Checking required packages...
python -c "import cv2, ultralytics, flask; print('All packages available')" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Required packages not installed
    echo Please run: pip install -r "Desktop UI for Drowsiness Detection\python-backend\requirements.txt"
    pause
    exit /b 1
)

echo.
echo Starting YOLO detection test...
python test_yolo_detection.py

echo.
echo Test completed. Press any key to exit...
pause
