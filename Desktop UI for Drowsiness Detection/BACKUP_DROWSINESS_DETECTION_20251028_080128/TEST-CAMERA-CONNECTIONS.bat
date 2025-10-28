@echo off
echo Testing Camera Connection System...
echo.

cd /d "%~dp0"

echo Building React app...
call npm run build

echo.
echo Starting Desktop App for Camera Connection Test...
echo.
echo Camera Connection Features:
echo - IP Camera Support (Hikvision, Dahua, Ezviz, KBVision)
echo - Webcam Support (USB devices)
echo - RTSP URL Generation
echo - Connection Testing
echo - Real-time YOLO Detection
echo.

echo Test Cases:
echo 1. IP Camera - Hikvision: rtsp://admin:admin123@192.168.1.100:554/Streaming/Channels/101
echo 2. IP Camera - Dahua: rtsp://admin:admin123@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0
echo 3. IP Camera - Ezviz: rtsp://admin:admin123@192.168.1.102:554/h264/ch1/main/av_stream
echo 4. Webcam - Device 0: Default webcam
echo 5. Webcam - Device 1: Secondary webcam
echo.

call npm run electron

pause

