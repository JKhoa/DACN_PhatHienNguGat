@echo off
echo ========================================
echo    KHOI PHUC BACKUP DROWSINESS SYSTEM
echo ========================================
echo.
echo Backup: BACKUP_DROWSINESS_DETECTION_20251028_080128
echo Ngay tao: 28/10/2025 - 08:01:28
echo.

echo [1/4] Kiem tra Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js chua duoc cai dat!
    echo Vui long cai dat Node.js tu https://nodejs.org/
    pause
    exit /b 1
)
echo ✓ Node.js da san sang

echo.
echo [2/4] Cai dat dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Cai dat dependencies that bai!
    pause
    exit /b 1
)
echo ✓ Dependencies da duoc cai dat

echo.
echo [3/4] Build frontend...
call npm run build
if %errorlevel% neq 0 (
    echo ERROR: Build frontend that bai!
    pause
    exit /b 1
)
echo ✓ Frontend da duoc build

echo.
echo [4/4] Khoi dong he thong...
echo ✓ He thong da san sang!
echo.
echo ========================================
echo    HE THONG DA DUOC KHOI PHUC THANH CONG!
echo ========================================
echo.
echo Cac buoc tiep theo:
echo 1. Chay "npm run electron" de khoi dong app
echo 2. Hoac chay cac script .bat co san
echo.
echo Backup chua:
echo - React frontend hoan chinh
echo - Python backend voi YOLO model
echo - Electron main process
echo - Tat ca dependencies va configs
echo - Camera detection da duoc fix
echo - Backend connection da duoc fix
echo.
pause
