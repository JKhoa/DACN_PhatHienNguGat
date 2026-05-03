# Start Drowsiness Detection Application
# This script starts both backend and frontend automatically

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Drowsiness Detection Application Launcher    " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Start backend
Write-Host "[1/2] Starting Python backend server..." -ForegroundColor Green
$backendPath = Join-Path $PSScriptRoot "python-backend"
$pythonExe = "D:\Study\DoAnChuyenNganh\DACN_PhatHienNguGat\.venv\Scripts\python.exe"
$serverScript = "server_with_tracking_backup.py"

Set-Location $backendPath
Start-Process -FilePath $pythonExe -ArgumentList $serverScript -WindowStyle Normal

Write-Host "   Backend starting (separate window)..." -ForegroundColor Yellow
Write-Host "   Waiting 5 seconds for initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start frontend
Write-Host ""
Write-Host "[2/2] Starting Electron app..." -ForegroundColor Green
Set-Location $PSScriptRoot
npm start

Write-Host ""
Write-Host "Application closed." -ForegroundColor Yellow
Write-Host "Remember to close the Python backend window!" -ForegroundColor Red
