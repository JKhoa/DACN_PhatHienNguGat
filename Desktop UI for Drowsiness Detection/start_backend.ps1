# Start Backend Server for Drowsiness Detection
# This script starts the Python backend in a separate window

Write-Host "Starting Python backend server..." -ForegroundColor Green
Write-Host "Backend will run in a separate window. Keep it open while using the app." -ForegroundColor Yellow

$backendPath = Join-Path $PSScriptRoot "python-backend"
$pythonExe = "D:\Study\DoAnChuyenNganh\DACN_PhatHienNguGat\.venv\Scripts\python.exe"
$serverScript = "server_with_tracking_backup.py"

# Check if backend directory exists
if (-not (Test-Path $backendPath)) {
    Write-Host "Error: Backend directory not found at $backendPath" -ForegroundColor Red
    exit 1
}

# Check if Python executable exists
if (-not (Test-Path $pythonExe)) {
    Write-Host "Error: Python executable not found at $pythonExe" -ForegroundColor Red
    Write-Host "Please ensure virtual environment is activated" -ForegroundColor Red
    exit 1
}

# Start backend in new window
Set-Location $backendPath
Start-Process -FilePath $pythonExe -ArgumentList $serverScript -WindowStyle Normal

Write-Host ""
Write-Host "Backend server starting..." -ForegroundColor Green
Write-Host "Wait 3-5 seconds for server to initialize" -ForegroundColor Yellow
Write-Host "Backend URL: http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop the backend, close the Python window or press Ctrl+C" -ForegroundColor Yellow
