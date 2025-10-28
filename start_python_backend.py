import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "Desktop UI for Drowsiness Detection" / "python-backend"
REQ = BACKEND_DIR / "requirements.txt"

if __name__ == "__main__":
    if not BACKEND_DIR.exists():
        print(f"Backend folder not found: {BACKEND_DIR}")
        sys.exit(1)

    # Install backend deps into current Python env/venv
    if REQ.exists():
        print("Installing backend requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQ)], check=False)
    else:
        print("No requirements.txt found for backend, skipping install.")

    print("Starting Flask backend server (python-backend/server.py) on http://127.0.0.1:5000 ...")
    # Use console-owning run so you can see logs
    subprocess.run([sys.executable, str(BACKEND_DIR / "server.py")], check=False)
