import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

ROOT = Path(__file__).parent
UI_DIR = ROOT / "Desktop UI for Drowsiness Detection"
DEV_URL = os.environ.get("DESKTOP_UI_URL", "http://localhost:3000/")

def run(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True)


def popen(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}  (background)", flush=True)
    # Start without blocking; let Vite own the console output
    return subprocess.Popen(cmd, cwd=cwd)


def wait_for(url: str, timeout_s: int = 30):
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    if last_err:
        print(f"Timed out waiting for {url}: {last_err}")
    return False


def main():
    if not UI_DIR.exists():
        print(f"Cannot find '{UI_DIR}'. Make sure the 'Desktop UI for Drowsiness Detection' folder exists.")
        sys.exit(1)

    # Ensure dependencies installed
    node_modules = UI_DIR / "node_modules"
    if not node_modules.exists():
        print("Installing web UI dependencies (npm install)...")
        run(["npm", "install"], cwd=str(UI_DIR))

    # Start dev server
    print("Starting Vite dev server (npm run dev)...")
    # Use '--' so windows terminals handle scripts consistently
    proc = popen(["npm", "run", "dev"], cwd=str(UI_DIR))

    # Wait for server to be ready
    if wait_for(DEV_URL, timeout_s=45):
        print(f"Desktop UI is up: {DEV_URL}")
        try:
            webbrowser.open(DEV_URL)
        except Exception:
            pass
        # Keep the script running while child process is alive
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("Stopping dev server...")
            proc.terminate()
    else:
        print("Dev server didn't come up in time. Check the terminal running Vite for logs.")
        try:
            proc.terminate()
        except Exception:
            pass
        sys.exit(2)


if __name__ == "__main__":
    main()
