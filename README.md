# Drowsiness Detection (Classroom)

This repo includes a complete desktop UI and CLI for multi-camera classroom drowsiness detection using Ultralytics YOLO pose.

## Quick start

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run the GUI (recommended):

```powershell
python standalone_app.py --gui
```

- Default model auto-resolves to `yolo11n-pose.pt` in the repo if present, otherwise it will auto-download via Ultralytics.
- In the GUI you can choose source (webcam, RTSP/HTTP, video, image), switch models, record annotated video, and use the Multi-Camera tab.

4) CLI examples:

```powershell
# Webcam
python standalone_app.py --cam 0 --res 1280x720

# IP camera (example for IMOU/Dahua)
python standalone_app.py --ip-camera --ip 192.168.1.100 --username admin --password 123456 --camera-brand imou --stream-quality main

# Video file
python standalone_app.py --video data_raw/cap_000000.jpg
```

## Project layout

- `standalone_app.py` — root launcher that dispatches to the full app under `yolo-sleepy-allinone-final/`.
- `yolo-sleepy-allinone-final/gui_app.py` — Complete desktop UI with tabs and a modern layout.
- `yolo-sleepy-allinone-final/multi_camera_gui.py` — Multi-camera manager (add/edit/test IP/Webcam, start/stop all, grid/single view).
- `yolo-sleepy-allinone-final/standalone_app.py` — CLI runner with video/webcam/IP camera support.
- `yolo-sleepy-allinone-final/enhanced_display.py` — Enhanced multi-person overlays.
- `yolo-sleepy-allinone-final/camera_core.py` — Shared capture utilities with threaded frame queue for low latency.

## Notes

- Torch (PyTorch) is not pinned in `requirements.txt` because it must match your GPU/CPU and platform. Install it first if needed from https://pytorch.org.
- If PyQt5 is missing or you run with `--cli`, the app falls back to console mode.
- For the multi-camera tab, configuration can be saved/loaded as YAML; `PyYAML` is included in the root requirements.

## Web Desktop UI (React/Vite)

You also have a separate, modern Desktop UI implemented with React + Vite under:

- `Desktop UI for Drowsiness Detection/`

This is independent from the PyQt5 GUI. To run the web UI:

```powershell
# Install Node.js if you don't have it (https://nodejs.org)

# From repo root
cd "Desktop UI for Drowsiness Detection"
npm install
npm run dev
# Open the URL shown (typically http://localhost:3000/)
```

Or launch it via the helper script from the repo root:

```powershell
python start_desktop_ui.py
```

Notes:
- The React/Vite UI is currently a separate frontend. To drive live detections from Python, expose a local API/WebSocket from the backend and connect to it from the web app. If you want, we can wire this next.

## Realtime WebSocket detection (frontend streams video, backend returns results)

This project now includes a low‑latency WebSocket path for webcam detection:

- Frontend: streams webcam via getUserMedia, captures frames, and sends them over a Socket.IO WebSocket.
- Backend (Flask + Socket.IO): receives frames, runs the drowsiness model (YOLO pose), and immediately emits structured results (IDs, head boxes, states).
- Frontend: overlays results (green = normal, red = buồn ngủ, purple = gục bàn) directly on top of the live video; video itself continues to play locally for minimal latency.

Run it:

```powershell
# 1) (In Python venv) start the backend with WebSocket support
python start_python_backend.py

# 2) In another terminal, start the React UI
cd "Desktop UI for Drowsiness Detection"
npm install
npm run dev
```

How it works:

- WebSocket namespace: `ws://127.0.0.1:5000/ws/detect`
- Frontend client: `src/lib/wsDetection.ts`
- Webcam component: `src/components/CameraCard.tsx` streams frames to WS and draws overlays from detection results.
- IP cameras: still managed by backend threads; UI polls `/api/camera/<id>/detection` for results and draws overlays. You can switch to WS per camera later if desired.

Tip: If you only want detection results (no annotated images over the wire), keep the video element local (webcam) and use the WS results to draw on a canvas overlay.
