
  # Desktop UI for Drowsiness Detection

  This is a code bundle for Desktop UI for Drowsiness Detection. The original project is available at https://www.figma.com/design/LSvpZ0P3ysfvfn4Ru2PieR/Desktop-UI-for-Drowsiness-Detection.

  ## Setup

  ```
  npm install
  pip install -r python-backend/requirements.txt
  ```

  ## Running — two modes

  **Desktop (Electron)** — one window, Electron auto-spawns the Python backend:

  ```
  start-desktop.bat
  ```

  **Web (localhost)** — Python backend + Vite dev server in browser:

  ```
  start-web.bat
  ```

  The frontend auto-detects which mode it's in (presence of `window.appApi`).
  In web mode it talks directly to `http://127.0.0.1:5000` via `fetch` and
  `socket.io-client`. Override the backend URL with `VITE_BACKEND_URL` if needed.
