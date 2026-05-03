const { app, BrowserWindow, ipcMain, session } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const wsBridge = require('./wsBridge');

let mainWindow;
let pythonProcess;

// Internal Python backend base URL — never exposed to renderer
const PYTHON_BASE = 'http://127.0.0.1:5000';

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,          // Security: no Node.js in renderer
            contextIsolation: true,           // Security: preload bridge only
            enableRemoteModule: false,
            // WS and HTTP now routed via IPC → renderer never opens localhost connections.
            webSecurity: true,
            allowRunningInsecureContent: false,
            preload: path.join(__dirname, 'preload.js'),
        },
        title: 'Hệ thống Phát hiện Ngủ gật - Classroom Monitoring'
    });

    const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
    console.log('Loading React app from:', indexPath);
    mainWindow.loadFile(indexPath);

    // DevTools enabled for smoke test — revert (comment out) before demo.
    mainWindow.webContents.openDevTools({ mode: 'detach' });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// ─── Python backend launcher ───────────────────────────────────────────────

function startPythonBackend() {
    const pythonPath = path.join(__dirname, '..', 'python-backend');
    const fs = require('fs');

    let serverPath = path.join(pythonPath, 'server.py');
    if (!fs.existsSync(serverPath)) {
        serverPath = path.join(pythonPath, 'server_with_tracking_backup.py');
        console.log('Using fallback server_with_tracking_backup.py');
    } else {
        console.log('✅ Using main server.py');
    }

    const reqsPath = path.join(pythonPath, 'requirements.txt');
    const repoRoot = path.join(__dirname, '..', '..');
    const venvPython = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
    const pythonExe = fs.existsSync(venvPython) ? venvPython : 'python';

    console.log('Starting Python backend...');
    console.log('Python executable:', pythonExe);
    console.log('Server path:', serverPath);

    // Install dependencies (no-op if already satisfied)
    const pipInstall = spawn(pythonExe, ['-m', 'pip', 'install', '-r', reqsPath], {
        cwd: pythonPath,
        stdio: ['pipe', 'pipe', 'pipe']
    });

    pipInstall.stdout.on('data', d => console.log(`pip: ${d}`));
    pipInstall.stderr.on('data', d => console.error(`pip err: ${d}`));

    pipInstall.on('close', (code) => {
        console.log(`pip install done (code ${code}), starting server...`);

        pythonProcess = spawn(pythonExe, [serverPath], {
            cwd: pythonPath,
            stdio: ['pipe', 'pipe', 'pipe']
        });

        pythonProcess.stdout.on('data', d => console.log(`Python: ${d}`));
        pythonProcess.stderr.on('data', d => console.error(`Python err: ${d}`));
        pythonProcess.on('close', code => console.log(`Python exited (code ${code})`));
        pythonProcess.on('error', err => console.error('Failed to start Python:', err));
    });
}

// ─── Health-check: wait for Python before opening window ──────────────────
// Polls /api/health every 1 s, max 60 retries (covers cold pip install).

async function waitForPythonBackend(maxRetries = 60, intervalMs = 1000) {
    for (let i = 1; i <= maxRetries; i++) {
        try {
            const res = await fetch(`${PYTHON_BASE}/api/health`,
                { signal: AbortSignal.timeout(800) });
            if (res.ok) {
                console.log(`✅ Python backend ready (attempt ${i})`);
                return;
            }
        } catch (_) { /* not yet listening — keep polling */ }
        console.log(`⏳ Waiting for Python backend... (${i}/${maxRetries})`);
        await new Promise(r => setTimeout(r, intervalMs));
    }
    console.warn('⚠️ Python backend did not respond in 60 s — opening window anyway');
}

// ─── IPC Gateway: generic HTTP proxy to Python backend ────────────────────
// Renderer calls window.appApi.invoke('api:request', { method, endpoint, data })
// Main proxies to Python and returns { status, data }.
// The renderer never sees the localhost URL.

ipcMain.handle('api:request', async (event, { method, endpoint, data }) => {
    try {
        const url = `${PYTHON_BASE}/${endpoint.replace(/^\//, '')}`;
        const opts = { method: method || 'GET', headers: {} };
        if (data !== undefined && data !== null) {
            opts.headers['Content-Type'] = 'application/json';
            opts.body = JSON.stringify(data);
        }
        const response = await fetch(url, opts);
        let responseData;
        try {
            responseData = await response.json();
        } catch {
            responseData = {};
        }
        return { status: response.status, data: responseData };
    } catch (error) {
        console.error('[IPC] api:request error:', error.message);
        return { status: 503, data: { success: false, error: error.message } };
    }
});

// ─── IPC Gateway: binary export (PDF / Excel) ─────────────────────────────
// Returns base64-encoded content so the renderer can reconstruct a Blob
// without ever connecting directly to localhost.

ipcMain.handle('api:export', async (event, { format, period, camera_ids }) => {
    try {
        const url = `${PYTHON_BASE}/api/logs/export/${format}`;
        const body = camera_ids ? { period, camera_ids } : { period };
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            return { status: response.status, base64: null, contentType: null };
        }
        const buffer = await response.arrayBuffer();
        const base64 = Buffer.from(buffer).toString('base64');
        const contentType = response.headers.get('content-type') || 'application/octet-stream';
        return { status: response.status, base64, contentType };
    } catch (error) {
        console.error('[IPC] api:export error:', error.message);
        return { status: 503, base64: null, contentType: null };
    }
});

// ─── App lifecycle ──────────────────────────────────────────────────────────

app.whenReady().then(async () => {
    // Auto-allow camera/microphone permission requests from renderer
    try {
        session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
            if (['media', 'camera', 'microphone'].includes(permission)) {
                return callback(true);
            }
            callback(true);
        });
    } catch (e) {
        console.warn('Could not configure permission handler:', e);
    }

    startPythonBackend();

    // Wait until Python is ready (replaces the old blind setTimeout(2000))
    await waitForPythonBackend();

    // Register WS IPC bridge BEFORE creating the window so renderer can
    // invoke ws:* channels immediately on mount.
    wsBridge.register(() => mainWindow);

    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    wsBridge.shutdown();
    if (pythonProcess) pythonProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
    wsBridge.shutdown();
    if (pythonProcess) pythonProcess.kill();
});
