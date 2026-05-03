/**
 * WebSocket Bridge — runs socket.io-client in the MAIN process and proxies
 * events to/from the renderer over IPC.
 *
 * Renderer never connects to 127.0.0.1 directly. This allows
 * webPreferences.webSecurity to be set to true.
 *
 * IPC channels (renderer → main, via invoke):
 *   ws:detect:connect          → ensure detect socket exists
 *   ws:detect:send-frame       → { frameBase64, cameraId, conf, preprocess }
 *   ws:detect:disconnect       → close detect socket
 *   ws:camera:connect          → ensure camera socket exists
 *   ws:camera:subscribe        → { cameraId }
 *   ws:camera:unsubscribe      → { cameraId }
 *   ws:camera:disconnect       → close camera socket
 *   ws:v1:connect              → ensure v1 realtime socket exists
 *   ws:v1:send-frame           → { image_base64, conf }
 *   ws:v1:disconnect           → close v1 socket
 *
 * Push events (main → renderer, via webContents.send):
 *   ws:detect:status           → { connected, reason? }
 *   ws:detect:hello            → server hello payload
 *   ws:detect:result           → detection result payload
 *   ws:camera:status           → { connected, reason? }
 *   ws:camera:update           → camera update payload
 *   ws:v1:status               → { connected, reason? }
 *   ws:v1:ready                → server ready payload
 *   ws:v1:result               → v1 detection result payload
 *   ws:v1:error                → v1 error payload
 */
const { ipcMain } = require('electron');
const { io } = require('socket.io-client');

const PYTHON_BASE = 'http://127.0.0.1:5000';
// Fix: Socket.io namespaces should be handled by passing the full URL to io() 
// but ensuring the path is correct. Flask-SocketIO expects /socket.io/ by default.
const DETECT_NAMESPACE = '/ws/detect';
const CAMERA_NAMESPACE = '/ws/camera';
const V1_NAMESPACE = '/api/v1/detect/realtime';

const RECONNECTION_OPTS = {
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
};

let detectSocket = null;
let cameraSocket = null;
let v1Socket = null;
// Track active camera subscriptions so we can re-subscribe on reconnect.
const cameraSubscriptions = new Set();
let getWindow = () => null;

/** Small helper: push an event to the renderer if the window is alive. */
function emitToRenderer(channel, payload) {
    try {
        const win = getWindow();
        if (win && !win.isDestroyed() && win.webContents) {
            win.webContents.send(channel, payload);
        }
    } catch (e) {
        // console.warn(`[wsBridge] Failed to push ${channel}:`, e.message);
    }
}

// ─── Detect socket ──────────────────────────────────────────────────────────

function ensureDetectSocket() {
    if (detectSocket && detectSocket.connected) return detectSocket;
    if (detectSocket) detectSocket.close();

    const url = `${PYTHON_BASE}${DETECT_NAMESPACE}`;
    console.log(`[wsBridge] Connecting detect socket → ${url}`);
    
    detectSocket = io(url, {
        path: '/socket.io/',
        transports: ['websocket'],
        ...RECONNECTION_OPTS,
    });

    detectSocket.on('connect', () => {
        console.log('[wsBridge] detect connected');
        emitToRenderer('ws:detect:status', { connected: true });
    });

    detectSocket.on('disconnect', (reason) => {
        console.log('[wsBridge] detect disconnected:', reason);
        emitToRenderer('ws:detect:status', { connected: false, reason });
    });

    detectSocket.on('connect_error', (err) => {
        console.warn('[wsBridge] detect connect_error:', err?.message || err);
        emitToRenderer('ws:detect:status', { connected: false, reason: 'connect_error' });
    });

    detectSocket.on('hello', (msg) => emitToRenderer('ws:detect:hello', msg));
    detectSocket.on('result', (msg) => emitToRenderer('ws:detect:result', msg));

    return detectSocket;
}

function closeDetectSocket() {
    if (detectSocket) {
        detectSocket.removeAllListeners();
        detectSocket.disconnect();
        detectSocket = null;
    }
}

// ─── Camera socket ──────────────────────────────────────────────────────────

function ensureCameraSocket() {
    if (cameraSocket && cameraSocket.connected) return cameraSocket;
    if (cameraSocket) cameraSocket.close();

    const url = `${PYTHON_BASE}${CAMERA_NAMESPACE}`;
    console.log(`[wsBridge] Connecting camera socket → ${url}`);
    cameraSocket = io(url, {
        path: '/socket.io/',
        transports: ['websocket'],
        ...RECONNECTION_OPTS,
    });

    cameraSocket.on('connect', () => {
        console.log('[wsBridge] camera connected');
        emitToRenderer('ws:camera:status', { connected: true });
        // Re-subscribe known rooms after reconnect.
        for (const camId of cameraSubscriptions) {
            cameraSocket.emit('subscribe', { camera_id: camId });
        }
    });

    cameraSocket.on('disconnect', (reason) => {
        console.log('[wsBridge] camera disconnected:', reason);
        emitToRenderer('ws:camera:status', { connected: false, reason });
    });

    cameraSocket.on('connect_error', (err) => {
        console.warn('[wsBridge] camera connect_error:', err?.message || err);
        emitToRenderer('ws:camera:status', { connected: false, reason: 'connect_error' });
    });

    cameraSocket.on('update', (msg) => emitToRenderer('ws:camera:update', msg));

    return cameraSocket;
}

function closeCameraSocket() {
    cameraSubscriptions.clear();
    if (cameraSocket) {
        cameraSocket.removeAllListeners();
        cameraSocket.disconnect();
        cameraSocket = null;
    }
}

// ─── V1 realtime socket (pipeline ensemble VN) ──────────────────────────────

function ensureV1Socket() {
    if (v1Socket && v1Socket.connected) return v1Socket;
    if (v1Socket) v1Socket.close();

    const url = `${PYTHON_BASE}${V1_NAMESPACE}`;
    console.log(`[wsBridge] Connecting v1 socket → ${url}`);

    v1Socket = io(url, {
        path: '/socket.io/',
        transports: ['websocket'],
        ...RECONNECTION_OPTS,
    });

    v1Socket.on('connect', () => {
        console.log('[wsBridge] v1 connected');
        emitToRenderer('ws:v1:status', { connected: true });
    });

    v1Socket.on('disconnect', (reason) => {
        console.log('[wsBridge] v1 disconnected:', reason);
        emitToRenderer('ws:v1:status', { connected: false, reason });
    });

    v1Socket.on('connect_error', (err) => {
        console.warn('[wsBridge] v1 connect_error:', err?.message || err);
        emitToRenderer('ws:v1:status', { connected: false, reason: 'connect_error' });
    });

    v1Socket.on('ready', (msg) => emitToRenderer('ws:v1:ready', msg));
    v1Socket.on('result', (msg) => emitToRenderer('ws:v1:result', msg));
    v1Socket.on('error', (msg) => emitToRenderer('ws:v1:error', msg));

    return v1Socket;
}

function closeV1Socket() {
    if (v1Socket) {
        v1Socket.removeAllListeners();
        v1Socket.disconnect();
        v1Socket = null;
    }
}

// ─── IPC registration ───────────────────────────────────────────────────────

function register(getMainWindow) {
    getWindow = typeof getMainWindow === 'function' ? getMainWindow : () => null;

    ipcMain.handle('ws:detect:connect', () => {
        ensureDetectSocket();
        return { ok: true, connected: !!detectSocket?.connected };
    });

    ipcMain.handle('ws:detect:send-frame', (_event, payload) => {
        const sock = ensureDetectSocket();
        if (!sock.connected) return { ok: false, reason: 'not-connected' };
        // Forward the whole payload (frame, camera_id, optional conf + preprocess).
        sock.emit('frame', payload);
        return { ok: true };
    });

    ipcMain.handle('ws:detect:disconnect', () => {
        closeDetectSocket();
        return { ok: true };
    });

    ipcMain.handle('ws:camera:connect', () => {
        ensureCameraSocket();
        return { ok: true, connected: !!cameraSocket?.connected };
    });

    ipcMain.handle('ws:camera:subscribe', (_event, { cameraId }) => {
        const sock = ensureCameraSocket();
        cameraSubscriptions.add(cameraId);
        if (sock.connected) sock.emit('subscribe', { camera_id: cameraId });
        return { ok: true };
    });

    ipcMain.handle('ws:camera:unsubscribe', (_event, { cameraId }) => {
        cameraSubscriptions.delete(cameraId);
        if (cameraSocket?.connected) {
            cameraSocket.emit('unsubscribe', { camera_id: cameraId });
        }
        return { ok: true };
    });

    ipcMain.handle('ws:camera:disconnect', () => {
        closeCameraSocket();
        return { ok: true };
    });

    ipcMain.handle('ws:v1:connect', () => {
        ensureV1Socket();
        return { ok: true, connected: !!v1Socket?.connected };
    });

    ipcMain.handle('ws:v1:send-frame', (_event, payload) => {
        const sock = ensureV1Socket();
        if (!sock.connected) return { ok: false, reason: 'not-connected' };
        sock.emit('frame', payload);
        return { ok: true };
    });

    ipcMain.handle('ws:v1:disconnect', () => {
        closeV1Socket();
        return { ok: true };
    });

    console.log('[wsBridge] IPC channels registered');
}

function shutdown() {
    closeDetectSocket();
    closeCameraSocket();
    closeV1Socket();
}

module.exports = { register, shutdown };
