/**
 * Camera WebSocket client — two transports:
 *   - Electron mode: IPC bridge (main owns the real socket.io connection)
 *   - Web mode (localhost): direct socket.io-client to backend /ws/camera
 * Public API unchanged: subscribe/unsubscribe/disconnect/isConnected/onStatusChange/getSubscriptionCount.
 */
import { CameraUpdate } from '../types/detection';
import { io, Socket } from 'socket.io-client';

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';
const CAMERA_NAMESPACE = '/ws/camera';

function hasIpc(): boolean {
  return typeof window !== 'undefined' && !!window.appApi;
}

class WSCamera {
  private handlers = new Map<string, (msg: CameraUpdate) => void>();
  private onStatusChangeCallback?: (connected: boolean) => void;
  private bridgeReady = false;
  connected = false;

  // IPC mode
  private unsubStatus?: () => void;
  private unsubUpdate?: () => void;

  // Web mode
  private socket?: Socket;

  get isConnected(): boolean {
    return this.connected;
  }

  onStatusChange(callback: (connected: boolean) => void): void {
    this.onStatusChangeCallback = callback;
  }

  private ensureBridge() {
    if (this.bridgeReady) return;

    if (hasIpc()) {
      this.unsubStatus = window.appApi!.on('ws:camera:status', (raw: unknown) => {
        const payload = raw as { connected: boolean; reason?: string };
        this.connected = !!payload?.connected;
        this.onStatusChangeCallback?.(this.connected);
      });

      this.unsubUpdate = window.appApi!.on('ws:camera:update', (raw: unknown) => {
        const msg = raw as CameraUpdate;
        try {
          if (!msg || !msg.success || !msg.camera_id) return;
          const handler = this.handlers.get(`cam:${msg.camera_id}`);
          if (handler) handler(msg);
        } catch {}
      });

      window.appApi!.invoke('ws:camera:connect').catch((err) =>
        console.warn('[WS-CAM] camera:connect failed:', err)
      );
    } else {
      console.log('[WS-CAM] Connecting camera socket directly to', BACKEND_URL + CAMERA_NAMESPACE);
      this.socket = io(BACKEND_URL + CAMERA_NAMESPACE, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 10,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 15000,
        randomizationFactor: 0.5,
        timeout: 10000,
      });

      this.socket.on('connect', () => {
        this.connected = true;
        this.onStatusChangeCallback?.(true);
        // Re-subscribe to existing handlers (e.g. after reconnect).
        for (const key of this.handlers.keys()) {
          const camId = key.replace(/^cam:/, '');
          this.socket?.emit('subscribe', { camera_id: camId });
        }
      });

      this.socket.on('disconnect', (reason) => {
        this.connected = false;
        this.onStatusChangeCallback?.(false);
        console.log('[WS-CAM] disconnected:', reason);
      });

      this.socket.on('connect_error', (err) => {
        this.connected = false;
        this.onStatusChangeCallback?.(false);
        console.warn('[WS-CAM] connect_error:', err.message);
      });

      this.socket.on('update', (msg: CameraUpdate) => {
        try {
          if (!msg || !msg.success || !msg.camera_id) return;
          const handler = this.handlers.get(`cam:${msg.camera_id}`);
          if (handler) handler(msg);
        } catch {}
      });
    }

    this.bridgeReady = true;
  }

  subscribe(cameraId: string, onUpdate: (msg: CameraUpdate) => void): () => void {
    this.ensureBridge();
    this.handlers.set(`cam:${cameraId}`, onUpdate);

    if (hasIpc()) {
      window.appApi?.invoke('ws:camera:subscribe', { cameraId }).catch(() => { /* ignore */ });
    } else if (this.socket?.connected) {
      this.socket.emit('subscribe', { camera_id: cameraId });
    }
    // If web socket not yet connected, the 'connect' handler will replay subscriptions.

    return () => this.unsubscribe(cameraId);
  }

  unsubscribe(cameraId: string): void {
    this.handlers.delete(`cam:${cameraId}`);

    if (hasIpc()) {
      window.appApi?.invoke('ws:camera:unsubscribe', { cameraId }).catch(() => { /* ignore */ });
    } else if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { camera_id: cameraId });
    }
  }

  disconnect(): void {
    try { this.unsubStatus?.(); } catch {}
    try { this.unsubUpdate?.(); } catch {}
    this.unsubStatus = undefined;
    this.unsubUpdate = undefined;

    if (this.socket) {
      try { this.socket.disconnect(); } catch {}
      this.socket = undefined;
    } else if (hasIpc()) {
      window.appApi?.invoke('ws:camera:disconnect').catch(() => { /* ignore */ });
    }

    this.handlers.clear();
    this.bridgeReady = false;
    this.connected = false;
    this.onStatusChangeCallback = undefined;
  }

  getSubscriptionCount(): number {
    return this.handlers.size;
  }
}

export const wsCamera = new WSCamera();
