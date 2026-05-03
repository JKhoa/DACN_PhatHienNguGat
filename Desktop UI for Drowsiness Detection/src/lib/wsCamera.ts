/**
 * Camera WebSocket client — localhost only.
 * Direct socket.io-client tới backend namespace /ws/camera.
 * Public API: subscribe / unsubscribe / disconnect / isConnected / onStatusChange / getSubscriptionCount.
 */
import { CameraUpdate } from '../types/detection';
import { io, Socket } from 'socket.io-client';

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';
const CAMERA_NAMESPACE = '/ws/camera';

class WSCamera {
  private handlers = new Map<string, (msg: CameraUpdate) => void>();
  private onStatusChangeCallback?: (connected: boolean) => void;
  private socket?: Socket;
  connected = false;

  get isConnected(): boolean {
    return this.connected;
  }

  onStatusChange(callback: (connected: boolean) => void): void {
    this.onStatusChangeCallback = callback;
  }

  private ensureSocket() {
    if (this.socket) return;

    console.log('[WS-CAM] Connecting camera socket to', BACKEND_URL + CAMERA_NAMESPACE);
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
      // Re-subscribe sau khi reconnect.
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

  subscribe(cameraId: string, onUpdate: (msg: CameraUpdate) => void): () => void {
    this.ensureSocket();
    this.handlers.set(`cam:${cameraId}`, onUpdate);

    if (this.socket?.connected) {
      this.socket.emit('subscribe', { camera_id: cameraId });
    }
    // Nếu socket chưa connect, 'connect' handler sẽ replay subscriptions.

    return () => this.unsubscribe(cameraId);
  }

  unsubscribe(cameraId: string): void {
    this.handlers.delete(`cam:${cameraId}`);
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe', { camera_id: cameraId });
    }
  }

  disconnect(): void {
    if (this.socket) {
      try { this.socket.disconnect(); } catch {}
      this.socket = undefined;
    }
    this.handlers.clear();
    this.connected = false;
    this.onStatusChangeCallback = undefined;
  }

  getSubscriptionCount(): number {
    return this.handlers.size;
  }
}

export const wsCamera = new WSCamera();
