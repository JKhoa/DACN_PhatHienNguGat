import { io, Socket } from 'socket.io-client';

export type CameraUpdate = {
  success: boolean;
  camera_id: string;
  frame_width: number;
  frame_height: number;
  fps: number;
  persons: Array<{
    id: number;
    track_id: number;
    bbox: number[];
    head_bbox?: number[] | null;
    confidence: number;
    keypoints?: Array<{ x: number; y: number; confidence: number; visible: boolean }>;
    drowsiness_score?: number;
    drowsiness_state?: string;
    last_update?: number;
  }>;
  timestamp: number;
};

class WSCamera {
  private socket: Socket | null = null;
  private handlers = new Map<string, (msg: CameraUpdate) => void>();
  connected = false;

  private ensureSocket() {
    if (this.socket) return;
    this.socket = io('http://127.0.0.1:5000/ws/camera', {
      path: '/socket.io/',
      transports: ['websocket'],
      withCredentials: false,
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 500,
      reconnectionDelayMax: 3000,
    });

    this.socket.on('connect', () => {
      this.connected = true;
      console.log('[WS-CAM] Connected to /ws/camera');
      // re-subscribe existing rooms after reconnect
      for (const room of this.handlers.keys()) {
        const camId = room.replace('cam:', '');
        this.socket?.emit('subscribe', { camera_id: camId });
      }
    });

    this.socket.on('disconnect', (reason) => {
      this.connected = false;
      console.log('[WS-CAM] Disconnected:', reason);
    });

    this.socket.on('connect_error', (err: any) => {
      console.warn('[WS-CAM] connect_error:', err?.message || err);
    });

    this.socket.on('update', (msg: CameraUpdate) => {
      try {
        if (!msg || !msg.success || !msg.camera_id) return;
        const key = `cam:${msg.camera_id}`;
        const handler = this.handlers.get(key);
        if (handler) handler(msg);
      } catch {}
    });
  }

  subscribe(cameraId: string, onUpdate: (msg: CameraUpdate) => void) {
    this.ensureSocket();
    const key = `cam:${cameraId}`;
    this.handlers.set(key, onUpdate);
    this.socket?.emit('subscribe', { camera_id: cameraId });
    return () => this.unsubscribe(cameraId);
  }

  unsubscribe(cameraId: string) {
    const key = `cam:${cameraId}`;
    this.handlers.delete(key);
    this.socket?.emit('unsubscribe', { camera_id: cameraId });
  }
}

export const wsCamera = new WSCamera();
