import { io, Socket } from 'socket.io-client';

export type DetectionResult = {
  success: boolean;
  frame_width?: number;
  frame_height?: number;
  fps?: number;
  persons?: any[];
  timestamp?: number;
  error?: string;
};

export class DetectionWSClient {
  private socket: Socket | null = null;
  private connected = false;
  private conf: number | null = null;
  private preprocess: { enabled?: boolean; gamma?: number; beta?: number } | null = null;

  connect(onResult: (res: DetectionResult) => void): void {
    if (this.socket) return;
    // Explicitly provide path and namespace to avoid mismatches across environments
    this.socket = io('http://127.0.0.1:5000/ws/detect', {
      // Allow the client to negotiate transports (websocket or polling) to maximize compatibility
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
      // eslint-disable-next-line no-console
      console.log('[WS] Connected to /ws/detect');
      // Notify caller that WS is ready so upstream can start sending frames
      try { onResult({ success: true }); } catch {}
    });
    this.socket.on('disconnect', (reason: string) => {
      this.connected = false;
      // eslint-disable-next-line no-console
      console.log('[WS] Disconnected:', reason);
    });
    this.socket.on('connect_error', (err: any) => {
      // eslint-disable-next-line no-console
      console.warn('[WS] connect_error:', err?.message || err);
    });
    this.socket.on('hello', (msg: any) => {
      // eslint-disable-next-line no-console
      console.log('[WS] hello:', msg);
      // Also signal readiness on hello to unblock senders waiting for first result
      try { onResult({ success: true }); } catch {}
    });
    this.socket.on('result', (msg: DetectionResult) => {
      // eslint-disable-next-line no-console
      try { console.log('[WS] result persons:', Array.isArray((msg as any)?.persons) ? (msg as any).persons.length : 0, 'fps:', (msg as any)?.fps ?? ''); } catch {}
      onResult(msg);
    });
  }

  updateConfig(cfg: { conf?: number; preprocess?: { enabled?: boolean; gamma?: number; beta?: number } }): void {
    if (cfg?.conf !== undefined) {
      this.conf = cfg.conf;
    }
    if (cfg?.preprocess !== undefined) {
      this.preprocess = cfg.preprocess;
    }
  }

  sendFrame(frameBase64: string, cameraId: string): void {
    if (!this.socket || !this.connected) return;
    const payload: any = { frame: frameBase64, camera_id: cameraId };
    if (typeof this.conf === 'number') payload.conf = this.conf;
    if (this.preprocess) payload.preprocess = this.preprocess;
    this.socket.emit('frame', payload);
  }

  close(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}




