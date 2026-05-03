/**
 * V1 realtime detection WebSocket client — localhost only.
 * Direct socket.io-client tới namespace /api/v1/detect/realtime.
 *
 * Server protocol (xem python-backend/api_v1.py):
 *   client → server: `frame` { image_base64, conf }
 *   server → client: `ready` {...}, `result` {...}, `error` {...}
 */
import { io, Socket } from 'socket.io-client';

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';
const V1_NAMESPACE = '/api/v1/detect/realtime';

export interface V1DetectionObject {
  class_name: string;
  display_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  severity: 'danger' | 'warn' | 'info';
  source: string;
}

export interface V1TopKItem {
  class_name: string;
  display_name: string;
  confidence: number;
  source: string;
}

export interface V1DetectionResult {
  objects: V1DetectionObject[];
  top_k: V1TopKItem[];
  inference_time_ms: number;
  image_size: [number, number];
}

type ResultCb = (msg: V1DetectionResult) => void;
type StatusCb = (connected: boolean) => void;
type ErrorCb = (msg: { error: string }) => void;

class WSDetectV1 {
  private resultCb?: ResultCb;
  private statusCb?: StatusCb;
  private errorCb?: ErrorCb;
  private socket?: Socket;
  connected = false;

  get isConnected(): boolean {
    return this.connected;
  }

  onStatus(cb: StatusCb): void { this.statusCb = cb; }
  onResult(cb: ResultCb): void { this.resultCb = cb; }
  onError(cb: ErrorCb): void { this.errorCb = cb; }

  connect(): void {
    if (this.socket) return;
    console.log('[WS-V1] Connecting to', BACKEND_URL + V1_NAMESPACE);

    this.socket = io(BACKEND_URL + V1_NAMESPACE, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 15000,
      timeout: 10000,
    });

    this.socket.on('connect', () => {
      this.connected = true;
      this.statusCb?.(true);
    });

    this.socket.on('disconnect', (reason) => {
      this.connected = false;
      this.statusCb?.(false);
      console.log('[WS-V1] disconnected:', reason);
    });

    this.socket.on('connect_error', (err) => {
      this.connected = false;
      this.statusCb?.(false);
      console.warn('[WS-V1] connect_error:', err.message);
    });

    this.socket.on('ready', () => {
      // Server hello — đã sẵn sàng nhận frame.
    });

    this.socket.on('result', (raw: V1DetectionResult) => {
      this.resultCb?.(raw);
    });

    this.socket.on('error', (raw: { error: string }) => {
      this.errorCb?.(raw);
    });
  }

  sendFrame(imageBase64: string, conf: number): void {
    if (!this.connected || !this.socket) return;
    this.socket.emit('frame', { image_base64: imageBase64, conf });
  }

  disconnect(): void {
    if (this.socket) {
      try { this.socket.disconnect(); } catch {}
      this.socket = undefined;
    }
    this.connected = false;
    this.resultCb = undefined;
    this.statusCb = undefined;
    this.errorCb = undefined;
  }
}

export const wsDetectV1 = new WSDetectV1();
