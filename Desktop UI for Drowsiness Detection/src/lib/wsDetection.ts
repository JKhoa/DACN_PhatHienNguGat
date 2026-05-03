/**
 * Detection WebSocket client — localhost only.
 * Direct socket.io-client tới backend namespace /ws/detect.
 * Public API: connect / updateConfig / sendFrame / close / isConnected / onStatusChange.
 */
import { DetectionResult } from '../types/detection';
import { io, Socket } from 'socket.io-client';

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';
const DETECT_NAMESPACE = '/ws/detect';

export class DetectionWSClient {
  private connected = false;
  private conf: number | null = null;
  private preprocess: { enabled?: boolean; gamma?: number; beta?: number } | null = null;
  private onStatusChangeCallback?: (connected: boolean) => void;
  private socket?: Socket;

  get isConnected(): boolean {
    return this.connected;
  }

  onStatusChange(callback: (connected: boolean) => void): void {
    this.onStatusChangeCallback = callback;
  }

  connect(onResult: (res: DetectionResult) => void): void {
    if (this.socket) return;
    console.log('[WS] Connecting detect socket to', BACKEND_URL + DETECT_NAMESPACE);

    this.socket = io(BACKEND_URL + DETECT_NAMESPACE, {
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
      try { onResult({ success: true } as DetectionResult); } catch {}
    });

    this.socket.on('disconnect', (reason) => {
      this.connected = false;
      this.onStatusChangeCallback?.(false);
      console.log('[WS] detect disconnected:', reason);
    });

    this.socket.on('connect_error', (err) => {
      this.connected = false;
      this.onStatusChangeCallback?.(false);
      console.warn('[WS] detect connect_error:', err.message);
    });

    this.socket.on('hello', () => {
      try { onResult({ success: true } as DetectionResult); } catch {}
    });

    this.socket.on('result', (msg: DetectionResult) => {
      onResult(msg);
    });
  }

  updateConfig(cfg: { conf?: number; preprocess?: { enabled?: boolean; gamma?: number; beta?: number } }): void {
    if (cfg?.conf !== undefined) this.conf = cfg.conf;
    if (cfg?.preprocess !== undefined) this.preprocess = cfg.preprocess;
  }

  sendFrame(frameBase64: string, cameraId: string): void {
    if (!this.connected || !this.socket) return;
    const payload: Record<string, unknown> = { frame: frameBase64, camera_id: cameraId };
    if (typeof this.conf === 'number') payload.conf = this.conf;
    if (this.preprocess) payload.preprocess = this.preprocess;
    this.socket.emit('frame', payload);
  }

  close(): void {
    if (this.socket) {
      try { this.socket.disconnect(); } catch {}
      this.socket = undefined;
    }
    this.connected = false;
    this.conf = null;
    this.preprocess = null;
    this.onStatusChangeCallback = undefined;
  }
}
