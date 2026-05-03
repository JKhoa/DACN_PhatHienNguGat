/**
 * Detection WebSocket client — two transports:
 *   - Electron mode: IPC bridge (main owns the real socket.io connection)
 *   - Web mode (localhost): direct socket.io-client to backend /ws/detect
 * Public API unchanged: connect/updateConfig/sendFrame/close/isConnected/onStatusChange.
 */
import { DetectionResult } from '../types/detection';
import { io, Socket } from 'socket.io-client';

const BACKEND_URL =
  (import.meta as any)?.env?.VITE_BACKEND_URL || 'http://127.0.0.1:5000';
const DETECT_NAMESPACE = '/ws/detect';

function hasIpc(): boolean {
  return typeof window !== 'undefined' && !!window.appApi;
}

export class DetectionWSClient {
  private connected = false;
  private conf: number | null = null;
  private preprocess: { enabled?: boolean; gamma?: number; beta?: number } | null = null;
  private onStatusChangeCallback?: (connected: boolean) => void;

  // IPC mode
  private unsubResult?: () => void;
  private unsubStatus?: () => void;
  private unsubHello?: () => void;

  // Web mode
  private socket?: Socket;

  get isConnected(): boolean {
    return this.connected;
  }

  onStatusChange(callback: (connected: boolean) => void): void {
    this.onStatusChangeCallback = callback;
  }

  connect(onResult: (res: DetectionResult) => void): void {
    if (hasIpc()) {
      this.connectIpc(onResult);
    } else {
      this.connectDirect(onResult);
    }
  }

  private connectIpc(onResult: (res: DetectionResult) => void): void {
    if (this.unsubResult) return;
    console.log('[WS] Connecting detect bridge via IPC');

    this.unsubStatus = window.appApi!.on('ws:detect:status', (raw: unknown) => {
      const payload = raw as { connected: boolean; reason?: string };
      this.connected = !!payload?.connected;
      this.onStatusChangeCallback?.(this.connected);
      if (this.connected) {
        try { onResult({ success: true } as DetectionResult); } catch {}
      }
    });

    this.unsubHello = window.appApi!.on('ws:detect:hello', (_raw: unknown) => {
      try { onResult({ success: true } as DetectionResult); } catch {}
    });

    this.unsubResult = window.appApi!.on('ws:detect:result', (raw: unknown) => {
      onResult(raw as DetectionResult);
    });

    window.appApi!.invoke('ws:detect:connect').catch((err) =>
      console.warn('[WS] detect:connect failed:', err)
    );
  }

  private connectDirect(onResult: (res: DetectionResult) => void): void {
    if (this.socket) return;
    console.log('[WS] Connecting detect socket directly to', BACKEND_URL + DETECT_NAMESPACE);

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
    if (!this.connected) return;
    const payload: Record<string, unknown> = { frame: frameBase64, camera_id: cameraId };
    if (typeof this.conf === 'number') payload.conf = this.conf;
    if (this.preprocess) payload.preprocess = this.preprocess;

    if (hasIpc()) {
      window.appApi!.invoke('ws:detect:send-frame', payload).catch(() => { /* ignore */ });
    } else if (this.socket) {
      this.socket.emit('frame', payload);
    }
  }

  close(): void {
    try { this.unsubResult?.(); } catch {}
    try { this.unsubStatus?.(); } catch {}
    try { this.unsubHello?.(); } catch {}
    this.unsubResult = undefined;
    this.unsubStatus = undefined;
    this.unsubHello = undefined;

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
