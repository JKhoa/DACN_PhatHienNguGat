/**
 * V1 realtime detection WebSocket client — IPC bridge to main process,
 * which owns the socket.io connection to Python namespace `/api/v1/detect/realtime`.
 *
 * Server protocol (see python-backend/api_v1.py):
 *   client → server: `frame` { image_base64, conf }
 *   server → client: `ready` {...}, `result` {...}, `error` {...}
 */

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
  private unsubStatus?: () => void;
  private unsubResult?: () => void;
  private unsubError?: () => void;
  private bridgeReady = false;
  connected = false;

  get isConnected(): boolean {
    return this.connected;
  }

  onStatus(cb: StatusCb): void { this.statusCb = cb; }
  onResult(cb: ResultCb): void { this.resultCb = cb; }
  onError(cb: ErrorCb): void { this.errorCb = cb; }

  connect(): void {
    if (this.bridgeReady) return;
    if (!window.appApi) {
      console.error('[WS-V1] window.appApi missing — preload.js not loaded?');
      return;
    }

    this.unsubStatus = window.appApi.on('ws:v1:status', (raw: unknown) => {
      const p = raw as { connected: boolean };
      this.connected = !!p?.connected;
      this.statusCb?.(this.connected);
    });
    this.unsubResult = window.appApi.on('ws:v1:result', (raw: unknown) => {
      this.resultCb?.(raw as V1DetectionResult);
    });
    this.unsubError = window.appApi.on('ws:v1:error', (raw: unknown) => {
      this.errorCb?.(raw as { error: string });
    });

    window.appApi.invoke('ws:v1:connect').catch((e) =>
      console.warn('[WS-V1] connect failed:', e)
    );
    this.bridgeReady = true;
  }

  sendFrame(imageBase64: string, conf: number): void {
    if (!this.bridgeReady) this.connect();
    window.appApi?.invoke('ws:v1:send-frame', { image_base64: imageBase64, conf })
      .catch(() => { /* ignore, status handler reports disconnect */ });
  }

  disconnect(): void {
    try { this.unsubStatus?.(); } catch { /* noop */ }
    try { this.unsubResult?.(); } catch { /* noop */ }
    try { this.unsubError?.(); } catch { /* noop */ }
    this.unsubStatus = undefined;
    this.unsubResult = undefined;
    this.unsubError = undefined;
    this.bridgeReady = false;
    this.connected = false;
    this.resultCb = undefined;
    this.statusCb = undefined;
    this.errorCb = undefined;
    window.appApi?.invoke('ws:v1:disconnect').catch(() => { /* ignore */ });
  }
}

export const wsDetectV1 = new WSDetectV1();
