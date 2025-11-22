import { io, Socket } from 'socket.io-client';
import { ENV } from '../config/env';
import { DetectionResult } from '../types/detection';

export class DetectionWSClient {
  private socket: Socket | null = null;
  private connected = false;
  private conf: number | null = null;
  private preprocess: { enabled?: boolean; gamma?: number; beta?: number } | null = null;
  private lastPingTime: number = Date.now();
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private onStatusChangeCallback?: (connected: boolean) => void;

  /**
   * Get current connection status
   */
  get isConnected(): boolean {
    return this.connected;
  }

  /**
   * Register callback for connection status changes
   */
  onStatusChange(callback: (connected: boolean) => void): void {
    this.onStatusChangeCallback = callback;
  }

  connect(onResult: (res: DetectionResult) => void): void {
    if (this.socket) return;
    // eslint-disable-next-line no-console
    console.log(`[WS] Creating socket.io client to ${ENV.WS_DETECT_URL}`);
    
    // Parse URL to extract server and namespace separately
    // ENV.WS_DETECT_URL = 'http://127.0.0.1:5000/ws/detect'
    // Need: server='http://127.0.0.1:5000', namespace='/ws/detect'
    const url = new URL(ENV.WS_DETECT_URL);
    const serverUrl = `${url.protocol}//${url.host}`;
    const namespace = url.pathname || '/';
    
    console.log(`[WS] Connecting to server=${serverUrl}, namespace=${namespace}`);
    
    // Connect to namespace explicitly
    this.socket = io(serverUrl + namespace, {
      path: '/socket.io/',
      transports: ['websocket', 'polling'],
      withCredentials: false,
      reconnection: true,
      reconnectionAttempts: ENV.WS_RECONNECTION_ATTEMPTS,
      reconnectionDelay: ENV.WS_RECONNECTION_DELAY,
      reconnectionDelayMax: ENV.WS_RECONNECTION_DELAY_MAX,
    });

    this.socket.on('connect', () => {
      this.connected = true;
      this.lastPingTime = Date.now();
      // eslint-disable-next-line no-console
      console.log('[WS] Connected to /ws/detect');
      // Notify caller that WS is ready so upstream can start sending frames
      try { 
        onResult({ success: true }); 
        this.onStatusChangeCallback?.(true);
      } catch {}
    });

    this.socket.on('disconnect', (reason: string) => {
      this.connected = false;
      // eslint-disable-next-line no-console
      console.log('[WS] Disconnected:', reason);
      this.onStatusChangeCallback?.(false);
    });

    this.socket.on('connect_error', (err: any) => {
      // eslint-disable-next-line no-console
      console.warn('[WS] connect_error:', err?.message || err);
      this.onStatusChangeCallback?.(false);
    });

    this.socket.on('hello', (msg: any) => {
      // eslint-disable-next-line no-console
      console.log('[WS] hello:', msg);
      // Also signal readiness on hello to unblock senders waiting for first result
      try { onResult({ success: true }); } catch {}
    });

    this.socket.on('result', (msg: DetectionResult) => {
      // eslint-disable-next-line no-console
      try { 
        console.log('[WS] 📥 result received:', {
          persons: Array.isArray(msg?.persons) ? msg.persons.length : 0,
          fps: msg?.fps ?? 0,
          success: msg?.success,
          fullData: msg
        }); 
      } catch {}
      onResult(msg);
    });

    // Monitor pong responses for health check
    this.socket.on('pong', () => {
      this.lastPingTime = Date.now();
    });

    // Start health check monitoring
    this.startHealthCheck();
  }

  /**
   * Start health check monitoring to detect stale connections
   */
  private startHealthCheck(): void {
    if (this.healthCheckInterval) return;

    this.healthCheckInterval = setInterval(() => {
      if (!this.socket || !this.connected) return;

      const timeSinceLastPing = Date.now() - this.lastPingTime;
      if (timeSinceLastPing > ENV.WS_HEALTH_CHECK_TIMEOUT) {
        console.warn('[WS] Connection might be stale (no pong), reconnecting...');
        this.socket.disconnect();
        this.socket.connect();
      }
    }, ENV.WS_HEALTH_CHECK_INTERVAL);
  }

  /**
   * Stop health check monitoring
   */
  private stopHealthCheck(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
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

  /**
   * Close connection and cleanup resources
   */
  close(): void {
    this.stopHealthCheck();
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
    this.connected = false;
    this.conf = null;
    this.preprocess = null;
    this.onStatusChangeCallback = undefined;
  }
}




