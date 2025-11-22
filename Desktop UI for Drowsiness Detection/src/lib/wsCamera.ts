import { io, Socket } from 'socket.io-client';
import { ENV } from '../config/env';
import { CameraUpdate } from '../types/detection';

class WSCamera {
  private socket: Socket | null = null;
  private handlers = new Map<string, (msg: CameraUpdate) => void>();
  private lastPingTime: number = Date.now();
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private onStatusChangeCallback?: (connected: boolean) => void;
  connected = false;

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

  private ensureSocket() {
    if (this.socket) return;
    this.socket = io(ENV.WS_CAMERA_URL, {
      path: '/socket.io/',
      transports: ['websocket'],
      withCredentials: false,
      reconnection: true,
      reconnectionAttempts: ENV.WS_RECONNECTION_ATTEMPTS,
      reconnectionDelay: ENV.WS_RECONNECTION_DELAY,
      reconnectionDelayMax: ENV.WS_RECONNECTION_DELAY_MAX,
    });

    this.socket.on('connect', () => {
      this.connected = true;
      this.lastPingTime = Date.now();
      console.log('[WS-CAM] Connected to /ws/camera');
      this.onStatusChangeCallback?.(true);
      
      // re-subscribe existing rooms after reconnect
      for (const room of this.handlers.keys()) {
        const camId = room.replace('cam:', '');
        this.socket?.emit('subscribe', { camera_id: camId });
      }
    });

    this.socket.on('disconnect', (reason) => {
      this.connected = false;
      console.log('[WS-CAM] Disconnected:', reason);
      this.onStatusChangeCallback?.(false);
    });

    this.socket.on('connect_error', (err: any) => {
      console.warn('[WS-CAM] connect_error:', err?.message || err);
      this.onStatusChangeCallback?.(false);
    });

    this.socket.on('update', (msg: CameraUpdate) => {
      try {
        if (!msg || !msg.success || !msg.camera_id) return;
        const key = `cam:${msg.camera_id}`;
        const handler = this.handlers.get(key);
        if (handler) handler(msg);
      } catch {}
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
        console.warn('[WS-CAM] Connection might be stale (no pong), reconnecting...');
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
    
    // If no more handlers, consider cleanup
    if (this.handlers.size === 0) {
      console.log('[WS-CAM] No more subscriptions, keeping connection alive for future use');
    }
  }

  /**
   * Disconnect and cleanup all resources
   */
  disconnect(): void {
    console.log('[WS-CAM] Disconnecting and cleaning up...');
    this.stopHealthCheck();
    this.handlers.clear();
    
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
    
    this.connected = false;
    this.onStatusChangeCallback = undefined;
  }

  /**
   * Get number of active subscriptions
   */
  getSubscriptionCount(): number {
    return this.handlers.size;
  }
}

export const wsCamera = new WSCamera();
