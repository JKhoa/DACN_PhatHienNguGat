/**
 * Environment configuration — localhost only.
 *
 * Frontend giao tiếp trực tiếp với Python backend (Flask + Socket.IO) qua
 * REST + WebSocket. Backend URL mặc định http://127.0.0.1:5000, override
 * bằng VITE_BACKEND_URL nếu cần. Các giá trị reconnect/health-check áp
 * dụng cho socket.io client.
 */
export const ENV = {
  WS_RECONNECTION_ATTEMPTS: import.meta.env.VITE_WS_RECONNECTION_ATTEMPTS === 'Infinity'
    ? Infinity
    : Number(import.meta.env.VITE_WS_RECONNECTION_ATTEMPTS) || Infinity,
  WS_RECONNECTION_DELAY: Number(import.meta.env.VITE_WS_RECONNECTION_DELAY) || 500,
  WS_RECONNECTION_DELAY_MAX: Number(import.meta.env.VITE_WS_RECONNECTION_DELAY_MAX) || 3000,
  WS_HEALTH_CHECK_INTERVAL: Number(import.meta.env.VITE_WS_HEALTH_CHECK_INTERVAL) || 5000,
  WS_HEALTH_CHECK_TIMEOUT: Number(import.meta.env.VITE_WS_HEALTH_CHECK_TIMEOUT) || 10000,
} as const;

export type EnvConfig = typeof ENV;
