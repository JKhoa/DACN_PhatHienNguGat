/**
 * Environment configuration.
 *
 * After the IPC migration, all WebSocket and HTTP traffic is routed through
 * the main process (electron/wsBridge.js + electron/main.js). The renderer no
 * longer needs the localhost URLs — only the reconnection/health-check knobs
 * remain, and they are now consumed inside the bridge if needed.
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
