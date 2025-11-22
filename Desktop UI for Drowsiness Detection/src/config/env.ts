// Environment configuration with fallbacks
export const ENV = {
  // WebSocket URLs
  WS_DETECT_URL: import.meta.env.VITE_WS_DETECT_URL || 'http://127.0.0.1:5000/ws/detect',
  WS_CAMERA_URL: import.meta.env.VITE_WS_CAMERA_URL || 'http://127.0.0.1:5000/ws/camera',
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000',
  
  // WebSocket Configuration
  WS_RECONNECTION_ATTEMPTS: import.meta.env.VITE_WS_RECONNECTION_ATTEMPTS === 'Infinity' 
    ? Infinity 
    : Number(import.meta.env.VITE_WS_RECONNECTION_ATTEMPTS) || Infinity,
  WS_RECONNECTION_DELAY: Number(import.meta.env.VITE_WS_RECONNECTION_DELAY) || 500,
  WS_RECONNECTION_DELAY_MAX: Number(import.meta.env.VITE_WS_RECONNECTION_DELAY_MAX) || 3000,
  WS_HEALTH_CHECK_INTERVAL: Number(import.meta.env.VITE_WS_HEALTH_CHECK_INTERVAL) || 5000,
  WS_HEALTH_CHECK_TIMEOUT: Number(import.meta.env.VITE_WS_HEALTH_CHECK_TIMEOUT) || 10000,
} as const;

// Type-safe environment config
export type EnvConfig = typeof ENV;
