export type CameraStatus = 'online' | 'offline' | 'reconnecting' | 'connecting' | 'error';
export type DetectionState = 'normal' | 'sleepy' | 'head_down';

export interface Student {
  id: string;
  position: { x: number; y: number };
  state: DetectionState;
  confidence: number;
  sleepDuration: number; // seconds
  lastUpdate: Date;
}

export interface Camera {
  id: string;
  name: string;
  type: 'webcam' | 'ip';
  status: CameraStatus;
  fps: number;
  isRunning: boolean;
  students: Student[];
  totalStudents: number;
  sleepyStudents: number;
  deviceId?: number;
  brand?: string;
  ip?: string;
  port?: number;
  username?: string;
  password?: string;
  streamQuality?: 'main' | 'sub';
  rtspUrl?: string;
  config: CameraConfig;
  lastConnectAttempt?: Date;
  errorMessage?: string;
}

export interface CameraConfig {
  decorators: {
    reconnect: boolean;
    frameQueue: boolean;
    performance: boolean;
    detection: boolean;
    overlay: boolean;
    logging: boolean;
  };
  model: string;
  confidence: number;
  strategy: 'YOLO' | 'Mediapipe' | 'EAR';
  showFPS: boolean;
  showOverlay: boolean;
  maxQueueSize: number;
}

export interface LogEvent {
  id: string;
  timestamp: Date;
  cameraId: string;
  cameraName: string;
  studentId?: string;
  studentPosition?: string;
  type: 'sleepy' | 'wake_up' | 'head_down' | 'connection' | 'error' | 'detection_start' | 'detection_end';
  message: string;
  duration?: number;
  studentCount?: number;
}

export interface SystemStats {
  totalFPS: number;
  runningCameras: number;
  totalCameras: number;
  totalStudents: number;
  sleepyStudents: number;
  gpuUsage?: number;
  cpuUsage: number;
  reconnectCount: number;
}
