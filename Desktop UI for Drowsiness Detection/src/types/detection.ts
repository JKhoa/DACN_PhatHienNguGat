// Detailed types for detection results

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
  visible: boolean;
}

export interface Person {
  id: number;
  track_id: number;
  bbox: number[]; // [x1, y1, x2, y2]
  head_bbox?: number[] | null;
  confidence: number;
  keypoints?: Keypoint[];
  drowsiness_score?: number;
  drowsiness_state?: 'awake' | 'drowsy' | 'sleeping';
  last_update?: number;
}

export interface DetectionResult {
  success: boolean;
  schema?: string;
  frame_width?: number;
  frame_height?: number;
  fps?: number;
  processing_time?: number;
  camera_id?: string;
  persons?: Person[];
  timestamp?: number;
  error?: string;
}

export interface CameraUpdate {
  success: boolean;
  schema?: string;
  camera_id: string;
  frame_width: number;
  frame_height: number;
  fps: number;
  processing_time?: number;
  persons: Person[];
  timestamp: number;
}
