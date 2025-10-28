import { Camera, LogEvent, CameraConfig, Student } from '../types';

export const defaultCameraConfig: CameraConfig = {
  decorators: {
    reconnect: true,
    frameQueue: true,
    performance: true,
    detection: true,
    overlay: true,
    logging: true,
  },
  model: 'yolo11n-pose.pt',
  confidence: 0.5,
  strategy: 'YOLO',
  showFPS: true,
  showOverlay: true,
  maxQueueSize: 2,
};

// Generate RTSP URL based on brand
export const generateRTSPUrl = (
  brand: string,
  ip: string,
  port: number,
  username: string,
  password: string,
  streamQuality: 'main' | 'sub'
): string => {
  const channel = '1';
  const subtype = streamQuality === 'main' ? '0' : '1';
  
  switch (brand.toLowerCase()) {
    case 'hikvision':
      return `rtsp://${username}:${password}@${ip}:${port}/Streaming/Channels/${channel}${subtype === '0' ? '01' : '02'}`;
    case 'dahua':
      return `rtsp://${username}:${password}@${ip}:${port}/cam/realmonitor?channel=${channel}&subtype=${subtype}`;
    case 'ezviz':
      return `rtsp://${username}:${password}@${ip}:${port}/h264/ch${channel}/${streamQuality}/av_stream`;
    case 'kbvision':
      return `rtsp://${username}:${password}@${ip}:${port}/stream${subtype}`;
    default:
      return `rtsp://${username}:${password}@${ip}:${port}/stream`;
  }
};

// Generate students for real-time tracking (empty initially)
export const generateStudents = (count: number): Student[] => {
  return []; // Empty array - will be populated by real-time detection
};

// Empty camera slots ready for real camera connection
export const mockCameras: Camera[] = [
  // Empty array - no cameras initially
];

export const generateMockLog = (
  cameraId: string, 
  cameraName: string, 
  student?: Student,
  totalStudents?: number
): LogEvent => {
  const types: LogEvent['type'][] = ['sleepy', 'wake_up', 'head_down', 'detection_start'];
  const type = types[Math.floor(Math.random() * types.length)];
  
  const getPositionName = (studentId: string) => {
    const num = parseInt(studentId.split('-')[1]);
    const row = Math.floor((num - 1) / 8) + 1;
    const col = ((num - 1) % 8) + 1;
    return `Hàng ${row}, Vị trí ${col}`;
  };
  
  const messages = {
    sleepy: student 
      ? `Học sinh ${getPositionName(student.id)} có dấu hiệu buồn ngủ`
      : 'Phát hiện dấu hiệu buồn ngủ',
    wake_up: student
      ? `Học sinh ${getPositionName(student.id)} tỉnh táo trở lại`
      : 'Học sinh tỉnh táo trở lại',
    head_down: student
      ? `Học sinh ${getPositionName(student.id)} gục xuống bàn`
      : 'Phát hiện gục xuống bàn',
    connection: 'Kết nối camera thành công',
    detection_start: `Bắt đầu giám sát ${totalStudents || 0} học sinh`,
    detection_end: 'Dừng giám sát',
    error: 'Lỗi kết nối',
  };

  return {
    id: `log-${Date.now()}-${Math.random()}`,
    timestamp: new Date(),
    cameraId,
    cameraName,
    studentId: student?.id,
    studentPosition: student ? getPositionName(student.id) : undefined,
    type,
    message: messages[type],
    duration: type === 'sleepy' && student ? student.sleepDuration : undefined,
    studentCount: type === 'detection_start' ? totalStudents : undefined,
  };
};
