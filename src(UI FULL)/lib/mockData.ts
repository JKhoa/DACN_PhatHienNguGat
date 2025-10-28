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

// Generate random students for a camera
export const generateStudents = (count: number): Student[] => {
  const students: Student[] = [];
  const states: Student['state'][] = ['normal', 'normal', 'normal', 'normal', 'sleepy', 'head_down'];
  
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / 8); // 8 students per row
    const col = i % 8;
    const state = states[Math.floor(Math.random() * states.length)];
    
    students.push({
      id: `student-${i + 1}`,
      position: {
        x: 50 + col * 120 + Math.random() * 20,
        y: 80 + row * 100 + Math.random() * 20,
      },
      state,
      confidence: 0.7 + Math.random() * 0.25,
      sleepDuration: state === 'sleepy' ? Math.floor(Math.random() * 300) : 0,
      lastUpdate: new Date(),
    });
  }
  
  return students;
};

export const mockCameras: Camera[] = [
  {
    id: 'cam-1',
    name: 'Camera Lớp 12A1 - Phòng 101',
    type: 'ip',
    status: 'online',
    fps: 30,
    isRunning: true,
    students: generateStudents(32),
    totalStudents: 32,
    sleepyStudents: 0,
    brand: 'Hikvision',
    ip: '192.168.1.101',
    port: 554,
    username: 'admin',
    password: 'admin123',
    streamQuality: 'main',
    rtspUrl: generateRTSPUrl('Hikvision', '192.168.1.101', 554, 'admin', 'admin123', 'main'),
    config: { ...defaultCameraConfig },
  },
  {
    id: 'cam-2',
    name: 'Camera Lớp 12A2 - Phòng 102',
    type: 'ip',
    status: 'online',
    fps: 28,
    isRunning: true,
    students: generateStudents(28),
    totalStudents: 28,
    sleepyStudents: 0,
    brand: 'Dahua',
    ip: '192.168.1.102',
    port: 554,
    username: 'admin',
    password: 'admin123',
    streamQuality: 'main',
    rtspUrl: generateRTSPUrl('Dahua', '192.168.1.102', 554, 'admin', 'admin123', 'main'),
    config: { ...defaultCameraConfig },
  },
  {
    id: 'cam-3',
    name: 'Camera Lớp 11A1 - Phòng 201',
    type: 'ip',
    status: 'online',
    fps: 30,
    isRunning: true,
    students: generateStudents(35),
    totalStudents: 35,
    sleepyStudents: 0,
    brand: 'Hikvision',
    ip: '192.168.1.103',
    port: 554,
    username: 'admin',
    password: 'admin123',
    streamQuality: 'sub',
    rtspUrl: generateRTSPUrl('Hikvision', '192.168.1.103', 554, 'admin', 'admin123', 'sub'),
    config: { ...defaultCameraConfig },
  },
  {
    id: 'cam-4',
    name: 'Camera Lớp 11A2 - Phòng 202',
    type: 'ip',
    status: 'offline',
    fps: 0,
    isRunning: false,
    students: [],
    totalStudents: 30,
    sleepyStudents: 0,
    brand: 'Ezviz',
    ip: '192.168.1.104',
    port: 554,
    username: 'admin',
    password: 'admin123',
    streamQuality: 'main',
    errorMessage: 'Không thể kết nối đến camera',
    config: { ...defaultCameraConfig },
  },
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
