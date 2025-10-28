# Hướng dẫn Sửa chữa Camera - Desktop UI for Drowsiness Detection

## Tổng quan
App này là một hệ thống phát hiện ngủ gật học sinh sử dụng:
- **Frontend**: React + TypeScript + Electron
- **Backend**: Python Flask + OpenCV + YOLO
- **Camera**: Webcam và IP Camera

## Cấu trúc Project
```
Desktop UI for Drowsiness Detection/
├── src/
│   ├── components/
│   │   ├── CameraCard.tsx          # Component hiển thị camera feed
│   │   ├── CameraGrid.tsx          # Grid layout cho cameras
│   │   ├── CameraSidebar.tsx       # Sidebar danh sách cameras
│   │   └── CameraDialog.tsx        # Dialog thêm/sửa camera
│   ├── App.tsx                     # Main app component
│   ├── types/index.ts              # TypeScript interfaces
│   └── lib/mockData.ts             # Mock data và config
├── electron/
│   └── main.js                     # Electron main process
├── python-backend/
│   ├── server.py                   # Flask API server
│   ├── main.py                     # Camera management & detection
│   └── requirements.txt            # Python dependencies
└── dist/                           # Built frontend files
```

## Các Vấn đề Camera Thường Gặp

### 1. Camera hiển thị màn hình đen
**Triệu chứng**: Camera card hiển thị "Đang hoạt động" nhưng video feed là màn hình đen

**Nguyên nhân có thể**:
- Stream endpoint trả về 404 NOT FOUND
- Camera không được add vào Python backend
- Camera không có frame data
- getUserMedia không hoạt động

**Cách sửa**:

#### Bước 1: Kiểm tra Python Backend
```bash
# Kiểm tra cameras có trong backend không
curl http://127.0.0.1:5000/api/cameras

# Kiểm tra camera cụ thể
curl http://127.0.0.1:5000/api/camera/{camera_id}

# Test stream endpoint
curl http://127.0.0.1:5000/api/camera/{camera_id}/stream
```

#### Bước 2: Kiểm tra CameraCard.tsx
File: `src/components/CameraCard.tsx`

**Vấn đề**: Code đang fetch từ stream endpoint nhưng endpoint không hoạt động
**Giải pháp**: Sử dụng getUserMedia trực tiếp cho webcam

```typescript
// Thay thế fetch stream bằng getUserMedia
useEffect(() => {
  if (camera.isRunning && camera.status === 'online' && camera.type === 'webcam') {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: camera.deviceId !== undefined ? { exact: camera.deviceId } : undefined,
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setVideoStream(stream);
        }
      } catch (error) {
        console.error(`Error starting webcam:`, error);
      }
    };
    
    startWebcam();
  }
}, [camera.isRunning, camera.status, camera.type, camera.deviceId]);
```

#### Bước 3: Cập nhật Render
```typescript
// Thay canvas bằng video element
{camera.isRunning && camera.status === 'online' ? (
  <div className="relative w-full h-full">
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className="w-full h-full object-cover"
    />
    {/* Canvas overlay cho tracking */}
    <canvas
      ref={canvasRef}
      className="absolute top-0 left-0 w-full h-full pointer-events-none"
    />
  </div>
) : (
  <div className="w-full h-full flex items-center justify-center">
    <div className="text-center text-gray-400">
      <AlertCircle className="h-12 w-12 mx-auto mb-2" />
      <div className="text-lg font-semibold">Camera offline</div>
    </div>
  </div>
)}
```

### 2. Camera không được add vào backend
**Triệu chứng**: Console hiển thị lỗi "Camera not found"

**Cách sửa**:

#### Kiểm tra App.tsx
File: `src/App.tsx`

```typescript
const handleSaveCamera = async (cameraData: Partial<Camera>) => {
  try {
    // Gửi camera data đến Python backend
    const response = await fetch('http://127.0.0.1:5000/api/camera/add', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        id: cameraData.id,
        config: cameraData
      })
    });

    const result = await response.json();
    
    if (result.success) {
      // Cập nhật local state
      if (editingCamera) {
        setCameras(prev => prev.map(c => 
          c.id === editingCamera.id ? { ...c, ...cameraData } : c
        ));
      } else {
        setCameras(prev => [...prev, cameraData as Camera]);
      }
    }
  } catch (error) {
    console.error('Error saving camera:', error);
  }
};
```

### 3. Camera không start được
**Triệu chứng**: Camera được add nhưng không start

**Cách sửa**:

#### Kiểm tra Python Backend
File: `python-backend/main.py`

```python
def start_camera(self, camera_id):
    """Start camera processing"""
    if camera_id not in self.cameras:
        logger.error(f"Camera {camera_id} not found")
        return False
    
    camera = self.cameras[camera_id]
    camera['running'] = True
    
    # Start processing thread
    thread = threading.Thread(target=self._process_camera, args=(camera_id,))
    thread.daemon = True
    thread.start()
    
    return True
```

#### Kiểm tra Webcam Backend
```python
def add_camera(self, camera_id, camera_config):
    """Add a new camera"""
    try:
        if camera_config['type'] == 'webcam':
            device_id = int(camera_config.get('deviceId', 0))
            
            # Thử các backend khác nhau
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            cap = None
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(device_id, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            logger.info(f"Successfully opened webcam {device_id}")
                            break
                        else:
                            cap.release()
                            cap = None
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    if cap:
                        cap.release()
                        cap = None
            
            if not cap or not cap.isOpened():
                raise Exception(f"Cannot open webcam device: {device_id}")
```

### 4. Stream endpoint trả về 404
**Triệu chứng**: Console hiển thị "GET /api/camera/{id}/stream 404 (NOT FOUND)"

**Cách sửa**:

#### Kiểm tra server.py
File: `python-backend/server.py`

```python
@app.route('/api/camera/<camera_id>/stream', methods=['GET'])
def get_camera_stream(camera_id):
    """Get camera video stream as base64 encoded frames"""
    try:
        print(f"Stream request for camera {camera_id}")
        
        if camera_id not in camera_manager.cameras:
            return jsonify({'success': False, 'error': 'Camera not found'}), 404
        
        camera = camera_manager.cameras[camera_id]
        
        if not camera.get('running', False):
            return jsonify({'success': False, 'error': 'Camera not running'}), 400
        
        frame = camera.get('last_frame')
        if frame is None:
            return jsonify({'success': False, 'error': 'No frame available'}), 400
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return jsonify({'success': False, 'error': 'Failed to encode frame'}), 500
        
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'frame': frame_base64,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Error in stream endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

## Quy trình Debug Camera

### 1. Kiểm tra Backend
```bash
# 1. Kiểm tra Python backend có chạy không
curl http://127.0.0.1:5000/api/cameras

# 2. Add camera test
curl -X POST http://127.0.0.1:5000/api/camera/add \
  -H "Content-Type: application/json" \
  -d '{"id":"test-cam","config":{"name":"Test","type":"webcam","deviceId":0}}'

# 3. Start camera
curl -X POST http://127.0.0.1:5000/api/camera/test-cam/start

# 4. Test stream
curl http://127.0.0.1:5000/api/camera/test-cam/stream
```

### 2. Kiểm tra Frontend
```typescript
// Trong CameraCard.tsx, thêm debug logging
useEffect(() => {
  console.log('Camera state changed:', {
    id: camera.id,
    isRunning: camera.isRunning,
    status: camera.status,
    type: camera.type,
    deviceId: camera.deviceId
  });
}, [camera.isRunning, camera.status, camera.type, camera.deviceId]);
```

### 3. Kiểm tra Electron
```javascript
// Trong electron/main.js
function startPythonBackend() {
  const pythonPath = path.join(__dirname, '..', 'python-backend');
  const serverPath = path.join(pythonPath, 'server.py');
  
  console.log('Starting Python backend from:', serverPath);
  
  pythonProcess = spawn('python', [serverPath], {
    cwd: pythonPath,
    stdio: ['pipe', 'pipe', 'pipe']
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
  });
}
```

## Các Lỗi Thường Gặp và Giải pháp

### 1. "Camera not found" Error
**Nguyên nhân**: Camera không được add vào backend
**Giải pháp**: 
- Kiểm tra `handleSaveCamera` trong App.tsx
- Đảm bảo camera data được gửi đúng format
- Kiểm tra Python backend có nhận được request không

### 2. "Camera not running" Error
**Nguyên nhân**: Camera được add nhưng không start
**Giải pháp**:
- Kiểm tra `handleToggleCamera` trong App.tsx
- Đảm bảo camera được start sau khi add
- Kiểm tra Python backend có start camera không

### 3. "No frame available" Error
**Nguyên nhân**: Camera start nhưng không có frame data
**Giải pháp**:
- Kiểm tra `_process_camera` trong main.py
- Đảm bảo camera có thể đọc frame
- Kiểm tra OpenCV backend compatibility

### 4. getUserMedia Error
**Nguyên nhân**: Browser không thể truy cập webcam
**Giải pháp**:
- Kiểm tra camera permissions
- Thử deviceId khác (0, 1, 2...)
- Kiểm tra camera có bị sử dụng bởi app khác không

## Cấu hình Camera

### Webcam Configuration
```typescript
// Trong CameraDialog.tsx
const cameraData = {
  id: `cam-${Date.now()}`,
  name: "Webcam",
  type: "webcam",
  deviceId: 0, // Thử 0, 1, 2...
  config: {
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
  }
};
```

### IP Camera Configuration
```typescript
const cameraData = {
  id: `cam-${Date.now()}`,
  name: "IP Camera",
  type: "ip",
  brand: "Hikvision",
  ip: "192.168.1.100",
  port: 554,
  username: "admin",
  password: "password",
  streamQuality: "main",
  rtspUrl: "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"
};
```

## Testing Camera

### 1. Test Webcam
```bash
# Test webcam với OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam available:', cap.isOpened()); cap.release()"
```

### 2. Test IP Camera
```bash
# Test RTSP stream
python -c "import cv2; cap = cv2.VideoCapture('rtsp://admin:password@192.168.1.100:554/stream'); print('RTSP available:', cap.isOpened()); cap.release()"
```

### 3. Test Frontend
```typescript
// Test getUserMedia
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    console.log('Webcam access granted');
    stream.getTracks().forEach(track => track.stop());
  })
  .catch(error => {
    console.error('Webcam access denied:', error);
  });
```

## Troubleshooting Checklist

- [ ] Python backend đang chạy trên port 5000
- [ ] Camera được add vào backend thành công
- [ ] Camera được start thành công
- [ ] Camera có frame data
- [ ] Frontend có thể fetch camera data
- [ ] getUserMedia hoạt động cho webcam
- [ ] Video element hiển thị stream
- [ ] Canvas overlay hiển thị tracking

## Liên hệ Hỗ trợ

Nếu vẫn gặp vấn đề, hãy:
1. Kiểm tra console logs trong DevTools
2. Kiểm tra Python backend logs
3. Test camera với OpenCV trực tiếp
4. Kiểm tra camera permissions trong browser
5. Thử với camera khác để isolate vấn đề

