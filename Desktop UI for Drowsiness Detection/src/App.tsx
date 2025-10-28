import { useState, useEffect } from 'react';
import { Camera, LogEvent, SystemStats } from './types';
import { mockCameras, generateMockLog, defaultCameraConfig } from './lib/mockData';
import { detectWorkingCamera } from './lib/webcamRegistry';
import { Toolbar } from './components/Toolbar';
import { StatusBar } from './components/StatusBar';
import { CameraSidebar } from './components/CameraSidebar';
import { CameraGrid } from './components/CameraGrid';
import { LogPanel } from './components/LogPanel';
import { CameraDialog } from './components/CameraDialog';
import { SettingsDialog } from './components/SettingsDialog';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from './components/ui/resizable';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

export default function App() {
  const [cameras, setCameras] = useState<Camera[]>(mockCameras);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>();
  const [gridSize, setGridSize] = useState<'1x1' | '2x2' | '3x3' | '4x4'>('2x2');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const [performanceEnabled, setPerformanceEnabled] = useState(true);
  const [loggingEnabled, setLoggingEnabled] = useState(true);
  const [cameraDialogOpen, setCameraDialogOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera>();

  // System stats
  const [stats, setStats] = useState<SystemStats>({
    totalFPS: 0,
    runningCameras: 0,
    totalCameras: cameras.length,
    cpuUsage: 0,
    gpuUsage: 45,
    reconnectCount: 0,
  });

  // Initialize YOLO detector on app startup
  useEffect(() => {
    const initializeYOLO = async () => {
      try {
        console.log('Initializing YOLO detector...');
        const response = await fetch('http://127.0.0.1:5000/api/detection/initialize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ model_path: 'yolo11n-pose.pt' }),
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            console.log('YOLO detector initialized successfully');
            toast.success('YOLO detector đã được khởi tạo');
          } else {
            console.error('Failed to initialize YOLO detector:', data.error);
            toast.error('Không thể khởi tạo YOLO detector');
          }
        } else {
          console.error('Failed to initialize YOLO detector');
          toast.error('Không thể kết nối với backend để khởi tạo YOLO');
        }
      } catch (error) {
        console.error('Error initializing YOLO detector:', error);
        toast.error('Lỗi khi khởi tạo YOLO detector');
      }
    };

    // Initialize YOLO after a short delay to ensure backend is ready
    const timer = setTimeout(initializeYOLO, 3000);
    return () => clearTimeout(timer);
  }, []);

  // Auto-detect camera on app startup
  useEffect(() => {
    const autoDetectCamera = async () => {
      try {
        console.log('Auto-detecting camera on startup...');
        const workingCamera = await detectWorkingCamera();
        
        if (workingCamera) {
          console.log(`Auto-detected camera: ${workingCamera.label}`);
          
          // Check if this camera is already added
          const existingCamera = cameras.find(c => c.deviceId?.toString() === workingCamera.deviceId);
          
          if (!existingCamera) {
            // Add the detected camera
            const newCamera: Camera = {
              id: `webcam-${workingCamera.deviceId}`,
              name: workingCamera.label,
              type: 'webcam',
              status: 'offline',
              fps: 0,
              isRunning: false,
              students: [],
              totalStudents: 30,
              sleepyStudents: 0,
              deviceId: parseInt(workingCamera.deviceId),
              config: defaultCameraConfig,
            };
            
            setCameras(prev => [...prev, newCamera]);
            toast.success(`Đã tự động phát hiện camera: ${workingCamera.label}`);
          } else {
            console.log('Camera already exists in list');
          }
        } else {
          console.log('No working camera detected on startup');
        }
      } catch (error) {
        console.error('Error auto-detecting camera:', error);
      }
    };
    
    // Delay auto-detection to allow app to fully load
    const timer = setTimeout(autoDetectCamera, 2000);
    return () => clearTimeout(timer);
  }, []); // Only run once on mount

  // Toggle dark mode
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Sync camera data from Python backend
  useEffect(() => {
    const syncCameras = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/api/cameras');
        const result = await response.json();
        
        if (result.success && result.cameras) {
          // Convert Python backend format to frontend format
          const backendCameras = Object.values(result.cameras).map((cam: any) => ({
            id: cam.id,
            name: cam.name,
            type: cam.config?.type || 'webcam',
            status: cam.status,
            fps: cam.fps || 0,
            isRunning: cam.status === 'online',
            students: cam.students || [],
            totalStudents: cam.totalStudents || 0,
            sleepyStudents: cam.sleepyStudents || 0,
            deviceId: cam.config?.deviceId,
            brand: cam.config?.brand,
            ip: cam.config?.ip,
            port: cam.config?.port,
            username: cam.config?.username,
            password: cam.config?.password,
            streamQuality: cam.config?.streamQuality,
            rtspUrl: cam.config?.rtspUrl,
            config: cam.config || { ...defaultCameraConfig },
            errorMessage: cam.status === 'offline' ? 'Camera offline' : undefined,
          }));
          // Merge: keep local webcams (handled by browser) + backend IP cams
          setCameras(prev => {
            const localWebcams = prev.filter(c => c.type === 'webcam');
            // For IP cams, backend is source of truth. Remove existing IP cams and replace with backend list.
            return [...localWebcams, ...backendCameras.filter((c: any) => c.type !== 'webcam')];
          });
        }
      } catch (error) {
        console.error('Error syncing cameras:', error);
      }
    };

    // Initial sync
    syncCameras();
    
    // Sync every 2 seconds
    const interval = setInterval(syncCameras, 2000);
    
    return () => clearInterval(interval);
  }, []);

  // Generate mock logs
  useEffect(() => {
    const interval = setInterval(() => {
      const runningCameras = cameras.filter(c => c.isRunning && c.status === 'online');
      if (runningCameras.length > 0 && loggingEnabled) {
        const randomCamera = runningCameras[Math.floor(Math.random() * runningCameras.length)];
        
        // Pick a random student from the camera if available
        const student = randomCamera.students.length > 0
          ? randomCamera.students[Math.floor(Math.random() * randomCamera.students.length)]
          : undefined;
        
        const newLog = generateMockLog(
          randomCamera.id, 
          randomCamera.name, 
          student,
          randomCamera.totalStudents
        );
        setLogs(prev => [newLog, ...prev].slice(0, 100));

        if (newLog.type === 'sleepy' && student) {
          toast.warning(`${randomCamera.name}: Phát hiện buồn ngủ!`, {
            description: newLog.message,
          });
        }
      }
    }, 8000);

    return () => clearInterval(interval);
  }, [cameras, loggingEnabled]);

  // Simulate student state changes
  useEffect(() => {
    const interval = setInterval(() => {
      setCameras(prev => prev.map(camera => {
        if (!camera.isRunning || camera.status !== 'online') return camera;

        // Update each student's state
        const updatedStudents = camera.students.map(student => {
          const rand = Math.random();
          let newState = student.state;
          let newSleepDuration = student.sleepDuration;

          // State transitions
          if (student.state === 'sleepy') {
            newSleepDuration += 3; // 3 seconds per update
            if (rand < 0.1) newState = 'normal'; // 10% chance to wake up
            if (rand > 0.95) newState = 'head_down'; // 5% chance to head down
          } else if (student.state === 'head_down') {
            if (rand < 0.15) newState = 'normal'; // 15% chance to wake up
          } else {
            // Normal state
            if (rand < 0.02) newState = 'sleepy'; // 2% chance to get sleepy
            newSleepDuration = 0;
          }

          return {
            ...student,
            state: newState,
            sleepDuration: newSleepDuration,
            confidence: 0.7 + Math.random() * 0.25,
            lastUpdate: new Date(),
          };
        });

        const sleepyStudents = updatedStudents.filter(s => s.state === 'sleepy' || s.state === 'head_down').length;

        return {
          ...camera,
          students: updatedStudents,
          sleepyStudents,
          fps: Math.floor(Math.random() * 3 + 28),
        };
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const handleStartAll = async () => {
    // Start IP cameras in backend; webcams are handled in browser
    try {
      for (const cam of cameras) {
        if (cam.type !== 'webcam') {
          try {
            await fetch(`http://127.0.0.1:5000/api/camera/${cam.id}/start`, { method: 'POST' });
          } catch {}
        }
      }
    } catch {}

    setCameras(prev => prev.map(c => {
      // Generate students when starting
      const students = c.type === 'ip' && c.ip 
        ? Array.from({ length: Math.floor(Math.random() * 10 + (c.totalStudents - 5)) }, (_, i) => ({
            id: `student-${i + 1}`,
            position: {
              x: 50 + (i % 8) * 75 + Math.random() * 20,
              y: 70 + Math.floor(i / 8) * 60 + Math.random() * 20,
            },
            state: 'normal' as const,
            confidence: 0.7 + Math.random() * 0.25,
            sleepDuration: 0,
            lastUpdate: new Date(),
          }))
        : [];
      
      return { 
        ...c, 
        isRunning: true,
        // For webcams, we rely on getUserMedia in the browser instead of backend capture to avoid device locking
        status: 'online',
        students,
        sleepyStudents: 0,
      };
    }));
    toast.success('Đã khởi động tất cả camera');
  };

  const handleStopAll = async () => {
    try {
      // Stop only non-webcam cameras via Python backend
      for (const camera of cameras) {
        if (camera.isRunning && camera.type !== 'webcam') {
          const response = await fetch(`http://127.0.0.1:5000/api/camera/${camera.id}/stop`, {
            method: 'POST'
          });
          const result = await response.json();
          console.log(`Stop camera ${camera.id}:`, result);
        }
      }
      
      // Update local state
      setCameras(prev => prev.map(c => ({ ...c, isRunning: false, students: [], sleepyStudents: 0 })));
      toast.info('Đã dừng tất cả camera');
    } catch (error) {
      console.error('Error stopping all cameras:', error);
      toast.error('Lỗi dừng camera');
    }
  };

  const handleToggleCamera = async (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    if (!camera) return;

    try {
      if (!camera.isRunning) {
        if (camera.type === 'webcam') {
          // Do NOT start webcam in backend to prevent "device in use" conflicts. Use getUserMedia only.
          setCameras(prev => prev.map(c => 
            c.id === cameraId 
              ? { ...c, isRunning: true, status: 'online' as const }
              : c
          ));
          toast.success(`Đã khởi động ${camera.name} (webcam trong trình duyệt)`);
          return;
        }
        // Start camera via Python backend
        const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/start`, {
          method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
          setCameras(prev => prev.map(c => 
            c.id === cameraId 
              ? { ...c, isRunning: true, status: 'online' as const }
              : c
          ));
          toast.success(`Đã khởi động ${camera.name}`);
        } else {
          toast.error(`Lỗi khởi động camera: ${result.error}`);
        }
      } else {
        if (camera.type === 'webcam') {
          setCameras(prev => prev.map(c => 
            c.id === cameraId 
              ? { ...c, isRunning: false, students: [], sleepyStudents: 0 }
              : c
          ));
          toast.info(`Đã dừng ${camera.name} (webcam trong trình duyệt)`);
          return;
        }
        // Stop camera via Python backend
        const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/stop`, {
          method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
          setCameras(prev => prev.map(c => 
            c.id === cameraId 
              ? { ...c, isRunning: false, students: [], sleepyStudents: 0 }
              : c
          ));
          toast.info(`Đã dừng ${camera.name}`);
        } else {
          toast.error(`Lỗi dừng camera: ${result.error}`);
        }
      }
    } catch (error) {
      console.error('Error toggling camera:', error);
      toast.error('Lỗi kết nối đến backend');
    }
  };

  const handleDeleteCamera = async () => {
    if (selectedCameraId) {
      try {
        const response = await fetch(`http://127.0.0.1:5000/api/camera/${selectedCameraId}/remove`, {
          method: 'DELETE'
        });
        const result = await response.json();
        
        if (result.success) {
          setCameras(prev => prev.filter(c => c.id !== selectedCameraId));
          setSelectedCameraId(undefined);
          toast.success('Đã xóa camera');
        } else {
          toast.error(`Lỗi xóa camera: ${result.error}`);
        }
      } catch (error) {
        console.error('Error deleting camera:', error);
        toast.error('Lỗi kết nối đến backend');
      }
    } else {
      toast.error('Vui lòng chọn camera để xóa');
    }
  };

  const handleClearAllCameras = async () => {
    try {
      // Remove all cameras from Python backend
      for (const camera of cameras) {
        if (camera.type !== 'webcam') {
          const response = await fetch(`http://127.0.0.1:5000/api/camera/${camera.id}/remove`, {
            method: 'DELETE'
          });
          const result = await response.json();
          console.log(`Remove camera ${camera.id}:`, result);
        }
      }
      
      // Clear local state
      setCameras([]);
      setSelectedCameraId(undefined);
      toast.success('Đã xóa tất cả camera');
    } catch (error) {
      console.error('Error clearing all cameras:', error);
      toast.error('Lỗi xóa camera');
    }
  };

  const handleAddCamera = () => {
    setEditingCamera(undefined);
    setCameraDialogOpen(true);
  };

  const handleConfigureCamera = (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    setEditingCamera(camera);
    setCameraDialogOpen(true);
  };

  const handleSaveCamera = async (cameraData: Partial<Camera>) => {
    try {
      console.log('Saving camera data:', cameraData);
      
      // Only sync to backend for non-webcam cameras to avoid backend grabbing the webcam device
      let result: any = { success: true };
      if (cameraData.type !== 'webcam') {
        const response = await fetch('http://127.0.0.1:5000/api/camera/add', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            id: cameraData.id,
            url: cameraData.rtspUrl, // Send rtspUrl as url field
            name: cameraData.name
          })
        });
        result = await response.json();
        console.log('Backend response:', result);
      }
      
      if (result.success) {
        // Update local state
        if (editingCamera) {
          setCameras(prev => prev.map(c => 
            c.id === editingCamera.id ? { ...c, ...cameraData } : c
          ));
          toast.success('Đã cập nhật camera');
        } else {
          setCameras(prev => [...prev, cameraData as Camera]);
          toast.success('Đã thêm camera mới');
        }
      } else {
        toast.error(`Lỗi lưu camera: ${result.error}`);
      }
    } catch (error) {
      console.error('Error saving camera:', error);
      toast.error('Lỗi kết nối đến backend');
    }
  };

  const handlePopOut = (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    if (camera) {
      // Create a new window for the camera
      const popupWindow = window.open('', `camera-${cameraId}`, 'width=800,height=600');
      if (popupWindow) {
        popupWindow.document.write(`
          <html>
            <head><title>${camera.name}</title></head>
            <body style="margin:0;padding:0;background:black;">
              <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:white;">
                <div style="text-align:center;">
                  <h2>${camera.name}</h2>
                  <p>Camera Feed: ${camera.status}</p>
                  <p>Students: ${camera.students.length}</p>
                  <p>FPS: ${camera.fps}</p>
                </div>
              </div>
            </body>
          </html>
        `);
        toast.success(`Đã mở popup cho ${camera.name}`);
      } else {
        toast.error('Không thể mở popup window');
      }
    }
  };

  const handleSaveLayout = () => {
    const layout = {
      gridSize,
      cameras: cameras.map(c => c.id),
    };
    localStorage.setItem('camera-layout', JSON.stringify(layout));
    toast.success('Đã lưu bố cục');
  };

  const handleRestoreLayout = () => {
    const saved = localStorage.getItem('camera-layout');
    if (saved) {
      const layout = JSON.parse(saved);
      setGridSize(layout.gridSize);
      toast.success('Đã khôi phục bố cục');
    } else {
      toast.info('Không tìm thấy bố cục đã lưu');
    }
  };

  const handleExportConfig = () => {
    const config = {
      cameras: cameras.map(c => ({
        name: c.name,
        type: c.type,
        config: c.config,
        ...(c.type === 'webcam' ? { deviceId: c.deviceId } : {
          brand: c.brand,
          ip: c.ip,
          port: c.port,
          streamQuality: c.streamQuality,
        }),
      })),
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'camera-config.yaml';
    a.click();
    toast.success('Đã export cấu hình');
  };

  const handleImportConfig = () => {
    toast.info('Import YAML', {
      description: 'Tính năng đang được phát triển',
    });
  };

  const handleToggleOverlay = () => {
    setOverlayEnabled(!overlayEnabled);
    toast.success(`Overlay ${!overlayEnabled ? 'bật' : 'tắt'}`);
  };

  const handleToggleLogging = () => {
    setLoggingEnabled(!loggingEnabled);
    toast.success(`Logging ${!loggingEnabled ? 'bật' : 'tắt'}`);
  };

  const handleCapturePhoto = (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    if (camera && camera.status === 'online') {
      // Simulate photo capture
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `camera-${cameraId}-${timestamp}.jpg`;
      
      // In a real implementation, this would capture the actual frame
      toast.success(`Đã chụp ảnh: ${filename}`);
    } else {
      toast.error('Camera không hoạt động');
    }
  };

  const handleRecordVideo = (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    if (camera && camera.status === 'online') {
      // Simulate video recording
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `camera-${cameraId}-${timestamp}.mp4`;
      
      // In a real implementation, this would start/stop video recording
      toast.success(`Đã bắt đầu ghi video: ${filename}`);
    } else {
      toast.error('Camera không hoạt động');
    }
  };

  const handleExportLogs = () => {
    const csv = [
      'Timestamp,Camera,Type,Message,Duration',
      ...logs.map(log => 
        `${log.timestamp.toISOString()},${log.cameraName},${log.type},${log.message},${log.duration || ''}`
      ),
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'logs.csv';
    a.click();
    toast.success('Đã export logs');
  };

  const isAllRunning = cameras.every(c => c.isRunning);

  return (
    <div className="h-screen flex flex-col bg-background">
            <Toolbar
              isAllRunning={isAllRunning}
              onStartAll={handleStartAll}
              onStopAll={handleStopAll}
              onAddCamera={handleAddCamera}
              onDeleteCamera={handleDeleteCamera}
              onClearAllCameras={handleClearAllCameras}
              onImportConfig={handleImportConfig}
              onExportConfig={handleExportConfig}
              onSaveLayout={handleSaveLayout}
              onRestoreLayout={handleRestoreLayout}
              onToggleOverlay={() => setOverlayEnabled(!overlayEnabled)}
              onTogglePerformance={() => setPerformanceEnabled(!performanceEnabled)}
              onToggleLogging={() => setLoggingEnabled(!loggingEnabled)}
              onToggleTheme={() => setIsDarkMode(!isDarkMode)}
              onOpenSettings={() => setSettingsDialogOpen(true)}
              isDarkMode={isDarkMode}
              overlayEnabled={overlayEnabled}
              performanceEnabled={performanceEnabled}
              loggingEnabled={loggingEnabled}
            />

      <ResizablePanelGroup direction="horizontal" className="flex-1">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
          <CameraSidebar
            cameras={cameras}
            selectedCameraId={selectedCameraId}
            onSelectCamera={setSelectedCameraId}
            onAddCamera={handleAddCamera}
            onDeleteCamera={handleDeleteCamera}
            onConfigureCamera={handleConfigureCamera}
          />
        </ResizablePanel>

        <ResizableHandle />

        <ResizablePanel defaultSize={55} minSize={30}>
          <CameraGrid
            cameras={cameras}
            gridSize={gridSize}
            onGridSizeChange={setGridSize}
            onToggleCamera={handleToggleCamera}
            onPopOut={handlePopOut}
            onConfigure={handleConfigureCamera}
            onToggleOverlay={handleToggleOverlay}
            onToggleLogging={handleToggleLogging}
            onCapturePhoto={handleCapturePhoto}
            onRecordVideo={handleRecordVideo}
            showOverlay={overlayEnabled}
            showPerformance={performanceEnabled}
          />
        </ResizablePanel>

        <ResizableHandle />

        <ResizablePanel defaultSize={25} minSize={20} maxSize={35}>
          <LogPanel
            logs={logs}
            cameras={cameras}
            onExport={handleExportLogs}
          />
        </ResizablePanel>
      </ResizablePanelGroup>

      <StatusBar stats={stats} />

      <CameraDialog
        open={cameraDialogOpen}
        onClose={() => setCameraDialogOpen(false)}
        onSave={handleSaveCamera}
        camera={editingCamera}
      />

      <SettingsDialog
        open={settingsDialogOpen}
        onClose={() => setSettingsDialogOpen(false)}
      />

      <Toaster />
    </div>
  );
}
