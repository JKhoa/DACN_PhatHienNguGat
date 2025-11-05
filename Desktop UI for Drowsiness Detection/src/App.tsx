import React, { useState, useEffect, useRef } from 'react';
import { Camera, LogEvent, SystemStats } from './types';
import { mockCameras, generateMockLog, defaultCameraConfig } from './lib/mockData';
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
import { initAutoFix } from './utils/autoFixErrors';

export default function App() {
  // Initialize auto-fix on mount
  useEffect(() => {
    initAutoFix();
    console.log('[App] Auto-fix system initialized');
  }, []);
  // Start with no cameras; sync from backend or add via dialog to avoid mock confusion
  const [cameras, setCameras] = useState<Camera[]>([]);
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
    totalStudents: 0,
    sleepyStudents: 0,
    cpuUsage: 0,
    gpuUsage: 45,
    reconnectCount: 0,
  });

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
          const backendCameras = (Array.isArray(result.cameras) ? result.cameras : Object.values(result.cameras)).map((cam: any) => {
            const mappedStatus = cam.status === 'running' ? 'online' : (cam.status === 'stopped' ? 'offline' : cam.status);
            return {
            id: cam.id,
            name: cam.name,
              type: cam.type || cam.config?.type || 'ip',
              status: mappedStatus,
              fps: cam.fps || 0,
              isRunning: mappedStatus === 'online',
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
              rtspUrl: cam.config?.rtspUrl || cam.url,
            config: cam.config || { ...defaultCameraConfig },
              errorMessage: mappedStatus === 'offline' ? 'Camera offline' : undefined,
            } as Camera;
          });
          
          // Merge: keep local webcams (handled by browser) + backend IP cams
          // Also preserve local IP cameras that are not in backend (they might be in process of being added)
          setCameras((prev: Camera[]) => {
            const localWebcams = prev.filter((c: Camera) => c.type === 'webcam');
            const localIpCams = prev.filter((c: Camera) => c.type === 'ip');
            const backendCamIds = new Set(backendCameras.map((c: any) => c.id));
            
            // Keep local IP cams that are not in backend yet (will be synced later)
            const missingLocalCams = localIpCams.filter((c: Camera) => !backendCamIds.has(c.id));
            
            // Update existing cameras with backend status, add new ones
            const updatedCameras = prev.map((c: Camera) => {
              if (c.type === 'webcam') return c;
              const backendCam = backendCameras.find((bc: any) => bc.id === c.id);
              if (backendCam) {
                return { ...c, ...backendCam, rtspUrl: c.rtspUrl || backendCam.rtspUrl };
              }
              return c;
            });
            
            // Add new backend cameras that don't exist locally
            const newBackendCams = backendCameras.filter((bc: any) => 
              !prev.find((c: Camera) => c.id === bc.id)
            );
            
            // Combine: webcams + missing local IP cams + updated IP cams + new backend cams
            const updatedIpCams = updatedCameras.filter((c: Camera) => c.type === 'ip');
            return [...localWebcams, ...missingLocalCams, ...updatedIpCams, ...newBackendCams];
          });
        }
      } catch (error) {
        console.error('Error syncing cameras:', error);
      }
    };

    // Initial sync
    syncCameras();
    
    // Sync every 5 seconds (reduced frequency to avoid conflicts)
    const interval = setInterval(syncCameras, 5000);
    
    return () => clearInterval(interval);
  }, []);

  // Generate mock logs
  useEffect(() => {
    const interval = setInterval(() => {
  const runningCameras = cameras.filter((c: Camera) => c.isRunning && c.status === 'online');
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
  setLogs((prev: LogEvent[]) => [newLog, ...prev].slice(0, 100));

        if (newLog.type === 'sleepy' && student) {
          toast.warning(`${randomCamera.name}: Phát hiện buồn ngủ!`, {
            description: newLog.message,
          });
        }
      }
    }, 8000);

    return () => clearInterval(interval);
  }, [cameras, loggingEnabled]);

  // Track cameras being auto-fixed to avoid repeated attempts
  const autoFixInProgress = useRef<Set<string>>(new Set());
  const lastAutoFixAttempt = useRef<Record<string, number>>({});

  // Recompute system stats whenever cameras state changes (driven by WS updates)
  useEffect(() => {
    const running = cameras.filter((c: Camera) => c.isRunning && c.status === 'online');
    const totalFPS = running.reduce((sum: number, c: Camera) => sum + (c.fps || 0), 0);
    const totalStudents = running.reduce((sum: number, c: Camera) => sum + (c.students?.length || 0), 0);
    const sleepyStudents = running.reduce((sum: number, c: Camera) => sum + (c.sleepyStudents || 0), 0);

    setStats((prevStats) => ({
      ...prevStats,
      totalFPS,
      runningCameras: running.length,
      totalCameras: cameras.length,
      totalStudents,
      sleepyStudents,
    }));
  }, [cameras]);

  const handleStartAll = async () => {
    // Start IP cameras in backend; webcams are handled in browser
    try {
      for (const cam of cameras) {
        if (cam.type !== 'webcam') {
          try {
            await fetch(`http://127.0.0.1:5000/api/camera/${cam.id}/start`, { 
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ enable_detection: true })
            });
          } catch {}
        }
      }
    } catch {}

  setCameras((prev: Camera[]) => prev.map((c: Camera) => {
      // Generate students when starting
      const students = c.type === 'ip' && c.ip 
        ? Array.from({ length: Math.floor(Math.random() * 10 + (c.totalStudents - 5)) }, (_, i) => {
            const px = 50 + (i % 8) * 75 + Math.random() * 20;
            const py = 70 + Math.floor(i / 8) * 60 + Math.random() * 20;
            const bw = 60 + Math.random() * 30; // bbox width
            const bh = 80 + Math.random() * 40; // bbox height
            const x1 = Math.max(0, Math.floor(px - bw / 2));
            const y1 = Math.max(0, Math.floor(py - bh / 2));
            const x2 = Math.floor(x1 + bw);
            const y2 = Math.floor(y1 + bh);
            const headH = Math.floor(bh * 0.3);
            return {
              id: `student-${i + 1}`,
              position: { x: px, y: py },
              state: 'normal' as const,
              confidence: 0.7 + Math.random() * 0.25,
              sleepDuration: 0,
              lastUpdate: new Date(),
              bbox: [x1, y1, x2, y2] as [number, number, number, number],
              headBbox: [x1, y1, x2, y1 + headH] as [number, number, number, number],
            };
          })
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
  setCameras((prev: Camera[]) => prev.map((c: Camera) => ({ ...c, isRunning: false, students: [], sleepyStudents: 0 })));
      toast.info('Đã dừng tất cả camera');
    } catch (error) {
      console.error('Error stopping all cameras:', error);
      toast.error('Lỗi dừng camera');
    }
  };

  const handleToggleCamera = async (cameraId: string) => {
  const camera = cameras.find((c: Camera) => c.id === cameraId);
    if (!camera) return;

    try {
      if (!camera.isRunning) {
        if (camera.type === 'webcam') {
          // Do NOT start webcam in backend to prevent "device in use" conflicts. Use getUserMedia only.
          setCameras((prev: Camera[]) => prev.map((c: Camera) => 
            c.id === cameraId 
              ? { ...c, isRunning: true, status: 'online' as const }
              : c
          ));
          toast.success(`Đã khởi động ${camera.name} (webcam trong trình duyệt)`);
          return;
        }
        // Start camera via Python backend
        const response = await fetch(`http://127.0.0.1:5000/api/camera/${cameraId}/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enable_detection: true })
        });
        
        const result = await response.json();
        
        if (result.success) {
          setCameras((prev: Camera[]) => prev.map((c: Camera) => 
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
          setCameras((prev: Camera[]) => prev.map((c: Camera) => 
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
          setCameras((prev: Camera[]) => prev.map((c: Camera) => 
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
        const camera = cameras.find((c: Camera) => c.id === selectedCameraId);
        if (!camera) {
          toast.error('Camera không tồn tại');
          return;
        }

        // Only call backend API for IP cameras, webcams are handled locally
        if (camera.type !== 'webcam') {
        const response = await fetch(`http://127.0.0.1:5000/api/camera/${selectedCameraId}/remove`, {
          method: 'DELETE'
        });
          
          // Check if response is JSON
          const contentType = response.headers.get('content-type');
          if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            console.error('Non-JSON response:', text);
            throw new Error('Backend trả về dữ liệu không hợp lệ');
          }
          
        const result = await response.json();
          
          if (!result.success) {
            toast.error(`Lỗi xóa camera: ${result.error || 'Unknown error'}`);
            return;
          }
        }
        
        // Remove from local state regardless of type
        setCameras((prev: Camera[]) => prev.filter((c: Camera) => c.id !== selectedCameraId));
          setSelectedCameraId(undefined);
          toast.success('Đã xóa camera');
      } catch (error: any) {
        console.error('Error deleting camera:', error);
        if (error.message) {
          toast.error(`Lỗi xóa camera: ${error.message}`);
        } else {
          toast.error('Lỗi kết nối đến backend');
        }
      }
    } else {
      toast.error('Vui lòng chọn camera để xóa');
    }
  };

  const handleClearAllCameras = async () => {
    try {
      // Remove all cameras from Python backend (only IP cameras)
      const errors: string[] = [];
      for (const camera of cameras) {
        if (camera.type !== 'webcam') {
          try {
          const response = await fetch(`http://127.0.0.1:5000/api/camera/${camera.id}/remove`, {
            method: 'DELETE'
          });
            
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
          const result = await response.json();
              if (!result.success) {
                errors.push(`${camera.name}: ${result.error || 'Unknown error'}`);
              }
            } else {
              errors.push(`${camera.name}: Backend trả về dữ liệu không hợp lệ`);
            }
          } catch (error: any) {
            console.error(`Error removing camera ${camera.id}:`, error);
            errors.push(`${camera.name}: ${error.message || 'Connection error'}`);
          }
        }
      }
      
      // Clear local state regardless of backend errors
      setCameras([]);
      setSelectedCameraId(undefined);
      
      if (errors.length > 0) {
        toast.warning(`Đã xóa camera nhưng có ${errors.length} lỗi: ${errors.slice(0, 3).join(', ')}`);
      } else {
      toast.success('Đã xóa tất cả camera');
      }
    } catch (error: any) {
      console.error('Error clearing all cameras:', error);
      toast.error(`Lỗi xóa camera: ${error.message || 'Unknown error'}`);
    }
  };

  const handleAddCamera = () => {
    setEditingCamera(undefined);
    setCameraDialogOpen(true);
  };

  const handleConfigureCamera = (cameraId: string) => {
    const camera = cameras.find((c: Camera) => c.id === cameraId);
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
        
        // If add succeeded (even if camera already exists), try to start it automatically
        if (result.success && cameraData.id) {
          // Wait a bit for backend to process
          await new Promise(resolve => setTimeout(resolve, 300));
          
          // Start camera with detection enabled
          try {
            const startResponse = await fetch(`http://127.0.0.1:5000/api/camera/${cameraData.id}/start`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ enable_detection: true }),
            });
            
            if (startResponse.ok) {
              console.log(`[App] Auto-started camera ${cameraData.id} with detection`);
              // Set status to online and running
              cameraData.status = 'online';
              cameraData.isRunning = true;
            } else {
              console.warn(`[App] Failed to auto-start camera ${cameraData.id}`);
            }
          } catch (startError) {
            console.warn(`[App] Error starting camera ${cameraData.id}:`, startError);
          }
        }
      }
      
      if (result.success) {
        // Update local state
        if (editingCamera) {
          setCameras((prev: Camera[]) => prev.map((c: Camera) => 
            c.id === editingCamera.id ? { ...c, ...cameraData } : c
          ));
          toast.success('Đã cập nhật camera');
        } else {
          setCameras((prev: Camera[]) => [...prev, cameraData as Camera]);
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
    const camera = cameras.find((c: Camera) => c.id === cameraId);
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
  cameras: cameras.map((c: Camera) => c.id),
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
  cameras: cameras.map((c: Camera) => ({
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
    const camera = cameras.find((c: Camera) => c.id === cameraId);
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
    const camera = cameras.find((c: Camera) => c.id === cameraId);
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
      ...logs.map((log: LogEvent) => 
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

  const isAllRunning = cameras.every((c: Camera) => c.isRunning);

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
            onUpdateStudents={(cameraId: string, students: any[], fps: number) => {
              setCameras((prev: Camera[]) => prev.map((c: Camera) => 
                c.id === cameraId 
                  ? { 
                      ...c, 
                      students,
                      sleepyStudents: students.filter((s: any) => s.state === 'sleepy' || s.state === 'head_down').length,
                      totalStudents: students.length,
                      fps,
                    }
                  : c
              ));
            }}
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
