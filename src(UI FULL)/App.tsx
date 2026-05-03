import { useState, useEffect } from 'react';
import { Camera, LogEvent, SystemStats } from './types';
import { mockCameras, generateMockLog } from './lib/mockData';
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
import { toast } from 'sonner@2.0.3';

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

  // Toggle dark mode
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Update stats
  useEffect(() => {
    const interval = setInterval(() => {
      const runningCameras = cameras.filter(c => c.isRunning && c.status === 'online').length;
      const totalFPS = cameras
        .filter(c => c.isRunning && c.status === 'online')
        .reduce((sum, c) => sum + c.fps, 0);
      const reconnectCount = cameras.filter(c => c.status === 'reconnecting').length;
      const totalStudents = cameras.reduce((sum, c) => sum + c.students.length, 0);
      const sleepyStudents = cameras.reduce((sum, c) => sum + c.sleepyStudents, 0);

      setStats({
        totalFPS,
        runningCameras,
        totalCameras: cameras.length,
        totalStudents,
        sleepyStudents,
        cpuUsage: Math.floor(Math.random() * 20 + 40),
        gpuUsage: Math.floor(Math.random() * 15 + 35),
        reconnectCount,
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [cameras]);

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

  const handleStartAll = () => {
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
        status: 'online',
        students,
        sleepyStudents: 0,
      };
    }));
    toast.success('Đã khởi động tất cả camera');
  };

  const handleStopAll = () => {
    setCameras(prev => prev.map(c => ({ ...c, isRunning: false })));
    toast.info('Đã dừng tất cả camera');
  };

  const handleToggleCamera = (cameraId: string) => {
    setCameras(prev => prev.map(c => {
      if (c.id !== cameraId) return c;
      
      if (!c.isRunning) {
        // Starting camera - generate students
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
          status: 'online',
          students,
          sleepyStudents: 0,
        };
      } else {
        // Stopping camera
        return { 
          ...c, 
          isRunning: false,
          students: [],
          sleepyStudents: 0,
        };
      }
    }));
  };

  const handleAddCamera = () => {
    setEditingCamera(undefined);
    setCameraDialogOpen(true);
  };

  const handleDeleteCamera = () => {
    if (selectedCameraId) {
      setCameras(prev => prev.filter(c => c.id !== selectedCameraId));
      setSelectedCameraId(undefined);
      toast.success('Đã xóa camera');
    } else {
      toast.error('Vui lòng chọn camera để xóa');
    }
  };

  const handleConfigureCamera = (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    setEditingCamera(camera);
    setCameraDialogOpen(true);
  };

  const handleSaveCamera = (cameraData: Partial<Camera>) => {
    if (editingCamera) {
      setCameras(prev => prev.map(c => 
        c.id === editingCamera.id ? { ...c, ...cameraData } : c
      ));
      toast.success('Đã cập nhật camera');
    } else {
      setCameras(prev => [...prev, cameraData as Camera]);
      toast.success('Đã thêm camera mới');
    }
  };

  const handlePopOut = (cameraId: string) => {
    const camera = cameras.find(c => c.id === cameraId);
    toast.info(`Pop out: ${camera?.name}`, {
      description: 'Tính năng đang được phát triển',
    });
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
