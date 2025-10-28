import React, { useState, useEffect } from 'react';
import { RealCameraCard } from './RealCameraCard';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { AlertTriangle, Users, Camera, Activity } from 'lucide-react';

interface ClassroomStats {
  totalCameras: number;
  activeCameras: number;
  totalStudents: number;
  drowsyStudents: number;
  alertLevel: 'low' | 'medium' | 'high';
}

interface CameraInfo {
  id: string;
  name: string;
  room: string;
  isActive: boolean;
}

export const ClassroomDashboard: React.FC = () => {
  const [cameras, setCameras] = useState<CameraInfo[]>([
    { id: 'camera-1', name: 'Camera Phòng học 1A', room: 'Phòng 1A', isActive: false },
    { id: 'camera-2', name: 'Camera Phòng học 1B', room: 'Phòng 1B', isActive: false },
    { id: 'camera-3', name: 'Camera Phòng học 2A', room: 'Phòng 2A', isActive: false },
    { id: 'camera-4', name: 'Camera Phòng học 2B', room: 'Phòng 2B', isActive: false },
    { id: 'camera-5', name: 'Camera Phòng học 3A', room: 'Phòng 3A', isActive: false },
    { id: 'camera-6', name: 'Camera Phòng học 3B', room: 'Phòng 3B', isActive: false },
  ]);

  const [stats, setStats] = useState<ClassroomStats>({
    totalCameras: 6,
    activeCameras: 0,
    totalStudents: 0,
    drowsyStudents: 0,
    alertLevel: 'low'
  });

  const [isMonitoring, setIsMonitoring] = useState(false);

  // Update stats when cameras change
  useEffect(() => {
    const activeCameras = cameras.filter(c => c.isActive).length;
    setStats(prev => ({
      ...prev,
      activeCameras
    }));
  }, [cameras]);

  const handleCameraToggle = (cameraId: string) => {
    setCameras(prev => prev.map(camera => 
      camera.id === cameraId 
        ? { ...camera, isActive: !camera.isActive }
        : camera
    ));
  };

  const handleStartAllMonitoring = () => {
    setCameras(prev => prev.map(camera => ({ ...camera, isActive: true })));
    setIsMonitoring(true);
  };

  const handleStopAllMonitoring = () => {
    setCameras(prev => prev.map(camera => ({ ...camera, isActive: false })));
    setIsMonitoring(false);
  };

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      default: return 'secondary';
    }
  };

  const getAlertText = (level: string) => {
    switch (level) {
      case 'high': return 'Cảnh báo cao';
      case 'medium': return 'Cảnh báo trung bình';
      default: return 'Bình thường';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Hệ thống Giám sát Phòng học
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Phát hiện và theo dõi tình trạng ngủ gật của học sinh
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button
            onClick={handleStartAllMonitoring}
            disabled={isMonitoring}
            className="bg-green-600 hover:bg-green-700"
          >
            <Activity className="h-4 w-4 mr-2" />
            Bật tất cả
          </Button>
          <Button
            onClick={handleStopAllMonitoring}
            variant="destructive"
            disabled={!isMonitoring}
          >
            <Camera className="h-4 w-4 mr-2" />
            Tắt tất cả
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tổng Camera</CardTitle>
            <Camera className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalCameras}</div>
            <p className="text-xs text-muted-foreground">
              {stats.activeCameras} đang hoạt động
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tổng Học sinh</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalStudents}</div>
            <p className="text-xs text-muted-foreground">
              Được phát hiện
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Học sinh Buồn ngủ</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{stats.drowsyStudents}</div>
            <p className="text-xs text-muted-foreground">
              Cần chú ý
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Mức cảnh báo</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <Badge variant={getAlertColor(stats.alertLevel)}>
              {getAlertText(stats.alertLevel)}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Camera Grid */}
      <div>
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Danh sách Camera Phòng học
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {cameras.map((camera) => (
            <RealCameraCard
              key={camera.id}
              cameraId={camera.id}
              cameraName={camera.name}
              isActive={camera.isActive}
              onToggle={handleCameraToggle}
            />
          ))}
        </div>
      </div>

      {/* Instructions */}
      <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <CardHeader>
          <CardTitle className="text-blue-900 dark:text-blue-100 flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Hướng dẫn sử dụng
          </CardTitle>
        </CardHeader>
        <CardContent className="text-blue-800 dark:text-blue-200">
          <ul className="space-y-2 text-sm">
            <li>• Nhấn "Bật" trên camera để bắt đầu giám sát phòng học</li>
            <li>• Hệ thống sẽ tự động phát hiện và theo dõi học sinh</li>
            <li>• Các học sinh buồn ngủ sẽ được đánh dấu màu đỏ</li>
            <li>• Sử dụng "Bật tất cả" để giám sát toàn bộ phòng học</li>
            <li>• Dữ liệu sẽ được lưu trữ và báo cáo tự động</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

