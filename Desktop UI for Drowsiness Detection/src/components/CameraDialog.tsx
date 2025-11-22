import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Switch } from './ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Camera, CameraConfig } from '../types';
import { defaultCameraConfig, generateRTSPUrl } from '../lib/mockData';
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { getAvailableCameras } from '../lib/webcamRegistry';

interface CameraDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (camera: Partial<Camera>) => void;
  camera?: Camera;
}

export function CameraDialog({ open, onClose, onSave, camera }: CameraDialogProps) {
  const [cameraType, setCameraType] = useState<'webcam' | 'ip'>(camera?.type || 'ip');
  const [name, setName] = useState(camera?.name || '');
  const [totalStudents, setTotalStudents] = useState(camera?.totalStudents?.toString() || '30');
  const [deviceId, setDeviceId] = useState(camera?.deviceId?.toString() || '');
  const [availableCameras, setAvailableCameras] = useState<Array<{ deviceId: string; label: string }>>([]);
  const [brand, setBrand] = useState(camera?.brand || 'Hikvision');
  const [ip, setIp] = useState(camera?.ip || '');
  const [port, setPort] = useState(camera?.port?.toString() || '554');
  const [username, setUsername] = useState(camera?.username || 'admin');
  const [password, setPassword] = useState(camera?.password || 'admin123');
  const [streamQuality, setStreamQuality] = useState<'main' | 'sub'>(camera?.streamQuality || 'main');
  const [config, setConfig] = useState<CameraConfig>(camera?.config || { ...defaultCameraConfig });
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [rtspUrl, setRtspUrl] = useState(camera?.rtspUrl || '');

  // Load available cameras when dialog opens
  useEffect(() => {
    if (open && cameraType === 'webcam') {
      getAvailableCameras().then(cameras => {
        setAvailableCameras(cameras);
        // Auto-select first camera if no deviceId set
        if (!deviceId && cameras.length > 0) {
          setDeviceId(cameras[0].deviceId);
        }
      });
    }
  }, [open, cameraType]);

  const handleTestConnection = () => {
    setTestStatus('testing');
    
    // Generate RTSP URL
    const url = cameraType === 'ip'
      ? generateRTSPUrl(brand, ip, parseInt(port), username, password, streamQuality)
      : `video${deviceId}`;
    
    setRtspUrl(url);
    
    // Simulate connection test
    setTimeout(() => {
      // Mock validation
      if (cameraType === 'ip' && (!ip || !username || !password)) {
        setTestStatus('error');
      } else {
        setTestStatus('success');
      }
    }, 2000);
  };

  const handleSave = () => {
    const cameraData: Partial<Camera> = {
      id: camera?.id || `cam-${Date.now()}`,
      name,
      type: cameraType,
      config,
      status: 'offline',
      fps: 0,
      students: [],
      totalStudents: parseInt(totalStudents) || 30,
      sleepyStudents: 0,
      isRunning: false,
    };

    if (cameraType === 'webcam') {
      // Store deviceId as string (browser API returns string)
      cameraData.deviceId = deviceId;
    } else {
      cameraData.brand = brand;
      cameraData.ip = ip;
      cameraData.port = parseInt(port);
      cameraData.username = username;
      cameraData.password = password;
      cameraData.streamQuality = streamQuality;
      cameraData.rtspUrl = generateRTSPUrl(brand, ip, parseInt(port), username, password, streamQuality);
    }

    onSave(cameraData);
    onClose();
  };

  const updateDecorator = (key: keyof CameraConfig['decorators'], value: boolean) => {
    setConfig(prev => ({
      ...prev,
      decorators: {
        ...prev.decorators,
        [key]: value,
      },
    }));
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{camera ? 'Cấu hình Camera' : 'Thêm Camera Mới'}</DialogTitle>
          <DialogDescription>
            Cấu hình camera và decorators để phát hiện ngủ gật
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="basic">Cơ bản</TabsTrigger>
            <TabsTrigger value="decorators">Decorators</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-4">
            <div className="space-y-2">
              <Label>Tên Camera / Lớp học</Label>
              <Input
                placeholder="Ví dụ: Camera Lớp 12A1 - Phòng 101"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Số lượng học sinh</Label>
              <Input
                type="number"
                placeholder="30"
                min="1"
                max="50"
                value={totalStudents}
                onChange={(e) => setTotalStudents(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Số lượng học sinh dự kiến trong lớp (20-40 học sinh)
              </p>
            </div>

            <div className="space-y-2">
              <Label>Loại Camera</Label>
              <Select value={cameraType} onValueChange={(v: 'webcam' | 'ip') => setCameraType(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ip">IP Camera (Khuyến nghị)</SelectItem>
                  <SelectItem value="webcam">Webcam</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {cameraType === 'webcam' ? (
              <div className="space-y-2">
                <Label>Chọn Camera</Label>
                {availableCameras.length > 0 ? (
                  <Select value={deviceId} onValueChange={setDeviceId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Chọn camera..." />
                    </SelectTrigger>
                    <SelectContent>
                      {availableCameras.map((cam) => (
                        <SelectItem key={cam.deviceId} value={cam.deviceId}>
                          {cam.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <div className="text-sm text-muted-foreground">
                    Đang tải danh sách camera...
                  </div>
                )}
                <p className="text-xs text-muted-foreground">
                  Tìm thấy {availableCameras.length} camera. Chọn camera bạn muốn sử dụng.
                </p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Thương hiệu</Label>
                    <Select value={brand} onValueChange={setBrand}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Hikvision">Hikvision</SelectItem>
                        <SelectItem value="Dahua">Dahua</SelectItem>
                        <SelectItem value="Ezviz">Ezviz</SelectItem>
                        <SelectItem value="Kbvision">Kbvision</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Chất lượng Stream</Label>
                    <Select value={streamQuality} onValueChange={(v: 'main' | 'sub') => setStreamQuality(v)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="main">Main Stream</SelectItem>
                        <SelectItem value="sub">Sub Stream</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Địa chỉ IP</Label>
                    <Input
                      placeholder="192.168.1.100"
                      value={ip}
                      onChange={(e) => setIp(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Port</Label>
                    <Input
                      placeholder="554"
                      value={port}
                      onChange={(e) => setPort(e.target.value)}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Username</Label>
                    <Input
                      placeholder="admin"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Password</Label>
                    <Input
                      type="password"
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                    />
                  </div>
                </div>

                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={handleTestConnection}
                  disabled={testStatus === 'testing'}
                >
                  {testStatus === 'testing' && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  {testStatus === 'success' && <CheckCircle2 className="h-4 w-4 mr-2 text-green-500" />}
                  {testStatus === 'error' && <XCircle className="h-4 w-4 mr-2 text-red-500" />}
                  Test Connection
                </Button>
                
                {testStatus === 'success' && (
                  <Alert className="mt-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <AlertDescription>
                      Kết nối thành công!
                    </AlertDescription>
                  </Alert>
                )}
                
                {testStatus === 'error' && (
                  <Alert variant="destructive" className="mt-2">
                    <XCircle className="h-4 w-4" />
                    <AlertDescription>
                      Không thể kết nối. Vui lòng kiểm tra lại thông tin.
                    </AlertDescription>
                  </Alert>
                )}
                
                {rtspUrl && testStatus === 'success' && (
                  <div className="mt-2 p-2 bg-muted rounded text-xs font-mono break-all">
                    <Label className="text-xs">RTSP URL:</Label>
                    <div className="mt-1">{rtspUrl}</div>
                  </div>
                )}
              </>
            )}
          </TabsContent>

          <TabsContent value="decorators" className="space-y-4">
            <div className="space-y-4">
              <div className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4>Reconnect Decorator</h4>
                      <Badge variant="outline">Layer 1</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">Tự động kết nối lại khi mất kết nối</p>
                  </div>
                  <Switch
                    checked={config.decorators.reconnect}
                    onCheckedChange={(v: boolean) => updateDecorator('reconnect', v)}
                  />
                </div>
              </div>

              <div className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4>Frame Queue Decorator</h4>
                      <Badge variant="outline">Layer 2</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">Quản lý hàng đợi khung hình</p>
                  </div>
                  <Switch
                    checked={config.decorators.frameQueue}
                    onCheckedChange={(v: boolean) => updateDecorator('frameQueue', v)}
                  />
                </div>
                {config.decorators.frameQueue && (
                  <div className="space-y-2">
                    <Label>Max Queue Size</Label>
                    <Input
                      type="number"
                      value={config.maxQueueSize}
                      onChange={(e) => setConfig(prev => ({ ...prev, maxQueueSize: parseInt(e.target.value) }))}
                    />
                  </div>
                )}
              </div>

              <div className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4>Performance Decorator</h4>
                      <Badge variant="outline">Layer 3</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">Đo FPS và hiệu năng</p>
                  </div>
                  <Switch
                    checked={config.decorators.performance}
                    onCheckedChange={(v: boolean) => updateDecorator('performance', v)}
                  />
                </div>
              </div>

              <div className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4>Detection Decorator</h4>
                      <Badge variant="outline">Layer 4</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">Phát hiện pose và trạng thái</p>
                  </div>
                  <Switch
                    checked={config.decorators.detection}
                    onCheckedChange={(v: boolean) => updateDecorator('detection', v)}
                  />
                </div>
                {config.decorators.detection && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Model</Label>
                      <Select value={config.model} onValueChange={(v: string) => setConfig(prev => ({ ...prev, model: v }))}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yolo11n-pose.pt">yolo11n-pose.pt</SelectItem>
                          <SelectItem value="yolo11s-pose.pt">yolo11s-pose.pt</SelectItem>
                          <SelectItem value="yolo11m-pose.pt">yolo11m-pose.pt</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Strategy</Label>
                      <Select value={config.strategy} onValueChange={(v: any) => setConfig(prev => ({ ...prev, strategy: v }))}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="YOLO">YOLO</SelectItem>
                          <SelectItem value="Mediapipe">Mediapipe</SelectItem>
                          <SelectItem value="EAR">EAR (Eye Aspect Ratio)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label>Confidence Threshold: {config.confidence}</Label>
                      <Input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={config.confidence}
                        onChange={(e) => setConfig(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4>Overlay Decorator</h4>
                      <Badge variant="outline">Layer 5</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">Hiển thị keypoints và bbox</p>
                  </div>
                  <Switch
                    checked={config.decorators.overlay}
                    onCheckedChange={(v: boolean) => updateDecorator('overlay', v)}
                  />
                </div>
              </div>

              <div className="p-4 border rounded-lg space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4>Logging Decorator</h4>
                      <Badge variant="outline">Layer 6</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">Ghi log sự kiện</p>
                  </div>
                  <Switch
                    checked={config.decorators.logging}
                    onCheckedChange={(v: boolean) => updateDecorator('logging', v)}
                  />
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Hủy
          </Button>
          <Button onClick={handleSave}>
            {camera ? 'Cập nhật' : 'Thêm Camera'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
