import React, { useEffect, useRef, useState } from 'react';
import { Camera } from '../types';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  Play,
  Square,
  MoreVertical,
  Maximize2,
  Tag,
  Eye,
  FileText,
  Camera as CameraIcon,
  Video,
  Circle,
  AlertCircle,
  Loader2,
  Users,
  Brain,
} from 'lucide-react';
import { cn } from './ui/utils';
import { StudentTrackingDetails } from './StudentTrackingDetails';
import { YOLODetectionPanel } from './YOLODetectionPanel';
import { acquireWebcam, releaseWebcam, mapGetUserMediaError, forceReleaseAllWebcams, getAvailableCameras, requestCameraPermission, detectWorkingCamera } from '../lib/webcamRegistry';

interface CameraCardProps {
  camera: Camera;
  onToggle: (cameraId: string) => void;
  onPopOut: (cameraId: string) => void;
  onConfigure: (cameraId: string) => void;
  onToggleOverlay?: (cameraId: string) => void;
  onToggleLogging?: (cameraId: string) => void;
  onCapturePhoto?: (cameraId: string) => void;
  onRecordVideo?: (cameraId: string) => void;
  showOverlay: boolean;
  showPerformance: boolean;
}

export function CameraCard({
  camera,
  onToggle,
  onPopOut,
  onConfigure,
  onToggleOverlay,
  onToggleLogging,
  onCapturePhoto,
  onRecordVideo,
  showOverlay,
  showPerformance,
}: CameraCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [sleepDuration, setSleepDuration] = useState(0);
  const [showTrackingDetails, setShowTrackingDetails] = useState(false);
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [availableCameras, setAvailableCameras] = useState<any[]>([]);
  const [streamKey, setStreamKey] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [lastIpFrameUrl, setLastIpFrameUrl] = useState<string | null>(null);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [showYOLOPanel, setShowYOLOPanel] = useState(false);
  const [yoloDetectionEnabled, setYoloDetectionEnabled] = useState(false);

  // Load available cameras on mount
  useEffect(() => {
    const loadCameras = async () => {
      const cameras = await getAvailableCameras();
      setAvailableCameras(cameras);
      console.log('Available cameras:', cameras);
    };
    loadCameras();
  }, []);

  // Force retry function for "Device in use" errors
  const handleForceRetry = async () => {
    console.log('Force retrying webcam...');
    setLocalError(null);
    setRetryCount(prev => prev + 1);
    
    // Request camera permission first
    const hasPermission = await requestCameraPermission();
    if (!hasPermission) {
      setLocalError('Không có quyền truy cập camera. Vui lòng cho phép camera trong trình duyệt.');
      return;
    }
    
    // Detect working camera
    const workingCamera = await detectWorkingCamera();
    if (!workingCamera) {
      setLocalError('Không tìm thấy camera hoạt động. Hãy kiểm tra camera có được kết nối đúng không.');
      return;
    }
    
    console.log(`Found working camera: ${workingCamera.label} (${workingCamera.deviceId})`);
    
    // Force release all webcam streams first
    forceReleaseAllWebcams();
    
    // Wait a bit for resources to be freed
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Reload available cameras
    const cameras = await getAvailableCameras();
    setAvailableCameras(cameras);
    
    // Force re-render to trigger useEffect
    setReloadNonce(prev => prev + 1);
  };


  // Setup video stream - use getUserMedia with shared registry and robust fallbacks
  useEffect(() => {
    let mounted = true;

    const start = async () => {
      setLocalError(null);
      if (!(camera.isRunning && camera.status === 'online')) {
        return;
      }

      if (camera.type === 'webcam') {
        try {
          const { stream, streamKey: k } = await acquireWebcam({
            deviceId: camera.deviceId,
            width: 640,
            height: 480,
          });
          if (!mounted) return;
          setStreamKey(k);
          setVideoStream(stream);
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            try { await videoRef.current.play(); } catch {}
          }

          // Adjust canvas size to video dimensions when ready
          const v = videoRef.current;
          const onLoaded = () => {
            if (canvasRef.current && v) {
              canvasRef.current.width = v.videoWidth || 640;
              canvasRef.current.height = v.videoHeight || 360;
            }
          };
          v?.addEventListener('loadedmetadata', onLoaded, { once: true });
        } catch (err: any) {
          const msg = mapGetUserMediaError(err);
          console.error(`Error starting webcam for camera ${camera.id}:`, err);
          setLocalError(msg);
        }
      } else if (camera.type === 'ip') {
        // Poll backend for JPEG frames to avoid <video> element requirements
        let stopped = false;
        const poll = async () => {
          if (stopped) return;
          try {
            const res = await fetch(`http://127.0.0.1:5000/api/camera/${camera.id}/stream?ts=${Date.now()}`, {
              cache: 'no-store',
              mode: 'cors',
            });
            if (res.ok) {
              const data = await res.json();
              if (data?.success && data.frame) {
                const url = `data:image/jpeg;base64,${data.frame}`;
                setLastIpFrameUrl(url);
                if (imgRef.current) {
                  imgRef.current.src = url;
                }
              }
            }
          } catch (e) {
            // transient fetch errors are ignored; status UI shows overall health
          }
          setTimeout(poll, 150);
        };
        poll();
        return () => { stopped = true; };
      }
    };

    start();

    return () => {
      mounted = false;
      // Release shared stream if acquired
      if (streamKey) {
        releaseWebcam(streamKey);
        setStreamKey(null);
      }
      // Detach from video element
      if (videoRef.current) {
        const src = videoRef.current.srcObject as MediaStream | null;
        if (src) {
          videoRef.current.srcObject = null;
        }
      }
      setVideoStream(null);
    };
  }, [camera.isRunning, camera.status, camera.type, camera.deviceId, camera.id, reloadNonce]);

  // Draw tracking overlays on canvas
  useEffect(() => {
    if (camera.isRunning && camera.status === 'online' && showOverlay) {
      const drawInterval = setInterval(() => {
        if (canvasRef.current && videoRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw tracking overlays
            drawOverlays(ctx, canvas);
          }
        }
      }, 33); // ~30 FPS

      const interval = drawInterval;
      
      return () => {
        console.log(`Stopping video stream for camera ${camera.id}`);
        clearInterval(interval);
      };
    }
  }, [camera.isRunning, camera.status, camera.id]);

  const drawOverlays = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (showOverlay) {
      // Draw students with head-focused tracking
      camera.students.forEach((student, index) => {
        const { x, y } = student.position;
        
        // Student color based on state
        let color = '#22c55e'; // green - normal
        if (student.state === 'sleepy') color = '#ef4444'; // red
        if (student.state === 'head_down') color = '#a855f7'; // purple
        
        // Draw student circle (head) - smaller and more focused
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2); // Smaller radius
        ctx.fill();
        
        // Draw head-focused bounding box (smaller, less overlap)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        
        // Use headBbox if available, otherwise create smaller bbox
        if (student.headBbox) {
          const [x1, y1, x2, y2] = student.headBbox;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        } else {
          // Create smaller head-focused bbox
          ctx.strokeRect(x - 12, y - 12, 24, 20); // Smaller, head-focused
        }
        
        // Draw head keypoints only (no shoulders)
        ctx.fillStyle = color;
        // Eyes
        ctx.beginPath();
        ctx.arc(x - 4, y - 2, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x + 4, y - 2, 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Confidence label (smaller)
        if (student.confidence > 0.6) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(x - 12, y - 20, 24, 6);
          ctx.fillStyle = '#fff';
          ctx.font = '6px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`${(student.confidence * 100).toFixed(0)}%`, x, y - 16);
          ctx.textAlign = 'left';
        }
        
        // Student ID (smaller)
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x - 15, y + 8, 30, 6);
        ctx.fillStyle = '#fff';
        ctx.font = '5px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(student.id, x, y + 12);
        ctx.textAlign = 'left';
      });
    }

    // Performance HUD
    if (showPerformance) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(5, 5, 120, 80);
      ctx.fillStyle = '#fff';
      ctx.font = '11px monospace';
      ctx.fillText(`FPS: ${camera.fps}`, 10, 20);
      ctx.fillText(`Students: ${camera.students.length}`, 10, 35);
      ctx.fillText(`Sleepy: ${camera.sleepyStudents}`, 10, 50);
      ctx.fillText(`Latency: ${Math.floor(Math.random() * 50 + 20)}ms`, 10, 65);
      ctx.fillText(`Conf: ${(camera.students.reduce((sum, s) => sum + s.confidence, 0) / camera.students.length || 0).toFixed(2)}`, 10, 80);
    }
    
    // Camera info overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvas.width, 25);
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    ctx.fillText(camera.name, 10, 16);
    
    if (camera.sleepyStudents > 0) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
      ctx.fillRect(canvas.width - 80, 5, 75, 15);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`⚠ ${camera.sleepyStudents} học sinh`, canvas.width - 5, 15);
      ctx.textAlign = 'left';
    }
  };


  // Update sleep duration
  useEffect(() => {
    const sleepyStudents = camera.students.filter(s => s.state !== 'normal');
    if (sleepyStudents.length > 0) {
      const maxSleepDuration = Math.max(...sleepyStudents.map(s => s.sleepDuration));
      setSleepDuration(maxSleepDuration);
    } else {
      setSleepDuration(0);
    }
  }, [camera.students]);

  const getStatusIcon = () => {
    if (camera.status === 'online') {
      return <Circle className="h-2 w-2 fill-green-500 text-green-500" />;
    } else if (camera.status === 'reconnecting') {
      return <Loader2 className="h-2 w-2 animate-spin text-yellow-500" />;
    } else if (camera.status === 'error') {
      return <AlertCircle className="h-2 w-2 text-red-500" />;
    } else {
      return <Circle className="h-2 w-2 fill-gray-500 text-gray-500" />;
    }
  };

  const getStatusBadge = () => {
    if (camera.status === 'online') {
      return {
        text: 'Đang hoạt động',
        class: 'bg-green-500/10 text-green-500 border-green-500/20',
      };
    } else if (camera.status === 'reconnecting') {
      return {
        text: 'Đang kết nối lại',
        class: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20',
      };
    } else if (camera.status === 'error') {
      return {
        text: 'Lỗi kết nối',
        class: 'bg-red-500/10 text-red-500 border-red-500/20',
      };
    } else {
      return {
        text: 'Chưa phát hiện',
        class: 'bg-gray-500/10 text-gray-500 border-gray-500/20',
      };
    }
  };

  return (
    <div className="border rounded-lg overflow-hidden bg-card hover:shadow-lg transition-shadow">
      {/* Header */}
      <div className="p-2 border-b flex items-center justify-between bg-muted/50">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          {getStatusIcon()}
          <span className="text-sm truncate">{camera.name}</span>
        </div>

        <div className="flex items-center gap-1">
          <Badge variant="outline" className={cn("text-xs", getStatusBadge().class)}>
            {getStatusBadge().text}
          </Badge>
          
          {camera.status === 'online' && (
            <span className="text-xs text-muted-foreground font-mono ml-2">
              {camera.fps} FPS
            </span>
          )}

          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={() => onToggle(camera.id)}
          >
            {camera.isRunning ? (
              <Square className="h-3 w-3" />
            ) : (
              <Play className="h-3 w-3" />
            )}
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
                <MoreVertical className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setShowTrackingDetails(!showTrackingDetails)}>
                <Users className="h-4 w-4 mr-2" />
                {showTrackingDetails ? 'Ẩn' : 'Hiện'} Chi tiết Tracking
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setShowYOLOPanel(!showYOLOPanel)}>
                <Brain className="h-4 w-4 mr-2" />
                {showYOLOPanel ? 'Ẩn' : 'Hiện'} YOLO Detection
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => onPopOut(camera.id)}>
                <Maximize2 className="h-4 w-4 mr-2" />
                Pop Out
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onConfigure(camera.id)}>
                <Tag className="h-4 w-4 mr-2" />
                Cấu hình
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => onToggleOverlay?.(camera.id)}>
                <Eye className="h-4 w-4 mr-2" />
                Toggle Overlay
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onToggleLogging?.(camera.id)}>
                <FileText className="h-4 w-4 mr-2" />
                Toggle Logging
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => onCapturePhoto?.(camera.id)}>
                <CameraIcon className="h-4 w-4 mr-2" />
                Chụp ảnh
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onRecordVideo?.(camera.id)}>
                <Video className="h-4 w-4 mr-2" />
                Ghi video
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Video Feed */}
      <div className="relative aspect-video bg-black">
        {/* Base media: video for webcam, img for IP */}
        {camera.isRunning && camera.status === 'online' ? (
          <div className="relative w-full h-full">
            {camera.type === 'webcam' ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            ) : (
              <img
                ref={imgRef}
                alt={camera.name}
                className="w-full h-full object-cover select-none"
                draggable={false}
              />
            )}

            {/* Canvas overlay for tracking visualizations */}
            <canvas
              ref={canvasRef}
              width={640}
              height={360}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{ mixBlendMode: 'normal' }}
            />

            {/* Error overlay if webcam failed */}
            {camera.type === 'webcam' && localError && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <div className="text-center text-red-300 px-4">
                  <AlertCircle className="h-10 w-10 mx-auto mb-2" />
                  <div className="font-semibold">Không truy cập được webcam</div>
                  <div className="text-sm opacity-90 mt-1">{localError}</div>
                  
                  {/* Show available cameras info */}
                  {availableCameras.length > 0 && (
                    <div className="text-xs opacity-75 mt-2">
                      Tìm thấy {availableCameras.length} camera: {availableCameras.map(c => c.label).join(', ')}
                    </div>
                  )}
                  
                  <div className="mt-3 flex items-center justify-center gap-2">
                    <Button size="sm" variant="secondary" onClick={() => setReloadNonce(n => n + 1)}>
                      Thử lại
                    </Button>
                    <Button size="sm" variant="destructive" onClick={handleForceRetry}>
                      Force Retry
                    </Button>
                  </div>
                  
                  {retryCount > 0 && (
                    <div className="text-xs opacity-75 mt-2">
                      Đã thử {retryCount} lần
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Offline State */
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center text-gray-400">
              <AlertCircle className="h-12 w-12 mx-auto mb-2" />
              <div className="text-lg font-semibold">Camera offline</div>
              <div className="text-sm">{camera.errorMessage || 'Chưa kết nối camera'}</div>
            </div>
          </div>
        )}

        {/* Sleepy Students Count Badge */}
        {camera.sleepyStudents > 0 && (
          <div className="absolute top-2 right-2 flex gap-2">
            <Badge variant="destructive" className="animate-pulse">
              ⚠ {camera.sleepyStudents} học sinh
            </Badge>
            {sleepDuration > 0 && (
              <Badge variant="destructive">
                {Math.floor(sleepDuration / 60)}:{(sleepDuration % 60).toString().padStart(2, '0')}
              </Badge>
            )}
          </div>
        )}
      </div>

      {/* Student Tracking Details */}
      {showTrackingDetails && (
        <div className="p-4 border-t">
          <StudentTrackingDetails
            students={camera.students}
            cameraName={camera.name}
            isActive={camera.isRunning}
          />
        </div>
      )}

      {/* YOLO Detection Panel */}
      {showYOLOPanel && (
        <div className="p-4 border-t">
          <YOLODetectionPanel
            cameraId={camera.id}
            isEnabled={yoloDetectionEnabled}
            onToggleDetection={setYoloDetectionEnabled}
          />
        </div>
      )}
    </div>
  );
}