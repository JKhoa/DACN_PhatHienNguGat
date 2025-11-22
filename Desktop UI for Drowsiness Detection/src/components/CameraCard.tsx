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
} from 'lucide-react';
import { cn } from './ui/utils';
import { StudentTrackingDetails } from './StudentTrackingDetails';
import { acquireWebcam, releaseWebcam, mapGetUserMediaError, forceReleaseAllWebcams, getAvailableCameras, requestCameraPermission, testCameraAccess } from '../lib/webcamRegistry';
import { DetectionWSClient } from '../lib/wsDetection';
import { DetectionResult, Person } from '../types/detection';
import { wsCamera } from '../lib/wsCamera';

interface CameraCardProps {
  camera: Camera;
  onToggle: (cameraId: string) => void;
  onPopOut: (cameraId: string) => void;
  onConfigure: (cameraId: string) => void;
  onToggleOverlay?: (cameraId: string) => void;
  onToggleLogging?: (cameraId: string) => void;
  onCapturePhoto?: (cameraId: string) => void;
  onRecordVideo?: (cameraId: string) => void;
  onUpdateStudents?: (cameraId: string, students: any[], fps: number) => void;
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
  onUpdateStudents,
  showOverlay,
  showPerformance,
}: CameraCardProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [sleepDuration, setSleepDuration] = useState(0);
  const [showTrackingDetails, setShowTrackingDetails] = useState(false);
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [availableCameras, setAvailableCameras] = useState<Array<{ deviceId: string; label: string; groupId: string }>>([]);
  const [streamKey, setStreamKey] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [lastIpFrameUrl, setLastIpFrameUrl] = useState<string | null>(null);
  const [reloadNonce, setReloadNonce] = useState(0);
  const wsClientRef = useRef<DetectionWSClient | null>(null);
  const wsConnectedRef = useRef<boolean>(false);
  // Keep WS detection results locally so we can draw overlays even if parent doesn't wire onUpdateStudents
  const [wsStudents, setWsStudents] = useState<any[]>([]);
  const [wsFps, setWsFps] = useState<number>(0);
  // Last processing time from backend (ms)
  const [wsProcMs, setWsProcMs] = useState<number | null>(null);
  // Last detection frame dimensions from WS (used for correct overlay scaling)
  const [wsFrameDims, setWsFrameDims] = useState<{ w: number; h: number } | null>(null);
  // UI-adjustable detection sensitivity (0-100). Higher = more sensitive (lower YOLO conf)
  const [sensitivity, setSensitivity] = useState(75 as number);
  
  // Local tracking data (demo tracking boxes)
  const [localTrackingData, setLocalTrackingData] = useState([] as any[]);

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
  setRetryCount((prev: number) => prev + 1);
    
    // Request camera permission first
    const hasPermission = await requestCameraPermission();
    if (!hasPermission) {
      setLocalError('Không có quyền truy cập camera. Vui lòng cho phép camera trong trình duyệt.');
      return;
    }
    
    // Test camera access before retry
    const cameraAccessible = await testCameraAccess(camera.deviceId?.toString());
    if (!cameraAccessible) {
      setLocalError('Camera không thể truy cập được. Hãy kiểm tra camera có được kết nối đúng không.');
      return;
    }
    
    // Force release all webcam streams first
    forceReleaseAllWebcams();
    
    // Wait a bit for resources to be freed
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Reload available cameras
    const cameras = await getAvailableCameras();
    setAvailableCameras(cameras);
    
    // Force re-render to trigger useEffect
  setReloadNonce((prev: number) => prev + 1);
  };


  // Setup video stream - use getUserMedia with shared registry and robust fallbacks
  useEffect(() => {
    let mounted = true;

    const start = async () => {
      console.log(`[CameraCard ${camera.id}] Setup video stream - isRunning:${camera.isRunning}, status:${camera.status}, type:${camera.type}`);
      setLocalError(null);
      if (!(camera.isRunning && camera.status === 'online')) {
        console.log(`[CameraCard ${camera.id}] ⏸️ Skipping video setup - camera not ready`);
        return;
      }

      if (camera.type === 'webcam') {
        try {
          console.log(`[CameraCard ${camera.id}] Acquiring webcam with deviceId:`, camera.deviceId);
          const { stream, streamKey: k, actualDeviceId } = await acquireWebcam({
            deviceId: camera.deviceId,
            width: 640,
            height: 480,
          });
          console.log(`[CameraCard ${camera.id}] ✅ Webcam acquired successfully, streamKey:`, k);
          console.log(`[CameraCard ${camera.id}] Requested deviceId: ${camera.deviceId}, Actual deviceId: ${actualDeviceId}`);
          
          // Check for mismatch between requested and actual camera
          if (camera.deviceId !== undefined && actualDeviceId && String(camera.deviceId) !== actualDeviceId) {
            const errorMsg = `⚠️ Camera mismatch! Requested: ${camera.deviceId}, Got: ${actualDeviceId}`;
            console.error(`[CameraCard ${camera.id}] ${errorMsg}`);
            setLocalError(errorMsg);
            // Release the wrong camera
            releaseWebcam(k);
            return;
          }
          
          console.log(`[CameraCard ${camera.id}] Stream tracks:`, stream.getTracks().map(t => `${t.kind} (${t.readyState})`));
          if (!mounted) return;
          setStreamKey(k);
          setVideoStream(stream);
          if (videoRef.current) {
            console.log(`[CameraCard ${camera.id}] Setting srcObject on video element`);
            videoRef.current.srcObject = stream;
            try { 
              await videoRef.current.play();
              console.log(`[CameraCard ${camera.id}] ✅ Video playing`);
            } catch (playErr) {
              console.error(`[CameraCard ${camera.id}] Video play failed:`, playErr);
            }
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

          // Connect WebSocket for realtime detections
          if (!wsClientRef.current) {
            const client = new DetectionWSClient();
            wsClientRef.current = client;
            
            // Monitor connection status
            client.onStatusChange((connected) => {
              console.log(`[CameraCard ${camera.id}] WS connection status:`, connected ? '✅ Connected' : '❌ Disconnected');
            });
            
            client.connect((msg: DetectionResult) => {
              // Set connection status to true when we receive any message
              wsConnectedRef.current = true;
              
              // DEBUG: Log raw message
              console.log(`[CameraCard ${camera.id}] 🔍 WS Result:`, {
                success: msg?.success,
                personsCount: Array.isArray(msg?.persons) ? msg.persons.length : 0,
                fps: msg?.fps,
                rawMsg: msg
              });
              
              try {
                if (!msg || !msg.success) {
                  console.warn(`[CameraCard ${camera.id}] ⚠️ WS result not success or empty`);
                  return;
                }
                const persons = Array.isArray(msg.persons) ? msg.persons : [];
                const backendFps = typeof msg.fps === 'number' ? msg.fps : 0;
                  const proc = typeof (msg as any).processing_time === 'number' ? (msg as any).processing_time : undefined;
                const fw = typeof (msg as any).frame_width === 'number' ? (msg as any).frame_width : undefined;
                const fh = typeof (msg as any).frame_height === 'number' ? (msg as any).frame_height : undefined;
                if (fw && fh) setWsFrameDims({ w: fw, h: fh });
                  if (typeof proc === 'number') setWsProcMs(Math.max(0, Math.round(proc * 1000)));
                // Deduplicate by track_id/id within a frame
                const seen = new Set<string>();
                const students = persons.map((p: any, idx: number) => {
                  const head = p.head_bbox || p.headBbox || p.bbox;
                  const [x1, y1, x2, y2] = head || p.bbox || [0,0,0,0];
                  const cx = Math.max(0, Math.floor((x1 + x2) / 2));
                  const cy = Math.max(0, Math.floor((y1 + y2) / 2));
                  // Normalize state to only 2 classes: awake | drowsy
                  const st = (p.drowsiness_state === 'awake') ? 'awake' : 'drowsy';
                  const pid = String(p.track_id || p.id || idx + 1);
                  if (seen.has(pid)) return null;
                  seen.add(pid);
                  return {
                    id: pid,
                    position: { x: cx, y: cy },
                    state: st,
                    confidence: typeof p.confidence === 'number' ? p.confidence : (p.drowsiness_score ?? 0.5),
                    sleepDuration: 0,
                    lastUpdate: new Date(),
                    bbox: p.bbox,
                    headBbox: head,
                    keypoints: p.keypoints || [], // ADD KEYPOINTS HERE
                  };
                }).filter(Boolean) as any[];
                
                // DEBUG: Log processed students
                console.log(`[CameraCard ${camera.id}] 📦 Processed students:`, students.length, students);
                
                // Store locally for overlay drawing immediately
                setWsStudents(students);
                setWsFps(Math.round(backendFps));
                // Also propagate to parent store if handler provided (for global stats etc.)
                if (onUpdateStudents) {
                  onUpdateStudents(camera.id, students, Math.round(backendFps));
                }
              } catch (e) {
                // ignore parse errors
              }
            });
            // Push initial config
            const confVal = mapSensitivityToConf(sensitivity);
            client.updateConfig({ conf: confVal, preprocess: { enabled: true } });
          }
          
          // Start sending frames to backend for detection
          let detectionStopped = false;
          const sendFrameForDetection = async () => {
            if (detectionStopped || !mounted || !videoRef.current || !canvasRef.current) return;
            
            try {
              const video = videoRef.current;
              if (video.readyState < 2) { // Not ready
                setTimeout(sendFrameForDetection, 100);
                return;
              }
              
              const canvas = canvasRef.current;
              const ctx = canvas.getContext('2d');
              if (!ctx) {
                setTimeout(sendFrameForDetection, 100);
                return;
              }
              
              // Capture frame from video to temporary canvas
              const tempCanvas = document.createElement('canvas');
              // Downscale to reduce encoding+WS payload (target ~480p)
              const srcW = video.videoWidth || 640;
              const srcH = video.videoHeight || 360;
              const targetW = Math.min(480, srcW);
              const scale = targetW / Math.max(1, srcW);
              const targetH = Math.max(1, Math.floor(srcH * scale));
              tempCanvas.width = targetW;
              tempCanvas.height = targetH;
              const tempCtx = tempCanvas.getContext('2d');
              if (tempCtx) {
                tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                
                // Convert to base64 and send via WebSocket (low-latency)
                const frameBase64 = tempCanvas.toDataURL('image/jpeg', 0.6);
                const ws = wsClientRef.current as DetectionWSClient | null;
                if (ws) {
                  // Send frame even if not yet received response - backend will process and respond
                  ws.sendFrame(frameBase64, camera.id);
                  
                  // Debug log frame sending (throttled)
                  (window as any).__lastFrameSendLog = (window as any).__lastFrameSendLog || 0;
                  if (Date.now() - (window as any).__lastFrameSendLog > 2000) {
                    console.log(`[CameraCard ${camera.id}] ✅ Sending frame ${tempCanvas.width}x${tempCanvas.height} via WS`);
                    (window as any).__lastFrameSendLog = Date.now();
                  }
                } else {
                  console.warn(`[CameraCard ${camera.id}] ⚠️ WebSocket client not initialized`);
                }
              }
            } catch (error) {
              // Silently ignore detection errors for webcam
              console.debug(`[CameraCard ${camera.id}] Detection error:`, error);
            }
            
            // Send next frame after delay
            if (!detectionStopped && mounted) {
              setTimeout(sendFrameForDetection, 180); // ~5-6 FPS detection via WS to reduce CPU
            }
          };
          
          // Start detection after video is ready
          setTimeout(sendFrameForDetection, 500);
          
          return () => {
            detectionStopped = true;
          };
        } catch (err: any) {
          const msg = mapGetUserMediaError(err);
          console.error(`Error starting webcam for camera ${camera.id}:`, err);
          setLocalError(msg);
        }
      } else if (camera.type === 'ip') {
        // Poll backend for JPEG frames to avoid <video> element requirements
        let stopped = false;
        // Subscribe WS updates for this camera to draw overlays without HTTP polling for detections
        const unsubscribe = wsCamera.subscribe(camera.id, (msg) => {
          try {
            const persons = Array.isArray(msg.persons) ? msg.persons : [];
            const backendFps = typeof msg.fps === 'number' ? msg.fps : 0;
            const proc = typeof (msg as any).processing_time === 'number' ? (msg as any).processing_time : undefined;
            if (typeof msg.frame_width === 'number' && typeof msg.frame_height === 'number') {
              setWsFrameDims({ w: msg.frame_width, h: msg.frame_height });
            }
            if (typeof proc === 'number') setWsProcMs(Math.max(0, Math.round(proc * 1000)));
            // Debug: confirm WS updates arriving
            if ((window as any).__lastWsCamLogTs === undefined || Date.now() - (window as any).__lastWsCamLogTs > 2000) {
              console.log(`[WS-CAM ${camera.id}] update: persons=${persons.length}, dims=${msg.frame_width}x${msg.frame_height}, fps=${backendFps}`);
              (window as any).__lastWsCamLogTs = Date.now();
            }
            const seen = new Set<string>();
            const students = persons.map((p: any, idx: number) => {
              const head = p.head_bbox || p.headBbox || p.bbox;
              const [x1, y1, x2, y2] = head || p.bbox || [0,0,0,0];
              const cx = Math.max(0, Math.floor((x1 + x2) / 2));
              const cy = Math.max(0, Math.floor((y1 + y2) / 2));
              const st = (p.drowsiness_state === 'awake') ? 'awake' : 'drowsy';
              const pid = String(p.track_id || p.id || idx + 1);
              if (seen.has(pid)) return null;
              seen.add(pid);
              return {
                id: pid,
                position: { x: cx, y: cy },
                state: st,
                confidence: typeof p.confidence === 'number' ? p.confidence : (p.drowsiness_score ?? 0.5),
                sleepDuration: 0,
                lastUpdate: new Date(),
                bbox: p.bbox,
                headBbox: head,
              };
            }).filter(Boolean) as any[];
            setWsStudents(students);
            setWsFps(Math.round(backendFps));
            if (onUpdateStudents) onUpdateStudents(camera.id, students, Math.round(backendFps));
          } catch {}
        });
        
        // ========== WEBSOCKET ONLY - NO HTTP POLLING ==========
        // Backend sẽ tự động gửi frame qua WebSocket khi có update
        // Không cần poll HTTP nữa, chỉ cần subscribe WS và nhận frame từ backend
        
        return () => { 
          stopped = true; 
          try { unsubscribe(); } catch {} 
        };
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
      // Close WS if open
      if (wsClientRef.current) {
        try { wsClientRef.current.close(); } catch {}
        wsClientRef.current = null;
        wsConnectedRef.current = false;
      }
      setWsStudents([]);
      setWsFps(0);
      setVideoStream(null);
    };
  }, [camera.isRunning, camera.status, camera.type, camera.deviceId, camera.id, reloadNonce]);

  // Update backend config when sensitivity changes
  useEffect(() => {
    const client = wsClientRef.current as DetectionWSClient | null;
    if (client) {
      client.updateConfig({ conf: mapSensitivityToConf(sensitivity) });
    }
  }, [sensitivity]);

  // Sync canvas dimensions with video/image dimensions
  useEffect(() => {
    const updateCanvasSize = () => {
      if (!canvasRef.current) return;
      
      const canvas = canvasRef.current;
      
      // Get the displayed size of the canvas container
      const rect = canvas.getBoundingClientRect();
      const displayWidth = Math.floor(rect.width);
      const displayHeight = Math.floor(rect.height);
      
      // Set canvas internal dimensions to match displayed size for 1:1 pixel mapping
      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        console.log(`[CameraCard ${camera.id}] Canvas dimensions updated to display size: ${displayWidth}x${displayHeight}`);
      }
    };
    
    if (camera.isRunning && camera.status === 'online') {
      updateCanvasSize();
      
      // Watch for dimension changes
      const checkInterval = setInterval(updateCanvasSize, 500);
      
      if (videoRef.current) {
        videoRef.current.addEventListener('loadedmetadata', updateCanvasSize);
      }
      if (imgRef.current) {
        imgRef.current.addEventListener('load', updateCanvasSize);
      }
      
      return () => {
        clearInterval(checkInterval);
        if (videoRef.current) {
          videoRef.current.removeEventListener('loadedmetadata', updateCanvasSize);
        }
        if (imgRef.current) {
          imgRef.current.removeEventListener('load', updateCanvasSize);
        }
      };
    }
  }, [camera.isRunning, camera.status, camera.type, camera.id]);

  // Draw tracking overlays on canvas - ALWAYS DRAW WHEN CAMERA IS RUNNING
  useEffect(() => {
    if (camera.isRunning && camera.status === 'online') {
      console.log(`[CameraCard ${camera.id}] 🎨 Starting canvas drawing loop`);
      
      const drawInterval = setInterval(() => {
        if (canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (ctx && (canvas.width > 0 || canvas.height > 0)) {
            // Always draw, even if video/image not ready yet
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width || 640, canvas.height || 360);
            
            // Draw tracking overlays (ALWAYS, regardless of showOverlay)
            drawOverlays(ctx, canvas);
          }
        }
      }, 120); // Draw every ~120ms to reduce CPU slightly
      
      // Initial draw after a short delay
      const initialDraw = setTimeout(() => {
        if (canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width || 640, canvas.height || 360);
            drawOverlays(ctx, canvas);
          }
        }
      }, 200);
      
      return () => {
        clearInterval(drawInterval);
        clearTimeout(initialDraw);
      };
    }
  }, [camera.isRunning, camera.status, camera.id, showOverlay, camera.students, camera.type, localTrackingData]);

  // Generate demo tracking boxes if no students from backend - ALWAYS CREATE FOR ACTIVE CAMERAS
  useEffect(() => {
    if (camera.isRunning && camera.status === 'online') {
      // Generate demo tracking boxes based on canvas size
      const generateDemoBoxes = () => {
        // Use canvas dimensions if available, otherwise default
        let width = 640;
        let height = 360;
        
        if (canvasRef.current) {
          width = canvasRef.current.width || 640;
          height = canvasRef.current.height || 360;
        } else if (videoRef.current && videoRef.current.videoWidth) {
          width = videoRef.current.videoWidth;
          height = videoRef.current.videoHeight;
        } else if (imgRef.current && imgRef.current.naturalWidth) {
          width = imgRef.current.naturalWidth;
          height = imgRef.current.naturalHeight;
        }
        
        // Only generate demo boxes if no backend students
        if (camera.students.length === 0) {
          // Generate 2-4 demo tracking boxes
          const numBoxes = 2 + Math.floor(Math.random() * 3);
          const boxes = [];
          
          for (let i = 0; i < numBoxes; i++) {
            const boxWidth = width * (0.15 + Math.random() * 0.1); // 15-25% of canvas width
            const boxHeight = boxWidth * 1.2; // Aspect ratio for head
            const x = Math.random() * (width - boxWidth);
            const y = Math.random() * (height - boxHeight) * 0.6; // Upper 60% of canvas (head area)
            const state = Math.random() > 0.7 ? (Math.random() > 0.5 ? 'sleepy' : 'head_down') : 'normal';
            
            boxes.push({
              id: `demo-${i + 1}`,
              x,
              y,
              width: boxWidth,
              height: boxHeight,
              state,
            });
          }
          
          console.log(`[CameraCard ${camera.id}] Generated ${boxes.length} demo tracking boxes at ${width}x${height}`);
          console.log('[CameraCard] Demo boxes:', boxes);
          setLocalTrackingData(boxes);
        } else {
          // Clear demo boxes if backend has students
          setLocalTrackingData([]);
        }
      };
      
      // Generate immediately with default dimensions, then retry with actual dimensions
      generateDemoBoxes(); // Generate immediately with default or current dimensions
      
      // Retry after canvas might be ready
      const timeout1 = setTimeout(() => {
        generateDemoBoxes();
      }, 300);
      
      const timeout2 = setTimeout(() => {
        generateDemoBoxes();
      }, 1000);
      
      // Regenerate boxes every 3-5 seconds for demo
      const demoInterval = setInterval(() => {
        if (camera.students.length === 0) {
          generateDemoBoxes();
        }
      }, 3000 + Math.random() * 2000);
      
      return () => {
        clearTimeout(timeout1);
        clearTimeout(timeout2);
        clearInterval(demoInterval);
      };
    } else {
      setLocalTrackingData([]);
    }
  }, [camera.isRunning, camera.status, camera.students.length, camera.id]);

  const drawOverlays = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    // Vẽ tracking box từ kết quả WebSocket với 2 màu: xanh (awake) và đỏ (drowsy)
    // Ưu tiên WS; nếu chưa có, hiển thị hướng dẫn chờ WS.
    const wsConnected = wsConnectedRef.current; // Check actual WS connection status
    const wsHasDetections = wsStudents.length > 0;
    const trackingBoxes = wsHasDetections
      ? wsStudents.map((student: any) => ({
          id: student.id,
          x: student.position?.x || 0,
          y: student.position?.y || 0,
          bbox: student.bbox,
          headBbox: student.headBbox,
          state: student.state === 'drowsy' ? 'drowsy' : 'awake',
        }))
      : [];
    
    // Debug logging - log every time to ensure tracking is happening
    // Throttle verbose logging to avoid UI lag
    (window as any).__lastOverlayLog = (window as any).__lastOverlayLog || 0;
    const nowTs = Date.now();
    if (nowTs - (window as any).__lastOverlayLog > 2000) {
      if (trackingBoxes.length > 0) {
        console.log(`[CameraCard ${camera.id}] ✅ Drawing ${trackingBoxes.length} boxes (ws:${wsStudents.length}, backend:${camera.students.length}, demo:${localTrackingData.length}) on ${canvas.width}x${canvas.height}`);
      } else if (camera.isRunning && camera.status === 'online') {
        console.log(`[CameraCard ${camera.id}] ⚠️ No tracking boxes to draw (wsConnected:${wsConnected}, wsDetections:${wsHasDetections})`);
      }
      (window as any).__lastOverlayLog = nowTs;
    }
    
    if (trackingBoxes.length === 0) {
      // Show status overlay based on connection state
      ctx.strokeStyle = '#ff1744';
      ctx.lineWidth = 2;
      ctx.strokeRect(10, 10, 280, 50);
      ctx.fillStyle = '#ff1744';
      ctx.font = 'bold 14px Arial';
      if (!wsConnected) {
        ctx.fillText('Đang kết nối WebSocket...', 15, 35);
      } else {
        ctx.fillText('Không phát hiện người trong khung hình', 15, 35);
      }
      ctx.font = '12px Arial';
      ctx.fillText(`Camera: ${camera.id}`, 15, 50);
      
      // Show debug info with better status indication
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(10, canvas.height - 50, 280, 45);
      ctx.font = '12px Arial';
      
      // WS Connection status
      if (wsConnected) {
        ctx.fillStyle = '#22c55e';
        ctx.fillText('WS: ✓ Kết nối', 15, canvas.height - 35);
      } else {
        ctx.fillStyle = '#ef4444';
        ctx.fillText('WS: ✗ Chưa kết nối', 15, canvas.height - 35);
      }
      
      // Detection status
      ctx.fillStyle = '#fff';
      ctx.fillText(`Detections: ${wsHasDetections ? wsStudents.length : '0 (chưa thấy người)'}`, 15, canvas.height - 20);
      ctx.fillText(`FPS: ${wsFps || camera.fps} | WebSocket`, 15, canvas.height - 8);
      return;
    }
    
    // Get actual source dimensions for scaling
    // Prefer WS-reported detection frame size for correct scaling; fallback to media element dims
    let sourceWidth = wsFrameDims?.w || 640;
    let sourceHeight = wsFrameDims?.h || 360;
    if (!wsFrameDims) {
      if (camera.type === 'webcam' && videoRef.current) {
        sourceWidth = videoRef.current.videoWidth || sourceWidth;
        sourceHeight = videoRef.current.videoHeight || sourceHeight;
      } else if (camera.type === 'ip' && imgRef.current) {
        sourceWidth = imgRef.current.naturalWidth || sourceWidth;
        sourceHeight = imgRef.current.naturalHeight || sourceHeight;
      }
    }
    
    // Calculate scale factors (coordinates from backend may be in different frame size)
    const scaleX = canvas.width / sourceWidth;
    const scaleY = canvas.height / sourceHeight;
    
    // Draw tracking boxes
    trackingBoxes.forEach((box: any) => {
      // Determine box coordinates
      let x1, y1, x2, y2, x, y;
      
      if (box.headBbox && Array.isArray(box.headBbox) && box.headBbox.length >= 4) {
        // Use headBbox from backend (already in backend frame coordinates)
        [x1, y1, x2, y2] = box.headBbox;
        x1 = x1 * scaleX;
        y1 = y1 * scaleY;
        x2 = x2 * scaleX;
        y2 = y2 * scaleY;
        x = (x1 + x2) / 2;
        y = (y1 + y2) / 2;
      } else if (box.bbox && Array.isArray(box.bbox) && box.bbox.length >= 4) {
        // Use body bbox if headBbox not available
        [x1, y1, x2, y2] = box.bbox;
        x1 = x1 * scaleX;
        y1 = y1 * scaleY;
        x2 = x2 * scaleX;
        y2 = y2 * scaleY;
        // Use top portion of body bbox as head approximation
        const headHeight = (y2 - y1) * 0.3;
        y2 = y1 + headHeight;
        x = (x1 + x2) / 2;
        y = (y1 + y2) / 2;
      } else if (box.width && box.height) {
        // Use demo tracking box dimensions directly (already in canvas coordinates)
        x1 = box.x;
        y1 = box.y;
        x2 = box.x + box.width;
        y2 = box.y + box.height;
        x = (x1 + x2) / 2;
        y = (y1 + y2) / 2;
      } else {
        // Fallback: use position with default size
        x = (box.x || 320) * scaleX;
        y = (box.y || 180) * scaleY;
        const defaultSize = Math.min(canvas.width, canvas.height) * 0.15;
        x1 = x - defaultSize / 2;
        y1 = y - defaultSize / 2;
        x2 = x + defaultSize / 2;
        y2 = y + defaultSize / 2;
      }
      
      // Ensure coordinates are within canvas bounds
      x1 = Math.max(0, Math.min(x1, canvas.width));
      y1 = Math.max(0, Math.min(y1, canvas.height));
      x2 = Math.max(0, Math.min(x2, canvas.width));
      y2 = Math.max(0, Math.min(y2, canvas.height));
      x = Math.max(0, Math.min(x, canvas.width));
      y = Math.max(0, Math.min(y, canvas.height));
        
      // Student color based on state
  const color = box.state === 'drowsy' ? '#ff1744' : '#00e676'; // đỏ | xanh
      
      // Draw head-focused bounding box
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(3, canvas.width / 220); // nét dày hơn, rõ ràng
      
      // Draw bounding box
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // Draw corner markers for better visibility
      const cornerSize = Math.max(8, canvas.width / 80);
      ctx.beginPath();
      // Top-left
      ctx.moveTo(x1, y1 + cornerSize);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerSize, y1);
      // Top-right
      ctx.moveTo(x2 - cornerSize, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cornerSize);
      // Bottom-left
      ctx.moveTo(x1, y2 - cornerSize);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + cornerSize, y2);
      // Bottom-right
      ctx.moveTo(x2 - cornerSize, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - cornerSize);
      ctx.stroke();
      
      // Draw center point circle for tracking
      const centerRadius = Math.max(5, canvas.width / 180);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, centerRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // ID label (to, rõ)
      const fontSize = Math.max(14, canvas.width / 42);
      ctx.font = `bold ${fontSize}px Arial`;
      ctx.textAlign = 'center';
      
      const idText = `#${box.id}`;
      const textMetrics = ctx.measureText(idText);
      const padding = Math.max(6, canvas.width / 180);
      const labelHeight = fontSize * 1.25;
      
      // Background for ID label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(
        x - textMetrics.width / 2 - padding,
        y - labelHeight - padding * 2 - centerRadius - 5,
        textMetrics.width + padding * 2,
        labelHeight + padding
      );
      
      // Draw ID text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(idText, x, y - padding - centerRadius - 5);
      
      // State label: chỉ hai trạng thái
      const stateText = box.state === 'drowsy' ? 'BUỒN NGỦ' : 'TỈNH';
      ctx.font = `bold ${fontSize * 0.9}px Arial`;
        const stateMetrics = ctx.measureText(stateText);
        
      ctx.fillStyle = color;
      ctx.fillRect(
        x - stateMetrics.width / 2 - padding,
        y + padding + centerRadius + 5,
        stateMetrics.width + padding * 2,
        labelHeight + padding
      );
      
      ctx.fillStyle = '#ffffff';
      ctx.fillText(stateText, x, y + labelHeight + padding + centerRadius + 5);
      
      ctx.textAlign = 'left';
      
      // Draw keypoints and skeleton if available
      const student = wsStudents.find((s: any) => s.id === box.id);
      if (student && student.keypoints && Array.isArray(student.keypoints) && student.keypoints.length >= 17) {
        const kpts = student.keypoints;
        
        // COCO skeleton connections (pose)
        const skeleton = [
          [0, 1], [0, 2], [1, 3], [2, 4], // head
          [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // arms
          [5, 11], [6, 12], [11, 12], // torso
          [11, 13], [13, 15], [12, 14], [14, 16] // legs
        ];
        
        // Draw skeleton lines
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, canvas.width / 320);
        skeleton.forEach(([i, j]) => {
          if (i < kpts.length && j < kpts.length) {
            const pt1 = kpts[i];
            const pt2 = kpts[j];
            if (pt1.visible && pt2.visible && pt1.confidence > 0.3 && pt2.confidence > 0.3) {
              ctx.beginPath();
              ctx.moveTo(pt1.x * scaleX, pt1.y * scaleY);
              ctx.lineTo(pt2.x * scaleX, pt2.y * scaleY);
              ctx.stroke();
            }
          }
        });
        
        // Draw keypoints
        kpts.forEach((kpt: any, idx: number) => {
          if (kpt.visible && kpt.confidence > 0.3) {
            const kx = kpt.x * scaleX;
            const ky = kpt.y * scaleY;
            const radius = Math.max(3, canvas.width / 200);
            
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(kx, ky, radius, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      }
    });

    // Performance HUD
      if (showPerformance) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(5, 5, 140, 80);
        ctx.fillStyle = '#fff';
        ctx.font = '11px monospace';
        ctx.fillText(`FPS: ${wsFps || camera.fps}`, 10, 20);
        ctx.fillText(`Students: ${wsStudents.length || camera.students.length}`, 10, 35);
        ctx.fillText(`Sleepy: ${camera.sleepyStudents}`, 10, 50);
        ctx.fillText(`Proc: ${wsProcMs != null ? wsProcMs : '-'}ms`, 10, 65);
        ctx.fillText(`Conf: ${(wsStudents.reduce((sum, s) => sum + (s.confidence || 0), 0) / (wsStudents.length || camera.students.length || 1) || 0).toFixed(2)}`, 10, 80);
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
      return <Loader2 className="h-2 w-2 animate-spin text-red-500" />;
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
        class: 'bg-red-500/10 text-red-500 border-red-500/20',
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
      <div className="relative aspect-video bg-gray-800">
        {/* Base media: video for webcam, img for IP */}
        {camera.isRunning && camera.status === 'online' ? (
          <div className="relative w-full h-full">
            {camera.type === 'webcam' ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute top-0 left-0 w-full h-full object-cover z-0 bg-blue-900"
                onLoadedMetadata={(e) => {
                  console.log(`[CameraCard ${camera.id}] ✅ Video metadata loaded:`, videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight);
                  console.log(`[CameraCard ${camera.id}] Video srcObject:`, videoRef.current?.srcObject);
                  console.log(`[CameraCard ${camera.id}] Video readyState:`, videoRef.current?.readyState);
                }}
                onPlay={() => console.log(`[CameraCard ${camera.id}] ✅ Video playing`)}
                onError={(e) => console.error(`[CameraCard ${camera.id}] ❌ Video error:`, e)}
                onLoadStart={() => console.log(`[CameraCard ${camera.id}] Video load start`)}
              />
            ) : (
              <img
                ref={imgRef}
                alt={camera.name}
                className="absolute top-0 left-0 w-full h-full object-cover select-none z-0"
                draggable={false}
              />
            )}

            {/* Canvas overlay for tracking visualizations - EXACTLY overlay video */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none z-10 border-2 border-red-500 bg-green-900/20"
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

      {/* Controls */}
      {camera.isRunning && camera.status === 'online' && (
        <div className="px-3 py-2 border-t bg-muted/30 flex items-center gap-3">
          <label className="text-xs text-muted-foreground whitespace-nowrap">Detection sensitivity</label>
          <input
            type="range"
            min={0}
            max={100}
            value={sensitivity}
            onChange={(e) => setSensitivity(Number(e.target.value))}
            className="flex-1"
            aria-label="Detection sensitivity"
            title="Detection sensitivity"
          />
          <span className="text-xs w-10 text-right font-mono">{sensitivity}</span>
        </div>
      )}

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
    </div>
  );
}

// Map UI sensitivity (0-100) to YOLO confidence threshold (0.05-0.6)
function mapSensitivityToConf(s: number): number {
  const sens = Math.max(0, Math.min(100, s || 0));
  const conf = 0.6 - (sens / 100) * 0.55; // higher sensitivity -> lower conf
  return Math.max(0.05, Math.min(0.6, conf));
}