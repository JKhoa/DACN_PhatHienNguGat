type AcquireOpts = {
  deviceId?: number | string;
  width?: number;
  height?: number;
};

// Simple shared registry to allow multiple components to consume the same webcam stream
const registry = new Map<string, { stream: MediaStream; refCount: number }>();

function keyFor(dev?: number | string) {
  return dev !== undefined ? String(dev) : 'default';
}

async function tryGetUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream> {
  // Some browsers require https except for localhost
  try {
    // Skip permission check in Electron (auto-granted by main.js)
    const isElectron = navigator.userAgent.includes('Electron');
    console.log('[webcamRegistry] tryGetUserMedia - isElectron:', isElectron);
    console.log('[webcamRegistry] constraints:', JSON.stringify(constraints, null, 2));
    
    if (!isElectron) {
      // Request permission first in browser
      try {
        const permission = await navigator.permissions.query({ name: 'camera' as PermissionName });
        console.log('Camera permission status:', permission.state);
        
        if (permission.state === 'denied') {
          throw new Error('Camera permission denied by user');
        }
      } catch (permError) {
        console.warn('Permission query not supported, proceeding anyway:', permError);
      }
    } else {
      console.log('[webcamRegistry] Running in Electron, skipping permission check');
    }
    
    console.log('[webcamRegistry] Calling getUserMedia...');
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    console.log('[webcamRegistry] ✅ getUserMedia SUCCESS, tracks:', stream.getTracks().map(t => `${t.kind}:${t.label} (${t.readyState})`));
    return stream;
  } catch (error) {
    console.error('[webcamRegistry] ❌ getUserMedia FAILED:', error);
    console.error('[webcamRegistry] Error name:', (error as any)?.name);
    console.error('[webcamRegistry] Error message:', (error as any)?.message);
    throw error;
  }
}

export function mapGetUserMediaError(err: any): string {
  const name = err?.name || '';
  if (name === 'NotAllowedError') return 'Trình duyệt đã chặn quyền webcam. Hãy cấp quyền sử dụng camera.';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'Không tìm thấy thiết bị webcam phù hợp với cấu hình. Hãy thử deviceId khác hoặc bỏ chọn deviceId.';
  if (name === 'NotReadableError') return 'Thiết bị đang được ứng dụng khác sử dụng. Hãy đóng ứng dụng đang dùng camera (Teams, Zoom, app khác) và thử lại.';
  if (name === 'SecurityError') return 'Yêu cầu HTTPS hoặc localhost để truy cập webcam.';
  return err?.message || 'Không thể truy cập webcam.';
}

export async function acquireWebcam(opts: AcquireOpts): Promise<{ stream: MediaStream; streamKey: string; actualDeviceId?: string }> {
  const k = keyFor(opts.deviceId);
  const cached = registry.get(k);
  if (cached) {
    cached.refCount += 1;
    // Return the actual device ID from the stream's video track
    const videoTrack = cached.stream.getVideoTracks()[0];
    const actualDeviceId = videoTrack?.getSettings().deviceId;
    return { stream: cached.stream, streamKey: k, actualDeviceId };
  }

  // Try ONLY the exact deviceId if provided - no fallback to other cameras
  if (opts.deviceId !== undefined) {
    const attempts: MediaStreamConstraints[] = [];
    
    // 1) exact deviceId with ideal dimensions
    attempts.push({ video: {
      deviceId: { exact: String(opts.deviceId) },
      width: { ideal: opts.width || 640 },
      height: { ideal: opts.height || 480 },
      frameRate: { ideal: 30, max: 30 },
    } });
    
    // 2) exact deviceId with permissive dimensions
    attempts.push({ video: {
      deviceId: { exact: String(opts.deviceId) },
    } });
    
    // 3) exact deviceId with low-res fallback
    attempts.push({ video: {
      deviceId: { exact: String(opts.deviceId) },
      width: { ideal: 320 },
      height: { ideal: 240 },
      frameRate: { ideal: 15, max: 15 },
    } });

    let lastErr: any;
    for (const c of attempts) {
      try {
        const s = await tryGetUserMedia(c);
        registry.set(k, { stream: s, refCount: 1 });
        
        // Get actual device ID from stream
        const videoTrack = s.getVideoTracks()[0];
        const actualDeviceId = videoTrack?.getSettings().deviceId;
        
        console.log(`✅ Acquired exact device ${opts.deviceId}, verified actualDeviceId: ${actualDeviceId}`);
        return { stream: s, streamKey: k, actualDeviceId };
      } catch (e: any) {
        lastErr = e;
        console.warn(`Webcam attempt failed for device ${opts.deviceId}:`, e);
        // Add small delay between attempts
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    // If exact device failed, throw error - DO NOT try alternative cameras
    console.error(`❌ Failed to acquire exact device ${opts.deviceId}, NOT falling back to alternatives`);
    throw lastErr;
  }
  
  // If no deviceId specified, let browser choose any camera
  const attempts: MediaStreamConstraints[] = [];
  
  // 1) Ideal dimensions
  attempts.push({ video: { 
    width: { ideal: opts.width || 640 }, 
    height: { ideal: opts.height || 480 }, 
    frameRate: { ideal: 30, max: 30 } 
  } });
  
  // 2) Very permissive
  attempts.push({ video: true });
  
  // 3) FacingMode user (laptop webcams)
  attempts.push({ video: { facingMode: 'user' } as any });
  
  // 4) Low-res fallback
  attempts.push({ video: { 
    width: { ideal: 320 }, 
    height: { ideal: 240 }, 
    frameRate: { ideal: 15, max: 15 } 
  } });

  let lastErr: any;
  for (const c of attempts) {
    try {
      const s = await tryGetUserMedia(c);
      registry.set(k, { stream: s, refCount: 1 });
      
      // Get actual device ID from stream
      const videoTrack = s.getVideoTracks()[0];
      const actualDeviceId = videoTrack?.getSettings().deviceId;
      
      console.log(`✅ Acquired default camera, actualDeviceId: ${actualDeviceId}`);
      return { stream: s, streamKey: k, actualDeviceId };
    } catch (e: any) {
      lastErr = e;
      console.warn(`Webcam attempt failed:`, e);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  // If no camera works, throw error
  console.log('❌ No cameras available');
  throw lastErr;
}

export function releaseWebcam(streamKey: string) {
  const entry = registry.get(streamKey);
  if (!entry) return;
  entry.refCount -= 1;
  if (entry.refCount <= 0) {
    entry.stream.getTracks().forEach(t => t.stop());
    registry.delete(streamKey);
  }
}

// Force release all webcam streams (useful for "Device in use" errors)
export function forceReleaseAllWebcams() {
  console.log('Force releasing all webcam streams...');
  for (const [key, entry] of registry.entries()) {
    entry.stream.getTracks().forEach(track => {
      track.stop();
      console.log(`Stopped track: ${track.kind}`);
    });
  }
  registry.clear();
  console.log('All webcam streams released');
}

// Request camera permission explicitly
export async function requestCameraPermission(): Promise<boolean> {
  try {
    console.log('Requesting camera permission...');
    
    // Try to get a temporary stream to trigger permission request
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 1, height: 1 } 
    });
    
    // Stop the stream immediately
    stream.getTracks().forEach(track => track.stop());
    
    console.log('Camera permission granted');
    return true;
  } catch (error) {
    console.error('Camera permission denied:', error);
    return false;
  }
}

// Mock camera for testing when no real camera is available
export async function createMockCamera(): Promise<MediaStream> {
  console.log('Creating mock camera for testing...');
  
  // Create a canvas with a simple pattern
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');
  
  if (!ctx) {
    throw new Error('Cannot create canvas context');
  }
  
  // Draw a simple pattern
  const drawFrame = () => {
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw moving circles
    const time = Date.now() / 1000;
    for (let i = 0; i < 5; i++) {
      const x = (canvas.width / 2) + Math.sin(time + i) * 100;
      const y = (canvas.height / 2) + Math.cos(time + i) * 80;
      ctx.fillStyle = `hsl(${(time * 50 + i * 60) % 360}, 70%, 50%)`;
      ctx.beginPath();
      ctx.arc(x, y, 20, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw text
    ctx.fillStyle = '#ffffff';
    ctx.font = '24px Arial';
    ctx.fillText('Mock Camera - No Real Camera Detected', 50, 50);
    ctx.fillText(`Time: ${new Date().toLocaleTimeString()}`, 50, 100);
  };
  
  // Create a MediaStream from canvas
  const stream = canvas.captureStream(30); // 30 FPS
  
  // Update canvas every frame
  const updateInterval = setInterval(drawFrame, 33); // ~30 FPS
  
  // Clean up when stream ends
  stream.addEventListener('ended', () => {
    clearInterval(updateInterval);
  });
  
  return stream;
}

// Get list of available camera devices
export async function getAvailableCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices
      .filter(d => d.kind === 'videoinput')
      .map(d => ({
        deviceId: d.deviceId,
        label: d.label || `Camera ${d.deviceId}`,
        groupId: d.groupId
      }));
    console.log(`Found ${cameras.length} camera devices:`, cameras);
    return cameras;
  } catch (error) {
    console.error('Failed to enumerate cameras:', error);
    return [];
  }
}

// Test if a specific camera device is actually accessible
export async function testCameraAccess(deviceId?: string): Promise<boolean> {
  try {
    console.log(`Testing camera access for device: ${deviceId || 'default'}`);
    
    const constraints: MediaStreamConstraints = deviceId 
      ? { video: { deviceId: { exact: deviceId } } }
      : { video: true };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    
    // Test if we can actually read frames
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    
    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        console.log(`Camera ${deviceId || 'default'} is accessible`);
        stream.getTracks().forEach(track => track.stop());
        resolve(true);
      };
      
      video.onerror = () => {
        console.log(`Camera ${deviceId || 'default'} failed to load`);
        stream.getTracks().forEach(track => track.stop());
        resolve(false);
      };
      
      video.load();
    });
  } catch (error) {
    console.log(`Camera ${deviceId || 'default'} access failed:`, error);
    return false;
  }
}

