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
    // Request permission first
    const permission = await navigator.permissions.query({ name: 'camera' as PermissionName });
    console.log('Camera permission status:', permission.state);
    
    if (permission.state === 'denied') {
      throw new Error('Camera permission denied by user');
    }
    
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    console.error('getUserMedia failed:', error);
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

export async function acquireWebcam(opts: AcquireOpts): Promise<{ stream: MediaStream; streamKey: string }> {
  const k = keyFor(opts.deviceId);
  const cached = registry.get(k);
  if (cached) {
    cached.refCount += 1;
    return { stream: cached.stream, streamKey: k };
  }

  // Try a sequence of constraints with graceful fallbacks
  const attempts: MediaStreamConstraints[] = [];
  
  // 1) exact deviceId if provided
  if (opts.deviceId !== undefined) {
    attempts.push({ video: {
      deviceId: { exact: String(opts.deviceId) },
      width: { ideal: opts.width || 640 },
      height: { ideal: opts.height || 480 },
      frameRate: { ideal: 30, max: 30 },
    } });
  }
  
  // 2) same without deviceId (let browser choose)
  attempts.push({ video: { 
    width: { ideal: opts.width || 640 }, 
    height: { ideal: opts.height || 480 }, 
    frameRate: { ideal: 30, max: 30 } 
  } });
  
  // 3) very permissive
  attempts.push({ video: true });
  
  // 4) facingMode user (laptop webcams)
  attempts.push({ video: { facingMode: 'user' } as any });
  
  // 5) low-res fallback to reduce hardware pressure
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
      return { stream: s, streamKey: k };
    } catch (e: any) {
      lastErr = e;
      console.warn(`Webcam attempt failed:`, e);
      // If it's "Device in use", try next attempt immediately
      if (e && (e as any).name === 'NotReadableError') {
        continue;
      }
      // For other errors, add a small delay
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  // If device is busy or unsuitable, try cycling through available cameras
  try {
    console.log('Trying alternative camera devices...');
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter(d => d.kind === 'videoinput');
    console.log(`Found ${videos.length} video devices:`, videos.map(d => ({ id: d.deviceId, label: d.label })));
    
    for (const d of videos) {
      // Skip the requested one if specified
      if (opts.deviceId && String(opts.deviceId) === d.deviceId) continue;
      
      try {
        console.log(`Trying alternative device: ${d.deviceId}`);
        const s = await tryGetUserMedia({
          video: {
            deviceId: { exact: d.deviceId },
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30, max: 30 },
          } as any,
        });
        const altKey = keyFor(d.deviceId);
        registry.set(altKey, { stream: s, refCount: 1 });
        console.log(`Successfully acquired alternative device: ${d.deviceId}`);
        return { stream: s, streamKey: altKey };
      } catch (altErr) {
        console.warn(`Alternative device ${d.deviceId} failed:`, altErr);
        // Continue to next device
      }
    }
  } catch (enumErr) {
    console.warn('Failed to enumerate devices:', enumErr);
  }
  
  // If no real camera works, throw error instead of using mock camera
  console.log('No real cameras available, refusing to use mock camera');
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

