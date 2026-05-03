# Camera Mismatch Bug Fix Report

## Ngày: 2025-01-25

## Vấn đề đã sửa: Camera không khớp (Camera Mismatch)

### Mô tả lỗi ban đầu:
- Frontend hiển thị camera A nhưng detection đang chạy trên camera B
- Người dùng chọn camera cụ thể nhưng hệ thống tự động chuyển sang camera khác
- Không có cảnh báo khi camera thực tế khác với camera yêu cầu

### Nguyên nhân gốc rễ:
1. **Fallback logic quá linh hoạt**: File `webcamRegistry.ts` cũ có logic tự động thử các camera thay thế khi camera yêu cầu thất bại, KHÔNG thông báo cho component biết camera nào thực sự được chọn.

2. **Không kiểm tra actualDeviceId**: Component `CameraCard.tsx` không xác minh camera thực tế có khớp với camera yêu cầu hay không.

3. **Type không nhất quán**: Interface `Camera` định nghĩa `deviceId?: number` nhưng browser APIs trả về string, gây nhầm lẫn type.

### Giải pháp đã triển khai:

#### 1. Cập nhật `webcamRegistry.ts`:
**THAY ĐỔI QUAN TRỌNG**: Loại bỏ fallback tự động sang camera khác

```typescript
// CŨ: Tự động thử camera khác khi camera yêu cầu thất bại
for (const d of videos) {
  if (opts.deviceId && String(opts.deviceId) === d.deviceId) continue;
  // Try alternative camera...
}

// MỚI: CHỈ thử camera yêu cầu, throw error nếu thất bại
if (opts.deviceId !== undefined) {
  // Try exact device ONLY with different constraint attempts
  // If all fail, throw error - DO NOT try alternative cameras
  throw lastErr;
}
```

**Thêm trường `actualDeviceId`** trong return type:
```typescript
export async function acquireWebcam(opts: AcquireOpts): 
  Promise<{ stream: MediaStream; streamKey: string; actualDeviceId?: string }> {
  // Get actual device ID from stream's video track
  const videoTrack = stream.getVideoTracks()[0];
  const actualDeviceId = videoTrack?.getSettings().deviceId;
  return { stream, streamKey: k, actualDeviceId };
}
```

#### 2. Cập nhật `CameraCard.tsx`:
**Thêm kiểm tra mismatch**:
```typescript
const { stream, streamKey: k, actualDeviceId } = await acquireWebcam({
  deviceId: camera.deviceId,
  width: 640,
  height: 480,
});

// Check for mismatch
if (camera.deviceId !== undefined && actualDeviceId && 
    String(camera.deviceId) !== actualDeviceId) {
  const errorMsg = `⚠️ Camera mismatch! Requested: ${camera.deviceId}, Got: ${actualDeviceId}`;
  console.error(`[CameraCard ${camera.id}] ${errorMsg}`);
  setLocalError(errorMsg);
  releaseWebcam(k); // Release wrong camera
  return;
}
```

#### 3. Cập nhật `types/index.ts`:
**Hỗ trợ cả number và string cho deviceId**:
```typescript
export interface Camera {
  // ... other fields
  deviceId?: number | string; // Support both for flexibility
}
```

### Hành vi mới:
1. **Strict camera selection**: Khi người dùng chọn camera cụ thể (deviceId), hệ thống CHỈ thử camera đó với các constraint khác nhau (độ phân giải, framerate).

2. **Fail fast**: Nếu camera yêu cầu không khả dụng, throw error ngay lập tức thay vì tự động chuyển sang camera khác.

3. **Mismatch detection**: Nếu có sự không khớp (do race condition hoặc lỗi driver), hiển thị error rõ ràng và release camera sai.

4. **Browser choice**: Nếu không chỉ định deviceId, cho phép browser tự chọn camera bất kỳ.

### Lợi ích:
✅ **Tính nhất quán**: Người dùng luôn thấy đúng camera họ chọn  
✅ **Tính minh bạch**: Lỗi camera được hiển thị rõ ràng thay vì âm thầm chuyển đổi  
✅ **Dễ debug**: Logs rõ ràng về requested vs actual deviceId  
✅ **Tránh confusion**: Không còn tình trạng video không khớp với detection  

### Không ảnh hưởng đến:
❌ **Offline camera handling**: Không thay đổi logic xử lý camera offline (theo yêu cầu người dùng)  
❌ **IP camera**: Chỉ sửa webcam logic, IP camera không bị ảnh hưởng  
❌ **Backend integration**: Không thay đổi Python backend  

### Testing:
1. Build thành công: `npx vite build` - 5.30s
2. Không có TypeScript errors
3. Sẵn sàng test với Electron app: `npm run electron`

### Files đã sửa:
1. `src/lib/webcamRegistry.ts` - Core camera acquisition logic
2. `src/components/CameraCard.tsx` - Mismatch detection
3. `src/types/index.ts` - Type definition cho deviceId

### Version:
- Build time: 5.30s
- Bundle size: 970.72 kB (gzipped: 280.51 kB)
- CSS size: 48.52 kB (gzipped: 8.60 kB)

---

## Hướng dẫn test:

### Test 1: Camera khả dụng
1. Mở app: `npm run electron`
2. Thêm webcam mới với deviceId cụ thể
3. Kiểm tra console logs: "Requested deviceId: X, Actual deviceId: X"
4. ✅ Expected: Video và detection khớp với camera được chọn

### Test 2: Camera bận/không khả dụng
1. Mở Teams/Zoom để chiếm camera
2. Thử khởi động camera đó trong app
3. ✅ Expected: Hiển thị error rõ ràng "Camera không thể truy cập"
4. ✅ Expected: KHÔNG tự động chuyển sang camera khác

### Test 3: Camera mismatch (edge case)
1. Nếu xảy ra mismatch (do bug driver/OS)
2. ✅ Expected: Error message: "⚠️ Camera mismatch! Requested: X, Got: Y"
3. ✅ Expected: Camera bị release và không hiển thị feed

---

**Status**: ✅ FIXED - Ready for testing in Electron app
