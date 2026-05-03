# 🚀 Frontend WebSocket Improvements

## Các cải tiến đã thực hiện

### ✅ 1. Environment Variables cho URLs
- Tạo file `.env` và `.env.example` với các biến môi trường:
  - `VITE_WS_DETECT_URL` - WebSocket detection endpoint
  - `VITE_WS_CAMERA_URL` - WebSocket camera endpoint  
  - `VITE_API_BASE_URL` - HTTP API base URL
  - Các config cho reconnection và health check

- Tạo `src/config/env.ts` để centralize environment config với type safety

### ✅ 2. Chi tiết hóa Types cho Persons
- Tạo `src/types/detection.ts` với các interface chi tiết:
  - `Keypoint` - Tọa độ keypoint với confidence và visible
  - `Person` - Thông tin đầy đủ về person với typed fields
  - `DetectionResult` - Type-safe detection result từ backend
  - `CameraUpdate` - Type-safe camera update message

### ✅ 3. Health Check Monitoring
- Thêm health check vào cả 2 WebSocket clients:
  - Monitor `pong` responses từ backend
  - Auto reconnect nếu không nhận được pong trong timeout period
  - Configurable interval và timeout qua env vars

### ✅ 4. Memory Cleanup
**wsDetection.ts:**
- Thêm `stopHealthCheck()` để clear interval
- `close()` method cleanup:
  - Stop health check interval
  - Remove all event listeners
  - Disconnect socket
  - Clear all references (conf, preprocess, callbacks)

**wsCamera.ts:**
- Thêm `disconnect()` method:
  - Stop health check interval
  - Clear all handlers Map
  - Remove all event listeners
  - Disconnect socket
  - Clear callbacks
- Thêm `getSubscriptionCount()` để monitor active subscriptions

### ✅ 5. Connection Status Tracking
- Thêm `isConnected` getter cho cả 2 clients
- Thêm `onStatusChange(callback)` để monitor connection changes
- Tích hợp vào CameraCard để log status changes

### ✅ 6. API Helper Utility
- Tạo `src/lib/api.ts` - Centralized API helper:
  - `api.get(endpoint)` - GET request
  - `api.post(endpoint, data)` - POST request
  - `api.put(endpoint, data)` - PUT request
  - `api.delete(endpoint)` - DELETE request
  - `api.getURL(endpoint)` - Get full URL for downloads
  - Tự động build URL từ `ENV.API_BASE_URL`

## 📁 File Structure

```
Desktop UI for Drowsiness Detection/
├── .env                          # ✅ NEW - Environment variables
├── .env.example                  # ✅ NEW - Example env file
├── src/
│   ├── config/
│   │   └── env.ts               # ✅ NEW - Environment config
│   ├── types/
│   │   └── detection.ts         # ✅ NEW - Detection types
│   ├── lib/
│   │   ├── api.ts               # ✅ NEW - API helper
│   │   ├── wsDetection.ts       # ✅ UPDATED - Health check + cleanup
│   │   └── wsCamera.ts          # ✅ UPDATED - Health check + cleanup
│   └── components/
│       └── CameraCard.tsx       # ✅ UPDATED - Use new types
```

## 🔧 Cách sử dụng

### 1. Environment Variables
```typescript
// src/config/env.ts
import { ENV } from '../config/env';

console.log(ENV.WS_DETECT_URL);  // http://127.0.0.1:5000/ws/detect
console.log(ENV.API_BASE_URL);   // http://127.0.0.1:5000
```

### 2. API Helper
```typescript
// Old way (hardcoded)
const response = await fetch('http://127.0.0.1:5000/api/cameras');

// ✅ New way (use API helper)
import { api } from '../lib/api';
const response = await api.get('api/cameras');
```

### 3. WebSocket Health Check
```typescript
// Health check tự động chạy khi connect
const client = new DetectionWSClient();
client.connect((msg) => {
  // Handle message
});

// Health check sẽ:
// - Monitor pong responses mỗi 5 giây (configurable)
// - Auto reconnect nếu không nhận pong trong 10 giây (configurable)
```

### 4. Connection Status Monitoring
```typescript
const client = new DetectionWSClient();

// Register status change callback
client.onStatusChange((connected) => {
  console.log('Connection status:', connected ? 'Connected' : 'Disconnected');
});

// Check connection status
if (client.isConnected) {
  // Send data
}
```

### 5. Memory Cleanup
```typescript
// Cleanup khi component unmount
useEffect(() => {
  const client = new DetectionWSClient();
  client.connect(handleResult);
  
  return () => {
    client.close(); // ✅ Proper cleanup
  };
}, []);

// Cleanup camera subscriptions
useEffect(() => {
  const unsubscribe = wsCamera.subscribe(cameraId, handleUpdate);
  
  return () => {
    unsubscribe(); // ✅ Proper cleanup
  };
}, [cameraId]);
```

## 🎯 Lợi ích

1. **Dễ config cho production:**
   - Chỉ cần thay đổi `.env` thay vì search/replace trong code
   - Có thể dùng `.env.production` cho production build

2. **Type Safety tốt hơn:**
   - TypeScript catch errors lúc compile time
   - Autocomplete đầy đủ cho Person properties
   - Không còn `any[]` cho persons

3. **Reliability cao hơn:**
   - Health check tự động phát hiện stale connections
   - Auto reconnect khi connection die
   - Proper cleanup tránh memory leaks

4. **Maintainability:**
   - Centralized API calls qua `api` helper
   - Easy to add interceptors, auth headers, etc.
   - Consistent error handling

5. **Performance:**
   - WebSocket-only transport (no polling fallback)
   - Proper cleanup release resources
   - Monitor subscription count

## 🔄 Migration Guide

### Migrate API calls
```typescript
// Before
const response = await fetch('http://127.0.0.1:5000/api/cameras');

// After
import { api } from '../lib/api';
const response = await api.get('api/cameras');
```

### Use typed Person
```typescript
// Before
const persons = msg.persons as any[];

// After
import { Person } from '../types/detection';
const persons = msg.persons as Person[];
// Now you have full autocomplete and type checking
```

### Cleanup on unmount
```typescript
// Before
useEffect(() => {
  const client = new DetectionWSClient();
  client.connect(handleResult);
  // ❌ No cleanup
}, []);

// After
useEffect(() => {
  const client = new DetectionWSClient();
  client.connect(handleResult);
  
  return () => {
    client.close(); // ✅ Proper cleanup
  };
}, []);
```

## 📝 TODO (Optional improvements)

- [ ] Migrate tất cả API calls trong components sang dùng `api` helper
- [ ] Thêm request/response interceptors cho logging
- [ ] Thêm retry logic cho failed API calls
- [ ] Implement request cancellation với AbortController
- [ ] Add rate limiting cho WebSocket sends
- [ ] Add metrics tracking (connection uptime, message count, etc.)

## ✅ Testing

### Test WebSocket connection
```bash
# Start backend
python start_python_backend.py

# Start frontend
cd "Desktop UI for Drowsiness Detection"
npm run dev
```

### Test health check
1. Start frontend + backend
2. Open DevTools console
3. Kill backend process
4. Observe `[WS] Connection might be stale (no pong), reconnecting...`
5. Restart backend
6. Observe auto reconnection

### Test cleanup
1. Open component with WebSocket
2. Monitor Chrome DevTools Memory tab
3. Navigate away (unmount component)
4. Force GC
5. Verify no memory leaks

## 📚 References

- Socket.IO Client Docs: https://socket.io/docs/v4/client-api/
- Vite Environment Variables: https://vitejs.dev/guide/env-and-mode.html
- TypeScript Best Practices: https://www.typescriptlang.org/docs/handbook/2/everyday-types.html
