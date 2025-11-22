# 🧪 TESTING CHECKLIST - WebSocket Improvements

## ✅ Pre-requisites
- [ ] Backend đang chạy (`python start_python_backend.py`)
- [ ] Frontend dependencies installed (`npm install`)
- [ ] `.env` file exists với correct URLs

---

## 📋 Test Cases

### 1. Environment Variables
**Goal:** Verify env vars are loaded correctly

**Steps:**
1. Check `.env` file exists
2. Start frontend: `npm run dev`
3. Open browser console
4. Type: `import.meta.env`
4. Verify you see:
   - `VITE_WS_DETECT_URL`
   - `VITE_WS_CAMERA_URL`
   - `VITE_API_BASE_URL`

**Expected Result:**
```
✅ All VITE_* variables are present
✅ URLs point to correct backend (127.0.0.1:5000)
```

---

### 2. WebSocket Connection
**Goal:** Verify WebSocket connects successfully

**Steps:**
1. Start backend + frontend
2. Open DevTools Console
3. Add a webcam camera
4. Start the camera
5. Look for console logs

**Expected Result:**
```
✅ See: "[WS] Creating socket.io client to http://127.0.0.1:5000/ws/detect"
✅ See: "[WS] Connected to /ws/detect"
✅ See: "[CameraCard xxx] WS connection status: ✅ Connected"
✅ See: "[WS] result persons: N fps: XX"
```

---

### 3. Type Safety
**Goal:** Verify TypeScript types work correctly

**Steps:**
1. Open `CameraCard.tsx` in VSCode
2. Find the line with `msg.persons`
3. Hover over `persons`
4. Try to access `persons[0].`

**Expected Result:**
```
✅ Hover shows: Person[]
✅ Autocomplete shows: id, track_id, bbox, confidence, keypoints, etc.
✅ No "any" type
✅ drowsiness_state shows union type: 'awake' | 'drowsy' | 'sleeping'
```

---

### 4. Health Check Monitoring
**Goal:** Verify health check detects stale connections

**Steps:**
1. Start backend + frontend
2. Add webcam, start camera
3. Wait for "Connected" in console
4. **Kill backend process** (Ctrl+C)
5. Wait 10-15 seconds
6. Look at console
7. Restart backend
8. Observe console

**Expected Result:**
```
✅ After ~10s: See "[WS] Connection might be stale (no pong), reconnecting..."
✅ After backend restart: See "[WS] Connected to /ws/detect"
✅ Camera continues working after reconnect
```

---

### 5. Memory Cleanup - Single Component
**Goal:** Verify no memory leaks when unmounting

**Steps:**
1. Open Chrome DevTools → Memory tab
2. Start frontend
3. Take heap snapshot (Baseline)
4. Add webcam, start camera
5. Wait 10 seconds
6. Stop camera, remove camera
7. Force GC (🗑️ icon in Memory tab)
8. Take heap snapshot (After cleanup)
9. Compare snapshots

**Expected Result:**
```
✅ Detached DOM nodes: 0 or minimal
✅ EventListeners removed
✅ No growing memory after GC
✅ Console shows: "Cleaning up WebSocket..." (if you added log)
```

---

### 6. Memory Cleanup - Multiple Cameras
**Goal:** Verify cleanup with multiple cameras

**Steps:**
1. Add 3 IP cameras (or webcams)
2. Start all 3
3. Check memory baseline
4. Stop all cameras
5. Force GC
6. Check memory again

**Expected Result:**
```
✅ All WebSocket connections closed
✅ All subscriptions removed
✅ Memory returns to baseline (±10%)
✅ No orphaned intervals/timers
```

---

### 7. Connection Status Tracking
**Goal:** Verify status callbacks work

**Steps:**
1. Start backend + frontend
2. Add camera, start
3. Look for status logs in console
4. Kill backend
5. Wait 5 seconds
6. Restart backend

**Expected Result:**
```
✅ See: "[CameraCard xxx] WS connection status: ✅ Connected"
✅ After kill: "[CameraCard xxx] WS connection status: ❌ Disconnected"
✅ After restart: "[CameraCard xxx] WS connection status: ✅ Connected"
```

---

### 8. API Helper Utility
**Goal:** Verify API helper works (if migrated)

**Steps:**
1. Open `src/lib/api.ts`
2. In browser console, import: `import { api } from './src/lib/api'`
3. Try: `await api.get('api/cameras')`

**Expected Result:**
```
✅ Request goes to correct URL (check Network tab)
✅ Response received successfully
✅ No CORS errors
```

---

### 9. Stress Test - Reconnection
**Goal:** Verify reconnection is robust

**Steps:**
1. Start backend + frontend
2. Add camera, start
3. Kill backend
4. Wait 5 seconds
5. Restart backend
6. Kill backend again
7. Wait 5 seconds
8. Restart backend
9. Repeat 3-4 times

**Expected Result:**
```
✅ Every time backend restarts, frontend reconnects
✅ No crashes or freezes
✅ Camera continues working after each reconnect
✅ Console shows reconnection attempts
```

---

### 10. Production Build Test
**Goal:** Verify works in production build

**Steps:**
1. Build: `npm run build`
2. Preview: `npm run preview`
3. Test WebSocket connection
4. Test health check
5. Test cleanup

**Expected Result:**
```
✅ Build succeeds without errors
✅ WebSocket connects in production build
✅ All features work same as dev
✅ No console errors
```

---

## 🔍 Visual Inspection Checklist

### In Chrome DevTools

**Console Tab:**
- [ ] No unhandled errors
- [ ] WebSocket connection logs visible
- [ ] Status change logs visible
- [ ] Health check logs visible (if connection drops)

**Network Tab:**
- [ ] WebSocket connection shows as "101 Switching Protocols"
- [ ] WebSocket stays connected (green indicator)
- [ ] No failed requests to backend

**Memory Tab:**
- [ ] Memory usage stable after GC
- [ ] No growing heap size
- [ ] EventListeners decrease after unmount

**Performance Tab:**
- [ ] No long tasks (>50ms)
- [ ] Frame rate stable
- [ ] No memory leaks in timeline

---

## 🐛 Common Issues & Solutions

### Issue 1: WebSocket won't connect
**Symptoms:** No "[WS] Connected" log

**Solutions:**
- [ ] Check backend is running
- [ ] Check `.env` has correct URL
- [ ] Check firewall not blocking port 5000
- [ ] Try `http://localhost:5000` instead of `127.0.0.1`

### Issue 2: Health check not working
**Symptoms:** No "Connection stale" message when killing backend

**Solutions:**
- [ ] Check env vars: `VITE_WS_HEALTH_CHECK_INTERVAL`
- [ ] Wait full timeout period (default 10s)
- [ ] Check console for errors
- [ ] Verify pong events in Network tab

### Issue 3: Types not working
**Symptoms:** Still seeing `any` in VSCode

**Solutions:**
- [ ] Restart TypeScript server (Cmd+Shift+P → "Restart TS Server")
- [ ] Check `src/types/detection.ts` exists
- [ ] Verify import statement is correct
- [ ] Clear VSCode cache

### Issue 4: Memory leak
**Symptoms:** Memory keeps growing

**Solutions:**
- [ ] Verify `close()` called in useEffect cleanup
- [ ] Check `unsubscribe()` called for wsCamera
- [ ] Look for orphaned intervals (use browser profiler)
- [ ] Check for circular references

---

## ✅ Success Criteria

All tests should pass:
- [x] Environment variables load correctly
- [x] WebSocket connects successfully
- [x] Types work with autocomplete
- [x] Health check detects stale connections
- [x] Memory cleanup works (no leaks)
- [x] Status tracking works
- [x] API helper works (if migrated)
- [x] Reconnection is robust
- [x] Production build works

---

## 📊 Performance Benchmarks

Record these metrics for baseline:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Initial connection time | <500ms | ___ | ⏳ |
| Reconnection time | <1000ms | ___ | ⏳ |
| Memory usage (1 camera) | <50MB | ___ | ⏳ |
| Memory usage (4 cameras) | <100MB | ___ | ⏳ |
| Frame send frequency | 5-6 FPS | ___ | ⏳ |
| Health check overhead | <1% CPU | ___ | ⏳ |

---

## 📝 Notes Section

Use this to record issues or observations:

```
Date: ___________
Tester: ___________

Issues found:
1. 
2. 
3. 

Observations:
- 
- 
- 

Questions:
- 
- 
```

---

## 🎯 Final Checklist

Before marking complete:
- [ ] All 10 test cases passed
- [ ] No console errors
- [ ] No memory leaks
- [ ] Performance is acceptable
- [ ] Documentation reviewed
- [ ] Team notified of changes

**Status:** ⏳ Pending Testing

---

**Ready to test!** 🚀

If all tests pass, mark this file as:
**Status:** ✅ All Tests Passed - Ready for Production
