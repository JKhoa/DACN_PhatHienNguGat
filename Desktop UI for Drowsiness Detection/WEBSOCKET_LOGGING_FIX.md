# ✅ WebSocket Logging Fix - Đã Sửa!

## 🔥 **Vấn Đề:**
WebSocket detection (`/ws/detect`) **KHÔNG LOG** drowsiness events vào logger → Log panel trống!

## ✅ **Giải Pháp:**
Đã thêm **state tracking + logging support** vào WebSocket handler

---

## 📝 **Changes Made:**

### **File Modified:** `server_with_tracking_backup.py`

### **1. Added State Tracking Variables** (Lines ~1406-1412)
```python
# 🔥 WebSocket state tracking (per track_id)
_ws_per_id_state = {}  # track_id -> "awake" | "drowsy" | "sleeping"  
_ws_per_id_sleep_start = {}  # track_id -> timestamp
_ws_sleep_frames_required = 8  # ~1.6s at 5fps (same as camera worker)
_ws_awake_frames_required = 5  # ~1s at 5fps
_ws_per_id_sleep_count = {}  # track_id -> int
_ws_per_id_awake_count = {}  # track_id -> int
```

### **2. Added State Tracking Logic** (Lines ~1475-1570)
```python
# 🔥 Track state changes and log drowsiness events
now = time.time()
for p in det.persons:
    tid = int(getattr(p, 'track_id', getattr(p, 'id', 0)) or 0)
    state_now = str(getattr(p, 'drowsiness_state', 'awake') or 'awake')
    
    # Initialize tracking for new person
    if tid not in _ws_per_id_state:
        _ws_per_id_state[tid] = 'awake'
        _ws_per_id_sleep_count[tid] = 0
        _ws_per_id_awake_count[tid] = 0
    
    prev_state = _ws_per_id_state[tid]
    
    # Update counters
    if state_now in ('drowsy', 'sleeping'):
        _ws_per_id_sleep_count[tid] += 1
        _ws_per_id_awake_count[tid] = 0
    else:  # awake
        _ws_per_id_awake_count[tid] += 1
        _ws_per_id_sleep_count[tid] = 0
    
    # Temporal smoothing (same logic as camera worker)
    sleep_cnt = _ws_per_id_sleep_count[tid]
    awake_cnt = _ws_per_id_awake_count[tid]
    
    # Determine effective state
    eff_state = prev_state
    if prev_state in ('drowsy', 'sleeping'):
        if state_now == 'awake' and awake_cnt >= _ws_awake_frames_required:
            eff_state = 'wake_up'
    elif prev_state == 'wake_up':
        if awake_cnt >= _ws_awake_frames_required:
            eff_state = 'awake'
    else:
        if state_now in ('drowsy', 'sleeping') and sleep_cnt >= _ws_sleep_frames_required:
            eff_state = state_now
    
    # 🔥 LOG STATE TRANSITIONS
    if eff_state != prev_state:
        if eff_state in ('drowsy', 'sleeping'):
            # Started drowsiness
            _ws_per_id_sleep_start[tid] = now
            append_log({
                'camera_id': cam_id,
                'track_id': tid,
                'type': 'sleepy' if eff_state == 'drowsy' else 'head_down',
                'state': 'Ngủ gật' if eff_state == 'drowsy' else 'Gục xuống bàn',
                'ts': now
            })
            
            # 🔥 Log to drowsiness logger
            if LOGGER_AVAILABLE:
                logger = get_global_logger()
                # Register webcam if not already
                if not hasattr(logger, '_registered_webcam'):
                    logger.register_camera(cam_id, "WebSocket Camera")
                    logger._registered_webcam = True
                
                logger.update_student_state(cam_id, tid, True)
                app.logger.info(f"[WS] 🔴 Học sinh #{tid} BẮT ĐẦU {eff_state}")
        
        elif eff_state == 'wake_up':
            # Waking up
            dur = now - _ws_per_id_sleep_start.get(tid, now)
            if tid in _ws_per_id_sleep_start:
                del _ws_per_id_sleep_start[tid]
            
            append_log({
                'camera_id': cam_id,
                'track_id': tid,
                'type': 'wake_up',
                'state': 'Thức dậy',
                'duration': dur,
                'ts': now
            })
            
            # 🔥 Log wake up
            if LOGGER_AVAILABLE:
                logger = get_global_logger()
                logger.update_student_state(cam_id, tid, False)
                app.logger.info(f"[WS] 🟢 Học sinh #{tid} THỨC DẬY sau {dur:.1f}s")
    
    # Update state
    _ws_per_id_state[tid] = eff_state
```

---

## 🧪 **How to Test:**

### **Step 1: Start App**
```powershell
cd "Desktop UI for Drowsiness Detection"
npm start
```

### **Step 2: Trigger Drowsy Detection**
**IMPORTANT:** Bạn cần GỤC ĐẦU XUỐNG **LIÊN TỤC 2 GIÂY** để trigger!

- ❌ **SAI:** Gục đầu 0.5s rồi ngẩng lên → KHÔNG ĐỦ frames
- ✅ **ĐÚNG:** Gục đầu xuống giữ nguyên 2-3 giây → ĐỦ 8 frames ở 5fps

**Test scenarios:**

1. **Ngồi bình thường 5 giây:**
   - Mong đợi: KHÔNG có log nào
   - State: `awake` (xanh lá)

2. **Gục đầu xuống GIỮ NGUYÊN 2-3 giây:**
   - Mong đợi: Sau ~1.6s thấy console log: `[WS] 🔴 Học sinh #1 BẮT ĐẦU drowsy/sleeping`
   - State: `drowsy` (cam) hoặc `sleeping` (đỏ)
   - Log Panel: Hiển thị "Ngủ gật" hoặc "Gục xuống bàn"

3. **Ngẩng đầu lên GIỮ NGUYÊN 1 giây:**
   - Mong đợi: Sau ~1s thấy console log: `[WS] 🟢 Học sinh #1 THỨC DẬY sau X.Xs`
   - State: `wake_up` → `awake`
   - Log Panel: Hiển thị "Thức dậy" với duration

### **Step 3: Check Logs**

#### **A. Backend Console (Python terminal):**
Nên thấy:
```
[WS] 🔴 Học sinh #1 BẮT ĐẦU drowsy (camera: webcam)
[WS] 🟢 Học sinh #1 THỨC DẬY sau 2.5s (camera: webcam)
```

#### **B. API Endpoint:**
```powershell
$logs = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/logs"
$logs.logs | Select-Object -Last 5
```

Nên thấy:
```json
{
  "camera_id": "webcam",
  "track_id": 1,
  "type": "sleepy",
  "state": "Ngủ gật",
  "ts": 1699999999.123
},
{
  "camera_id": "webcam",
  "track_id": 1,
  "type": "wake_up",
  "state": "Thức dậy",
  "duration": 2.5,
  "ts": 1699999999.456
}
```

#### **C. Drowsiness Logger (SQLite):**
```powershell
cd python-backend
python -c "from drowsiness_logger import get_global_logger; logger = get_global_logger(); events = logger.get_all_events(); print(f'Total events: {len(events)}'); print(events[-2:])"
```

#### **D. Log Panel (Frontend):**
Vào tab "Dashboard" hoặc "Logs" → Nên thấy:
- **Recent Events**: Danh sách events mới nhất
- **Student #1**: "Ngủ gật" hoặc "Gục xuống bàn"
- **Duration**: X.Xs

---

## ⚙️ **Tuning Parameters:**

Nếu cần điều chỉnh độ nhạy:

### **File:** `server_with_tracking_backup.py`, lines ~1408-1410

```python
_ws_sleep_frames_required = 8  # ~1.6s at 5fps
_ws_awake_frames_required = 5  # ~1s at 5fps
```

**Nếu quá khó trigger (cần gục lâu hơn):**
```python
_ws_sleep_frames_required = 6  # ~1.2s (dễ hơn)
_ws_awake_frames_required = 4  # ~0.8s
```

**Nếu quá dễ trigger (false positives):**
```python
_ws_sleep_frames_required = 10  # ~2s (khó hơn)
_ws_awake_frames_required = 6   # ~1.2s
```

---

## 🔍 **Troubleshooting:**

### **Problem 1: "Console không thấy log [WS] 🔴"**
**Nguyên nhân:** Chưa đủ frames liên tục  
**Giải pháp:** Gục đầu GIỮ NGUYÊN 2-3 giây, KHÔNG ngẩng lên giữa chừng

### **Problem 2: "Log Panel trống"**
**Nguyên nhân:** Frontend chưa poll API hoặc API trả về empty  
**Kiểm tra:**
```powershell
curl http://127.0.0.1:5000/api/logs
```
Nếu `{"success": true, "logs": []}` → Chưa trigger drowsy event

### **Problem 3: "Backend crash khi gửi frame"**
**Nguyên nhân:** Runtime error trong tracking logic  
**Kiểm tra:** Xem Python console có traceback không

### **Problem 4: "State nhảy liên tục giữa drowsy và awake"**
**Nguyên nhân:** Temporal smoothing chưa đủ mạnh  
**Giải pháp:** Tăng `_ws_sleep_frames_required` và `_ws_awake_frames_required`

---

## 📊 **Expected Behavior:**

| Action | Frames Required | Expected Log | Backend Console |
|--------|----------------|--------------|-----------------|
| Ngồi bình thường | N/A | None | None |
| Gục đầu 0.5s | 0/8 frames | None | None |
| Gục đầu 1.0s | 5/8 frames | None | None |
| **Gục đầu 1.6s** | **8/8 frames** | **"Ngủ gật"** | **🔴 BẮT ĐẦU drowsy** |
| Tiếp tục gục 3.0s | N/A | None (already logged) | None |
| Ngẩng đầu 0.5s | 2/5 frames | None | None |
| **Ngẩng đầu 1.0s** | **5/5 frames** | **"Thức dậy"** | **🟢 THỨC DẬY** |

---

## ✅ **Success Criteria:**

- ✅ Backend console shows `[WS] 🔴 Học sinh #X BẮT ĐẦU drowsy` when gục đầu 2s
- ✅ Backend console shows `[WS] 🟢 Học sinh #X THỨC DẬY sau X.Xs` when ngẩng đầu
- ✅ API `/api/logs` returns events with `type: "sleepy"` and `type: "wake_up"`
- ✅ Drowsiness logger SQLite database has events
- ✅ Log Panel (Frontend) displays events in real-time
- ✅ Dashboard stats update (Drowsy Students count)

---

## 🎯 **Next Steps:**

1. **Test với WebSocket detection:**
   - Mở app
   - Gục đầu xuống 2-3 giây
   - Kiểm tra console + Log Panel

2. **Test với Camera Worker (optional):**
   - Vào tab "Camera"
   - Click "Add Camera" → Chọn webcam
   - Click "Start"
   - Gục đầu xuống 2-3 giây
   - Kiểm tra console + Log Panel

3. **Verify logging integration:**
   - Check SQLite database: `drowsiness_logs/events.db`
   - Check console logs: Backend terminal
   - Check frontend: Log Panel + Dashboard

---

**Status:** ✅ FIXED - WebSocket now logs drowsiness events!  
**Date:** November 10, 2025  
**Version:** 2.1 - WebSocket Logging Support
