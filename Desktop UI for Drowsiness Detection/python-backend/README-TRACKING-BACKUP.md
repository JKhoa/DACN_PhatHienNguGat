# 🎯 **SERVER VỚI ENHANCED TRACKING - BACKUP**

## ✅ **Mô tả**

File `server_with_tracking_backup.py` là bản backup tích hợp tracking logic từ `yolo-sleepy-allinone-final/gui_app.py`, được tối ưu cho IP camera với khả năng tracking **20+ học sinh**.

## 🔧 **Tính năng chính**

### **1. Enhanced Tracker**
- **Greedy IoU Matching**: Sử dụng greedy matching với IoU threshold 0.35
- **Head-Focused Tracking**: Ưu tiên tracking dựa trên head bbox để tránh overlap
- **Max Age**: 25 frames - giữ track lâu hơn để ổn định
- **Multi-Person Support**: Hỗ trợ tracking 20+ người cùng lúc

### **2. Camera Worker Enhancements**
- **EnhancedCameraWorker**: Worker riêng với tracker per camera
- **Frame-by-frame Tracking**: Tracking mỗi frame với YOLO detection
- **Performance Optimized**: Annotate mỗi 30 frames để tiết kiệm CPU
- **FPS Tracking**: Real-time FPS calculation

### **3. API Endpoints**
Giữ nguyên tất cả API endpoints như `server.py`:
- `GET /api/cameras`: Danh sách camera
- `POST /api/camera/add`: Thêm camera
- `POST /api/camera/<id>/start`: Khởi động với detection
- `GET /api/camera/<id>/detection`: Lấy detection results với tracking IDs
- `GET /api/system/stats`: Thống kê hệ thống

## 📊 **Tracking Output**

### **Detection Result Format:**
```json
{
  "success": true,
  "persons": [
    {
      "id": 1,
      "track_id": 5,  // Persistent tracking ID
      "bbox": [100, 150, 200, 350],
      "head_bbox": [120, 150, 180, 200],
      "confidence": 0.85,
      "drowsiness_state": "drowsy",
      "drowsiness_score": 0.6,
      "keypoints": [...]
    }
  ],
  "fps": 30.0,
  "frame_width": 1920,
  "frame_height": 1080
}
```

## 🚀 **Cách sử dụng**

### **Option 1: Auto-switch (Recommended)**
Electron sẽ tự động dùng `server_with_tracking_backup.py` nếu file tồn tại.

### **Option 2: Manual Switch**
Trong `electron/main.js`, thay đổi:
```javascript
const serverPath = path.join(__dirname, '..', 'python-backend', 'server_with_tracking_backup.py');
```

### **Option 3: Command Line**
```bash
cd "Desktop UI for Drowsiness Detection/python-backend"
python server_with_tracking_backup.py
```

## ⚙️ **Configuration**

### **Tracker Parameters:**
- `iou_thr=0.35`: IoU threshold cho matching (có thể điều chỉnh 0.3-0.4)
- `max_age=25`: Số frames tối đa giữ track không match (có thể tăng lên 30-40 cho ổn định hơn)

### **Performance Tuning:**
- Annotate interval: Mỗi 30 frames (có thể tăng lên 60 để tiết kiệm CPU)
- FPS limit: ~30 FPS max per camera
- Detection interval: Mỗi frame (có thể giảm xuống mỗi 2-3 frames nếu cần)

## 📈 **Performance cho 20+ người**

### **Expected Performance:**
- **20-30 người**: 15-25 FPS (tùy hardware)
- **30-50 người**: 10-20 FPS (tùy hardware)
- **50+ người**: 8-15 FPS (cần GPU hoặc giảm detection interval)

### **Optimization Tips:**
1. **Reduce Detection Interval**: Chỉ detect mỗi 2-3 frames
2. **Lower Resolution**: Giảm frame resolution (640x360 thay vì 1920x1080)
3. **GPU Acceleration**: Sử dụng GPU cho YOLO inference
4. **Head-only Tracking**: Chỉ track head region thay vì full body

## 🔄 **Fallback**

Nếu `server_with_tracking_backup.py` không tìm thấy, Electron sẽ tự động fallback về `server.py` mặc định.

## ✅ **Testing**

Test với IP camera:
1. Start server: `python server_with_tracking_backup.py`
2. Add IP camera qua UI
3. Start camera với `enable_detection=true`
4. Check `/api/camera/<id>/detection` để xem tracking IDs
5. Verify tracking ổn định với nhiều người

## 📝 **Notes**

- File này là backup, không thay thế `server.py` mặc định
- Có thể chạy song song với `server.py` trên port khác để test
- Tracking IDs sẽ persistent qua các frames nếu matching tốt
- Head bbox matching giúp tránh overlap khi có nhiều người




