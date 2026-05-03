# 🧠 YOLO Drowsiness Detection Integration Guide

## 📋 Tổng Quan

Hệ thống YOLO detection đã được tích hợp hoàn chỉnh vào ứng dụng Desktop UI để phát hiện ngủ gật học sinh thời gian thực.

## 🚀 Tính Năng Mới

### ✅ YOLO Detection System
- **Pose Detection**: Sử dụng YOLO11n-pose model để phát hiện tư thế người
- **Drowsiness Analysis**: Phân tích các chỉ số ngủ gật (mắt nhắm, đầu nghiêng, tư thế)
- **Real-time Processing**: Xử lý thời gian thực với FPS cao
- **Multi-person Tracking**: Theo dõi nhiều học sinh cùng lúc

### ✅ Backend Integration
- **YOLO Detector Class**: Class chuyên dụng cho detection
- **Drowsiness Analyzer**: Logic phân tích trạng thái ngủ gật
- **API Endpoints**: RESTful API cho detection results
- **Camera Worker Updates**: Tích hợp detection vào camera streams

### ✅ Frontend Integration
- **YOLO Detection Panel**: UI component hiển thị kết quả detection
- **Real-time Updates**: Cập nhật kết quả detection theo thời gian thực
- **Toggle Controls**: Bật/tắt detection cho từng camera
- **Visual Indicators**: Hiển thị trạng thái ngủ gật với màu sắc

## 🛠️ Cài Đặt và Sử Dụng

### 1. Cài Đặt Dependencies
```bash
cd "Desktop UI for Drowsiness Detection/python-backend"
pip install -r requirements.txt
```

### 2. Kiểm Tra Model
Đảm bảo file `yolo11n-pose.pt` có trong thư mục `python-backend/`

### 3. Test YOLO Detection
```bash
# Chạy test script
python test_yolo_detection.py

# Hoặc sử dụng batch file
TEST-YOLO-DETECTION.bat
```

### 4. Chạy Ứng Dụng
```bash
# Terminal 1: Start Python backend
cd "Desktop UI for Drowsiness Detection/python-backend"
python server.py

# Terminal 2: Start Desktop app
cd "Desktop UI for Drowsiness Detection"
npm run electron
```

## 🎯 Cách Sử Dụng YOLO Detection

### 1. Khởi Tạo Detection
- App sẽ tự động khởi tạo YOLO detector khi khởi động
- Kiểm tra notification để xác nhận khởi tạo thành công

### 2. Bật Detection cho Camera
1. Mở camera card
2. Click menu "..." → "Hiện YOLO Detection"
3. Toggle switch "Detection Enabled" để bật detection
4. Xem kết quả detection trong panel

### 3. Theo Dõi Kết Quả
- **Performance Stats**: FPS và processing time
- **Person Detection**: Số lượng người được phát hiện
- **Drowsiness States**: Trạng thái từng người (awake/drowsy/sleeping)
- **Keypoints Info**: Thông tin về các điểm keypoint

## 📊 Detection Results

### Drowsiness States
- **🟢 AWAKE**: Học sinh tỉnh táo (drowsiness_score < 0.4)
- **🟠 DROWSY**: Học sinh buồn ngủ (drowsiness_score 0.4-0.7)
- **🔴 SLEEPING**: Học sinh ngủ gật (drowsiness_score > 0.7)

### Detection Indicators
- **Eyes Closed**: Mắt nhắm (confidence < 0.3)
- **Head Tilted**: Đầu nghiêng (> 30 độ)
- **Head Down**: Đầu cúi xuống

### Performance Metrics
- **FPS**: Frames per second của detection
- **Processing Time**: Thời gian xử lý mỗi frame (ms)
- **Confidence**: Độ tin cậy của detection

## 🔧 API Endpoints

### Detection Control
- `POST /api/detection/initialize` - Khởi tạo YOLO detector
- `POST /api/camera/{id}/detection/toggle` - Bật/tắt detection

### Detection Results
- `GET /api/camera/{id}/detection` - Lấy kết quả detection
- `GET /api/camera/{id}/stream?annotated=true` - Stream với annotations

### System Stats
- `GET /api/system/stats` - Thống kê hệ thống bao gồm detection stats

## 🎨 UI Components

### YOLODetectionPanel
- **Location**: `src/components/YOLODetectionPanel.tsx`
- **Features**: 
  - Real-time detection results
  - Performance metrics
  - Person detection details
  - Drowsiness state indicators

### CameraCard Integration
- **New Menu Item**: "Hiện YOLO Detection"
- **Toggle Control**: Enable/disable detection
- **Visual Integration**: Seamless integration với existing UI

## 🐛 Troubleshooting

### Common Issues

1. **YOLO Not Available**
   ```
   Error: Ultralytics YOLO is required
   Solution: pip install ultralytics
   ```

2. **Model Not Found**
   ```
   Error: Model file not found
   Solution: Ensure yolo11n-pose.pt is in python-backend/
   ```

3. **Detection Not Working**
   ```
   Check: Camera is running and detection is enabled
   Check: Backend is running on port 5000
   Check: YOLO detector is initialized
   ```

4. **Low FPS**
   ```
   Solution: Reduce detection frequency
   Solution: Use smaller model (yolo11n-pose)
   Solution: Optimize camera resolution
   ```

### Debug Steps
1. Check backend logs for YOLO initialization
2. Test detection with `test_yolo_detection.py`
3. Verify camera stream is working
4. Check browser console for API errors

## 📈 Performance Optimization

### Model Selection
- **yolo11n-pose**: Fastest, lower accuracy
- **yolo11s-pose**: Balanced speed/accuracy
- **yolo11m-pose**: Higher accuracy, slower

### Detection Settings
- **Inference FPS**: Adjust per camera (default: 10 FPS)
- **Confidence Threshold**: Adjust detection sensitivity
- **Keypoint Threshold**: Adjust keypoint visibility

### Hardware Requirements
- **CPU**: Multi-core recommended
- **GPU**: CUDA support for faster inference
- **RAM**: 4GB+ recommended
- **Storage**: 2GB+ for models

## 🔮 Future Enhancements

### Planned Features
- [ ] Custom model training interface
- [ ] Advanced drowsiness algorithms
- [ ] Multi-camera synchronization
- [ ] Export detection data
- [ ] Alert system integration
- [ ] Performance analytics dashboard

### Model Improvements
- [ ] Fine-tuned models for classroom scenarios
- [ ] Multi-pose detection
- [ ] Emotion recognition integration
- [ ] Attention level analysis

## 📚 Technical Details

### Architecture
```
Frontend (React) ←→ Backend (Flask) ←→ YOLO Detector
     ↓                    ↓                    ↓
UI Components      API Endpoints        Detection Engine
     ↓                    ↓                    ↓
Real-time Display  Camera Workers      Drowsiness Analysis
```

### Data Flow
1. Camera captures frame
2. CameraWorker processes frame
3. YOLO Detector analyzes frame
4. DrowsinessAnalyzer evaluates poses
5. Results sent to frontend via API
6. UI displays detection results

### Key Classes
- **YOLODetector**: Main detection engine
- **DrowsinessAnalyzer**: Drowsiness analysis logic
- **DetectionResult**: Detection result data structure
- **PersonDetection**: Individual person detection data

---

*Hệ thống YOLO detection đã được tích hợp hoàn chỉnh và sẵn sàng sử dụng!* 🎉
