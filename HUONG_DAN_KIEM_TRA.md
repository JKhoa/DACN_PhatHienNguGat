# 🎯 HƯỚNG DẪN KIỂM TRA PHÁT HIỆN NGỦ GẬT - TRACKING BOX ĐỎ

## ✅ Trạng thái hiện tại
- ✅ Backend đang chạy tại `http://127.0.0.1:5000`
- ✅ Frontend Electron app đã mở
- ✅ YOLO detector đã khởi tạo thành công
- ✅ WebSocket `/ws/detect` và `/ws/camera` sẵn sàng

## 🔧 Đã sửa lỗi
**Vấn đề ban đầu:** Chỉ hiển thị tracking box xanh "TỈNH", không có box đỏ "BUỒN NGỦ"

**Nguyên nhân:** 
- Drowsiness analysis chạy TRƯỚC tracking
- Mỗi frame có `id` mới → temporal smoothing không tích lũy được lịch sử
- Luôn kết luận "awake" vì chưa đủ frames

**Đã fix trong `yolo_detector.py`:**
```python
# CŨ (SAI):
create PersonDetection → analyze_drowsiness → tracker.update()

# MỚI (ĐÚNG):
create PersonDetection → tracker.update() → analyze_drowsiness (dùng track_id ổn định)
```

## 📋 CÁCH KIỂM TRA (3 BƯỚC)

### Bước 1: Thêm Camera trong UI

1. Trong cửa sổ Electron đã mở, tìm nút **"Add Camera"** hoặc **"+"**
2. Chọn:
   - **Camera Type**: Webcam
   - **Camera ID**: `webcam` hoặc `cam1`
   - **Device**: Chọn webcam của bạn từ dropdown
3. Click **"Add"** hoặc **"Save"**
4. Camera sẽ xuất hiện trong danh sách

### Bước 2: Start Camera

1. Tìm camera vừa thêm trong danh sách
2. Click nút **Play** (▶) để bắt đầu stream
3. Chờ 2-3 giây để:
   - Camera khởi động
   - YOLO detector bắt đầu phân tích
   - Tracking boxes xuất hiện

### Bước 3: Test Phát Hiện Ngủ Gật

#### Test A: Tư thế BÌNH THƯỜNG (awake)
1. **Ngồi thẳng** trước camera
2. **Đầu ngẩng**, nhìn thẳng vào camera
3. **Quan sát UI:**
   - ✅ Tracking box **XANH LÁ** (`#00e676`)
   - ✅ Nhãn: **"TỈNH"** hoặc **"#1"** (ID)
   - ✅ Center point màu xanh
   - ✅ Khung focused trên vùng đầu

#### Test B: Tư thế NGỦ GẬT (drowsy/sleeping)
1. **Cúi đầu xuống** như đang ngủ gật
2. **Giữ tư thế 3-5 giây** (cần ít nhất 10-15 frames @ 5 FPS)
3. **Quan sát UI:**
   - ⏳ Chờ 2-3 giây (temporal smoothing)
   - ✅ Tracking box chuyển sang **ĐỎ** (`#ff1744`)
   - ✅ Nhãn chuyển: **"BUỒN NGỦ"**
   - ✅ Badge cảnh báo: **"⚠ 1 học sinh"**

#### Test C: Tỉnh lại (wake up)
1. **Ngẩng đầu lên** trở lại tư thế bình thường
2. **Giữ 3-5 giây**
3. **Quan sát:**
   - ⏳ Chờ temporal smoothing (awake_threshold = 5 frames)
   - ✅ Box chuyển lại **XANH**
   - ✅ Nhãn: **"TỈNH"**

## 🎨 Màu sắc & Hiển thị Chi Tiết

### Tracking Box XANH (Awake)
```
┌────────────────┐
│    #1          │ ← ID người (trắng trên nền đen)
│                │
│       ●        │ ← Center point (xanh + viền trắng)
│                │
│     TỈNH       │ ← State label (trắng trên nền xanh)
└────────────────┘
Color: #00e676 (green)
```

### Tracking Box ĐỎ (Drowsy/Sleeping)
```
┌────────────────┐
│    #1          │ ← ID người (trắng trên nền đen)
│                │
│       ●        │ ← Center point (đỏ + viền trắng)
│                │
│   BUỒN NGỦ     │ ← State label (trắng trên nền đỏ)
└────────────────┘
Color: #ff1744 (red)
Badge: ⚠ 1 học sinh (đỏ, góc trên phải)
```

## 🔍 Troubleshooting

### Không thấy tracking box (xanh hoặc đỏ)?

1. **Kiểm tra camera đã start chưa:**
   - Có thấy video stream không?
   - Nút Play có đổi thành Stop không?

2. **Kiểm tra ánh sáng:**
   - Đủ sáng để nhìn rõ khuôn mặt
   - Không bị backlight (đèn sau lưng)

3. **Kiểm tra backend log:**
   ```
   [YOLO] Detected X persons in frame...
   Person 1: bbox=..., state=awake/drowsy, conf=...
   ```

### Chỉ thấy box XANH, không thấy ĐỎ?

1. **Tăng độ nhạy (sensitivity):**
   - Kéo slider **"Detection sensitivity"** lên **85-95**

2. **Cúi đầu RÕ RÀNG hơn:**
   - Đầu phải cúi sâu, gần chạm bàn
   - Giữ ít nhất **5 giây** để tích lũy frames

3. **Kiểm tra temporal smoothing:**
   - Hiện tại cần **10/15 frames drowsy** để chốt
   - Nếu muốn nhanh hơn → hạ threshold (xem phần dưới)

4. **Xem backend console:**
   - Tìm dòng: `Person X: state=drowsy` hoặc `state=sleeping`
   - Nếu có trong log mà UI vẫn xanh → vấn đề ở frontend mapping

## ⚙️ Điều Chỉnh Độ Nhạy (Nâng Cao)

### Trong UI (Runtime)
- **Detection Sensitivity Slider**: 0-100
  - Thấp (0-50): Chỉ phát hiện khi rất chắc chắn
  - Cao (75-100): Nhạy hơn, phát hiện sớm hơn
  - Recommended: **80-85**

### Trong Code (Build-time)

#### File: `yolo_detector.py` → Class `DrowsinessAnalyzer`

**Tăng tốc độ phản ứng (chuyển state nhanh hơn):**
```python
# Dòng ~228-230
self.history_length = 10  # Giảm từ 15 → 10 frames
self.drowsy_threshold = 6  # Giảm từ 10 → 6 frames (60%)
self.sleeping_threshold = 7  # Giảm từ 12 → 7 frames (70%)
```

**Giảm nhiễu (chuyển state chậm hơn, chính xác hơn):**
```python
self.history_length = 20  # Tăng lên 20 frames
self.drowsy_threshold = 14  # 70% frames
self.sleeping_threshold = 16  # 80% frames
```

Sau khi sửa, **restart backend**:
```powershell
# Stop backend (Ctrl+C trong terminal backend)
# Restart
python "Desktop UI for Drowsiness Detection\python-backend\server_with_tracking_backup.py"
```

## 📊 Kiểm Tra Chi Tiết Qua WebSocket

Nếu muốn xem raw data WebSocket (cho developer):

```powershell
# Terminal riêng
python "Desktop UI for Drowsiness Detection\python-backend\tools\ws_test_subscribe.py" webcam
```

Output mong muốn:
```
[update] cam=webcam persons=1 size=640x480 fps=5.2
  Person 1: state=awake
[update] cam=webcam persons=1 size=640x480 fps=5.4
  Person 1: state=drowsy ← BẮT ĐẦU CÚI ĐẦU
[update] cam=webcam persons=1 size=640x480 fps=5.3
  Person 1: state=drowsy
...
```

## ✅ Xác Nhận Thành Công

Hệ thống hoạt động đúng khi:

1. ✅ **Tư thế bình thường:** Box xanh + "TỈNH"
2. ✅ **Cúi đầu 3-5s:** Box chuyển đỏ + "BUỒN NGỦ"
3. ✅ **Ngẩng đầu:** Box chuyển lại xanh
4. ✅ **Badge cảnh báo:** Xuất hiện khi có người drowsy

## 🎥 Demo Workflow

```
1. Mở app → Backend + Frontend chạy
2. Add camera (webcam) → Start
3. Ngồi thẳng → Box xanh "TỈNH" xuất hiện
4. Cúi đầu → Sau 2-3s, box đỏ "BUỒN NGỦ"
5. Ngẩng đầu → Sau 2s, box xanh "TỈNH"
```

## 📝 Notes

- **FPS:** Backend xử lý ~5-6 FPS (để tiết kiệm CPU)
- **Latency:** ~200ms WebSocket + ~2s temporal smoothing
- **Độ chính xác:** Tùy thuộc ánh sáng, góc camera, model weights
- **Model:** Hiện dùng `yolo11n-pose.pt` (pretrained generic)
  - Nếu có trained model (`sleepy_pose_*_best.pt`) → độ chính xác tốt hơn

## 🚀 Next Steps

1. **Test với webcam thật** theo hướng dẫn trên
2. **Chụp screenshot** khi thấy box đỏ (để confirm)
3. **Report kết quả:**
   - ✅ Thành công: Thấy box đỏ khi cúi đầu
   - ❌ Chưa thành công: Chỉ thấy box xanh → cần điều chỉnh thêm

---

**App đang chạy sẵn sàng!** Hãy thử ngay trong UI Electron đã mở nhé! 🎯
