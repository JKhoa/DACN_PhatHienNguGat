# 🔧 FIX FALSE POSITIVE DROWSY DETECTION

**Ngày fix:** 10/11/2025  
**Vấn đề:** Tracking box hiển thị "BUỒN NGỦ" khi người dùng đang tỉnh táo (false positive)  
**Nguyên nhân:** Threshold detection quá thấp, temporal smoothing không đủ strict

---

## ❌ VẤN ĐỀ TRƯỚC ĐÂY

### Symptom:
- Tracking box hiển thị "buồn ngủ" màu đỏ ngay cả khi người dùng ngồi thẳng
- Trước đây chỉ hiển thị khi thật sự ngủ gật, giờ trigger quá dễ
- False positive rate cao, gây khó chịu cho người dùng

### Root Cause Analysis:

1. **Pose Detection Thresholds quá thấp:**
   ```python
   # CŨ (QUÁ DỄ TRIGGER):
   angle_thr=25.0,    # 25° head tilt
   drop_h_thr=0.12,   # 12% head drop 
   drop_sw_thr=0.40   # 40% shoulder drop
   ```

2. **Temporal Smoothing không đủ strict:**
   ```python
   # CŨ (QUÁ NHANH CHỐT KẾT QUẢ):
   history_length = 10      # Chỉ 10 frames (2 giây)
   drowsy_threshold = 7     # 7/10 = 70%
   min_frames = 8           # Chỉ cần 8 frames để quyết định
   ```

3. **Logic Decision cho phép quá nhiều awake frames:**
   ```python
   # CŨ (CHO PHÉP 3 AWAKE TRONG 7 DROWSY):
   elif drowsy_count >= 7 and awake_count <= 3:
   ```

---

## ✅ GIẢI PHÁP ĐÃ ÁP DỤNG

### 1. Tăng Pose Detection Thresholds (Stricter)

**File:** `yolo_detector.py` dòng 341-345

```python
# MỚI (KHÓ TRIGGER HẠI):
angle_thr=35.0,    # ⬆ Tăng từ 25° → 35° (chỉ phát hiện cúi đầu rõ ràng)
drop_h_thr=0.18,   # ⬆ Tăng từ 12% → 18% (cần gục đầu sâu hơn)
drop_sw_thr=0.50   # ⬆ Tăng từ 40% → 50% (vai phải rơi rõ ràng)
```

**Impact:**
- ✅ Viết bài (~20-25° head tilt) → KHÔNG trigger
- ✅ Nhìn điện thoại (~25-30°) → KHÔNG trigger  
- ✅ Chỉ ngủ gật thật (~35°+) mới trigger

### 2. Tăng Temporal Smoothing Requirements (Much Stricter)

**File:** `yolo_detector.py` dòng 262-266

```python
# MỚI (CONSERVATIVE HƠN NHIỀU):
history_length = 15       # ⬆ Tăng từ 10 → 15 frames (3 giây thay vì 2 giây)
drowsy_threshold = 12     # ⬆ Tăng từ 7 → 12 frames (80% thay vì 70%)
sleeping_threshold = 13   # ⬆ Tăng từ 8 → 13 frames (87% thay vì 80%)
min_frames = 12          # ⬆ Tăng từ 8 → 12 frames để bắt đầu quyết định
```

**Impact:**
- ✅ Cần cúi đầu liên tục **3 giây** thay vì 2 giây
- ✅ Cần **12/15 frames** drowsy (80%) thay vì 7/10 (70%)
- ✅ Chỉ cho phép **2 awake frames** thay vì 3

### 3. Stricter Decision Logic

**File:** `yolo_detector.py` dòng 381

```python
# MỚI (EXTREMELY CONSERVATIVE):
elif drowsy_count >= 12 and awake_count <= 2:
    # ⬆ Yêu cầu 12+ drowsy frames VÀ chỉ ≤2 awake frames
```

**Impact:**
- ✅ Gần như tất cả frames phải là drowsy
- ✅ Chỉ chấp nhận tối đa 2 frames "nhấp nháy" awake
- ✅ Ưu tiên "awake" nếu có bất kỳ nghi ngờ nào

---

## 📊 SO SÁNH THRESHOLD TRƯỚC/SAU

| Metric | TRƯỚC (Dễ trigger) | SAU (Khó trigger) | Delta |
|--------|-------------------|------------------|-------|
| **Head Tilt Angle** | 25° | 35° | +10° |
| **Head Drop Ratio** | 12% | 18% | +6% |
| **Shoulder Drop** | 40% | 50% | +10% |
| **History Length** | 10 frames (2s) | 15 frames (3s) | +5 frames |
| **Drowsy Threshold** | 7/10 (70%) | 12/15 (80%) | +10% |
| **Max Awake Allowed** | 3 frames | 2 frames | -1 frame |
| **Min History for Decision** | 8 frames | 12 frames | +4 frames |

---

## 🧪 EXPECTED BEHAVIOR AFTER FIX

### ✅ KHÔNG trigger false positive khi:
1. **Viết bài:** Cúi đầu nhẹ 20-25° → awake
2. **Đọc sách:** Nhìn xuống 15-20° → awake  
3. **Dùng điện thoại:** Cúi 25-30° trong thời gian ngắn → awake
4. **Nói chuyện:** Quay đầu qua lại → awake
5. **Nhấc đầu thỉnh thoảng:** Trong quá trình làm bài → awake

### ✅ VẪN trigger drowsy khi:
1. **Ngủ gật thật:** Đầu gục 35°+ trong 3+ giây → drowsy
2. **Buồn ngủ nặng:** Đầu rơi sâu 18%+ liên tục → drowsy
3. **Gục đầu xuống bàn:** Drop > 22% → sleeping

---

## 🔍 DEBUGGING & MONITORING

### Test Scenarios để xác minh:

1. **Normal Study Position (AWAKE expected):**
   - Ngồi thẳng, viết bài 2-3 phút
   - Expected: Box XANH "TỈNH" suốt thời gian

2. **Light Head Movement (AWAKE expected):**
   - Nhìn xuống giấy, lên bảng, qua lại 1-2 phút
   - Expected: Box XANH "TỈNH", không có flicker

3. **Simulated Real Drowsiness (DROWSY expected):**
   - Cúi đầu sâu 40°+, giữ 5+ giây
   - Expected: Box ĐỎ "BUỒN NGỦ" sau 3-4 giây

4. **Recovery Test (AWAKE expected):**
   - Từ drowsy về ngồi thẳng
   - Expected: Box chuyển XANH sau 2-3 giây

### Console Logs để kiểm tra:

```bash
# Check pose detection thresholds:
[POSE] → Bình thường (angle=22.1°, drop_h=0.08, drop_sw=0.35)
# ✅ angle < 35° AND drop_h < 0.18 → awake

# Check temporal smoothing:
[TEMPORAL] Person 1: awake[12], drowsy[3], sleeping[0] → AWAKE
# ✅ drowsy_count=3 < 12 → awake

# Check decision logic:
[DECISION] Person 1: drowsy[12], awake[2] → DROWSY ⚠
# ✅ 12 >= 12 AND 2 <= 2 → drowsy confirmed
```

---

## 📝 FILES MODIFIED

1. **`yolo_detector.py` (Line 262-266):** Temporal smoothing parameters
2. **`yolo_detector.py` (Line 341-345):** Pose detection thresholds  
3. **`yolo_detector.py` (Line 371-381):** Decision logic conditions

---

## 🚀 DEPLOYMENT

**Cách áp dụng fix:**

1. ✅ Code đã được update
2. Restart backend: `python start_python_backend.py`
3. Test với các scenario trên
4. Monitor false positive rate trong 1-2 ngày

**Rollback plan:** 
Nếu threshold quá strict (miss real drowsy cases), có thể điều chỉnh:
- `angle_thr`: 35° → 30°
- `drop_h_thr`: 0.18 → 0.15
- `drowsy_threshold`: 12 → 10

---

## ✅ VERIFICATION CHECKLIST

- [ ] Backend restart thành công
- [ ] Normal study position → Box XANH 
- [ ] Light head movement → Box XANH (no flicker)
- [ ] Real drowsiness (5s+ head drop) → Box ĐỎ
- [ ] Recovery from drowsy → Box XANH
- [ ] No false positives trong 30 phút test
- [ ] User feedback tích cực (không còn khó chịu với false alerts)

**Kết quả mong đợi:** Significant reduction trong false positive rate while maintaining true positive detection capability.