# 📏 Depth-Aware Bounding Box Scaling

## 🎯 **Tính Năng Mới:**

Bounding box giờ đây **tự động scale theo khoảng cách** từ camera đến người:
- **Người GẦN camera** → Box to, chữ to, line dày
- **Người XA camera** → Box nhỏ, chữ nhỏ, line mỏng

---

## ⚙️ **Cách Hoạt Động:**

### 1. **Ước Tính Khoảng Cách (Depth Estimation)**

Hệ thống ước tính khoảng cách dựa trên **tỷ lệ diện tích bbox** so với frame:

```python
bbox_area = bbox_width × bbox_height
frame_area = frame_width × frame_height
bbox_ratio = bbox_area / frame_area
```

**Phân loại depth levels (5 cấp độ):**

| Depth Level | Bbox Ratio | Khoảng Cách | Mô Tả |
|-------------|-----------|-------------|-------|
| **Level 5** | > 30% | **Very Close** | Người ngồi rất gần camera (<1m) |
| **Level 4** | 15-30% | **Close** | Người ngồi gần (1-2m) |
| **Level 3** | 5-15% | **Medium** | Khoảng cách trung bình (2-4m) |
| **Level 2** | 2-5% | **Far** | Người ngồi xa (4-6m) |
| **Level 1** | < 2% | **Very Far** | Người ngồi rất xa (>6m) |

---

### 2. **Adaptive Scaling Rules**

Các thành phần UI tự động scale theo depth level:

#### **A. Line Thickness (Độ dày viền box)**
```python
line_thickness = max(1, min(4, depth_level))
```
- Level 1 (Very Far): **1 pixel** - Viền mỏng
- Level 3 (Medium): **3 pixels** - Viền vừa
- Level 5 (Very Close): **4 pixels** - Viền dày

#### **B. Font Scale (Kích thước chữ)**
```python
base_font_scale = 0.3 + (depth_level - 1) × 0.1
```
- Level 1 (Very Far): **0.3** - Chữ rất nhỏ
- Level 3 (Medium): **0.5** - Chữ trung bình
- Level 5 (Very Close): **0.7** - Chữ lớn

#### **C. Circle Radius (Bán kính điểm center)**
```python
circle_radius = max(2, min(6, depth_level + 1))
```
- Level 1 (Very Far): **3 pixels** - Điểm nhỏ
- Level 3 (Medium): **4 pixels** - Điểm vừa
- Level 5 (Very Close): **6 pixels** - Điểm lớn

#### **D. Label Padding (Khoảng cách padding)**
```python
padding = max(2, depth_level)
```
- Level 1 (Very Far): **2 pixels** - Padding ít
- Level 5 (Very Close): **5 pixels** - Padding nhiều

---

## 📊 **So Sánh Trước & Sau:**

### ❌ **TRƯỚC (Fixed Size):**
```
Người gần camera:
  └─ Box: 2px viền, chữ 0.5, điểm 4px

Người xa camera:
  └─ Box: 2px viền, chữ 0.5, điểm 4px  ← GIỐNG NHAU (không phân biệt)
```

### ✅ **SAU (Depth-Aware Scaling):**
```
Người gần camera (Level 5):
  └─ Box: 4px viền, chữ 0.7, điểm 6px  ← TO HƠN

Người trung bình (Level 3):
  └─ Box: 3px viền, chữ 0.5, điểm 4px  ← VỪA

Người xa camera (Level 1):
  └─ Box: 1px viền, chữ 0.3, điểm 3px  ← NHỎ HƠN
```

---

## 🎨 **Visual Elements:**

### **1. Bounding Box**
- **Màu sắc:** Không đổi (vẫn dựa trên state: xanh/cam/đỏ)
- **Độ dày:** Adaptive (1-4px)
- **Vị trí:** Head bbox hoặc body bbox

### **2. Person ID Label**
- **Vị trí:** Trên box, center alignment
- **Format:** `#ID` (ví dụ: `#1`, `#2`)
- **Font scale:** Base + 0.2 (to hơn state label)
- **Background:** Đen solid, text trắng

### **3. Depth Badge (NEW!)**
- **Vị trí:** Bên phải ID label
- **Format:** `[Depth Text]` (ví dụ: `[Close]`, `[Far]`)
- **Font scale:** Base × 0.6 (nhỏ hơn ID)
- **Background:** Xám đậm (80,80,80)
- **Text:** Xám nhạt (200,200,200)
- **Điều kiện:** Chỉ hiển thị nếu đủ không gian (không vượt ra ngoài frame)

**Example:**
```
┌─────────────┐
│  #1 [Close] │ ← ID + Depth Badge
└─────────────┘
```

### **4. State Label (DROWSY/SLEEPING)**
- **Vị trí:** Dưới box, center alignment
- **Font scale:** Adaptive base scale
- **Background:** Màu state (cam/đỏ), text trắng
- **Padding:** Adaptive (2-5px)
- **Điều kiện:** Chỉ hiển thị khi state ≠ "awake"

### **5. Center Point**
- **Hình dạng:** Circle (filled)
- **Màu:** State color + white outline
- **Radius:** Adaptive (3-6px)

---

## 🧪 **Cách Test:**

### **Test 1: Di Chuyển Gần Camera**
```
Hành động: Từ từ di chuyển GẦN camera
Mong đợi:
  ✅ Bbox ratio tăng (5% → 10% → 20% → 30%)
  ✅ Depth level tăng (Level 2 → 3 → 4 → 5)
  ✅ Box thickness tăng (2px → 3px → 4px)
  ✅ Font size to lên (0.4 → 0.5 → 0.6 → 0.7)
  ✅ Depth badge: "Far" → "Medium" → "Close" → "Very Close"
```

### **Test 2: Di Chuyển Xa Camera**
```
Hành động: Từ từ lùi XA camera
Mong đợi:
  ✅ Bbox ratio giảm (30% → 20% → 10% → 5% → 2%)
  ✅ Depth level giảm (Level 5 → 4 → 3 → 2 → 1)
  ✅ Box thickness giảm (4px → 3px → 2px → 1px)
  ✅ Font size nhỏ lại (0.7 → 0.6 → 0.5 → 0.4 → 0.3)
  ✅ Depth badge: "Very Close" → "Close" → "Medium" → "Far" → "Very Far"
```

### **Test 3: Nhiều Người Ở Khoảng Cách Khác Nhau**
```
Setup: 
  - Người A: Ngồi gần camera (1m)
  - Người B: Ngồi xa camera (4m)

Mong đợi:
  ✅ Người A: Box dày (4px), chữ to (0.7), badge "Very Close"
  ✅ Người B: Box mỏng (2px), chữ nhỏ (0.4), badge "Far"
  ✅ Phân biệt rõ ràng giữa 2 người
```

---

## 📈 **Lợi Ích:**

### ✅ **1. Improved Visibility**
- Người xa camera: Box nhỏ, ít chiếm diện tích → Dễ nhìn
- Người gần camera: Box lớn, dễ đọc → Dễ theo dõi

### ✅ **2. Better Multi-Person Tracking**
- Phân biệt rõ người gần vs xa
- Ưu tiên attention vào người gần (box nổi bật hơn)
- Tránh cluttered UI khi có nhiều người

### ✅ **3. Enhanced UX**
- Adaptive UI feels more natural
- Matches human perception (gần = to, xa = nhỏ)
- Professional appearance

### ✅ **4. Performance Optimization**
- Người xa: Labels nhỏ hơn → Ít tốn rendering
- Người gần: Chi tiết hơn → Chất lượng cao khi cần

---

## 🔧 **Tuning Parameters:**

Nếu muốn điều chỉnh depth classification:

### **File:** `yolo_detector.py`, lines ~660-680

```python
# Điều chỉnh ngưỡng depth levels
if bbox_ratio > 0.3:  # Very close (30%)
    depth_level = 5
elif bbox_ratio > 0.15:  # Close (15%)
    depth_level = 4
elif bbox_ratio > 0.05:  # Medium (5%)
    depth_level = 3
elif bbox_ratio > 0.02:  # Far (2%)
    depth_level = 2
else:  # Very far (<2%)
    depth_level = 1
```

**Suggestions:**
- **Giảm ngưỡng** → Cần gần hơn mới lên level cao (stricter)
- **Tăng ngưỡng** → Dễ lên level cao hơn (more lenient)

### **Ví dụ - Strict Mode:**
```python
if bbox_ratio > 0.4:  # Very close (40% - rất khó đạt)
    depth_level = 5
elif bbox_ratio > 0.2:  # Close (20%)
    depth_level = 4
# ...
```

### **Ví dụ - Lenient Mode:**
```python
if bbox_ratio > 0.2:  # Very close (20% - dễ đạt)
    depth_level = 5
elif bbox_ratio > 0.1:  # Close (10%)
    depth_level = 4
# ...
```

---

## 🎛️ **Enable/Disable Depth Badge:**

Nếu không muốn hiển thị depth badge `[Close]`, `[Far]`, etc:

### **Option 1: Comment out depth badge code**

**File:** `yolo_detector.py`, lines ~720-740

```python
# 🆕 DEPTH INDICATOR: Show estimated distance (optional - can be disabled)
# ← Thêm # vào đầu các dòng sau để disable
# depth_badge = f"[{depth_text}]"
# depth_font_scale = base_font_scale * 0.6
# ...
# cv2.putText(annotated_frame, depth_badge, ...)
```

### **Option 2: Set condition to always False**

```python
# Only show if there's enough space (don't overlap with edge)
if False:  # ← Đổi thành False để luôn skip
    cv2.rectangle(...)
    cv2.putText(...)
```

---

## 🐛 **Troubleshooting:**

### **Vấn đề 1: "Depth badge overlap với ID label"**
**Nguyên nhân:** Frame nhỏ, không đủ space  
**Giải pháp:** Badge đã có check `depth_x + depth_badge_size[0] + 4 < frame_width`

### **Vấn đề 2: "Box quá dày/mỏng"**
**Nguyên nhân:** Depth thresholds chưa phù hợp  
**Giải pháp:** Điều chỉnh `bbox_ratio` thresholds (xem phần Tuning)

### **Vấn đề 3: "Font quá to/nhỏ"**
**Nguyên nhân:** Font scale range không phù hợp  
**Giải pháp:**
```python
# Thay đổi base font scale formula
base_font_scale = 0.4 + (depth_level - 1) * 0.08  # Nhỏ hơn: 0.4-0.72
# hoặc
base_font_scale = 0.35 + (depth_level - 1) * 0.12  # To hơn: 0.35-0.83
```

### **Vấn đề 4: "Depth level nhảy liên tục"**
**Nguyên nhân:** Bbox size dao động (người di chuyển)  
**Giải pháp:** Thêm temporal smoothing:
```python
# Trong YOLODetector.__init__():
self.depth_history = {}  # track_id -> [depth_levels]

# Trong draw_detections():
if track_id not in self.depth_history:
    self.depth_history[track_id] = []
self.depth_history[track_id].append(depth_level)
if len(self.depth_history[track_id]) > 5:
    self.depth_history[track_id].pop(0)
# Use average
depth_level = int(sum(self.depth_history[track_id]) / len(self.depth_history[track_id]))
```

---

## 📝 **Code Changes Summary:**

### **Modified Files:**
- ✅ `yolo_detector.py` - Lines 642-740 (draw_detections method)

### **New Features:**
1. ✅ Depth estimation from bbox ratio
2. ✅ 5-level depth classification
3. ✅ Adaptive line thickness (1-4px)
4. ✅ Adaptive font scale (0.3-0.7)
5. ✅ Adaptive circle radius (3-6px)
6. ✅ Adaptive label padding (2-5px)
7. ✅ Depth indicator badge `[Very Close]`, `[Close]`, `[Medium]`, `[Far]`, `[Very Far]`

### **Backward Compatible:**
- ✅ Không thay đổi detection logic
- ✅ Không ảnh hưởng tracking
- ✅ Chỉ thay đổi visualization

---

## 🎯 **Use Cases:**

### **1. Classroom Monitoring (Nhiều học sinh)**
```
Người ngồi hàng đầu (gần): Box dày, dễ đọc
Người ngồi hàng giữa: Box trung bình
Người ngồi hàng cuối (xa): Box mỏng, không che khuất
```

### **2. Meeting Room (Vài người)**
```
Người presenter (đứng gần camera): Box lớn, nổi bật
Người ngồi xa: Box nhỏ, ít gây rối
```

### **3. Security Monitoring (Nhiều camera angles)**
```
Wide angle camera → Mọi người đều xa → All boxes nhỏ
Close-up camera → Mọi người gần → All boxes lớn
```

---

## ✅ **Checklist:**

- [x] Depth estimation implemented
- [x] 5-level classification working
- [x] Adaptive line thickness
- [x] Adaptive font scaling
- [x] Adaptive circle radius
- [x] Adaptive padding
- [x] Depth badge display
- [x] Edge case handling (frame boundary check)
- [x] Multi-person support
- [x] Backward compatible
- [x] Documentation complete

---

**Tác giả:** GitHub Copilot  
**Ngày:** 10/11/2025  
**Version:** 3.0 - Depth-Aware Scaling
