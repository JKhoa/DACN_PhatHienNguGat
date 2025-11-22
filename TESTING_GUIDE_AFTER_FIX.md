# 🎯 HƯỚNG DẪN KIỂM TRA SAU KHI FIX FALSE POSITIVE

**Backend đã được restart với threshold mới!** ✅  
**Thời gian:** 10/11/2025 15:32

---

## 🔧 NHỮNG GÌ ĐÃ ĐƯỢC SỬA

### ❌ Vấn đề trước đây:
- Tracking box hiển thị "BUỒN NGỦ" màu đỏ khi người dùng đang tỉnh táo
- Trigger quá dễ, gây false positive cao
- Người dùng khó chịu vì cảnh báo sai

### ✅ Giải pháp đã áp dụng:

1. **Tăng threshold phát hiện pose:**
   - Góc cúi đầu: 25° → **35°** (khó trigger hơn)
   - Head drop ratio: 12% → **18%** (cần gục sâu hơn)
   - Shoulder drop: 40% → **50%** (vai phải rơi rõ ràng)

2. **Temporal smoothing strict hơn:**
   - History length: 10 frames → **15 frames** (3 giây thay vì 2 giây)
   - Drowsy threshold: 7/10 → **12/15 frames** (80% thay vì 70%)
   - Minimum frames: 8 → **12 frames** để bắt đầu quyết định

3. **Logic quyết định conservative hơn:**
   - Chỉ cho phép **≤2 awake frames** thay vì ≤3
   - Ưu tiên "awake" khi có nghi ngờ

---

## 🧪 CÁCH KIỂM TRA

### Test Case 1: Normal Study (Expect XANH)
1. Ngồi thẳng, viết bài bình thường 2-3 phút
2. **Expected:** Box XANH "TỈNH" suốt quá trình
3. **Không** được thấy box đỏ "BUỒN NGỦ"

### Test Case 2: Light Head Movement (Expect XANH)  
1. Nhìn xuống giấy, lên bảng, qua trái/phải
2. Làm động tác này 1-2 phút
3. **Expected:** Box XANH "TỈNH", không flicker đỏ

### Test Case 3: Simulated Drowsiness (Expect ĐỎ)
1. Cúi đầu sâu khoảng 40°+ (như ngủ gật thật)
2. Giữ tư thế này 5+ giây
3. **Expected:** Box chuyển ĐỎ "BUỒN NGỦ" sau 3-4 giây

### Test Case 4: Recovery (Expect XANH)
1. Từ trạng thái drowsy, ngồi thẳng trở lại
2. **Expected:** Box chuyển về XANH sau 2-3 giây

---

## 📊 THRESHOLD COMPARISON

| Metric | CŨ (Dễ trigger) | MỚI (Khó trigger) |
|--------|----------------|------------------|
| Head Tilt | 25° | **35°** |
| Head Drop | 12% | **18%** |
| History Length | 2 seconds | **3 seconds** |
| Required Drowsy | 70% frames | **80% frames** |

---

## ⚠️ LƯU Ý

### Điều kiện để trigger "BUỒN NGỦ" giờ đây:
1. **Cúi đầu ≥35°** HOẶC **Đầu rơi ≥18%** HOẶC **Vai rơi ≥50%**
2. **Liên tục 3+ giây**
3. **≥12/15 frames** phải detect drowsy
4. **Chỉ ≤2 frames** awake được phép trong period

### Behavior mong đợi:
- ✅ **Viết bài** (20-25°) → KHÔNG trigger
- ✅ **Đọc điện thoại** (25-30°) → KHÔNG trigger  
- ✅ **Quay đầu nói chuyện** → KHÔNG trigger
- ✅ **Ngủ gật thật** (35°+, 3+ giây) → VẪN trigger

---

## 🎯 NEXT STEPS

1. **Test ngay:** Chạy các test cases trên
2. **Monitor 30 phút:** Xem có còn false positive không
3. **User feedback:** Thu thập ý kiến từ người dùng thực tế
4. **Fine-tune nếu cần:** Nếu threshold quá strict, có thể điều chỉnh

---

## 📞 SUPPORT

Nếu gặp vấn đề:
1. Kiểm tra backend logs trong terminal
2. Xem browser console (F12) để check WebSocket
3. Test với sensitivity slider ở mức 80-90
4. Đảm bảo ánh sáng đủ cho camera

**Expected result:** Significant reduction in false positive rate! 🚀