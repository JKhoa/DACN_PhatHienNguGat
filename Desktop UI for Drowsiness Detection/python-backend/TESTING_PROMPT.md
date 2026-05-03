# Prompt TEST pipeline phát hiện V1

Paste nguyên khối dưới đây vào session Claude Code mới (cùng thư mục
`D:\Study\DoAnChuyenNganh`) để kiểm thử end-to-end pipeline Phase 1-4.

---

TÁC VỤ: Test toàn bộ pipeline phát hiện ngủ gật + bấm điện thoại đã được tích hợp.

NGỮ CẢNH
- App root: `DACN_PhatHienNguGat/Desktop UI for Drowsiness Detection/`
- Backend: `python-backend/` (Flask + SocketIO, port 5000)
- Frontend: Electron + React, tab mới "⚠ Phát hiện V1"
- Pipeline mới: `api_v1.py` blueprint + `detectors/ensemble.py`
- Mapping EN→VN ở `detectors/label_mapping.py`
- README chi tiết: `python-backend/README_detection_v1.md`

YÊU CẦU KIỂM THỬ

## A. Smoke test backend (không cần UI)

1. Kiểm tra weights có đủ:
   ```bash
   cd "DACN_PhatHienNguGat/Desktop UI for Drowsiness Detection/python-backend"
   ls -la models/
   ```
   Phải có: `yolo11n.pt`, `drowsiness_cls.pt`, `phone_det.pt`.
   Nếu thiếu → `python download_models.py`.

2. Smoke test ensemble trực tiếp:
   ```bash
   python test_ensemble_smoke.py
   ```
   Kỳ vọng: 5 ảnh `test_samples/sample_*.jpg`, có ít nhất 3 ảnh detect
   `ngu_gat` với `display_name == "Ngủ gật"` và `severity == "danger"`.

3. Unit test mapping:
   ```bash
   python -c "
   from detectors.label_mapping import _map_any_name, passes_floor
   assert _map_any_name('Drowsy') == ('ngu_gat', 'Ngủ gật')
   assert _map_any_name('Phone (Object)') == ('dien_thoai', 'Điện thoại')
   assert _map_any_name('person') == (None, None)
   assert passes_floor('ngu_gat', 0.35) is True
   assert passes_floor('ngu_gat', 0.25) is False
   assert passes_floor('dien_thoai', 0.35) is False  # floor=0.40
   print('mapping OK')
   "
   ```

## B. Test Flask endpoints (chạy server thật)

1. Khởi server:
   ```bash
   python server_with_tracking_backup.py
   ```
   Để chạy background; đợi 10-15s để load models.

2. `/health`:
   ```bash
   curl http://127.0.0.1:5000/api/v1/detect/health
   ```
   Kỳ vọng: `{"status":"ok","hybrid_mode":true,"primary":"yolo11n.pt","secondary":"phone_det.pt", ...}`

3. `POST /image` (ảnh ngủ gật thật):
   ```bash
   PYTHONIOENCODING=utf-8 python -c "
   import base64, json, urllib.request, sys
   sys.stdout.reconfigure(encoding='utf-8')
   with open('test_samples/sample_00_10_002304.jpg','rb') as f:
       b64 = base64.b64encode(f.read()).decode()
   req = urllib.request.Request(
       'http://127.0.0.1:5000/api/v1/detect/image?conf=0.35',
       data=json.dumps({'image_base64': b64}).encode(),
       headers={'Content-Type':'application/json'}, method='POST')
   j = json.loads(urllib.request.urlopen(req, timeout=60).read())
   o = j['objects'][0]
   assert o['class_name'] == 'ngu_gat', o
   assert o['display_name'] == 'Ngủ gật', o
   assert o['severity'] == 'danger', o
   print('image OK:', len(j['objects']), 'objects,', j['inference_time_ms'], 'ms')
   "
   ```

4. Error path — body rỗng:
   ```bash
   curl -X POST http://127.0.0.1:5000/api/v1/detect/image \
     -H "Content-Type: application/json" -d "{}"
   ```
   Kỳ vọng: 400 với `error` tiếng Việt.

5. `POST /video` (tuỳ chọn — cần 1 file mp4 nhỏ):
   Multipart upload `file=@small.mp4`, verify response có `alerts[]` và `frames[]`.

## C. Test UI Electron

1. Cài deps nếu chưa:
   ```bash
   cd ..  # về "Desktop UI for Drowsiness Detection/"
   npm install
   ```

2. Khởi chế độ dev:
   ```bash
   ./START-DEV-MODE.bat   # hoặc npm run electron:dev
   ```
   Trước đó server Python ở bước B.1 phải đang chạy.

3. Trong Electron window, click tab "⚠ Phát hiện V1". Kiểm:
   - Card đầu hiển thị "Model primary: yolo11n.pt • secondary: phone_det.pt • hybrid mode".
   - Nếu hiển thị đỏ "Không kết nối backend" → backend chết hoặc IPC hỏng.

4. Tab **Upload Ảnh**:
   - Chọn ảnh ngủ gật (hoặc từ `python-backend/test_samples/`).
   - Verify: canvas hiện ảnh kèm bbox đỏ, badge "⚠ Ngủ gật (0.xx)".
   - Thử ảnh phong cảnh (không người) → thấy khối xanh "✓ Bình thường".
   - Thử ảnh mờ/khó → thấy dropdown "Có thể là..." với top_k.

5. Tab **Upload Video**:
   - Upload video người đang gục đầu > 2s.
   - Verify: xuất hiện dòng đỏ "⚠ Ngủ gật từ giây Xs (kéo dài Ys)".

6. Tab **Camera Realtime**:
   - Bấm "Bật camera" → cấp quyền webcam.
   - Giả vờ nhắm mắt / gục đầu → badge đỏ xuất hiện + beep kêu.
   - Cầm điện thoại trước camera → badge vàng "Bấm điện thoại"/"Điện thoại".

## D. Tiêu chí PASS

- [ ] Tất cả smoke test (A) pass không exception.
- [ ] `curl /health` trả `status=ok`.
- [ ] POST ảnh sample thật trả `class_name="ngu_gat"`, `display_name="Ngủ gật"`.
- [ ] UI hiển thị 3 tab, backend info hiện đúng.
- [ ] Upload ảnh → bbox + badge đúng màu.
- [ ] Realtime webcam → beep khi nhắm mắt.
- [ ] Không có chuỗi tiếng Anh lọt ra UI (mọi label, alert, error).
- [ ] `inference_time_ms` log có mặt trong response.

## E. Khi FAIL

- Đọc `README_detection_v1.md` + `C:\Users\Admin\.claude\projects\D--Study-DoAnChuyenNganh\memory\detection_pipeline_v1.md` để biết trạng thái.
- Log server Python + DevTools Electron (Ctrl+Shift+I).
- Kiểm `window.appApi` tồn tại → preload load OK chưa.
- Nếu weight `phone_det.pt` mất → `python download_models.py`.

BÁO CÁO (ngắn, ≤ 200 từ)
- Section nào fail, command nào lỗi, stack trace ngắn.
- Nếu pass toàn bộ: liệt kê thời gian inference thực đo + screenshot path.
