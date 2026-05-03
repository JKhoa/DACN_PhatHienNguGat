# Pipeline phát hiện V1 — Ngủ gật + Bấm điện thoại

Tiếng Việt 100%. Primary = ngủ gật (đỏ + beep). Secondary = điện thoại (vàng).

## Kiến trúc

```
┌────────────── Electron (React) ──────────────┐
│  DetectionV1Panel.tsx                         │
│    • Realtime: webcam → canvas → base64 JPEG  │
│    • Ảnh / Video: FileReader → base64         │
│    • Badge đỏ/vàng, beep, dropdown top_k      │
└──────────── window.appApi.invoke ─────────────┘
                     │ IPC
┌──────────── Electron main process ───────────┐
│  api:request → HTTP → Flask                   │
└────────────── http://localhost:5000 ──────────┘
                     │
┌──────────────── Flask Blueprint ──────────────┐
│  api_v1.py (/api/v1/detect)                   │
│    /health  • /image  • /video  • WS realtime │
└───────────── EnsembleDetector ────────────────┘
                     │
┌────────── detectors/ensemble.py ──────────────┐
│  Primary (hybrid):                             │
│    yolo11n.pt (person) → crop → drowsiness_cls │
│  Secondary:                                    │
│    phone_det.pt (IndUSV/yolov8n-mobile-phone)  │
│  Fallback:                                     │
│    transformers pipeline closed_eyes/yawn      │
│  Class-aware NMS: same 0.55 / cross 0.85       │
│  Auto-retry conf=0.05 khi rỗng                 │
│  Per-class floor (ngu_gat 0.30, phone 0.40)    │
└────────────────────────────────────────────────┘
```

## Chạy local

```bash
# 1) Tải weights (one-time). Cần internet.
cd python-backend
python download_models.py

# 2) (Optional) smoke test
python test_ensemble_smoke.py

# 3) Chạy server
python server_with_tracking_backup.py  # port 5000
```

Mở UI: `npm run electron:dev` ở thư mục gốc → tab **⚠ Phát hiện V1**.

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| GET  | `/api/v1/detect/health` | Ping + info weights |
| POST | `/api/v1/detect/image`  | Ảnh tĩnh. Body: multipart `file` hoặc JSON `{image_base64}` |
| POST | `/api/v1/detect/video`  | Video. Body: multipart `file` hoặc JSON `{video_base64, filename}` |
| WS   | `/api/v1/detect/realtime` | SocketIO namespace. Emit `frame {image_base64, conf}` → reply `result` |

Query: `conf` (0.35 mặc định), `use_secondary` (1), `use_hf` (1), `frame_stride` (video, 10).

### Response schema

```json
{
  "objects": [
    {
      "class_name": "ngu_gat",
      "display_name": "Ngủ gật",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2],
      "severity": "danger",
      "source": "primary"
    }
  ],
  "top_k": [],
  "inference_time_ms": 932.7,
  "image_size": [640, 480]
}
```

## Mapping slug → VN

| slug | display | severity | floor |
|---|---|---|---|
| `ngu_gat` | Ngủ gật | danger | 0.30 |
| `ngap` | Ngáp | warn | 0.30 |
| `mat_nham` | Mắt nhắm | danger | 0.30 |
| `tinh_tao` | Tỉnh táo | info | 0.25 |
| `dien_thoai` | Điện thoại | warn | 0.40 |
| `bam_dien_thoai` | Bấm điện thoại | warn | 0.40 |

## Weights

| File | Nguồn HF | Vai trò |
|---|---|---|
| `yolo11n.pt` | `Ultralytics/YOLO11` | Person detector (hybrid primary) |
| `drowsiness_cls.pt` | `mosesb/drowsiness-detection-yolo-cls` | Classifier crop → Drowsy/Non Drowsy |
| `phone_det.pt` | `IndUSV/yolov8n-mobile-phone` | Secondary phone detector |

Không có drowsiness-YOLO-detector open-access Ultralytics `.pt` tại thời điểm 2026-04 → primary = hybrid (detect + classify).

## Verified (2026-04-20)

- `curl /api/v1/detect/health` → 200 `{status:"ok", hybrid_mode:true, primary:"yolo11n.pt", secondary:"phone_det.pt"}`
- `POST /api/v1/detect/image` với ảnh lớp học `sample_00_10_002304.jpg` → 12 objects, class_name=`ngu_gat`, display=`Ngủ gật`, severity=`danger`, `inference_time_ms=932ms` (CPU).

## Ghi chú hiệu năng

- CPU hybrid ~900ms/ảnh cho lớp 20 người (nhiều crops). Realtime 1-2 FPS trên CPU.
- GPU (CUDA) đề xuất đạt 30+ FPS. Set `device=0` khi khởi `EnsembleDetector`.
- Realtime trong UI poll 500ms/frame qua IPC — chấp nhận được cho demo đồ án. Production nên migrate socket.io qua IPC streaming (xem `project_drowsiness.md`).
