# train_model.md — Auto-Training Runbook

> **Hướng dẫn cho Claude:** Khi user bảo "đọc `train_model.md` và train", thực thi **tuần tự** các bước dưới. Dừng và hỏi user nếu gặp block. Mỗi bước phải verify success trước khi sang bước kế.

**Mục tiêu:** Train model YOLO11 detection 2-class (`drowsy`, `distracted`) trên dataset `bow_turn_head`, nhanh nhất + hiệu quả cao nhất với phần cứng sẵn có.

**Pre-reqs:** `python-backend/` là working dir. `ultralytics`, `torch`, `huggingface_hub` đã cài.

---

## Bước 0 — Sanity check (30s)

Chạy 4 lệnh song song:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A')"
python -c "from pathlib import Path; p=Path('models/yolo11n.pt'); print('base model:', 'OK' if p.exists() else 'MISSING', p.stat().st_size/1e6 if p.exists() else '', 'MB')"
python -c "from pathlib import Path; p=Path('datasets/bow_turn_head/data.yaml'); print('dataset yaml:', 'OK' if p.exists() else 'MISSING')"
nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
```

**Quy tắc quyết định cấu hình dựa trên kết quả:**

| VRAM | Profile | batch | imgsz | epochs | cache | amp | model |
|---|---|---|---|---|---|---|---|
| ≥ 12 GB | `gpu-large` | 32 | 640 | 100 | `ram` | ✅ | yolo11n.pt |
| 6–12 GB | `gpu-medium` | 16 | 640 | 80 | `ram` | ✅ | yolo11n.pt |
| 4–6 GB | `gpu-small` | 8 | 512 | 60 | `disk` | ✅ | yolo11n.pt |
| 2–4 GB | `gpu-tiny` | 4 | 416 | 40 | `disk` | ✅ | yolo11n.pt |
| CPU only | `cpu-smoke` | 4 | 416 | 5 | `disk` | ❌ | yolo11n.pt |

**CPU profile chỉ dùng để smoke test pipeline**, không kỳ vọng chất lượng. Báo user: "CPU 5 epochs = proof pipeline hoạt động, không phải model dùng thật. Nên chạy lại trên GPU."

---

## Bước 1 — Prepare dataset (nếu chưa)

```bash
test -d datasets/bow_turn_head/images/train && echo "ALREADY PREPARED" || python training/prepare_dataset.py
```

**Verify:** `ls datasets/bow_turn_head/images/train | wc -l` phải ra ~1905, `val` ~505.

---

## Bước 2 — Train

Gọi `ultralytics` trực tiếp với **hyperparameters tối ưu cho fine-tuning nhanh + ít overfit**:

```python
python -c "
from ultralytics import YOLO
import torch

profile = {
    # EDIT IF NEEDED — chọn theo bảng ở Bước 0
    'batch': 16, 'imgsz': 640, 'epochs': 80,
    'cache': 'ram', 'amp': True, 'device': 0,
}
# Auto-fallback to CPU profile nếu không có GPU
if not torch.cuda.is_available():
    profile.update(batch=4, imgsz=416, epochs=5, cache='disk', amp=False, device='cpu')

model = YOLO('models/yolo11n.pt')
model.train(
    data='datasets/bow_turn_head/data.yaml',
    name='classroom_drowsy',
    exist_ok=True,
    # core
    epochs=profile['epochs'],
    batch=profile['batch'],
    imgsz=profile['imgsz'],
    device=profile['device'],
    cache=profile['cache'],
    amp=profile['amp'],
    # optimizer — AdamW hội tụ nhanh hơn SGD cho fine-tuning
    optimizer='AdamW',
    lr0=1e-3, lrf=0.01,
    cos_lr=True,
    warmup_epochs=3.0,
    weight_decay=5e-4,
    # augmentation — mạnh ở đầu, tắt dần cuối
    mosaic=1.0, mixup=0.15, copy_paste=0.0,
    close_mosaic=10,   # tắt mosaic 10 epochs cuối → cleaner val
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, flipud=0.0,
    degrees=5.0, translate=0.1, scale=0.5,
    # regularization & early stop
    patience=15,       # early stop khi val không cải thiện 15 epochs
    dropout=0.0,
    # speed
    workers=8,
    save_period=-1,    # chỉ lưu best + last
    plots=True, verbose=True,
)
print('TRAIN DONE — weights at runs/detect/classroom_drowsy/weights/best.pt')
"
```

**Lý do từng flag (giải thích nếu user hỏi):**
- `AdamW + lr0=1e-3` → hội tụ nhanh gấp ~2x so với SGD default cho transfer learning.
- `cos_lr=True` → cosine schedule, drop LR mượt cuối train → mAP tốt hơn flat LR.
- `amp=True` → mixed precision FP16, speed ~2x trên GPU có Tensor Cores (RTX 20xx+).
- `cache='ram'` → load toàn bộ ảnh vào RAM, eliminate disk I/O — đây là **tăng tốc lớn nhất** (epoch thứ 2 trở đi nhanh 3-5x). Cần ~3-5 GB RAM cho 1905 ảnh 640px.
- `patience=15` → early stop. Không train hoài tốn điện nếu plateau.
- `close_mosaic=10` → tắt mosaic augmentation 10 epochs cuối, để val metrics phản ánh realistic deployment.
- `mixup=0.15` → ít hơn default 0.1, giảm nhiễu vì dataset chỉ 1905 ảnh.
- `workers=8` → song song load data, bỏ bottleneck DataLoader.

---

## Bước 3 — Verify training chất lượng

```bash
ls -la runs/detect/classroom_drowsy/weights/
cat runs/detect/classroom_drowsy/results.csv 2>/dev/null | tail -5 || echo "no csv"
```

Kiểm tra **mAP@50** ở epoch cuối (cột `metrics/mAP50(B)` trong results.csv):

| mAP@50 | Đánh giá | Hành động |
|---|---|---|
| ≥ 0.80 | Xuất sắc | Deploy ngay |
| 0.65 – 0.80 | Tốt | Deploy được, có thể nâng sau với Roboflow data |
| 0.50 – 0.65 | Trung bình | Train thêm 40 epochs với `resume=True` |
| < 0.50 | Kém | Debug: xem `confusion_matrix.png`, check labels |

---

## Bước 4 — Deploy weights

Copy weights vào `models/`:

```bash
cp runs/detect/classroom_drowsy/weights/best.pt models/classroom_drowsy.pt
python -c "
from ultralytics import YOLO
m = YOLO('models/classroom_drowsy.pt')
print('names:', m.names)
print('task:', m.task)
"
```

**Verify:** output phải có `{0: 'drowsy', 1: 'distracted'}` và `task: detect`.

---

## Bước 5 — Smoke test trên ảnh thật

```bash
python -c "
from ultralytics import YOLO
from pathlib import Path
m = YOLO('models/classroom_drowsy.pt')
samples = [p for p in Path('../data_raw').iterdir() if p.suffix.lower() in {'.jpg','.png'}][:10]
for s in samples:
    r = m.predict(str(s), conf=0.35, verbose=False)[0]
    n = len(r.boxes) if r.boxes is not None else 0
    print(f'{s.name}: {n} detections')
"
```

Kỳ vọng: ảnh `ngu.jpg`, `pngtree-*-sleeping` phải có ≥1 detection class `drowsy`.

---

## Bước 6 — Update detector module

Sửa `detectors/multi_task_detector.py` để dùng model mới:

```python
# Thêm constant ở đầu file:
DROWSY_MODEL: str = "models/classroom_drowsy.pt"
```

Trong `MultiTaskDetector.__init__`, thêm:
```python
self.drowsy_model = YOLO(DROWSY_MODEL)
```

Trong `_process_frame`, thay rule góc đầu bằng direct detection từ `self.drowsy_model`. **Không xóa** logic cũ — đặt cờ `use_custom_model=True/False` để fallback được.

(Để user xác nhận trước khi edit — có thể họ muốn giữ rule-based song song.)

---

## Troubleshooting

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `CUDA out of memory` | Batch quá lớn | Giảm batch 16→8→4. Giảm imgsz 640→512→416 |
| `cache='ram'` OOM host RAM | Dataset > RAM | Đổi sang `cache='disk'` |
| mAP stuck < 0.3 sau 30 epochs | Label sai hoặc LR sai | Check `train_batch0.jpg`, thử `lr0=5e-4` |
| Train chậm (>5 phút/epoch trên GPU) | DataLoader bottleneck | Tăng `workers`, bật `cache='ram'` |
| Windows `num_workers` lỗi | Windows multiprocessing | Giảm `workers=4` hoặc `workers=2` |
| `killed` / OOM giữa train | Cache RAM quá lớn | `cache='disk'` hoặc giảm batch |

---

## Nếu muốn NHANH HƠN NỮA (≤ 15 phút train)

Đổi profile thành:
```python
epochs=30, imgsz=512, batch=32, close_mosaic=5, patience=8
```

mAP sẽ giảm 5-10% nhưng train trong 1/3 thời gian — tốt cho iteration nhanh khi debug dataset.

---

## Liên kết

- Prepare dataset: `training/prepare_dataset.py`
- Training script version khác: `training/train_drowsy.py` (hyperparams đơn giản hơn)
- Style guide: `D:/Claude_Code_Resources/claude-obsidian/00_System/YOLO_Standard_Style.md`
- Feature log: `D:/Claude_Code_Resources/claude-obsidian/01_Projects/Student_Monitoring/Feature_Log.md`
