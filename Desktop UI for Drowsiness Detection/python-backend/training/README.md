# Training Pipeline — Drowsy + Distracted Detection

## Cấu trúc

```
training/
├── prepare_dataset.py    # Giải nén BowTurnHead, tạo data.yaml
├── train_drowsy.py       # Fine-tune yolo11n.pt
└── README.md
```

## Quy trình

### Bước 1. Chuẩn bị dataset

```bash
cd python-backend/training
python prepare_dataset.py
```

Tạo ra:
- `datasets/bow_turn_head/images/{train,val}/` — 2410 ảnh lớp học
- `datasets/bow_turn_head/labels/{train,val}/` — label YOLO format
- `datasets/bow_turn_head/data.yaml` — config cho Ultralytics
- `datasets/data_raw_to_label/` — khung để label thêm `data_raw` (xem bước 3)

**Class mapping:**
- `0 = drowsy` (từ BowHead — học sinh cúi đầu)
- `1 = distracted` (từ TurnHead — học sinh quay đầu)

### Bước 2. Train

**Trên GPU (khuyến nghị):**
```bash
python train_drowsy.py --epochs 100 --batch 16 --device 0
```

**Trên CPU (chậm, chỉ để smoke test):**
```bash
python train_drowsy.py --epochs 5 --batch 4 --device cpu
```

Kết quả: `runs/detect/classroom_drowsy/weights/best.pt` (~5–10 MB).

Monitor qua TensorBoard:
```bash
tensorboard --logdir runs/detect
```

### Bước 3. (Optional) Mở rộng với `data_raw` và label thêm phone

BowTurnHead **không có** class `phone_usage`. Để detect điện thoại, cần:

1. Copy ảnh có điện thoại vào `datasets/data_raw_to_label/images/`.
2. Label 3 class: `drowsy`, `distracted`, `phone_usage` bằng:
   - **Roboflow** (online, dễ dùng, free tier): roboflow.com → Upload → Annotate → Export YOLO.
   - **LabelImg** (offline): `pip install labelImg && labelImg`.
3. Sửa `data.yaml`:
   ```yaml
   nc: 3
   names:
     0: drowsy
     1: distracted
     2: phone_usage
   ```
4. Gộp 2 dataset: viết `merge_datasets.py` hoặc đơn giản là dùng chung folder.
5. Train lại.

**Hoặc giữ đơn giản:** train với BowTurnHead 2 class, còn `phone_usage` detect bằng `yolo11n.pt` pretrained (COCO class 67) ghép vào pipeline — như `MultiTaskDetector` hiện tại.

## Tích hợp sau khi train xong

Copy `best.pt` vào `models/classroom_drowsy.pt`, sửa `detectors/multi_task_detector.py`:

```python
POSE_MODEL  = "yolo11n-pose.pt"
OBJECT_MODEL = "models/yolo11n.pt"           # giữ để detect phone
DROWSY_MODEL = "models/classroom_drowsy.pt"  # THÊM: 2 class drowsy/distracted
```

Thay logic "head angle" bằng direct detection từ `DROWSY_MODEL`.

## Tham khảo

- [[../../../../../Claude_Code_Resources/claude-obsidian/02_Skills/YOLOv11_Optimization|YOLOv11 Optimization]]
- [[../../../../../Claude_Code_Resources/claude-obsidian/00_System/YOLO_Standard_Style|Code Style]]
