# DACN PhatHienNguGat - 2026
"""Train YOLO11 detection cho lớp học drowsy + distracted.

Base: yolo11n.pt (pretrained COCO) → fine-tune 2-class trên BowTurnHead.

Chạy:
    python train_drowsy.py --epochs 100 --batch 16 --device 0

Sau khi xong, ``best.pt`` nằm ở ``runs/detect/classroom_drowsy/weights/best.pt``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_YAML: Path = BACKEND_ROOT / "datasets" / "bow_turn_head" / "data.yaml"
BASE_MODEL: Path = BACKEND_ROOT / "models" / "yolo11n.pt"
RUN_NAME: str = "classroom_drowsy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--model", type=Path, default=BASE_MODEL)
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"{args.data} — chạy `prepare_dataset.py` trước")
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} — download yolo11n.pt trước")

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=RUN_NAME,
        patience=20,
        save=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        resume=args.resume,
    )
    print(f"\nDone. Weights: runs/detect/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
