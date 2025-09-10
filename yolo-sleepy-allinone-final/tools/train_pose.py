import argparse
from pathlib import Path

from ultralytics import YOLO
import torch


def main():
    ap = argparse.ArgumentParser(description="Train a YOLO Pose model for sleepy detection")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "datasets" / "sleepy_pose" / "sleepy.yaml"), help="Dataset YAML for pose train/val")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "yolo11n-pose.pt"), help="Base pose weights to fine-tune (e.g. yolo11n-pose.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    ap.add_argument("--device", default="auto", help="cuda:0 / cpu / auto")
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[1] / "runs" / "pose-train"))
    ap.add_argument("--name", default="sleepy_pose_v11n")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--cos_lr", action="store_true", help="Use cosine LR schedule")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(args.model)

    # Resolve device: if 'auto' but no CUDA visible, force CPU to avoid Ultralytics error
    if args.device in (None, "", "auto"):
        effective_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        effective_device = args.device

    # Resolve batch/workers: Ultralytics may crash if batch is None on CPU during auto-estimation.
    # Use a conservative default on CPU when user leaves batch=-1 (auto).
    batch_val = None if args.batch == -1 else args.batch
    if batch_val is None and effective_device == "cpu":
        batch_val = 4

    # Fewer workers on CPU/Windows for stability; keep small non-zero workers for CUDA.
    workers_val = 0 if effective_device == "cpu" else 2

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_val,
        device=effective_device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        lr0=args.lr0,
        patience=args.patience,
        cos_lr=args.cos_lr,
        # Augmentation knobs (conservative for pose):
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.0, fliplr=0.5,
        mosaic=0.1, mixup=0.0,
        # Optimizations
        workers=workers_val,
        cache=False,
        verbose=True,
    )

    results = model.train(**train_kwargs)
    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
