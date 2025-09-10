import argparse
from pathlib import Path

from ultralytics import YOLO
import torch


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained YOLO Pose model on sleepy dataset")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "datasets" / "sleepy_pose" / "sleepy.yaml"))
    ap.add_argument("--weights", required=True, help="Path to trained weights (e.g. runs/pose-train/exp/weights/best.pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"Dataset YAML not found: {data_yaml}")

    model = YOLO(args.weights)
    # Resolve device like in training
    if args.device in (None, "", "auto"):
        effective_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        effective_device = args.device
    # Validate on val set defined in YAML
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=effective_device)
    print(metrics)


if __name__ == "__main__":
    main()
