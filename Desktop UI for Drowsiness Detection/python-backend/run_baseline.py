# DACN PhatHienNguGat - 2026
"""Chạy MultiTaskDetector baseline — dùng ngay không cần train.

Baseline dùng pose + rule góc đầu (không dùng classifier driver bị bias).
Kết hợp COCO cell phone detection từ yolo11n.pt.

Usage:
    python run_baseline.py --source 0                    # webcam
    python run_baseline.py --source path/to/video.mp4    # file
    python run_baseline.py --source data_raw/ngu.jpg     # 1 ảnh
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from detectors import BehaviorType, MultiTaskDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Webcam index, video path, or image path")
    ap.add_argument("--device", default="cpu", help="'cpu' or GPU index like '0'")
    ap.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    ap.add_argument("--vid-stride", type=int, default=2)
    ap.add_argument("--show", action="store_true", help="Hiển thị cửa sổ realtime")
    args = ap.parse_args()

    source: str | int
    try:
        source = int(args.source)  # webcam index
    except ValueError:
        source = args.source

    detector = MultiTaskDetector(device=args.device, half=args.half)
    print(f"Detector ready: device={args.device} half={args.half}")
    print(f"Source: {source}")
    print("Events below — Ctrl+C to stop:\n")

    counts: dict[BehaviorType, int] = {b: 0 for b in BehaviorType}
    t0 = time.time()
    frame_count = 0

    try:
        for events in detector.stream(source, vid_stride=args.vid_stride):
            frame_count += 1
            for ev in events:
                counts[ev.behavior] += 1
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"track={ev.track_id:>3}  {ev.behavior.value:<12} "
                    f"dur={ev.duration:5.1f}s  conf={ev.confidence:.2f}"
                )
    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY ({elapsed:.1f}s, {frame_count} frames) ===")
    for b, c in counts.items():
        print(f"  {b.value:<12}: {c}")


if __name__ == "__main__":
    main()
