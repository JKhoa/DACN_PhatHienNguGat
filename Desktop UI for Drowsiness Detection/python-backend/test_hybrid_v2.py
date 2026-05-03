# DACN PhatHienNguGat - 2026
"""Test HybridDetector v2 — so sánh 3 chiến lược classify drowsy.

Chạy trên ảnh thật trong ``data_raw/`` (ảnh ngủ gật rõ ràng).

3 biến thể:
    A) full-body crop + threshold 0.60 (baseline — như test trước)
    B) head-region crop (top 35% bbox) + threshold 0.60
    C) head-region crop + threshold 0.75

Dùng yolo11s.pt (nặng hơn nhưng detect tốt hơn) thay vì yolo11n.pt.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA_RAW = Path(r"D:/Study/DoAnChuyenNganh/DACN_PhatHienNguGat/data_raw")
OUT_DIR = Path(__file__).parent / "test_samples" / "v2_annotated"
DETECTOR = Path(__file__).parent / "models" / "yolo11n.pt"
CLASSIFIER = Path(__file__).parent / "models" / "drowsiness_cls.pt"

PERSON_ID = 0
PHONE_ID = 67


def head_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop top 35% of person bbox (head region)."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    y2 = y1 + int(h * 0.35)
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else np.zeros((32, 32, 3), dtype=frame.dtype)


def full_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else np.zeros((32, 32, 3), dtype=frame.dtype)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "A_fullbody_60").mkdir(exist_ok=True)
    (OUT_DIR / "B_head_60").mkdir(exist_ok=True)
    (OUT_DIR / "C_head_75").mkdir(exist_ok=True)

    detector = YOLO(str(DETECTOR))
    classifier = YOLO(str(CLASSIFIER))
    drowsy_idx = next(
        (i for i, n in classifier.names.items() if "drows" in n.lower() and "non" not in n.lower()),
        0,
    )

    images = sorted(p for p in DATA_RAW.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Testing on {len(images)} raw images\n")

    stats = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}  # [persons, drowsy]

    for img_path in images:  # process all
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        det = detector.predict(frame, classes=[PERSON_ID, PHONE_ID],
                               conf=0.25, iou=0.5, imgsz=640, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            print(f"{img_path.name}: NO PERSON")
            continue
        cls = det.boxes.cls.int().cpu().numpy()
        xyxy = det.boxes.xyxy.cpu().numpy()
        persons = xyxy[cls == PERSON_ID].astype(int)
        phones = xyxy[cls == PHONE_ID].astype(int)

        full_crops = [full_crop(frame, tuple(b)) for b in persons]
        head_crops = [head_crop(frame, tuple(b)) for b in persons]

        full_probs = [float(r.probs.data[drowsy_idx]) for r in classifier.predict(full_crops, imgsz=224, verbose=False)] if full_crops else []
        head_probs = [float(r.probs.data[drowsy_idx]) for r in classifier.predict(head_crops, imgsz=224, verbose=False)] if head_crops else []

        variants = {
            "A": ("A_fullbody_60", full_probs, 0.60),
            "B": ("B_head_60", head_probs, 0.60),
            "C": ("C_head_75", head_probs, 0.75),
        }

        per_image = {}
        for key, (folder, probs, thr) in variants.items():
            annotated = frame.copy()
            drowsy_count = 0
            for (x1, y1, x2, y2), p in zip(persons, probs):
                is_drowsy = p >= thr
                if is_drowsy: drowsy_count += 1
                color = (0, 0, 255) if is_drowsy else (0, 200, 0)
                lbl = f"{'DROWSY' if is_drowsy else 'awake'} {p:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, lbl, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            for x1, y1, x2, y2 in phones:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
                cv2.putText(annotated, "PHONE", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            cv2.imwrite(str(OUT_DIR / folder / img_path.name), annotated)
            stats[key][0] += len(persons)
            stats[key][1] += drowsy_count
            per_image[key] = drowsy_count
        print(f"{img_path.name}: persons={len(persons)}  phones={len(phones)}  "
              f"drowsy A={per_image['A']}  B={per_image['B']}  C={per_image['C']}")

    print("\n=== SUMMARY ===")
    for k, (p, d) in stats.items():
        pct = d / p * 100 if p else 0
        print(f"Variant {k}: {d}/{p} drowsy ({pct:.1f}%)")
    print(f"\nAnnotated: {OUT_DIR}")


if __name__ == "__main__":
    main()
