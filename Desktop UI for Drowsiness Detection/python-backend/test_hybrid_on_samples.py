# DACN PhatHienNguGat - 2026
"""Quick test HybridDetector on sample images from SCB_BowTurnHead dataset.

Chạy:
    python test_hybrid_on_samples.py

In ra: per-image người detect được, số phone, số drowsy, FPS. Lưu ảnh
annotated vào ``test_samples/annotated/``.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SAMPLES_DIR: Path = Path(__file__).parent / "test_samples"
OUT_DIR: Path = SAMPLES_DIR / "annotated"
DETECTOR: Path = Path(__file__).parent / "models" / "yolo11n.pt"
CLASSIFIER: Path = Path(__file__).parent / "models" / "drowsiness_cls.pt"

PERSON_ID: int = 0
PHONE_ID: int = 67
DROWSY_THRESHOLD: float = 0.60


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    detector = YOLO(str(DETECTOR))
    classifier = YOLO(str(CLASSIFIER))

    drowsy_idx = next(
        (i for i, n in classifier.names.items() if "drows" in n.lower() and "non" not in n.lower()),
        0,
    )
    print(f"classifier.names = {classifier.names}  drowsy_idx = {drowsy_idx}")

    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Testing on {len(images)} images\n")

    total_persons = 0
    total_phones = 0
    total_drowsy = 0
    t0 = time.time()

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"SKIP (unreadable): {img_path.name}")
            continue

        det_result = detector.predict(
            source=frame,
            classes=[PERSON_ID, PHONE_ID],
            conf=0.35, iou=0.5, imgsz=640, verbose=False,
        )[0]
        boxes = det_result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"{img_path.name}: no detections")
            continue
        cls = boxes.cls.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        person_boxes = xyxy[cls == PERSON_ID]
        phone_count = int((cls == PHONE_ID).sum())
        total_persons += len(person_boxes)
        total_phones += phone_count

        drowsy_in_image = 0
        annotated = frame.copy()

        # classify each person crop
        crops: list[np.ndarray] = []
        for x1, y1, x2, y2 in person_boxes.astype(int):
            y1c, y2c = max(0, y1), min(frame.shape[0], y2)
            x1c, x2c = max(0, x1), min(frame.shape[1], x2)
            crop = frame[y1c:y2c, x1c:x2c]
            crops.append(crop if crop.size else np.zeros((32, 32, 3), dtype=frame.dtype))

        if crops:
            cls_results = classifier.predict(
                crops, imgsz=224, verbose=False,
            )
        else:
            cls_results = []

        for (x1, y1, x2, y2), cr in zip(person_boxes.astype(int), cls_results):
            p_drowsy = float(cr.probs.data[drowsy_idx]) if cr.probs is not None else 0.0
            is_drowsy = p_drowsy >= DROWSY_THRESHOLD
            if is_drowsy:
                drowsy_in_image += 1
            color = (0, 0, 255) if is_drowsy else (0, 200, 0)
            label = f"{'DROWSY' if is_drowsy else 'awake'} {p_drowsy:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # annotate phones
        for x1, y1, x2, y2 in xyxy[cls == PHONE_ID].astype(int):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(annotated, "PHONE", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

        total_drowsy += drowsy_in_image
        cv2.imwrite(str(OUT_DIR / img_path.name), annotated)
        print(f"{img_path.name}: persons={len(person_boxes)}  phones={phone_count}  drowsy={drowsy_in_image}")

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY ===")
    print(f"Images processed: {len(images)}")
    print(f"Total persons detected: {total_persons}")
    print(f"Total phones detected:  {total_phones}")
    print(f"Total drowsy flagged:   {total_drowsy}")
    print(f"Drowsy rate: {total_drowsy/total_persons*100:.1f}%" if total_persons else "")
    print(f"Elapsed: {elapsed:.2f}s  ({len(images)/elapsed:.2f} images/sec)")
    print(f"Annotated outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
