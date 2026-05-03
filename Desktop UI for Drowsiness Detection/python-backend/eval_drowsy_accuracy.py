"""Đánh giá độ chính xác pose classifier ngủ gật trên val set bow_turn_head.

Chạy:  python eval_drowsy_accuracy.py [--limit N]

Pipeline:
  1) YOLO pose inference trên từng ảnh val (lấy keypoints + bbox người)
  2) Áp classify_pose_custom (logic cũ + ngưỡng đã tinh chỉnh) lên mỗi person
  3) So với label GT: class 0 = drowsy (BowHead), class 1 = distracted (TurnHead)
     Map từ pose classifier:
        "Gục xuống bàn" / "Ngủ gật" -> dự đoán = drowsy
        "Bình thường"               -> dự đoán = not_drowsy
  4) Match GT bbox <-> detected bbox bằng IoU >= 0.3, lấy mỗi GT 1 detection
  5) In ra confusion matrix + precision / recall / F1 cho lớp drowsy.

Lưu ý: KHÔNG dùng FSM (vì ảnh tĩnh), chỉ test logic raw pose. FSM chỉ smoothing
theo thời gian, không thay đổi semantic của 1 frame đơn lẻ.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Cho phép import sibling module
sys.path.insert(0, str(Path(__file__).parent))
from yolo_detector import classify_pose_custom  # noqa: E402


def load_yolo():
    """Load YOLO pose model giống production."""
    from ultralytics import YOLO
    here = Path(__file__).parent
    candidates = [
        here / 'models' / 'sleepy_pose_v11n_full_best.pt',
        here / 'models' / 'sleepy_pose_v11n3_best.pt',
        here / 'yolo11n-pose.pt',
        here.parent / 'yolo11n-pose.pt',
    ]
    for p in candidates:
        if p.exists():
            print(f"[load] Using model: {p}")
            return YOLO(str(p))
    raise FileNotFoundError("Không tìm thấy YOLO pose weights")


def parse_yolo_label(path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int]]:
    """Trả list (cls, x1, y1, x2, y2) pixel."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        out.append((cls, x1, y1, x2, y2))
    return out


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def predict_image(model, img_path: Path, conf: float = 0.20, iou_nms: float = 0.45):
    """Trả list (bbox, raw_state) cho mỗi người detect được."""
    img = cv2.imread(str(img_path))
    if img is None:
        return [], None
    h, w = img.shape[:2]
    results = model(img, conf=conf, iou=iou_nms, imgsz=640, verbose=False)
    persons = []
    for r in results:
        boxes = getattr(r, 'boxes', None)
        kpts_obj = getattr(r, 'keypoints', None)
        if boxes is None or kpts_obj is None or boxes.xyxy is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        kdata = kpts_obj.data.cpu().numpy() if hasattr(kpts_obj, 'data') else None
        if kdata is None:
            continue
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            kp = kdata[i]  # (17, 3) hoặc (17, 2)
            if kp.shape[0] < 7:
                continue
            k7 = kp[:7, :2]
            bbox_h = max(1, y2 - y1)
            bbox_w = max(1, x2 - x1)
            state, _, _ = classify_pose_custom(
                k7, bbox_h, bbox_w,
                angle_thr=25.0, drop_h_thr=0.12, drop_sw_thr=0.40,
            )
            persons.append(((x1, y1, x2, y2), state))
    return persons, (w, h)


def gt_inside_person(gt_box, person_box) -> bool:
    """GT (thường là head bbox) coi như match nếu tâm GT nằm trong person bbox
    và phần lớn diện tích GT bị person bbox chứa.
    """
    gx1, gy1, gx2, gy2 = gt_box
    px1, py1, px2, py2 = person_box
    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    if not (px1 <= cx <= px2 and py1 <= cy <= py2):
        return False
    # Containment ratio: phần GT nằm trong person box / diện tích GT
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    gt_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    return inter / gt_area >= 0.6


def evaluate(val_dir: Path, label_dir: Path, model, limit: int, conf: float = 0.20):
    images = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if limit > 0:
        images = images[:limit]
    print(f"[eval] {len(images)} images in val set\n")

    tp = fp = fn = tn = 0
    matched_gt = 0
    unmatched_gt = 0
    cls_breakdown = {0: {'tp': 0, 'fn': 0}, 1: {'tp_distracted_as_notdrowsy': 0, 'fp_distracted_as_drowsy': 0}}

    t0 = time.time()
    for idx, img_path in enumerate(images):
        label_path = label_dir / (img_path.stem + '.txt')
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = parse_yolo_label(label_path, w, h)
        persons, _ = predict_image(model, img_path, conf=conf)

        used = set()
        for (cls, gx1, gy1, gx2, gy2) in gts:
            best_idx = -1
            best_metric = 0.0
            for i, (pbox, _) in enumerate(persons):
                if i in used:
                    continue
                # Match theo containment (GT head nằm trong person body box)
                if gt_inside_person((gx1, gy1, gx2, gy2), pbox):
                    # Person bbox càng nhỏ (gần GT) càng tốt
                    px1, py1, px2, py2 = pbox
                    p_area = max(1, (px2 - px1) * (py2 - py1))
                    metric = 1.0 / p_area  # nhỏ hơn = ưu tiên hơn
                    if metric > best_metric:
                        best_metric = metric
                        best_idx = i
            if best_idx < 0:
                unmatched_gt += 1
                if cls == 0:
                    fn += 1
                    cls_breakdown[0]['fn'] += 1
                continue
            used.add(best_idx)
            matched_gt += 1
            _, pred_state = persons[best_idx]
            pred_drowsy = pred_state in ('Gục xuống bàn', 'Ngủ gật')
            gt_drowsy = (cls == 0)  # class 0 = BowHead/drowsy
            if pred_drowsy and gt_drowsy:
                tp += 1; cls_breakdown[0]['tp'] += 1
            elif pred_drowsy and not gt_drowsy:
                fp += 1; cls_breakdown[1]['fp_distracted_as_drowsy'] += 1
            elif (not pred_drowsy) and gt_drowsy:
                fn += 1; cls_breakdown[0]['fn'] += 1
            else:
                tn += 1; cls_breakdown[1]['tp_distracted_as_notdrowsy'] += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ...{idx + 1}/{len(images)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n[done] {len(images)} images in {elapsed:.1f}s ({len(images)/max(elapsed,1):.1f} img/s)\n")

    print("=== Confusion (lớp drowsy là positive) ===")
    print(f"  TP (drowsy đúng):           {tp}")
    print(f"  FP (báo nhầm drowsy):       {fp}")
    print(f"  FN (bỏ sót drowsy):         {fn}")
    print(f"  TN (not-drowsy đúng):       {tn}")
    print(f"  Matched GT/det: {matched_gt} | Unmatched GT (det miss): {unmatched_gt}")

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    print()
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Accuracy:  {acc:.3f}")

    print("\n=== Breakdown theo class GT ===")
    n_gt0 = cls_breakdown[0]['tp'] + cls_breakdown[0]['fn']
    n_gt1 = cls_breakdown[1]['tp_distracted_as_notdrowsy'] + cls_breakdown[1]['fp_distracted_as_drowsy']
    if n_gt0:
        print(f"  Class 0 (drowsy/BowHead, n={n_gt0}): "
              f"detect đúng={cls_breakdown[0]['tp']} ({cls_breakdown[0]['tp']/n_gt0:.1%})")
    if n_gt1:
        print(f"  Class 1 (distracted/TurnHead, n={n_gt1}): "
              f"không nhầm thành drowsy={cls_breakdown[1]['tp_distracted_as_notdrowsy']} "
              f"({cls_breakdown[1]['tp_distracted_as_notdrowsy']/n_gt1:.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='Giới hạn số ảnh (0 = chạy hết)')
    ap.add_argument('--conf', type=float, default=0.20, help='YOLO confidence threshold')
    args = ap.parse_args()

    here = Path(__file__).parent
    val_img = here / 'datasets' / 'bow_turn_head' / 'images' / 'val'
    val_lbl = here / 'datasets' / 'bow_turn_head' / 'labels' / 'val'
    if not val_img.exists():
        print(f"Không tìm thấy {val_img}")
        sys.exit(1)

    print(f"[cfg] conf={args.conf}")
    model = load_yolo()
    evaluate(val_img, val_lbl, model, args.limit, conf=args.conf)


if __name__ == '__main__':
    main()
