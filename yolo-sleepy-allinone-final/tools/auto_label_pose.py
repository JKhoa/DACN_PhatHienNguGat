import os
import cv2
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


def ensure_dirs(base: Path):
    (base / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (base / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (base / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (base / 'val' / 'labels').mkdir(parents=True, exist_ok=True)


def classify_state(kp_xy: np.ndarray, img_h: int) -> int:
    """
    Heuristic classification using keypoints.
    Returns class id: 0=binhthuong, 1=ngugat, 2=gucxuongban
    kp_xy: (17,2) in pixels
    """
    try:
        nose = kp_xy[0]
        l_sh = kp_xy[5]
        r_sh = kp_xy[6]
    except Exception:
        return 0

    neck = np.array([(l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0], dtype=float)
    dx, dy = nose[0] - neck[0], nose[1] - neck[1]
    # Angle of head relative to vertical axis; larger => more tilted/down
    ang = abs(math.degrees(math.atan2(dx, dy)))
    # Vertical drop of nose below neck as fraction of image height
    drop = (nose[1] - neck[1]) / max(img_h, 1)

    # thresholds (tunable)
    if ang > 75 or drop > 0.20:
        return 2  # guc xuong ban
    if ang > 55 or drop > 0.12:
        return 1  # ngu gat
    return 0


def save_label(label_path: Path, cls_id: int, box_xywhn: np.ndarray, kps_xy: np.ndarray, w: int, h: int):
    # Normalize keypoints and set visibility=2 (labeled)
    kps = []
    for j in range(17):
        x = float(np.clip(kps_xy[j, 0] / max(w, 1), 0, 1))
        y = float(np.clip(kps_xy[j, 1] / max(h, 1), 0, 1))
        v = 2
        kps.extend([x, y, v])

    xc, yc, bw, bh = box_xywhn.tolist()
    line = [cls_id, float(xc), float(yc), float(bw), float(bh)] + kps
    with open(label_path, 'a', encoding='utf-8') as f:
        f.write(' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in line) + '\n')


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def iter_media(source: Path, frame_stride: int = 5):
    """Yield (image_bgr, name) from images/videos inside source.
    For videos, sample every frame_stride frames.
    """
    if source.is_dir():
        for p in sorted(source.rglob('*')):
            if is_image_file(p):
                img = cv2.imread(str(p))
                if img is not None:
                    yield img, p.stem
            elif is_video_file(p):
                cap = cv2.VideoCapture(str(p))
                idx = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % frame_stride == 0:
                        yield frame, f"{p.stem}_{idx:06d}"
                    idx += 1
                cap.release()
    else:
        if is_image_file(source):
            img = cv2.imread(str(source))
            if img is not None:
                yield img, source.stem
        elif is_video_file(source):
            cap = cv2.VideoCapture(str(source))
            idx = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_stride == 0:
                    yield frame, f"{source.stem}_{idx:06d}"
                idx += 1
            cap.release()


def main():
    ap = argparse.ArgumentParser(description='Auto-label dataset with YOLO pose + heuristics')
    ap.add_argument('--source', required=True, help='Image/Video file or folder containing media')
    ap.add_argument('--out', default='datasets/sleepy_pose', help='Output dataset base folder')
    ap.add_argument('--model', default='yolo11n-pose.pt', help='Pose model weights')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--frame-stride', type=int, default=5, help='Sample every N frames from videos')
    ap.add_argument('--limit', type=int, default=0, help='Stop after N frames (0 = no limit)')
    args = ap.parse_args()

    base = Path(args.out)
    ensure_dirs(base)

    model = YOLO(args.model)

    src = Path(args.source)
    n_written = 0
    t0 = time.time()

    for frame, name in iter_media(src, frame_stride=args.frame_stride):
        h, w = frame.shape[:2]
        results = model(frame, imgsz=args.imgsz, conf=0.25, verbose=False)
        r0 = results[0]
        if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
            continue

        # Decide split
        split = 'val' if random.random() < args.val_ratio else 'train'
        img_out = base / split / 'images' / f"{name}.jpg"
        lab_out = base / split / 'labels' / f"{name}.txt"
        # Save image once
        if not img_out.exists():
            cv2.imwrite(str(img_out), frame)
        # Clear label file if exists to avoid appending across multiple runs on same name
        if lab_out.exists():
            lab_out.unlink()

        # Gather predictions
        boxes_xywhn = r0.boxes.xywhn.cpu().numpy() if hasattr(r0.boxes, 'xywhn') else None
        keypoints = r0.keypoints.xy.cpu().numpy() if hasattr(r0, 'keypoints') and r0.keypoints is not None else None
        if boxes_xywhn is None or keypoints is None:
            continue

        for i in range(min(len(boxes_xywhn), len(keypoints))):
            kp_xy = keypoints[i]
            cls_id = classify_state(kp_xy, h)
            save_label(lab_out, cls_id, boxes_xywhn[i], kp_xy, w, h)
            n_written += 1

        if args.limit and n_written >= args.limit:
            break

    dt = time.time() - t0
    print(f"Done. Wrote {n_written} annotations in {dt:.1f}s to {base}")


if __name__ == '__main__':
    main()
