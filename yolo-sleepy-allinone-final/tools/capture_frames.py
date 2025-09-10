import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser(description="Capture frames from webcam to a folder")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default="datasets/raw_capture")
    ap.add_argument("--secs", type=int, default=15, help="Duration to capture (seconds)")
    ap.add_argument("--fps", type=float, default=2.0, help="Frames per second to save")
    ap.add_argument("--res", default="640x480")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = [int(x) for x in args.res.lower().split("x")]
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    interval = 1.0 / max(args.fps, 0.1)
    end_time = time.time() + max(args.secs, 1)
    idx = 0
    last = 0.0
    while time.time() < end_time:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        t = time.time()
        if t - last >= interval:
            out_path = out_dir / f"cap_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            idx += 1
            last = t
    cap.release()
    print(f"Saved {idx} frames to {out_dir}")


if __name__ == "__main__":
    main()
