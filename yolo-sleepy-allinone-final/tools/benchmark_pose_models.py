import argparse
import time
import cv2
from ultralytics import YOLO


def warmup(model, img, n=5):
    for _ in range(n):
        _ = model(img, imgsz=640, conf=0.25, verbose=False)


def bench(model, img, imgsz: int, iters: int):
    t0 = time.time()
    for _ in range(iters):
        _ = model(img, imgsz=imgsz, conf=0.25, verbose=False)
    t1 = time.time()
    dt = t1 - t0
    fps = iters / dt if dt > 0 else 0.0
    return dt, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11n-pose.pt"], help="List of model weights to test")
    ap.add_argument("--img", default=None, help="Optional path to a test image; if not provided uses webcam frame")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    # Get a frame
    if args.img:
        frame = cv2.imread(args.img)
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.img}")
    else:
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Cannot open camera to fetch a frame. Please provide --img.")

    print(f"Benchmarking on frame shape: {frame.shape}, imgsz={args.imgsz}, iters={args.iters}")
    results = []
    for w in args.models:
        try:
            model = YOLO(w)
        except Exception as e:
            print(f"- {w}: failed to load: {e}")
            continue
        warmup(model, frame, n=3)
        dt, fps = bench(model, frame, args.imgsz, args.iters)
        params = getattr(model, 'model', None)
        n_params = None
        if params is not None and hasattr(params, 'nparams'):
            try:
                n_params = params.nparams
            except Exception:
                n_params = None
        print(f"- {w}: {fps:.2f} FPS over {args.iters} iters; total {dt:.2f}s")
        results.append((w, fps, dt, n_params))

    print("\nSummary:")
    for w, fps, dt, n_params in results:
        extra = f", params={n_params}" if n_params is not None else ""
        print(f"  {w:20s}  {fps:7.2f} FPS  ({dt:.2f}s){extra}")


if __name__ == "__main__":
    main()
