"""
Run a minimal Decorator pipeline demo:
- WebcamStream -> FrameQueueDecorator -> PerformanceDecorator
- Reads frames ~5 seconds, computes instantaneous and average FPS
- Saves last frame (if any) and a small log under docs/

If webcam is not available, falls back to a DummyImageStream that
repeats a local image with a controlled interval.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Any

ROOT = os.path.dirname(os.path.dirname(__file__))
DOCS_DIR = os.path.join(ROOT, 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)


class ICameraStream:
    def read(self) -> Optional[Any]:
        raise NotImplementedError


class WebcamStream(ICameraStream):
    def __init__(self, device_id: int = 0):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device_id)
        # Reduce buffering to lower latency if supported
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[Any]:
        ok, frame = self.cap.read()
        return frame if ok else None


class DummyImageStream(ICameraStream):
    """Repeats a static image at a target interval to simulate a camera."""
    def __init__(self, image_path: str, interval_s: float = 1.0/30.0):
        import cv2
        self.cv2 = cv2
        self.frame = cv2.imread(image_path)
        self.interval_s = interval_s
        self._last = 0.0

    def read(self) -> Optional[Any]:
        now = time.time()
        if now - self._last < self.interval_s:
            time.sleep(max(0.0, self.interval_s - (now - self._last)))
        self._last = time.time()
        return self.frame.copy() if self.frame is not None else None


class StreamDecorator(ICameraStream):
    def __init__(self, inner: ICameraStream):
        self.inner = inner

    def read(self) -> Optional[Any]:
        return self.inner.read()


class FrameQueueDecorator(StreamDecorator):
    def __init__(self, inner: ICameraStream, maxsize: int = 3):
        super().__init__(inner)
        from queue import Queue
        import threading
        self.q = Queue(maxsize=maxsize)
        self.running = True

        def loop():
            while self.running:
                f = self.inner.read()
                if f is None:
                    time.sleep(0.02)
                    continue
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except Exception:
                        pass
                self.q.put_nowait(f)

        self.th = threading.Thread(target=loop, daemon=True)
        self.th.start()

    def read(self) -> Optional[Any]:
        f = None
        # Drain to get the freshest
        while not self.q.empty():
            f = self.q.get()
        return f


class PerformanceDecorator(StreamDecorator):
    def __init__(self, inner: ICameraStream):
        super().__init__(inner)
        self._last = time.time()
        self.fps = 0.0

    def read(self) -> Optional[Any]:
        frame = self.inner.read()
        now = time.time()
        dt = max(now - self._last, 1e-6)
        self.fps = 1.0 / dt
        self._last = now
        return frame


def choose_stream() -> ICameraStream:
    """Try webcam first; if no frames, fallback to DummyImageStream."""
    test_stream = WebcamStream(0)
    # Try a few reads to see if frames arrive
    none_count = 0
    for _ in range(10):
        f = test_stream.read()
        if f is None:
            none_count += 1
            time.sleep(0.05)
    if none_count >= 10:
        # Fallback image path
        img_path = os.path.join(ROOT, 'data_raw', 'cap_000000.jpg')
        return DummyImageStream(img_path, interval_s=1.0/30.0)
    return test_stream


def main():
    # Build pipeline: Source -> FrameQueue -> Performance
    source = choose_stream()
    pipeline: ICameraStream = PerformanceDecorator(FrameQueueDecorator(source, maxsize=3))
    perf = pipeline  # type: ignore

    start = time.time()
    duration = 5.0
    frames = 0
    last_frame = None
    while time.time() - start < duration:
        f = pipeline.read()
        if f is not None:
            frames += 1
            last_frame = f
        else:
            time.sleep(0.01)

    elapsed = time.time() - start
    avg_fps = frames / elapsed if elapsed > 0 else 0.0

    # Save results
    log_path = os.path.join(DOCS_DIR, 'decorator_demo_log.txt')
    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(f'Thời lượng chạy: {elapsed:.2f}s\n')
        fp.write(f'Tổng số frame: {frames}\n')
        fp.write(f'FPS trung bình: {avg_fps:.2f}\n')
        # Instantaneous fps from decorator (if available)
        if isinstance(perf, PerformanceDecorator):
            fp.write(f'FPS tức thời cuối: {perf.fps:.2f}\n')

    if last_frame is not None:
        try:
            import cv2
            out_img = os.path.join(DOCS_DIR, 'decorator_demo_frame.jpg')
            cv2.imwrite(out_img, last_frame)
        except Exception:
            pass

    print(f"Decorator demo complete. Avg FPS: {avg_fps:.2f}. See {log_path}.")


if __name__ == '__main__':
    main()
