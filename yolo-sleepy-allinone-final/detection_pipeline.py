import time
import threading
from collections import deque
from typing import List, Optional

import numpy as np


class RoundRobinDetector:
    """Shared detector that throttles per-camera inference using round-robin scheduling.

    This class is model-agnostic: pass a callable(model_input) -> result1 per call.
    """
    def __init__(self, infer_fn, max_fps_per_cam: int = 10):
        self.infer_fn = infer_fn
        self.max_fps_per_cam = max_fps_per_cam
        self.inputs: List[dict] = []  # { 'name': str, 'get_frame': callable, 'set_result': callable, 'last_ts': float }
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def add_stream(self, name: str, get_frame, set_result) -> None:
        with self.lock:
            self.inputs.append({'name': name, 'get_frame': get_frame, 'set_result': set_result, 'last_ts': 0.0})

    def remove_stream(self, name: str) -> None:
        with self.lock:
            self.inputs = [x for x in self.inputs if x['name'] != name]

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)

    def _loop(self):
        idx = 0
        min_interval = 1.0 / max(1, self.max_fps_per_cam)
        while self.running:
            with self.lock:
                if not self.inputs:
                    time.sleep(0.05)
                    continue
                item = self.inputs[idx % len(self.inputs)]
            now = time.time()
            if now - item['last_ts'] < min_interval:
                idx += 1
                time.sleep(0.005)
                continue
            frame = item['get_frame']()
            if frame is None:
                idx += 1
                time.sleep(0.005)
                continue
            try:
                result = self.infer_fn(frame)
            except Exception:
                result = None
            item['set_result'](result)
            item['last_ts'] = time.time()
            idx += 1



