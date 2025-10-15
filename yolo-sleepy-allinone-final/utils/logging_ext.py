import csv
import time
from typing import Optional, Dict


class CsvEventLogger:
    def __init__(self, path: str, flush_interval_s: float = 60.0):
        self.path = path
        self.flush_interval_s = flush_interval_s
        self._last_flush = time.time()
        # create header if not exists
        try:
            with open(self.path, 'x', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["timestamp","camera","track_id","state","duration_s"])
        except FileExistsError:
            pass

    def log_event(self, camera_name: str, track_id: int, state: str, duration_s: float) -> None:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([ts, camera_name, track_id, state, f"{duration_s:.2f}"])
        now = time.time()
        if now - self._last_flush > self.flush_interval_s:
            self._last_flush = now


def send_alert(event: Dict) -> None:
    """Stub for webhook alert. Integrate with requests.post later."""
    # Example: requests.post(WEBHOOK_URL, json=event, timeout=2)
    return



