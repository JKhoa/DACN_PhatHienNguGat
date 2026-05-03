# DACN PhatHienNguGat - 2026
"""Smoke test EnsembleDetector với vài ảnh mẫu thật trong ``test_samples/``."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from detectors import EnsembleDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SAMPLES_DIR = Path(__file__).parent / "test_samples"


def main() -> None:
    samples = sorted(SAMPLES_DIR.glob("sample_*.jpg"))[:5]
    assert samples, "Không có ảnh mẫu"

    # Tắt HF fallback để smoke test nhanh
    ens = EnsembleDetector(enable_hf_fallback=False)

    for img in samples:
        res = ens.detect_image(str(img), conf=0.30)
        summary = {
            "image": img.name,
            "time_ms": round(res.inference_time_ms, 1),
            "n_objects": len(res.objects),
            "objects": [
                {
                    "class": o.class_name,
                    "display": o.display_name,
                    "conf": round(o.confidence, 3),
                    "severity": o.severity,
                    "src": o.source,
                }
                for o in res.objects[:5]
            ],
            "top_k": res.top_k[:3],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
