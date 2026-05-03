# DACN PhatHienNguGat - 2026
"""Tải weight mã nguồn mở cho pipeline phát hiện ngủ gật + bấm điện thoại.

Chiến lược:

1. **Drowsiness classifier** (đã có sẵn ``models/drowsiness_cls.pt`` — yolo11x-cls
   fine-tuned bởi ``mosesb/drowsiness-detection-yolo-cls``). Nếu thiếu → tải lại.
2. **Drowsiness / face detector** (YOLO .pt): thử các repo ứng viên trên HF Hub,
   dừng ở repo đầu tiên tải được file ``.pt``.
3. **Phone detector**: COCO ``yolo11n.pt`` đã phủ class 67 (cell phone); script
   xác nhận có mặt và không tải mô hình fine-tune trùng lặp.
4. **HF image-classifier fallback** (transformers) cho nhắm mắt / ngáp — tải
   snapshot offline để ``pipeline("image-classification")`` dùng được khi mất mạng.

Chạy:
    python download_models.py
    python download_models.py --force  # tải lại toàn bộ
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

LOGGER = logging.getLogger("download_models")

MODELS_DIR: Path = Path(__file__).resolve().parent / "models"
HF_CACHE_DIR: Path = MODELS_DIR / "hf_cache"


@dataclass(frozen=True)
class WeightSpec:
    """Mô tả một file weight cần đảm bảo tồn tại trong ``models/``.

    Attributes:
        local_name: Tên file lưu trong ``models/``.
        hf_candidates: Danh sách (repo_id, filename) trên HF Hub — thử lần lượt.
        direct_url: URL fallback (ví dụ GitHub release) khi HF fail.
        required: ``True`` nếu thiếu file này là lỗi chặn.
    """

    local_name: str
    hf_candidates: tuple[tuple[str, str], ...]
    direct_url: str | None = None
    required: bool = True


DROWSY_CLS: WeightSpec = WeightSpec(
    local_name="drowsiness_cls.pt",
    hf_candidates=(("mosesb/drowsiness-detection-yolo-cls", "best.pt"),),
    required=True,
)

PERSON_DETECTOR: WeightSpec = WeightSpec(
    local_name="yolo11n.pt",
    hf_candidates=(("Ultralytics/YOLO11", "yolo11n.pt"),),
    direct_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    required=True,
)

# Detector chuyên ngủ gật (YOLO Ultralytics-compatible). Tại thời điểm 2026-04,
# khảo sát HF Hub cho thấy không có repo drowsiness YOLO .pt nào Ultralytics-format
# và open-access. Do đó primary = hybrid (person YOLO + classifier crops) — xem
# ``detectors/ensemble.py``. Vẫn giữ spec để tương lai có repo mới thì thêm vào.
DROWSY_DETECTOR: WeightSpec = WeightSpec(
    local_name="drowsiness_det.pt",
    hf_candidates=(),
    required=False,
)

# Phone in-hand detector — confirmed: IndUSV/yolov8n-mobile-phone (Ultralytics .pt).
PHONE_DETECTOR: WeightSpec = WeightSpec(
    local_name="phone_det.pt",
    hf_candidates=(
        ("IndUSV/yolov8n-mobile-phone", "yolov8n-mobile-phone.pt"),
    ),
    required=False,
)

HF_CLASSIFIERS: tuple[str, ...] = (
    "dima806/closed_eyes_image_detection",
    "dima806/yawn_image_detection",
)

ALL_WEIGHTS: tuple[WeightSpec, ...] = (
    DROWSY_CLS,
    PERSON_DETECTOR,
    DROWSY_DETECTOR,
    PHONE_DETECTOR,
)


def _ensure_dirs() -> None:
    """Tạo các thư mục đích nếu chưa có."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _try_hf_download(repo_id: str, filename: str, dest: Path) -> bool:
    """Tải 1 file từ HF Hub về ``dest``. Trả về True nếu thành công."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        LOGGER.error("Thiếu huggingface_hub — chạy: pip install huggingface_hub")
        return False

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(HF_CACHE_DIR),
        )
    except Exception as exc:  # noqa: BLE001 — HF có vô số lỗi khác nhau
        LOGGER.info("  HF %s/%s → %s", repo_id, filename, exc.__class__.__name__)
        return False

    shutil.copy(path, dest)
    LOGGER.info("  ✓ HF %s/%s → %s", repo_id, filename, dest.name)
    return True


def _try_direct_download(url: str, dest: Path) -> bool:
    """Tải 1 file qua HTTP GET (fallback khi HF không có)."""
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("  Direct %s → %s", url, exc.__class__.__name__)
        return False

    LOGGER.info("  ✓ Direct %s → %s", url, dest.name)
    return True


def ensure_weight(spec: WeightSpec, force: bool = False) -> bool:
    """Bảo đảm ``models/<spec.local_name>`` tồn tại. Trả về True nếu có file."""
    dest = MODELS_DIR / spec.local_name
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1 << 20)
        LOGGER.info("• %s đã có (%.1f MB) — bỏ qua", spec.local_name, size_mb)
        return True

    LOGGER.info("• Tải %s …", spec.local_name)
    for repo_id, filename in spec.hf_candidates:
        if _try_hf_download(repo_id, filename, dest):
            return True

    if spec.direct_url and _try_direct_download(spec.direct_url, dest):
        return True

    level = logging.ERROR if spec.required else logging.WARNING
    LOGGER.log(level, "  ✗ Không tải được %s (required=%s)", spec.local_name, spec.required)
    return False


def ensure_hf_classifier(repo_id: str) -> bool:
    """Tải snapshot transformers classifier về HF cache local."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        LOGGER.error("Thiếu huggingface_hub cho classifier %s", repo_id)
        return False

    try:
        snapshot_download(repo_id=repo_id, cache_dir=str(HF_CACHE_DIR))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("  ✗ HF classifier %s → %s", repo_id, exc.__class__.__name__)
        return False

    LOGGER.info("  ✓ HF classifier %s sẵn sàng (cache)", repo_id)
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point — trả về mã lỗi (0 nếu mọi required OK)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Tải lại cả khi đã có")
    parser.add_argument("--skip-hf-cls", action="store_true", help="Bỏ qua HF classifiers")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _ensure_dirs()

    LOGGER.info("=== Weight YOLO (.pt) ===")
    ok_required = True
    for spec in ALL_WEIGHTS:
        got = ensure_weight(spec, force=args.force)
        if spec.required and not got:
            ok_required = False

    if not args.skip_hf_cls:
        LOGGER.info("\n=== Classifier transformers (fallback) ===")
        for repo in HF_CLASSIFIERS:
            ensure_hf_classifier(repo)

    LOGGER.info("\nHoàn tất. Thư mục weight: %s", MODELS_DIR)
    return 0 if ok_required else 1


if __name__ == "__main__":
    sys.exit(main())
