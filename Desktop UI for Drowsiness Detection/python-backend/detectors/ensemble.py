# DACN PhatHienNguGat - 2026
"""Ensemble đa-model cho phát hiện ngủ gật (primary) + bấm điện thoại (secondary).

Kiến trúc:

1. **Primary YOLO** (drowsiness detector) — luôn chạy, conf do user chọn.
2. **Secondary YOLO** (phone detector) — chạy song song, conf ≈ 0.7 × primary.
3. **Class-aware NMS merge**:
   - Cùng slug, IoU > ``NMS_SAME_IOU`` → giữ box conf cao hơn.
   - Khác slug, IoU > ``NMS_CROSS_IOU`` → dedupe (ưu tiên primary).
4. **Auto-retry** conf = 0.05 nếu cả 2 đều không ra box nào.
5. **HF classifier fallback** (``transformers.pipeline``) — khi YOLO không bbox
   nào. Chỉ dùng cho ảnh tĩnh (KHÔNG cho realtime).
6. Mapping EN → slug VN qua :mod:`detectors.label_mapping`.

Example:
    >>> ens = EnsembleDetector(primary_path="models/drowsiness_det.pt")
    >>> res = ens.detect_image("test_samples/drowsy.jpg", conf=0.35)
    >>> res.objects[0].class_name, res.objects[0].display_name
    ('ngu_gat', 'Ngủ gật')
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .label_mapping import (
    ALLOWED_SLUGS,
    CONF_FLOOR,
    DISPLAY_NAME,
    SEVERITY,
    _map_any_name,
    passes_floor,
)

__all__ = [
    "Detection",
    "DetectionResult",
    "EnsembleDetector",
]

LOGGER = logging.getLogger("ensemble")

MODELS_DIR: Path = Path(__file__).resolve().parents[1] / "models"

NMS_SAME_IOU: float = 0.55
NMS_CROSS_IOU: float = 0.85
RETRY_CONF: float = 0.05
TOPK_FALLBACK: int = 3

# COCO class id → slug cho trường hợp secondary dùng yolo11n.pt (COCO)
_COCO_SLUG: dict[int, str] = {67: "dien_thoai"}  # cell phone


@dataclass
class Detection:
    """Một bbox đã map sang slug VN.

    Attributes:
        class_name: Slug không dấu (``ngu_gat``, ``dien_thoai``, …).
        display_name: Tiếng Việt có dấu.
        confidence: Xác suất 0–1.
        bbox: [x1, y1, x2, y2] pixel.
        severity: ``danger`` | ``warn`` | ``info``.
        source: ``primary`` | ``secondary`` | ``hf_cls``.
    """

    class_name: str
    display_name: str
    confidence: float
    bbox: list[float]
    severity: str
    source: str


@dataclass
class DetectionResult:
    """Kết quả cho một frame / ảnh.

    Attributes:
        objects: Các bbox đã qua NMS + floor.
        top_k: Dự đoán conf thấp để UI hỏi user xác nhận (khi ``objects`` rỗng).
        inference_time_ms: Tổng thời gian inference của ensemble.
        image_size: ``[width, height]`` px.
    """

    objects: list[Detection] = field(default_factory=list)
    top_k: list[dict[str, Any]] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_size: list[int] = field(default_factory=lambda: [0, 0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [asdict(o) for o in self.objects],
            "top_k": self.top_k,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_size": self.image_size,
        }


def _iou(a: list[float], b: list[float]) -> float:
    """IoU giữa 2 bbox [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _class_aware_merge(dets: list[Detection]) -> list[Detection]:
    """NMS có biết slug: cùng slug dùng ngưỡng thấp, khác slug dùng ngưỡng cao."""
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        drop = False
        for k in kept:
            iou = _iou(d.bbox, k.bbox)
            same = d.class_name == k.class_name
            if same and iou > NMS_SAME_IOU:
                drop = True
                break
            if not same and iou > NMS_CROSS_IOU:
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


class EnsembleDetector:
    """Primary + secondary YOLO với fallback HF classifier.

    Attributes:
        primary: YOLO chính phát hiện ngủ gật.
        secondary: YOLO phụ phát hiện điện thoại (có thể là None).
        hf_cls_pipelines: Dict slug_family → transformers pipeline.
        device: 'cpu' hoặc GPU index.
    """

    def __init__(
        self,
        primary_path: str | Path | None = None,
        classifier_path: str | Path | None = None,
        secondary_path: str | Path | None = None,
        device: str | int = "cpu",
        enable_hf_fallback: bool = True,
    ) -> None:
        from ultralytics import YOLO  # import trễ để script nhẹ khi chỉ test mapping

        # Primary: ưu tiên drowsiness_det.pt nếu có, nếu không → hybrid
        # (person detector yolo11n + classifier drowsiness_cls.pt).
        primary_path = Path(primary_path or (MODELS_DIR / "drowsiness_det.pt"))
        if primary_path.exists():
            self.primary = YOLO(str(primary_path))
            self.primary_path = primary_path
            self.classifier: Any | None = None
            self.hybrid_mode = False
        else:
            person = MODELS_DIR / "yolo11n.pt"
            cls_path = Path(classifier_path or (MODELS_DIR / "drowsiness_cls.pt"))
            assert person.exists(), f"Thiếu person detector: {person}"
            assert cls_path.exists(), f"Thiếu classifier: {cls_path}"
            self.primary = YOLO(str(person))
            self.classifier = YOLO(str(cls_path))
            self.primary_path = person
            self.hybrid_mode = True
            LOGGER.info("Primary = hybrid: %s + %s", person.name, cls_path.name)

        secondary_path = Path(secondary_path or (MODELS_DIR / "phone_det.pt"))
        if secondary_path.exists() and secondary_path != self.primary_path:
            self.secondary = YOLO(str(secondary_path))
            self.secondary_path: Path | None = secondary_path
        else:
            coco = MODELS_DIR / "yolo11n.pt"
            if coco.exists() and coco != self.primary_path:
                self.secondary = YOLO(str(coco))
                self.secondary_path = coco
                LOGGER.info("Secondary fallback: yolo11n.pt COCO (class 67 = phone)")
            else:
                self.secondary = None
                self.secondary_path = None

        self.device = device
        self.hf_cls_pipelines: dict[str, Any] = {}
        if enable_hf_fallback:
            self._load_hf_fallback()

    def _load_hf_fallback(self) -> None:
        """Load transformers pipeline classifier (offline nếu đã snapshot)."""
        try:
            from transformers import pipeline  # type: ignore
        except ImportError:
            LOGGER.info("transformers chưa cài → bỏ HF fallback")
            return

        cache = str(MODELS_DIR / "hf_cache")
        for family, repo in (("mat_nham", "dima806/closed_eyes_image_detection"),
                             ("ngap", "dima806/yawn_image_detection")):
            try:
                self.hf_cls_pipelines[family] = pipeline(
                    "image-classification",
                    model=repo,
                    device=-1 if self.device == "cpu" else 0,
                    model_kwargs={"cache_dir": cache},
                )
                LOGGER.info("HF classifier sẵn sàng: %s", repo)
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("HF classifier %s không load được: %s", repo, exc.__class__.__name__)

    def _yolo_to_dets(self, result: Any, source: str, names: dict[int, str]) -> list[Detection]:
        """Convert 1 YOLO result → list[Detection] đã map slug + floor."""
        out: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for bbox, cls_id, conf in zip(xyxy, cls_ids, confs):
            raw_name = names.get(int(cls_id), "")
            slug, display = _map_any_name(raw_name)
            if slug is None:
                # Thử map theo COCO id (ví dụ secondary = yolo11n.pt COCO)
                slug = _COCO_SLUG.get(int(cls_id))
                display = DISPLAY_NAME.get(slug) if slug else None
            if slug is None or slug not in ALLOWED_SLUGS:
                continue
            if not passes_floor(slug, float(conf)):
                continue
            out.append(Detection(
                class_name=slug,
                display_name=display or slug,
                confidence=float(conf),
                bbox=[float(v) for v in bbox.tolist()],
                severity=SEVERITY.get(slug, "info"),
                source=source,
            ))
        return out

    def _run_yolo(self, model: Any, image: Any, conf: float) -> tuple[list[Detection], dict[int, str]]:
        results = model.predict(source=image, conf=conf, device=self.device, verbose=False)
        if not results:
            return [], {}
        r = results[0]
        names: dict[int, str] = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
        return self._yolo_to_dets(r, source="primary", names=names), names

    def _run_hybrid_primary(self, image: Any, conf: float) -> list[Detection]:
        """Hybrid: person YOLO → crop → drowsiness classifier → emit bbox slug VN."""
        import cv2

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, np.ndarray):
            img = image
        else:
            # PIL.Image → ndarray
            img = np.array(image)[:, :, ::-1].copy()
        if img is None:
            return []

        person_results = self.primary.predict(
            source=img, conf=conf, classes=[0], device=self.device, verbose=False,
        )
        out: list[Detection] = []
        if not person_results:
            return out
        pr = person_results[0]
        if pr.boxes is None or len(pr.boxes) == 0:
            return out

        xyxy = pr.boxes.xyxy.cpu().numpy()
        person_confs = pr.boxes.conf.cpu().numpy()
        for bbox, pconf in zip(xyxy, person_confs):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            cls_res = self.classifier.predict(source=crop, device=self.device, verbose=False)
            if not cls_res:
                continue
            probs = cls_res[0].probs
            if probs is None:
                continue
            top1 = int(probs.top1)
            top1_conf = float(probs.top1conf)
            cls_names = cls_res[0].names
            raw = cls_names[top1] if isinstance(cls_names, dict) else cls_names[top1]
            slug, display = _map_any_name(raw)
            if slug is None or slug not in ALLOWED_SLUGS:
                continue
            if not passes_floor(slug, top1_conf):
                continue
            out.append(Detection(
                class_name=slug,
                display_name=display or slug,
                confidence=top1_conf * float(pconf),
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                severity=SEVERITY.get(slug, "info"),
                source="primary",
            ))
        return out

    def _run_hf_fallback(self, image: Any) -> list[dict[str, Any]]:
        """Chạy HF classifier khi YOLO rỗng — dùng cho ``top_k`` gợi ý."""
        candidates: list[dict[str, Any]] = []
        for family, pipe in self.hf_cls_pipelines.items():
            try:
                preds = pipe(image, top_k=TOPK_FALLBACK)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("HF cls %s lỗi: %s", family, exc.__class__.__name__)
                continue
            for p in preds:
                slug, display = _map_any_name(p.get("label", ""))
                if slug is None:
                    # fallback: nếu family là mat_nham/ngap và label dương
                    low = str(p.get("label", "")).lower()
                    if family == "mat_nham" and ("close" in low or "closed" in low):
                        slug, display = "mat_nham", DISPLAY_NAME["mat_nham"]
                    elif family == "ngap" and "yawn" in low:
                        slug, display = "ngap", DISPLAY_NAME["ngap"]
                if slug is None:
                    continue
                candidates.append({
                    "class_name": slug,
                    "display_name": display,
                    "confidence": float(p.get("score", 0.0)),
                    "source": "hf_cls",
                })
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates[:TOPK_FALLBACK]

    def detect_image(
        self,
        image: Any,
        conf: float = 0.35,
        use_secondary: bool = True,
        use_hf_fallback: bool = True,
    ) -> DetectionResult:
        """Phát hiện trên 1 ảnh (path, np.ndarray hoặc PIL).

        Args:
            image: đường dẫn / ndarray BGR / PIL Image.
            conf: ngưỡng primary (secondary dùng 0.7 × conf).
            use_secondary: tắt để đo riêng primary.
            use_hf_fallback: tắt cho realtime (quá chậm).

        Returns:
            DetectionResult có ``objects``, ``top_k`` (khi empty), ``inference_time_ms``.
        """
        t0 = time.perf_counter()
        result = DetectionResult()

        # Lấy image size (phục vụ UI scale)
        try:
            if isinstance(image, (str, Path)):
                import cv2
                im = cv2.imread(str(image))
                if im is not None:
                    result.image_size = [int(im.shape[1]), int(im.shape[0])]
            elif isinstance(image, np.ndarray):
                result.image_size = [int(image.shape[1]), int(image.shape[0])]
        except Exception:  # noqa: BLE001
            pass

        if self.hybrid_mode:
            primary_dets = self._run_hybrid_primary(image, conf)
        else:
            primary_dets, _ = self._run_yolo(self.primary, image, conf)
            for d in primary_dets:
                d.source = "primary"

        secondary_dets: list[Detection] = []
        if use_secondary and self.secondary is not None:
            sec_conf = max(0.05, conf * 0.7)
            secondary_dets, _ = self._run_yolo(self.secondary, image, sec_conf)
            for d in secondary_dets:
                d.source = "secondary"

        merged = _class_aware_merge(primary_dets + secondary_dets)

        # Auto-retry conf thấp nếu rỗng
        if not merged:
            LOGGER.info("Ensemble rỗng → retry conf=%.2f", RETRY_CONF)
            if self.hybrid_mode:
                primary_retry = self._run_hybrid_primary(image, RETRY_CONF)
            else:
                primary_retry, _ = self._run_yolo(self.primary, image, RETRY_CONF)
            secondary_retry: list[Detection] = []
            if use_secondary and self.secondary is not None:
                secondary_retry, _ = self._run_yolo(self.secondary, image, RETRY_CONF)
            # Ở retry, KHÔNG áp per-class floor — dùng làm top_k gợi ý.
            retries = primary_retry + secondary_retry
            retries.sort(key=lambda d: d.confidence, reverse=True)
            result.top_k = [asdict(d) for d in retries[:TOPK_FALLBACK]]

        # HF fallback khi vẫn rỗng và cho phép
        if not merged and not result.top_k and use_hf_fallback and self.hf_cls_pipelines:
            result.top_k = self._run_hf_fallback(image)

        result.objects = merged
        result.inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return result
