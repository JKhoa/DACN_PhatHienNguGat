# DACN PhatHienNguGat - 2026
"""Hybrid multi-student drowsiness + phone usage detector.

Kiến trúc 2-stage:

1. **Detect** bằng ``yolo11n.pt`` (COCO): bbox mọi học sinh (class 0) + điện thoại (class 67).
2. **Classify drowsy** từng học sinh bằng ``drowsiness_cls.pt``
   (yolo11x-cls fine-tuned, ``mosesb/drowsiness-detection-yolo-cls``):
   crop từng person bbox → predict Drowsy / Non Drowsy.

Cùng lúc matching phone ↔ person bằng IoU → phát hiện bấm điện thoại.

Style guide: ``00_System/YOLO_Standard_Style`` trong Obsidian vault.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .multi_task_detector import BehaviorEvent, BehaviorType, _bbox_iou


DEFAULT_MODELS_DIR: Path = Path(__file__).resolve().parents[1] / "models"
DETECTOR_WEIGHTS: Path = DEFAULT_MODELS_DIR / "yolo11n.pt"
DROWSY_CLS_WEIGHTS: Path = DEFAULT_MODELS_DIR / "drowsiness_cls.pt"

PERSON_CLASS_ID: int = 0
PHONE_CLASS_ID: int = 67

CONF_THRESHOLD: float = 0.35
IOU_THRESHOLD: float = 0.5
IMGSZ: int = 640
MAX_DET: int = 50

DROWSY_CONF_THRESHOLD: float = 0.60   # min prob để tin Drowsy
CONSECUTIVE_DROWSY_FRAMES: int = 10   # ~5s @ vid_stride=2, input 30fps
PHONE_PROXIMITY_IOU: float = 0.02
PHONE_SECONDS: float = 3.0


@dataclass
class _TrackState:
    """State per track_id qua nhiều frame."""

    drowsy_streak: int = 0
    phone_near_since: float | None = None
    emitted: set[BehaviorType] = field(default_factory=set)


class HybridDetector:
    """Multi-student drowsiness + phone usage detector.

    Attributes:
        detector (YOLO): yolo11n.pt — bbox person + phone.
        classifier (YOLO): drowsiness_cls.pt — per-person Drowsy/Non Drowsy.
        device (str | int): 'cpu' hoặc GPU index.
        half (bool): FP16 inference.

    Methods:
        stream: Generator yield list[BehaviorEvent] per frame.

    Examples:
        >>> det = HybridDetector(device=0, half=True)
        >>> for events in det.stream("classroom.mp4"):
        ...     for ev in events:
        ...         print(ev.track_id, ev.behavior.value, ev.confidence)
    """

    def __init__(
        self,
        detector_path: str | Path = DETECTOR_WEIGHTS,
        classifier_path: str | Path = DROWSY_CLS_WEIGHTS,
        device: str | int = "cpu",
        half: bool = False,
    ) -> None:
        self.detector = YOLO(str(detector_path))
        self.classifier = YOLO(str(classifier_path))
        self.device = device
        self.half = half
        self._tracks: dict[int, _TrackState] = defaultdict(_TrackState)

        # Verify class names match expectations
        drowsy_names = self.classifier.names
        if not any("drows" in str(v).lower() for v in drowsy_names.values()):
            raise ValueError(
                f"Classifier không có class 'Drowsy'. Got: {drowsy_names}"
            )

    def stream(
        self,
        source: str | int,
        vid_stride: int = 2,
    ) -> "Iterator[list[BehaviorEvent]]":
        """Stream camera và phát hiện hành vi từng frame.

        Args:
            source (str | int): Video path hoặc webcam index.
            vid_stride (int): Bỏ qua frame. Default 2 → ~15 FPS trên input 30 FPS.

        Yields:
            (list[BehaviorEvent]): Sự kiện phát hiện ở frame hiện tại.
        """
        tracker = self.detector.track(
            source=source,
            stream=True,
            persist=True,
            classes=[PERSON_CLASS_ID, PHONE_CLASS_ID],
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            half=self.half,
            device=self.device,
            vid_stride=vid_stride,
            verbose=False,
        )

        for result in tracker:
            yield self._process(result, time.time())

    def _process(self, result, now: float) -> list[BehaviorEvent]:
        """Phân tích 1 frame, trả về events mới emit."""
        events: list[BehaviorEvent] = []
        if result.boxes is None or len(result.boxes) == 0:
            return events

        cls = result.boxes.cls.int().cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id
        if ids is None:
            return events
        ids = ids.int().cpu().numpy()

        person_mask = cls == PERSON_CLASS_ID
        phone_mask = cls == PHONE_CLASS_ID

        person_boxes = xyxy[person_mask]
        person_ids = ids[person_mask]
        person_conf = conf[person_mask]
        phone_boxes = xyxy[phone_mask]

        frame = result.orig_img
        if len(person_boxes) == 0:
            return events

        drowsy_flags = self._classify_persons(frame, person_boxes)

        for pbox, tid, pconf, is_drowsy in zip(
            person_boxes, person_ids, person_conf, drowsy_flags
        ):
            bbox = tuple(float(v) for v in pbox)
            state = self._tracks[int(tid)]

            phone_near = any(
                _bbox_iou(bbox, tuple(pb.tolist())) > PHONE_PROXIMITY_IOU
                for pb in phone_boxes
            )

            if phone_near:
                if state.phone_near_since is None:
                    state.phone_near_since = now
            else:
                state.phone_near_since = None
                state.emitted.discard(BehaviorType.PHONE_USAGE)

            if is_drowsy is True:
                state.drowsy_streak += 1
            else:
                state.drowsy_streak = 0
                state.emitted.discard(BehaviorType.DROWSY)

            ev = self._classify(int(tid), state, phone_near, bbox, float(pconf), now)
            if ev is not None:
                events.append(ev)

        return events

    def _classify_persons(
        self,
        frame: np.ndarray,
        person_boxes: np.ndarray,
    ) -> list[bool | None]:
        """Crop mỗi person bbox và chạy classifier.

        Returns:
            (list[bool | None]): True = Drowsy confident, False = Awake, None = low conf.
        """
        if len(person_boxes) == 0:
            return []

        crops: list[np.ndarray] = []
        for x1, y1, x2, y2 in person_boxes.astype(int):
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crops.append(np.zeros((32, 32, 3), dtype=frame.dtype))
            else:
                crops.append(crop)

        results = self.classifier.predict(
            crops,
            imgsz=224,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        flags: list[bool | None] = []
        for r in results:
            probs = r.probs
            if probs is None:
                flags.append(None)
                continue
            drowsy_idx = next(
                (i for i, n in self.classifier.names.items() if "drows" in n.lower() and "non" not in n.lower()),
                0,
            )
            p_drowsy = float(probs.data[drowsy_idx])
            if p_drowsy >= DROWSY_CONF_THRESHOLD:
                flags.append(True)
            elif p_drowsy <= 1 - DROWSY_CONF_THRESHOLD:
                flags.append(False)
            else:
                flags.append(None)
        return flags

    def _classify(
        self,
        tid: int,
        state: _TrackState,
        phone_near: bool,
        bbox: tuple[float, float, float, float],
        conf: float,
        now: float,
    ) -> BehaviorEvent | None:
        """Quyết định event dựa trên state."""
        if phone_near and state.phone_near_since is not None:
            dur = now - state.phone_near_since
            if dur >= PHONE_SECONDS and BehaviorType.PHONE_USAGE not in state.emitted:
                state.emitted.add(BehaviorType.PHONE_USAGE)
                return BehaviorEvent(tid, BehaviorType.PHONE_USAGE, state.phone_near_since, dur, conf, bbox)

        if (
            state.drowsy_streak >= CONSECUTIVE_DROWSY_FRAMES
            and not phone_near
            and BehaviorType.DROWSY not in state.emitted
        ):
            state.emitted.add(BehaviorType.DROWSY)
            duration = state.drowsy_streak * 2 / 15.0  # ~ số giây (vid_stride=2, fps=15 sau stride)
            return BehaviorEvent(tid, BehaviorType.DROWSY, now - duration, duration, conf, bbox)

        return None


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
