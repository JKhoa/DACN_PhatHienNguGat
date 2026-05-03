# DACN PhatHienNguGat - 2026
"""Multi-task classroom behavior detector.

Detects three behaviors simultaneously on a single frame:

1. ``drowsy``      — học sinh cúi mặt > DROWSY_SECONDS, không có điện thoại gần.
2. ``phone_usage`` — có điện thoại (COCO class 67) trong vùng IoU của học sinh > 0.
3. ``distracted``  — học sinh cúi mặt > DISTRACTED_SECONDS nhưng dưới ngưỡng drowsy
                     và không có phone.

Style guide: ``00_System/YOLO_Standard_Style`` trong Obsidian vault.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics chưa được cài. Chạy: pip install ultralytics"
    ) from exc


# ----- Hằng số ngưỡng (hiệu chỉnh khi test thực tế) --------------------------

POSE_MODEL: str = "yolo11n-pose.pt"
OBJECT_MODEL: str = "yolo11n.pt"

PERSON_CLASS_ID: int = 0
PHONE_CLASS_ID: int = 67

CONF_THRESHOLD: float = 0.35
IOU_THRESHOLD: float = 0.5
IMGSZ: int = 640
MAX_DET: int = 50

HEAD_DOWN_ANGLE: float = 35.0        # độ, nose-shoulder-hip (gần 90 = ngồi thẳng)
DROWSY_SECONDS: float = 5.0
DISTRACTED_SECONDS: float = 3.0
PHONE_PROXIMITY_IOU: float = 0.02    # IoU phone ↔ person bbox > ngưỡng này = gần
HISTORY_SECONDS: float = 10.0


class BehaviorType(str, Enum):
    """Phân loại hành vi bất thường."""

    DROWSY = "drowsy"
    PHONE_USAGE = "phone_usage"
    DISTRACTED = "distracted"


@dataclass
class BehaviorEvent:
    """Một sự kiện hành vi bất thường được phát hiện.

    Attributes:
        track_id (int): ID tracking của học sinh (từ YOLO tracker).
        behavior (BehaviorType): Loại hành vi.
        start_time (float): Unix timestamp bắt đầu.
        duration (float): Số giây hành vi đã kéo dài.
        confidence (float): Độ tin cậy 0..1.
        bbox (tuple[float, float, float, float]): (x1, y1, x2, y2) của học sinh.
    """

    track_id: int
    behavior: BehaviorType
    start_time: float
    duration: float
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class _TrackState:
    """Trạng thái nội bộ của 1 track qua nhiều frame."""

    head_down_since: float | None = None
    phone_near_since: float | None = None
    recent_angles: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    emitted: set[BehaviorType] = field(default_factory=set)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Tính IoU giữa hai bbox (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _head_tilt_angle(keypoints_xy: np.ndarray) -> float | None:
    """Tính góc nghiêng đầu dựa vào nose và trung điểm vai.

    Args:
        keypoints_xy (np.ndarray): Shape (17, 2) COCO pose keypoints.

    Returns:
        (float | None): Góc (độ) giữa vector nose→shoulder và trục dọc,
            hoặc None nếu keypoint không đủ tin cậy.
    """
    nose = keypoints_xy[0]
    l_sh, r_sh = keypoints_xy[5], keypoints_xy[6]
    if np.any(nose == 0) or np.any(l_sh == 0) or np.any(r_sh == 0):
        return None
    mid_sh = (l_sh + r_sh) / 2.0
    dx = nose[0] - mid_sh[0]
    dy = mid_sh[1] - nose[1]  # y tăng xuống → đảo dấu
    if dy <= 0:
        return 0.0
    return float(np.degrees(np.arctan2(abs(dx), dy)))


class MultiTaskDetector:
    """Phát hiện đồng thời drowsy, phone_usage, distracted trên stream camera.

    Attributes:
        pose_model (YOLO): Model pose để lấy keypoints học sinh.
        object_model (YOLO): Model object để phát hiện điện thoại.
        device (str | int): 'cpu', hoặc GPU index như 0.
        half (bool): FP16 — bật khi có GPU.

    Methods:
        stream: Generator yield list[BehaviorEvent] mỗi frame.

    Examples:
        >>> det = MultiTaskDetector(device=0, half=True)
        >>> for events in det.stream("classroom_cam1.mp4"):
        ...     for ev in events:
        ...         print(ev.track_id, ev.behavior.value, ev.duration)
    """

    def __init__(
        self,
        pose_model_path: str = POSE_MODEL,
        object_model_path: str = OBJECT_MODEL,
        device: str | int = "cpu",
        half: bool = False,
    ) -> None:
        self.pose_model = YOLO(pose_model_path)
        self.object_model = YOLO(object_model_path)
        self.device = device
        self.half = half
        self._tracks: dict[int, _TrackState] = defaultdict(_TrackState)

    def stream(
        self,
        source: str | int,
        vid_stride: int = 2,
    ) -> "Iterator[list[BehaviorEvent]]":
        """Stream camera và yield sự kiện mỗi frame.

        Args:
            source (str | int): Đường dẫn video hoặc webcam index.
            vid_stride (int): Bỏ qua frame. Default 2 → ~15 FPS cho input 30 FPS.

        Yields:
            (list[BehaviorEvent]): Các sự kiện phát hiện ở frame hiện tại.
        """
        pose_stream = self.pose_model.track(
            source=source,
            stream=True,
            persist=True,
            classes=[PERSON_CLASS_ID],
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            half=self.half,
            device=self.device,
            vid_stride=vid_stride,
            verbose=False,
        )

        for pose_result in pose_stream:
            now = time.time()
            frame = pose_result.orig_img

            obj_result = self.object_model.predict(
                source=frame,
                classes=[PHONE_CLASS_ID],
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                half=self.half,
                device=self.device,
                verbose=False,
            )[0]

            phone_boxes = self._extract_bboxes(obj_result)
            events = self._process_frame(pose_result, phone_boxes, now)
            yield events

    def _extract_bboxes(self, result) -> list[tuple[float, ...]]:
        """Lấy bbox (x1, y1, x2, y2) từ result.boxes."""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return [tuple(b.tolist()) for b in result.boxes.xyxy.cpu().numpy()]

    def _process_frame(
        self,
        pose_result,
        phone_boxes: list[tuple[float, ...]],
        now: float,
    ) -> list[BehaviorEvent]:
        """Phân tích 1 frame và trả về danh sách sự kiện mới.

        Args:
            pose_result: Kết quả từ pose_model (có keypoints + track id).
            phone_boxes (list): Bbox điện thoại trên cùng frame.
            now (float): Timestamp hiện tại.

        Returns:
            (list[BehaviorEvent]): Sự kiện được emit lần đầu khi vượt ngưỡng.
        """
        events: list[BehaviorEvent] = []
        if pose_result.keypoints is None or pose_result.boxes is None:
            return events
        if pose_result.boxes.id is None:
            return events

        track_ids = pose_result.boxes.id.int().cpu().tolist()
        person_boxes = pose_result.boxes.xyxy.cpu().numpy()
        confidences = pose_result.boxes.conf.cpu().numpy()
        keypoints = pose_result.keypoints.xy.cpu().numpy()

        for tid, pbox, conf, kpts in zip(
            track_ids, person_boxes, confidences, keypoints
        ):
            state = self._tracks[tid]
            bbox = tuple(float(v) for v in pbox)

            phone_near = any(
                _bbox_iou(bbox, pb) > PHONE_PROXIMITY_IOU for pb in phone_boxes
            )

            angle = _head_tilt_angle(kpts)
            head_down = angle is not None and angle > HEAD_DOWN_ANGLE
            if angle is not None:
                state.recent_angles.append(angle)

            if head_down:
                if state.head_down_since is None:
                    state.head_down_since = now
            else:
                state.head_down_since = None
                state.emitted.discard(BehaviorType.DROWSY)
                state.emitted.discard(BehaviorType.DISTRACTED)

            if phone_near:
                if state.phone_near_since is None:
                    state.phone_near_since = now
            else:
                state.phone_near_since = None
                state.emitted.discard(BehaviorType.PHONE_USAGE)

            event = self._classify(tid, state, phone_near, bbox, float(conf), now)
            if event is not None:
                events.append(event)

        return events

    def _classify(
        self,
        tid: int,
        state: _TrackState,
        phone_near: bool,
        bbox: tuple[float, float, float, float],
        conf: float,
        now: float,
    ) -> BehaviorEvent | None:
        """Quyết định loại hành vi dựa trên state và điều kiện hiện tại."""
        if phone_near and state.phone_near_since is not None:
            dur = now - state.phone_near_since
            if dur >= DISTRACTED_SECONDS and BehaviorType.PHONE_USAGE not in state.emitted:
                state.emitted.add(BehaviorType.PHONE_USAGE)
                return BehaviorEvent(tid, BehaviorType.PHONE_USAGE, state.phone_near_since, dur, conf, bbox)

        if state.head_down_since is None:
            return None

        dur = now - state.head_down_since
        if phone_near:
            return None  # đã xử lý ở nhánh phone_usage

        if dur >= DROWSY_SECONDS and BehaviorType.DROWSY not in state.emitted:
            state.emitted.add(BehaviorType.DROWSY)
            return BehaviorEvent(tid, BehaviorType.DROWSY, state.head_down_since, dur, conf, bbox)

        if dur >= DISTRACTED_SECONDS and BehaviorType.DISTRACTED not in state.emitted:
            state.emitted.add(BehaviorType.DISTRACTED)
            return BehaviorEvent(tid, BehaviorType.DISTRACTED, state.head_down_since, dur, conf, bbox)

        return None


# Type-only import để tránh circular cost tại runtime ---------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
