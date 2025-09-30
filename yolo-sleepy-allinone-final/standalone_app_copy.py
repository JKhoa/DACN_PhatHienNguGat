import argparse
import time
import os
import math
from collections import deque
from typing import Dict, Tuple, Optional, cast, Any

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import logging

# Import enhanced display functions for multi-person detection
try:
    from enhanced_display import draw_enhanced_multi_person_display, draw_person_id_circles
except ImportError:
    # Fallback if enhanced_display module not available
    def draw_enhanced_multi_person_display(frame, track_data, sleep_status, sleep_start_time, max_sleep_duration, current_time):
        return frame
    def draw_person_id_circles(frame, track_data, sleep_status):
        return frame
try:  # Optional MediaPipe for eye/yawn ROI
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # type: ignore

# Optional GUI imports (PyQt5)
try:  # pragma: no cover
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QGroupBox,
        QFormLayout,
        QSpinBox,
        QDoubleSpinBox,
        QCheckBox,
        QComboBox,
        QLineEdit,
        QFileDialog,
        QTextEdit,
        QTabWidget,
    )
    HAS_QT = True
except Exception:
    HAS_QT = False
    # Minimal stubs to satisfy static analyzers when PyQt5 is missing
    class _QtStub:
        AlignCenter = 0
        KeepAspectRatio = 0
        SmoothTransformation = 0
    Qt = _QtStub()  # type: ignore
    QTimer = object  # type: ignore
    QImage = object  # type: ignore
    QPixmap = object  # type: ignore
    QApplication = object  # type: ignore
    QMainWindow = object  # type: ignore
    QWidget = object  # type: ignore
    QLabel = object  # type: ignore
    QPushButton = object  # type: ignore
    QVBoxLayout = object  # type: ignore
    QHBoxLayout = object  # type: ignore
    QGroupBox = object  # type: ignore
    QFormLayout = object  # type: ignore
    QSpinBox = object  # type: ignore
    QDoubleSpinBox = object  # type: ignore
    QCheckBox = object  # type: ignore
    QComboBox = object  # type: ignore
    QLineEdit = object  # type: ignore
    QFileDialog = object  # type: ignore
    QTextEdit = object  # type: ignore
    QTabWidget = object  # type: ignore


def draw_text_unicode(img, text, pos, color=(255, 255, 255), font_size=20):
    """Draw Unicode text on a numpy image using PIL and return numpy image back."""
    img_pil = Image.fromarray(img)
    # Try common fonts for Vietnamese; fallback to default
    font = None
    for f in [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            font = ImageFont.truetype(f, font_size)
            break
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)


def draw_panel(img, x, y, w, h, bg=(0, 0, 0), alpha=0.55, border=(255, 255, 255)):
    """Draw a semi-transparent rectangle panel with optional border."""
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(img.shape[1] - 1, x + w)), int(min(img.shape[0] - 1, y + h))
    if x2 <= x1 or y2 <= y1:
        return img
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)
    return img


def parse_res(txt: str) -> Tuple[int, int]:
    w, h = txt.lower().split("x")
    return int(w), int(h)

# ---------------- Tracking helpers (IoU + SimpleTracker) ---------------- #
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(a_area + b_area - inter, 1e-6)


class SimpleTracker:
    def __init__(self, iou_thr: float = 0.35, max_age: int = 25):
        self.iou_thr = iou_thr
        self.max_age = max_age
        # Instance track store: tid -> { 'bbox': [x1,y1,x2,y2], 'age': int }
        self.tracks = {}  # type: Dict[int, Dict[str, Any]]
        self.next_id = 1

    def update(self, dets):
        # dets: list of [x1,y1,x2,y2]
        assignments: Dict[int, int] = {}
        used_dets = set()
        # age all tracks
        for tid in list(self.tracks.keys()):
            age_val = self.tracks[tid].get("age", 0)
            try:
                age_int = int(age_val)
            except Exception:
                age_int = 0
            self.tracks[tid]["age"] = age_int + 1

        # greedy matching by IoU
        while True:
            best_tid, best_di, best_iou = None, None, 0.0
            for tid, tr in self.tracks.items():
                try:
                    age = int(tr.get("age", 0))
                except Exception:
                    age = 0
                if age > self.max_age:
                    continue
                tb = tr.get("bbox", None)
                if tb is None:
                    continue
                for di, dbox in enumerate(dets):
                    if di in used_dets:
                        continue
                    ov = iou_xyxy(tb, dbox)
                    if ov > best_iou:
                        best_tid, best_di, best_iou = tid, di, ov
            if best_tid is None or best_di is None or best_iou < self.iou_thr:
                break
            self.tracks[best_tid]["bbox"] = dets[best_di]
            self.tracks[best_tid]["age"] = 0
            assignments[best_tid] = best_di
            used_dets.add(best_di)

        # new tracks for unmatched dets
        for di, dbox in enumerate(dets):
            if di in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": dbox, "age": 0}
            assignments[tid] = di

        # prune old tracks
        for tid in list(self.tracks.keys()):
            try:
                age = int(self.tracks[tid].get("age", 0))
            except Exception:
                age = 0
            if tid not in assignments and age > self.max_age:
                del self.tracks[tid]
        return assignments


def open_capture(idx: int, res_txt: str, use_mjpg: bool = True):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(int(idx), backend)
        if cap.isOpened():
            w, h = parse_res(res_txt)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if use_mjpg:
                try:
                    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
                    if callable(fourcc_fn):
                        fourcc = cast(int, fourcc_fn(*"MJPG"))
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                except Exception:
                    pass
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass
            return cap
    return None


def classify_pose(k: np.ndarray, img_h: int, img_w: int, box_xyxy: Optional[Tuple[float, float, float, float]] = None) -> Tuple[str, float, float]:
    """
    Heuristic classification using keypoints (+ optional bbox for head-drop vs body alignment).
    Returns (state, angle_v_deg, drop_ratio_h)
    - state in {"Bình thường", "Ngủ gật", "Gục xuống bàn"}
    """
    # Need at least nose (0) and shoulders (5,6)
    if len(k) < 7:
        return "Bình thường", 0.0, 0.0

    nose = k[0]
    l_sh = k[5]
    r_sh = k[6]

    # If shoulder(s) missing (0,0), treat as unavailable
    def valid(p):
        return p[0] > 0 and p[1] > 0

    have_l = valid(l_sh)
    have_r = valid(r_sh)
    if have_l and have_r:
        neck = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        shoulder_w = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
    elif have_l:
        neck = (l_sh[0], l_sh[1])
        shoulder_w = img_w * 0.18
    elif have_r:
        neck = (r_sh[0], r_sh[1])
        shoulder_w = img_w * 0.18
    else:
        # no shoulders -> use image center height as proxy
        neck = (nose[0], nose[1] - img_h * 0.12)
        shoulder_w = img_w * 0.2

    dx = nose[0] - neck[0]
    dy = nose[1] - neck[1]
    # Angle vs vertical axis (bigger -> more tilt sideways/forward)
    angle_v = abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))
    drop_pix = dy  # nose below neck -> positive
    drop_h_ratio = float(drop_pix) / max(img_h, 1)
    drop_sw_ratio = float(drop_pix) / max(shoulder_w, 1e-6)
    drop_bb_ratio = 0.0
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        box_h = max(1.0, (y2 - y1))
        drop_bb_ratio = float(drop_pix) / box_h

    # Tuned thresholds: include drop_bb_ratio for "cúi mặt xuống nhưng ngồi thẳng"
    if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65 or drop_bb_ratio > 0.70:
        return "Gục xuống bàn", angle_v, drop_h_ratio
    if angle_v > 25 or drop_h_ratio > 0.12 or drop_sw_ratio > 0.35 or drop_bb_ratio > 0.40:
        return "Ngủ gật", angle_v, drop_h_ratio
    return "Bình thường", angle_v, drop_h_ratio


def main():
    ap = argparse.ArgumentParser(description="Real-time Sleepy Detection (YOLO Pose)")
    ap.add_argument(
        "--model",
        default=os.path.join(
            os.path.dirname(__file__),
            "runs",
            "pose-train",
            "sleepy_pose_v11n3",
            "weights",
            "best.pt",
        ),
        help="Pose model weights",
    )
    ap.add_argument(
        "--model-version", 
        choices=["v5", "v8", "v11"], 
        default="v11",
        help="YOLO model version to use (v5, v8, v11)"
    )
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--res", default="1280x720", help="camera resolution WxH")
    ap.add_argument("--video", default=None, help="Path to input video file")
    ap.add_argument("--save", default=None, help="Path to save annotated video (e.g. out.mp4)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--flip", choices=["none", "h", "v", "180"], default="none")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--lw", type=int, default=2, help="line width")
    ap.add_argument("--mjpg", action="store_true", help="prefer MJPG")
    ap.add_argument("--image", default=None, help="Run on a single image path instead of webcam")
    ap.add_argument("--max-people", type=int, default=5, help="Max number of persons to track/overlay (default: 5)")
    ap.add_argument("--hide-boxes", action="store_true", help="Hide default YOLO boxes/labels (show only our overlays)")
    ap.add_argument("--enhanced-display", action="store_true", help="Use enhanced multi-person display with statistics panel")
    ap.add_argument("--person-circles", action="store_true", help="Show person ID circles for easier tracking")
    # Optional: integrate eye/yawn techniques (from driving project)
    ap.add_argument("--enable-eyes", action="store_true", help="Enable eye/yawn detectors with ROI via MediaPipe")
    ap.add_argument(
        "--eye-weights",
        default=os.path.join("..", "real-time-drowsy-driving-detection", "runs", "detecteye", "train", "weights", "best.pt"),
        help="Path to eye detector weights (Open/Close)",
    )
    ap.add_argument(
        "--yawn-weights",
        default=os.path.join("..", "real-time-drowsy-driving-detection", "runs", "detectyawn", "train", "weights", "best.pt"),
        help="Path to yawn detector weights",
    )
    ap.add_argument("--secondary-interval", type=int, default=2, help="Run eye/yawn every N frames to save time")
    ap.add_argument("--microsleep-thresh", type=float, default=3.0, help="Seconds of eyes-closed to trigger sleepy")
    ap.add_argument("--yawn-thresh", type=float, default=7.0, help="Seconds of continuous yawn to warn")
    ap.add_argument("--gui", action="store_true", help="Launch full GUI application (default if PyQt5 available)")
    ap.add_argument("--cli", action="store_true", help="Run in console mode instead of GUI")
    ap.add_argument("--yolo-verbose", action="store_true", help="Show Ultralytics per-frame logs in console")
    args = ap.parse_args()

    # Optionally silence Ultralytics logger
    if not args.yolo_verbose:
        try:
            from ultralytics.utils import LOGGER as ULTRA_LOGGER
            ULTRA_LOGGER.setLevel(logging.ERROR)
        except Exception:
            pass

    # Auto-select model based on version if default model is not found
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"⚠️  Model không tìm thấy: {model_path}")
        print(f"🔄 Tự động chọn model {args.model_version}...")
        
        if args.model_version == "v5":
            # Try YOLOv5 models (available locally)
            v5_models = [
                "../yolov5/yolov5n.pt",
                "yolov5nu.pt",  # YOLOv5n Ultralytics version
                "yolov5n.pt"
            ]
            for v5_model in v5_models:
                test_path = os.path.join(os.path.dirname(__file__), v5_model)
                if os.path.exists(test_path):
                    model_path = test_path
                    break
            else:
                model_path = "yolov5nu.pt"  # YOLOv5n Ultralytics - will auto-download
        elif args.model_version == "v8":
            # Try YOLOv8 models
            v8_models = [
                "yolo8n-pose.pt",  
                "yolov8n-pose.pt"
            ]
            for v8_model in v8_models:
                test_path = os.path.join(os.path.dirname(__file__), v8_model)
                if os.path.exists(test_path):
                    model_path = test_path
                    break
            else:
                model_path = "yolov8n-pose.pt"  # Will auto-download
        else:  # v11 
            # Try YOLOv11 models (default)
            v11_models = [
                "yolo11n-pose.pt",
                "yolo11s-pose.pt", 
                "yolo11m-pose.pt"
            ]
            for v11_model in v11_models:
                test_path = os.path.join(os.path.dirname(__file__), v11_model)
                if os.path.exists(test_path):
                    model_path = test_path
                    break
            else:
                model_path = "yolo11n-pose.pt"  # Will auto-download
        
        print(f"✅ Sử dụng model: {model_path}")

    model = YOLO(model_path)

    # States for video mode
    SLEEP_FRAMES = 15
    AWAKE_FRAMES = 5  # confirm wake-up faster than going sleepy
    sleep_states: Dict[int, int] = {}
    awake_states: Dict[int, int] = {}
    sleep_status: Dict[int, str] = {}
    sleep_start_time: Dict[int, float] = {}
    max_sleep_duration: Dict[int, float] = {}
    log = deque(maxlen=20)
    ema_fps = None

    # Default to GUI when available, but only when not using --image/--video and not --cli
    if not args.cli and not args.image and not args.video:
        if HAS_QT:
            run_gui(args)
            return
        else:
            print("PyQt5 chưa cài. Tự động chạy chế độ console.")

    if args.image:
        # Single image mode
        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: Không đọc được ảnh: {args.image}")
            return
        res = model(img, imgsz=args.imgsz, conf=args.conf, verbose=False)
        r0 = res[0] if res and len(res) > 0 else None
        vis = img.copy() if (r0 is not None and args.hide_boxes) else (r0.plot(line_width=args.lw, conf=True) if r0 is not None else img.copy())
        if r0 is not None and hasattr(r0, "keypoints"):
            kps_all = list(r0.keypoints)
            boxes_xyxy = (
                r0.boxes.xyxy.cpu().numpy().tolist()
                if (hasattr(r0, "boxes") and r0.boxes is not None and hasattr(r0.boxes, "xyxy"))
                else []
            )
            confs = (
                r0.boxes.conf.cpu().numpy()
                if (hasattr(r0, "boxes") and r0.boxes is not None and hasattr(r0.boxes, "conf"))
                else None
            )
            idxs = list(range(len(kps_all)))
            if confs is not None:
                idxs = np.argsort(-confs).tolist()
            idxs = idxs[: max(1, args.max_people)]
            kps = [kps_all[i] for i in idxs]
            sel_boxes = [boxes_xyxy[i] for i in idxs] if boxes_xyxy else [None] * len(kps)
            for si, kp in enumerate(kps):
                k = kp.xy[0].cpu().numpy()
                box = sel_boxes[si] if sel_boxes and si < len(sel_boxes) else None
                state, ang, drop = classify_pose(k, vis.shape[0], vis.shape[1], tuple(box) if box else None)
                color = (0, 255, 0)
                if state == "Ngủ gật":
                    color = (0, 0, 255)
                elif state == "Gục xuống bàn":
                    color = (255, 0, 255)
                label = f"{state} ({ang:.1f}°, {drop*100:.0f}%)"
                if box:
                    x1, y1, x2, y2 = map(int, box)
                    y_label = max(0, y1 - 28)
                    cv2.rectangle(vis, (x1, y_label), (x2, y_label + 26), color, -1)
                    vis = draw_text_unicode(vis, label, (x1 + 5, y_label + 3), color=(255, 255, 255), font_size=18)
                else:
                    nose = k[0]
                    vis = draw_text_unicode(
                        vis, label, (int(nose[0]), max(20, int(nose[1]) - 10)), color=color, font_size=24
                    )
        cv2.imshow("Sleepy Detection - Image", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video/Webcam mode
    writer = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"ERROR: Không mở được video: {args.video}")
            return
        win_name = f"Sleepy Detection — {os.path.basename(args.video)} (Press Q to quit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if args.save:
            fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
            fourcc: int = 0
            if callable(fourcc_fn):
                val = fourcc_fn(*"mp4v")  # type: ignore[no-any-return]
                try:
                    fourcc = int(val)  # type: ignore[arg-type]
                except Exception:
                    fourcc = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
    else:
        cap = open_capture(args.cam, args.res, args.mjpg)
        if cap is None:
            print("ERROR: Không mở được webcam. Thử --cam 1 hoặc đổi --res.")
            return
        win_name = "Sleepy Detection — Press Q to quit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Optional eye/yawn components
    eye_model: Optional[YOLO] = None
    yawn_model: Optional[YOLO] = None
    face_mesh = None
    if args.enable_eyes:
        if mp is None:
            print("[eyes] mediapipe chưa cài. Vui lòng: pip install mediapipe")
        else:
            try:
                face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except Exception as e:
                print(f"[eyes] Lỗi khởi tạo FaceMesh: {e}")
                face_mesh = None
        if os.path.exists(args.eye_weights):
            try:
                eye_model = YOLO(args.eye_weights)
            except Exception as e:
                print(f"[eyes] Lỗi tải eye model: {e}")
        else:
            print(f"[eyes] Không tìm thấy eye weights: {args.eye_weights}")
        if os.path.exists(args.yawn_weights):
            try:
                yawn_model = YOLO(args.yawn_weights)
            except Exception as e:
                print(f"[eyes] Lỗi tải yawn model: {e}")
        else:
            print(f"[eyes] Không tìm thấy yawn weights: {args.yawn_weights}")

    # Eye/yawn state variables
    left_closed = False
    right_closed = False
    yawn_in_progress = False
    blinks = 0
    microsleeps = 0.0  # seconds both eyes closed
    yawns = 0
    yawn_duration = 0.0
    frame_idx = 0

    def safe_crop(img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        if x2i <= x1i or y2i <= y1i:
            return None
        roi = img[y1i:y2i, x1i:x2i]
        return roi if roi.size > 0 else None

    def predict_eye(roi) -> Optional[str]:
        if eye_model is None or roi is None:
            return None
        try:
            res = eye_model(roi)
            boxes = res[0].boxes if res and len(res) > 0 else None
            if boxes is None or len(boxes) == 0:
                return None
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            j = int(np.argmax(confs))
            if cls[j] == 1:
                return "Close Eye"
            if cls[j] == 0 and confs[j] > 0.30:
                return "Open Eye"
        except Exception:
            return None
        return None

    def predict_yawn(roi) -> Optional[str]:
        if yawn_model is None or roi is None:
            return None
        try:
            res = yawn_model(roi)
            boxes = res[0].boxes if res and len(res) > 0 else None
            if boxes is None or len(boxes) == 0:
                return None
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            j = int(np.argmax(confs))
            if cls[j] == 0:
                return "Yawn"
            if cls[j] == 1 and confs[j] > 0.50:
                return "No Yawn"
        except Exception:
            return None
        return None

    # Tracking and per-ID states
    tracker = SimpleTracker(iou_thr=0.35, max_age=25)
    angle_hist: Dict[int, deque] = {}

    try:
        while cap.isOpened():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera lỗi hoặc đã ngắt.")
                break

            if args.flip == "h":
                frame = cv2.flip(frame, 1)
            elif args.flip == "v":
                frame = cv2.flip(frame, 0)
            elif args.flip == "180":
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            res = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            r0 = res[0] if res and len(res) > 0 else None
            vis = frame.copy() if (r0 is not None and args.hide_boxes) else (r0.plot(line_width=args.lw, conf=True) if r0 is not None else frame.copy())

            sleepy_count = 0
            if r0 is not None and hasattr(r0, "keypoints"):
                # Collect keypoints and boxes
                kps_all = list(r0.keypoints)
                boxes_xyxy = (
                    r0.boxes.xyxy.cpu().numpy().tolist()
                    if (hasattr(r0, "boxes") and r0.boxes is not None and hasattr(r0.boxes, "xyxy"))
                    else []
                )
                confs = (
                    r0.boxes.conf.cpu().numpy()
                    if (hasattr(r0, "boxes") and r0.boxes is not None and hasattr(r0.boxes, "conf"))
                    else None
                )
                # Select top-N by confidence
                idxs = list(range(len(kps_all)))
                if confs is not None:
                    idxs = np.argsort(-confs).tolist()
                idxs = idxs[: max(1, args.max_people)]
                kps = [kps_all[i] for i in idxs]
                sel_boxes = [boxes_xyxy[i] for i in idxs] if boxes_xyxy else [None] * len(kps)

                # Update tracker
                dets = [b for b in sel_boxes if b is not None]
                assign: Dict[int, int] = {}
                if dets:
                    assign = tracker.update(dets)  # track_id -> det_index

                # Map track_id -> selected index
                id_to_sel: Dict[int, int] = {}
                if dets:
                    det_to_sel: Dict[int, int] = {}
                    di = 0
                    for si, b in enumerate(sel_boxes):
                        if b is not None:
                            det_to_sel[di] = si
                            di += 1
                    for tid, di in assign.items():
                        si = det_to_sel.get(di, None)
                        if si is not None:
                            id_to_sel[tid] = si
                else:
                    # Fallback ids when no boxes available
                    for i in range(len(kps)):
                        id_to_sel[-(i + 1)] = i

                # Optional classes override
                classes = []
                if hasattr(r0, "boxes") and r0.boxes is not None and hasattr(r0.boxes, "cls"):
                    try:
                        classes = r0.boxes.cls.cpu().numpy().tolist()
                    except Exception:
                        classes = []

                # Iterate by track ids
                for tid, si in id_to_sel.items():
                    kp = kps[si]
                    k = kp.xy[0].cpu().numpy()
                    box = sel_boxes[si] if sel_boxes and si < len(sel_boxes) else None

                    # Smooth angle by median of short history per track
                    hq = angle_hist.setdefault(tid, deque(maxlen=8))
                    state_raw, ang_raw, drop = classify_pose(k, vis.shape[0], vis.shape[1], tuple(box) if box else None)
                    hq.append(ang_raw)
                    ang = float(np.median(hq)) if len(hq) >= 3 else ang_raw

                    eff_from_cls = None
                    try:
                        if classes and idxs and si < len(idxs):
                            cls = int(r0.boxes.cls[idxs[si]].item())
                            if cls == 2:
                                eff_from_cls = "Gục xuống bàn"
                            elif cls == 1:
                                eff_from_cls = "Ngủ gật"
                    except Exception:
                        pass

                    base_state = eff_from_cls if eff_from_cls else state_raw

                    # Hysteresis per track id
                    prev_sleep = sleep_states.get(tid, 0)
                    prev_awake = awake_states.get(tid, 0)
                    if base_state in ("Ngủ gật", "Gục xuống bàn"):
                        sleep_states[tid] = prev_sleep + 1
                        awake_states[tid] = 0
                    else:
                        awake_states[tid] = prev_awake + 1
                        sleep_states[tid] = 0

                    prev_status = sleep_status.get(tid, "Bình thường")
                    now = time.time()
                    eff_state = prev_status
                    if prev_status in ("Ngủ gật", "Gục xuống bàn"):
                        if base_state not in ("Ngủ gật", "Gục xuống bàn") and awake_states[tid] >= AWAKE_FRAMES:
                            eff_state = "Bình thường"
                    else:
                        if base_state in ("Ngủ gật", "Gục xuống bàn") and sleep_states[tid] >= SLEEP_FRAMES:
                            eff_state = base_state

                    if prev_status != eff_state:
                        if eff_state in ("Ngủ gật", "Gục xuống bàn"):
                            sleep_start_time[tid] = now
                            log.appendleft(f"[{time.strftime('%M:%S', time.gmtime(now - t0))}] ID {tid}: {eff_state}")
                        elif prev_status in ("Ngủ gật", "Gục xuống bàn") and eff_state == "Bình thường":
                            if tid in sleep_start_time:
                                duration = now - sleep_start_time[tid]
                                max_sleep_duration[tid] = max(max_sleep_duration.get(tid, 0.0), duration)
                                log.appendleft(f"[{time.strftime('%M:%S', time.gmtime(now - t0))}] ID {tid}: Thức dậy ({duration:.1f}s)")
                                del sleep_start_time[tid]
                    sleep_status[tid] = eff_state
                    if eff_state in ("Ngủ gật", "Gục xuống bàn"):
                        sleepy_count += 1

                    # Draw highlighted label near bbox for clarity
                    label = f"ID {tid} — {eff_state}"
                    color = (0, 255, 0)
                    if eff_state == "Ngủ gật":
                        color = (0, 0, 255)
                    elif eff_state == "Gục xuống bàn":
                        color = (255, 0, 255)

                    if box:
                        x1, y1, x2, y2 = map(int, box)
                        y_label = max(0, y1 - 26)
                        cv2.rectangle(vis, (x1, y_label), (x2, y_label + 24), color, -1)
                        vis = draw_text_unicode(vis, label, (x1 + 5, y_label + 3), color=(255, 255, 255), font_size=18)
                    else:
                        nose = k[0]
                        vis = draw_text_unicode(
                            vis,
                            label,
                            (int(nose[0]), max(20, int(nose[1]) - 10)),
                            color=color,
                            font_size=22,
                        )

            # Secondary eye/yawn pipeline (optional)
            sec_info = []
            if args.enable_eyes and face_mesh is not None and (eye_model is not None or yawn_model is not None):
                do_run = (frame_idx % max(1, args.secondary_interval) == 0)
                ih, iw = frame.shape[:2]
                if do_run:
                    try:
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        fm_res = face_mesh.process(image_rgb)
                        if fm_res.multi_face_landmarks:
                            fmarks = fm_res.multi_face_landmarks[0]
                            ids = [187, 411, 152, 68, 174, 399, 298]
                            pts = []
                            for pid in ids:
                                lm = fmarks.landmark[pid]
                                pts.append((lm.x * iw, lm.y * ih))
                            if len(pts) == 7:
                                (x1, y1), (x2, _), (_, y3), (x4, y4), (x5, y5), (x6, y6), (x7, y7) = pts
                                x6a, x7a = min(x6, x7), max(x6, x7)
                                y6a, y7a = min(y6, y7), max(y6, y7)
                                mouth_roi = safe_crop(frame, x1, y1, x2, y3)
                                right_eye_roi = safe_crop(frame, x4, y4, x5, y5)
                                left_eye_roi = safe_crop(frame, x6a, y6a, x7a, y7a)

                                le = predict_eye(left_eye_roi)
                                re = predict_eye(right_eye_roi)
                                yaw = predict_yawn(mouth_roi)

                                prev_both_closed = (left_closed and right_closed)
                                if le == "Close Eye" and re == "Close Eye":
                                    if not prev_both_closed:
                                        blinks += 1
                                    left_closed = True
                                    right_closed = True
                                else:
                                    left_closed = (le == "Close Eye") if le is not None else False
                                    right_closed = (re == "Close Eye") if re is not None else False

                                dt = max(time.time() - t0, 0.0)
                                if left_closed and right_closed:
                                    microsleeps += dt
                                else:
                                    microsleeps = 0.0

                                if yaw == "Yawn":
                                    if not yawn_in_progress:
                                        yawns += 1
                                        yawn_in_progress = True
                                    yawn_duration += dt
                                else:
                                    yawn_in_progress = False
                                    yawn_duration = 0.0

                                sec_info.append(f"👁️ Blinks: {blinks}")
                                sec_info.append(f"💤 Microsleeps: {microsleeps:.1f}s")
                                sec_info.append(f"😮 Yawns: {yawns}")
                                sec_info.append(f"⏳ Yawn Dur: {yawn_duration:.1f}s")

                                if microsleeps >= args.microsleep_thresh or yawn_duration >= args.yawn_thresh:
                                    vis = draw_text_unicode(
                                        vis, "Cảnh báo: buồn ngủ!", (20, 90), color=(0, 0, 255), font_size=24
                                    )
                    except Exception:
                        pass
                else:
                    sec_info.append(f"👁️ Blinks: {blinks}")
                    sec_info.append(f"💤 Microsleeps: {microsleeps:.1f}s")
                    sec_info.append(f"😮 Yawns: {yawns}")
                    sec_info.append(f"⏳ Yawn Dur: {yawn_duration:.1f}s")

            # Max duration panel
            if max_sleep_duration:
                max_time = max(max_sleep_duration.values())
                vis = draw_text_unicode(
                    vis, f"Ngủ gật lâu nhất: {max_time:.1f}s", (20, 60), color=(255, 0, 0), font_size=22
                )

            # HUD
            t1 = time.time()
            fps_now = 1.0 / max(t1 - t0, 1e-6)
            ema_fps = fps_now if ema_fps is None else 0.8 * ema_fps + 0.2 * fps_now
            hud = f"FPS: {ema_fps:.1f}  |  Sleepy: {sleepy_count}"
            vis = draw_text_unicode(vis, hud, (12, 10), color=(50, 255, 50), font_size=22)

            # Log box (adaptive size, top-right)
            vw, vh = vis.shape[1], vis.shape[0]
            lines = list(log)[:8]
            line_h = 22
            pad = 10
            log_w = max(220, min(int(vw * 0.30), 320))
            log_h = 28 + len(lines) * line_h + pad
            log_x = vw - log_w - 12
            log_y = 12
            vis = draw_panel(vis, log_x, log_y, log_w, log_h)
            vis = draw_text_unicode(vis, "Log", (log_x + 10, log_y + 6), color=(255, 255, 0), font_size=18)
            for i, s in enumerate(lines):
                vis = draw_text_unicode(vis, s, (log_x + 10, log_y + 28 + i * line_h), color=(255, 255, 255), font_size=18)

            # Eye/yawn info panel (stacked below log)
            if 'sec_info' in locals() and sec_info:
                info_lines = sec_info
                info_line_h = 20
                info_w = log_w
                info_h = 20 + len(info_lines) * info_line_h + pad
                info_x = log_x
                info_y = min(log_y + log_h + 8, vh - info_h - 8)
                vis = draw_panel(vis, info_x, info_y, info_w, info_h, bg=(0, 30, 0))
                for k, line in enumerate(info_lines):
                    vis = draw_text_unicode(vis, line, (info_x + 10, info_y + 12 + k * info_line_h), color=(200, 255, 200), font_size=18)

            # Enhanced multi-person display
            if args.enhanced_display and len(id_to_sel) > 0:
                # Prepare track data for enhanced display
                track_data = {}
                for tid, si in id_to_sel.items():
                    if si < len(sel_boxes) and sel_boxes[si] is not None:
                        track_data[tid] = {
                            'bbox': sel_boxes[si],
                            'keypoints': kps[si] if si < len(kps) else None
                        }
                
                # Use enhanced display instead of default
                vis = draw_enhanced_multi_person_display(
                    vis, track_data, sleep_status, sleep_start_time, 
                    max_sleep_duration, time.time()
                )
            
            # Add person ID circles if requested
            if args.person_circles and len(id_to_sel) > 0:
                track_data = {}
                for tid, si in id_to_sel.items():
                    if si < len(sel_boxes) and sel_boxes[si] is not None:
                        track_data[tid] = {'bbox': sel_boxes[si]}
                vis = draw_person_id_circles(vis, track_data, sleep_status)

            cv2.imshow(win_name, vis)
            if writer is not None:
                writer.write(vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


#############################
# GUI Application (PyQt5)   #
#############################

def run_gui(args: argparse.Namespace):  # pragma: no cover
    """Load and run the PyQt5 GUI from a separate module to avoid static lint errors here."""
    try:
        import importlib.util
        gui_path = os.path.join(os.path.dirname(__file__), "gui_app.py")
        if not os.path.exists(gui_path):
            print("ERROR: gui_app.py không tồn tại.")
            return
        spec = importlib.util.spec_from_file_location("sleepy_gui_app", gui_path)
        if spec is None or spec.loader is None:
            print("ERROR: Không tải được GUI module.")
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "launch_gui"):
            mod.launch_gui(args)
        else:
            print("ERROR: GUI module không có hàm launch_gui().")
    except ImportError:
        print("ERROR: Cần cài PyQt5: pip install PyQt5")


def classify_pose_custom(k: np.ndarray, img_h: int, img_w: int, angle_thr: float, drop_h_thr: float, drop_sw_thr: float):
    if len(k) < 7:
        return "Bình thường", 0.0, 0.0
    nose, l_sh, r_sh = k[0], k[5], k[6]
    def valid(p):
        return p[0] > 0 and p[1] > 0
    have_l, have_r = valid(l_sh), valid(r_sh)
    if have_l and have_r:
        neck = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        shoulder_w = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
    elif have_l:
        neck = (l_sh[0], l_sh[1]); shoulder_w = img_w * 0.18
    elif have_r:
        neck = (r_sh[0], r_sh[1]); shoulder_w = img_w * 0.18
    else:
        neck = (nose[0], nose[1] - img_h * 0.12); shoulder_w = img_w * 0.2
    dx = nose[0] - neck[0]; dy = nose[1] - neck[1]
    angle_v = abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))
    drop_pix = dy
    drop_h_ratio = float(drop_pix) / max(img_h, 1)
    drop_sw_ratio = float(drop_pix) / max(shoulder_w, 1e-6)
    if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65:
        return "Gục xuống bàn", angle_v, drop_h_ratio
    if angle_v > angle_thr or drop_h_ratio > drop_h_thr or drop_sw_ratio > drop_sw_thr:
        return "Ngủ gật", angle_v, drop_h_ratio
    return "Bình thường", angle_v, drop_h_ratio


def draw_grid(img: np.ndarray, color=(60, 60, 60)):
    h, w = img.shape[:2]
    for i in range(1, 3):
        x = w * i // 3
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for i in range(1, 3):
        y = h * i // 3
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()