import os
import argparse
import time
from collections import deque
from typing import Optional, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # type: ignore

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QTextEdit,
    QTabWidget,
    QToolBar,
    QAction,
    QStyle,
    QStatusBar,
    QSplitter,
)


# Helpers

def draw_text_unicode(img, text, pos, color=(255, 255, 255), font_size=20):
    img_pil = Image.fromarray(img)
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


def draw_grid(img: np.ndarray, color=(60, 60, 60)):
    h, w = img.shape[:2]
    for i in range(1, 3):
        x = w * i // 3
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for i in range(1, 3):
        y = h * i // 3
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)


def draw_panel(img, x, y, w, h, bg=(0, 0, 0), alpha=0.55, border=(255, 255, 255)):
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(img.shape[1] - 1, x + w)), int(min(img.shape[0] - 1, y + h))
    if x2 <= x1 or y2 <= y1:
        return img
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)
    return img


def classify_pose_custom(k: np.ndarray, img_h: int, img_w: int, angle_thr: float, drop_h_thr: float, drop_sw_thr: float):
    if len(k) < 7:
        return "Bình thường", 0.0, 0.0
    nose, l_sh, r_sh = k[0], k[5], k[6]
    def valid(p):
        return p[0] > 0 and p[1] > 0
    have_l, have_r = valid(l_sh), valid(r_sh)
    if have_l and have_r:
        neck = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        shoulder_w = float(np.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1]))
    elif have_l:
        neck = (l_sh[0], l_sh[1]); shoulder_w = img_w * 0.18
    elif have_r:
        neck = (r_sh[0], r_sh[1]); shoulder_w = img_w * 0.18
    else:
        neck = (nose[0], nose[1] - img_h * 0.12); shoulder_w = img_w * 0.2
    dx = nose[0] - neck[0]; dy = nose[1] - neck[1]
    angle_v = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    drop_pix = dy
    drop_h_ratio = float(drop_pix) / max(img_h, 1)
    drop_sw_ratio = float(drop_pix) / max(shoulder_w, 1e-6)
    if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65:
        return "Gục xuống bàn", float(angle_v), float(drop_h_ratio)
    if angle_v > angle_thr or drop_h_ratio > drop_h_thr or drop_sw_ratio > drop_sw_thr:
        return "Ngủ gật", float(angle_v), float(drop_h_ratio)
    return "Bình thường", float(angle_v), float(drop_h_ratio)


def classify_pose_bbox(
    k: np.ndarray,
    img_h: int,
    img_w: int,
    box_xyxy: Optional[tuple],
    angle_thr: float,
    drop_h_thr: float,
    drop_sw_thr: float,
    drop_bb_ngugat: float = 0.40,
    drop_bb_guc: float = 0.70,
):
    if len(k) < 7:
        return "Bình thường", 0.0, 0.0
    nose, l_sh, r_sh = k[0], k[5], k[6]
    def valid(p):
        return p[0] > 0 and p[1] > 0
    have_l, have_r = valid(l_sh), valid(r_sh)
    if have_l and have_r:
        neck = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        shoulder_w = float(np.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1]))
    elif have_l:
        neck = (l_sh[0], l_sh[1]); shoulder_w = img_w * 0.18
    elif have_r:
        neck = (r_sh[0], r_sh[1]); shoulder_w = img_w * 0.18
    else:
        neck = (nose[0], nose[1] - img_h * 0.12); shoulder_w = img_w * 0.2
    dx = nose[0] - neck[0]; dy = nose[1] - neck[1]
    angle_v = abs(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    drop_pix = dy
    drop_h_ratio = float(drop_pix) / max(img_h, 1)
    drop_sw_ratio = float(drop_pix) / max(shoulder_w, 1e-6)
    drop_bb_ratio = 0.0
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        box_h = max(1.0, (y2 - y1))
        drop_bb_ratio = float(drop_pix) / box_h
    if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65 or drop_bb_ratio > drop_bb_guc:
        return "Gục xuống bàn", float(angle_v), float(drop_h_ratio)
    if (
        angle_v > angle_thr
        or drop_h_ratio > drop_h_thr
        or drop_sw_ratio > drop_sw_thr
        or drop_bb_ratio > drop_bb_ngugat
    ):
        return "Ngủ gật", float(angle_v), float(drop_h_ratio)
    return "Bình thường", float(angle_v), float(drop_h_ratio)


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
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1

    def update(self, dets):
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
    def parse_res(txt: str):
        w, h = txt.lower().split("x")
        return int(w), int(h)
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
                    fourcc = getattr(cv2, "VideoWriter_fourcc", None)
                    if callable(fourcc):
                        four = fourcc(*"MJPG")
                        cap.set(cv2.CAP_PROP_FOURCC, four)  # type: ignore[arg-type]
                except Exception:
                    pass
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass
            return cap
    return None


def qimage_from_bgr(img: np.ndarray):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)


class SleepyWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("Sleepy Detection — Classroom Edition")
        self.args = args

        # Models
        self.pose_model = YOLO(self.args.model)
        self.eye_model = None
        self.yawn_model = None
        self.face_mesh = None
        if self.args.enable_eyes and mp is not None:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self.face_mesh = None
        if self.args.enable_eyes and os.path.exists(self.args.eye_weights):
            try:
                self.eye_model = YOLO(self.args.eye_weights)
            except Exception:
                self.eye_model = None
        if self.args.enable_eyes and os.path.exists(self.args.yawn_weights):
            try:
                self.yawn_model = YOLO(self.args.yawn_weights)
            except Exception:
                self.yawn_model = None

        # Video/capture state
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.source_mode = "camera"  # camera|video|image
        self.video_path = None
        self.image_path = None
        self.frame = None

        # Detection state
        self.SLEEP_FRAMES = 15
        self.AWAKE_FRAMES = 5
        self.sleep_states = {}
        self.awake_states = {}
        self.sleep_status = {}
        self.sleep_start_time = {}
        self.max_sleep_duration = {}
        self.log = deque(maxlen=40)
        self.ema_fps = None
        self.frame_t0 = time.time()
        self.last_proc_time = time.time()
        self.secondary_interval = 2  # run eye/yawn every N frames

        # Eye/yawn counters
        self.left_closed = False
        self.right_closed = False
        self.yawn_in_progress = False
        self.blinks = 0
        self.microsleeps = 0.0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.frame_idx = 0

        # Recording state
        self.writer = None
        self.record_enabled = False
        self.save_path = None
        self.last_frame_size = None

        # Tracker
        self.tracker = SimpleTracker(iou_thr=0.35, max_age=25)
        self.angle_hist = {}

        # Build chrome: toolbar, status bar, theme
        self._init_theme()
        self._init_toolbar()
        self._init_statusbar()

        # UI setup
        self._build_ui()

        # Show current model info on status bar
        try:
            self.sb_left.setText(f"Loaded: {os.path.basename(self.args.model)}")
        except Exception:
            self.sb_left.setText("Model loaded")

    def _safe_crop(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        if x2i <= x1i or y2i <= y1i:
            return None
        roi = img[y1i:y2i, x1i:x2i]
        return roi if roi.size > 0 else None

    def _predict_eye(self, roi) -> Optional[str]:
        if self.eye_model is None or roi is None:
            return None
        try:
            res = self.eye_model(roi)
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

    def _predict_yawn(self, roi) -> Optional[str]:
        if self.yawn_model is None or roi is None:
            return None
        try:
            res = self.yawn_model(roi)
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

    def _build_ui(self):
        cw = QWidget(self)
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)

        # Left: video canvas (in splitter)
        self.video_label = QLabel()
        self.video_label.setObjectName("videoCanvas")
        self.video_label.setMinimumSize(900, 500)
        self.video_label.setStyleSheet("")  # applied via theme
        self.video_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]

        # Right: header + control tabs
        tabs = QTabWidget()
        tabs.setStyleSheet(
            """
            QTabWidget::pane { border: 1px solid #cbd5e1; border-radius: 6px; }
            QTabBar::tab { background: #e2e8f0; padding: 8px 14px; margin: 2px; border-radius: 6px; }
            QTabBar::tab:selected { background: #cfe8ff; }
            QTabBar::tab:hover { background: #ddebf7; }
            """
        )

        # Header
        header = QWidget()
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(0, 0, 0, 6)
        self.logo_label = QLabel("")
        self.logo_label.setFixedHeight(48)
        self.logo_label.setStyleSheet("background: transparent;")
        self.header_title = QLabel("Sleepy Detection — Classroom Edition")
        self.header_title.setStyleSheet("font-size: 18px; font-weight: 600; color: #0f172a;")
        # Quick model selector in header (synced with Settings)
        self.header_model_combo = QComboBox()
        for it in ["YOLOv11n-pose (default)", "YOLOv11s-pose", "YOLOv8n-pose", "YOLOv5n-pose", "Custom…"]:
            self.header_model_combo.addItem(it)
        self.header_model_combo.setCurrentText("YOLOv11n-pose (default)")
        self.btn_choose_logo = QPushButton("Chọn Logo…")
        self.btn_choose_logo.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
        self.btn_choose_logo.clicked.connect(self.on_choose_logo)
        hbox.addWidget(self.logo_label, 0)
        hbox.addWidget(self.header_title, 1)
        hbox.addWidget(self.header_model_combo, 0)
        hbox.addWidget(self.btn_choose_logo, 0)

        # Camera Tab
        cam_tab = QWidget()
        cam_form = QFormLayout(cam_tab)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Camera device", "RTSP/HTTP URL", "Video file", "Image file"])
        cam_form.addRow("Source", self.source_combo)
        self.dev_spin = QSpinBox(); self.dev_spin.setRange(0, 64); self.dev_spin.setValue(self.args.cam)
        cam_form.addRow("Device ID", self.dev_spin)
        self.url_edit = QLineEdit(); self.url_edit.setPlaceholderText("rtsp://user:pass@host:554/stream or http://...")
        cam_form.addRow("URL", self.url_edit)
        self.res_combo = QComboBox()
        for r in ["640x480", "960x540", "1280x720", "1920x1080"]:
            self.res_combo.addItem(r)
        self.res_combo.setCurrentText(self.args.res)
        cam_form.addRow("Resolution", self.res_combo)
        self.mjpg_check = QCheckBox("Prefer MJPG"); self.mjpg_check.setChecked(self.args.mjpg)
        cam_form.addRow(self.mjpg_check)
        self.btn_browse_video = QPushButton("Browse Video...")
        self.btn_browse_image = QPushButton("Browse Image...")
        cam_form.addRow(self.btn_browse_video, self.btn_browse_image)
        # Recording controls
        self.record_check = QCheckBox("Record annotated video")
        self.save_path_edit = QLineEdit(); self.save_path_edit.setPlaceholderText("out.mp4")
        self.btn_browse_save = QPushButton("Save to...")
        row_rec = QHBoxLayout(); row_rec.addWidget(self.record_check); row_rec.addWidget(self.save_path_edit); row_rec.addWidget(self.btn_browse_save)
        cam_form.addRow(row_rec)
        # Snapshot
        self.btn_snapshot = QPushButton("Snapshot")
        cam_form.addRow(self.btn_snapshot)
        self.btn_connect = QPushButton("Connect / Start")
        self.btn_disconnect = QPushButton("Stop")
        row = QHBoxLayout(); row.addWidget(self.btn_connect); row.addWidget(self.btn_disconnect)
        cam_form.addRow(row)
        tabs.addTab(cam_tab, "Source")

        # Settings Tab
        set_tab = QWidget(); set_form = QFormLayout(set_tab)
        # Model selection
        self.model_combo = QComboBox()
        # Presets
        preset_items = [
            "YOLOv11n-pose (default)",
            "YOLOv11s-pose",
            "YOLOv8n-pose",
            "YOLOv5n-pose",
            "Custom…",
        ]
        for it in preset_items:
            self.model_combo.addItem(it)
        self.model_combo.setCurrentText("YOLOv11n-pose (default)")
        self.model_path_edit = QLineEdit(); self.model_path_edit.setText(self.args.model)
        self.btn_browse_model = QPushButton("Browse…")
        row_model = QHBoxLayout(); row_model.addWidget(self.model_combo); row_model.addWidget(self.model_path_edit); row_model.addWidget(self.btn_browse_model)
        set_form.addRow("Pose model", row_model)
        self.conf_spin = QDoubleSpinBox(); self.conf_spin.setRange(0.05, 0.99); self.conf_spin.setSingleStep(0.05); self.conf_spin.setValue(self.args.conf)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(256, 1280); self.imgsz_spin.setSingleStep(32); self.imgsz_spin.setValue(self.args.imgsz)
        self.flip_combo = QComboBox(); self.flip_combo.addItems(["none", "h", "v", "180"]); self.flip_combo.setCurrentText(self.args.flip)
        self.angle_spin = QDoubleSpinBox(); self.angle_spin.setRange(5, 80); self.angle_spin.setValue(25.0)
        self.dropH_spin = QDoubleSpinBox(); self.dropH_spin.setRange(0.05, 0.8); self.dropH_spin.setSingleStep(0.01); self.dropH_spin.setValue(0.12)
        self.dropSW_spin = QDoubleSpinBox(); self.dropSW_spin.setRange(0.1, 1.2); self.dropSW_spin.setSingleStep(0.01); self.dropSW_spin.setValue(0.35)
        self.sleep_frames_spin = QSpinBox(); self.sleep_frames_spin.setRange(1, 90); self.sleep_frames_spin.setValue(self.SLEEP_FRAMES)
        self.awake_frames_spin = QSpinBox(); self.awake_frames_spin.setRange(1, 60); self.awake_frames_spin.setValue(self.AWAKE_FRAMES)
        self.max_people_spin = QSpinBox(); self.max_people_spin.setRange(1, 20); self.max_people_spin.setValue(int(getattr(self.args, "max_people", 1)))
        self.show_grid_check = QCheckBox("Show FOV grid")
        self.enable_eyes_check = QCheckBox("Enable eyes/yawn"); self.enable_eyes_check.setChecked(self.args.enable_eyes)
        set_form.addRow("Conf", self.conf_spin)
        set_form.addRow("imgsz", self.imgsz_spin)
        set_form.addRow("Flip", self.flip_combo)
        set_form.addRow("Angle (°)", self.angle_spin)
        set_form.addRow("Drop H", self.dropH_spin)
        set_form.addRow("Drop ShoulderW", self.dropSW_spin)
        set_form.addRow("Sleep frames", self.sleep_frames_spin)
        set_form.addRow("Awake frames", self.awake_frames_spin)
        set_form.addRow("Max persons", self.max_people_spin)
        set_form.addRow(self.show_grid_check)
        set_form.addRow(self.enable_eyes_check)
        tabs.addTab(set_tab, "Settings")

        # Logs/Stats Tab
        log_tab = QWidget(); v = QVBoxLayout(log_tab)
        self.stats_label = QLabel("Stats will appear here")
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        self.log_text.setObjectName("logText")
        v.addWidget(self.stats_label); v.addWidget(self.log_text)
        tabs.addTab(log_tab, "Logs")

        # Assemble right side
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setObjectName("rightPanel")
        right_layout.addWidget(header)
        right_layout.addWidget(tabs)

        # Splitter for modern resizable layout
        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)  # type: ignore[attr-defined]
        splitter.addWidget(self.video_label)
        splitter.addWidget(right_widget)
        splitter.setSizes([1000, 480])
        root.addWidget(splitter)

        # Signals
        self.btn_browse_video.clicked.connect(self.on_browse_video)
        self.btn_browse_image.clicked.connect(self.on_browse_image)
        self.btn_browse_save.clicked.connect(self.on_browse_save)
        self.btn_connect.clicked.connect(self.on_start)
        self.btn_disconnect.clicked.connect(self.on_stop)
        self.btn_snapshot.clicked.connect(self.on_snapshot)
        self.btn_browse_model.clicked.connect(self.on_browse_model)
        self.model_combo.currentTextChanged.connect(self.on_model_preset_changed)
        self.model_path_edit.editingFinished.connect(self.on_model_path_changed)
        # Sync header and settings combos both ways without loops
        self.header_model_combo.currentTextChanged.connect(self.on_header_model_changed)
        self.model_combo.currentTextChanged.connect(self.on_settings_model_changed)

    def _resolve_preset_path(self, preset_name: str) -> str:
        # Returns a local path or model alias; auto-fallback to existing local weights when possible
        base_dir = os.path.dirname(__file__)
        # Default paths in repo root for v11
        repo_root = os.path.abspath(os.path.join(base_dir, os.pardir))
        v11n = os.path.join(repo_root, "yolo11n-pose.pt")
        v11s = os.path.join(repo_root, "yolo11s-pose.pt")
        v5n = os.path.join(repo_root, "yolov5n-pose.pt")
        if preset_name.startswith("YOLOv11n") and os.path.exists(v11n):
            return v11n
        if preset_name.startswith("YOLOv11s") and os.path.exists(v11s):
            return v11s
        if preset_name.startswith("YOLOv8n"):
            # Ultralytics will auto-download if not present
            return "yolov8n-pose.pt"
        if preset_name.startswith("YOLOv5n"):
            # Check local file first, otherwise use Ultralytics alias
            if os.path.exists(v5n):
                return v5n
            else:
                return "yolov5n-pose.pt"
        # Fallback to current text field
        return self.model_path_edit.text().strip()

    def _load_pose_model(self, path: str):
        try:
            self.pose_model = YOLO(path)
            self.append_log(f"Đã tải model: {os.path.basename(path)}")
            self.sb_left.setText(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self.append_log(f"Lỗi tải model: {e}")

    # Handlers: model selection
    def on_browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn weights .pt", "", "PyTorch Weights (*.pt)")
        if p:
            self.model_combo.setCurrentText("Custom…")
            self.model_path_edit.setText(p)
            self._load_pose_model(p)

    def on_model_preset_changed(self, text: str):
        if text == "Custom…":
            return
        path = self._resolve_preset_path(text)
        if path:
            self.model_path_edit.setText(path)
            self._load_pose_model(path)

    def on_header_model_changed(self, text: str):
        # Update Settings combo without triggering loop
        if self.model_combo.currentText() != text:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentText(text)
            self.model_combo.blockSignals(False)
        self.on_model_preset_changed(text)

    def on_settings_model_changed(self, text: str):
        # Update header combo without triggering loop
        if self.header_model_combo.currentText() != text:
            self.header_model_combo.blockSignals(True)
            self.header_model_combo.setCurrentText(text)
            self.header_model_combo.blockSignals(False)

    def on_model_path_changed(self):
        p = self.model_path_edit.text().strip()
        if p:
            self._load_pose_model(p)

    # Theme & Chrome
    def _init_theme(self):
        # default theme
        self.theme = getattr(self, "theme", "light")
        self.apply_theme(self.theme)

    def _get_light_qss(self) -> str:
        return (
            """
            QMainWindow { background: #f8fafc; }
            QLabel { color: #0f172a; font-size: 13px; }
            QPushButton { background: #e2e8f0; border: 1px solid #cbd5e1; padding: 6px 10px; border-radius: 8px; }
            QPushButton:hover { background: #ddebf7; }
            QPushButton:pressed { background: #cfe8ff; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px; }
            QGroupBox { border: 1px solid #cbd5e1; border-radius: 8px; margin-top: 8px; }
            QStatusBar { background: #eef2ff; color: #1e293b; }
            #videoCanvas { border: 2px solid #d0e2ff; background: #0b1b2b; border-radius: 10px; }
            #rightPanel { background: transparent; }
            #logText { font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 12px; }
            """
        )

    def _get_dark_qss(self) -> str:
        return (
            """
            QMainWindow { background: #0b1220; }
            QLabel { color: #e2e8f0; font-size: 13px; }
            QPushButton { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 6px 10px; border-radius: 8px; }
            QPushButton:hover { background: #243447; }
            QPushButton:pressed { background: #334155; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { background: #0f172a; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px; padding: 6px; }
            QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 8px; }
            QStatusBar { background: #0f172a; color: #93c5fd; }
            #videoCanvas { border: 2px solid #1d4ed8; background: #0b1b2b; border-radius: 10px; }
            #rightPanel { background: transparent; }
            #logText { font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 12px; }
            """
        )

    def apply_theme(self, theme: str = "light"):
        self.theme = theme if theme in ("light", "dark") else "light"
        qss = self._get_light_qss() if self.theme == "light" else self._get_dark_qss()
        self.setStyleSheet(qss)

    def toggle_theme(self):
        self.apply_theme("dark" if self.theme == "light" else "light")

    def _init_toolbar(self):
        tb = QToolBar("Main Toolbar", self)
        tb.setIconSize(QSize(24, 24))
        self.addToolBar(tb)

        # Use text-only actions for broad compatibility
        act_start = QAction("Start", self)
        act_stop = QAction("Stop", self)
        act_snapshot = QAction("Snapshot", self)
        act_open_video = QAction("Open Video", self)
        act_open_image = QAction("Open Image", self)
        act_fullscreen = QAction("Fullscreen", self)
        act_theme = QAction("Theme", self)

        act_start.triggered.connect(self.on_start)
        act_stop.triggered.connect(self.on_stop)
        act_snapshot.triggered.connect(self.on_snapshot)
        act_open_video.triggered.connect(self.on_browse_video)
        act_open_image.triggered.connect(self.on_browse_image)
        act_fullscreen.triggered.connect(self.toggle_fullscreen)
        act_theme.triggered.connect(self.toggle_theme)

        tb.addAction(act_start)
        tb.addAction(act_stop)
        tb.addSeparator()
        tb.addAction(act_snapshot)
        tb.addSeparator()
        tb.addAction(act_open_video)
        tb.addAction(act_open_image)
        tb.addSeparator()
        tb.addAction(act_fullscreen)
        tb.addAction(act_theme)

        self.addAction(act_fullscreen)
        act_fullscreen.setShortcut(QKeySequence("F11"))
        act_start.setShortcut(QKeySequence("Ctrl+R"))
        act_stop.setShortcut(QKeySequence("Ctrl+S"))
        act_snapshot.setShortcut(QKeySequence("Ctrl+P"))

    def _init_statusbar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.sb_left = QLabel("Ready")
        self.sb_right = QLabel("")
        self.sb_right.setObjectName("recPill")
        sb.addWidget(self.sb_left, 1)
        sb.addPermanentWidget(self.sb_right, 0)

    # UI handlers
    def on_browse_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Videos (*.mp4 *.avi *.mkv)")
        if p:
            self.video_path = p
            self.source_combo.setCurrentText("Video file")

    def on_browse_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.jpg *.png *.jpeg)")
        if p:
            self.image_path = p
            self.source_combo.setCurrentText("Image file")

    def on_choose_logo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Chọn logo", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not p:
            return
        pix = QPixmap(p)
        if not pix.isNull():
            self.logo_label.setPixmap(pix.scaledToHeight(48, Qt.SmoothTransformation))  # type: ignore[attr-defined]

    def on_browse_save(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save annotated video", "out.mp4", "MP4 (*.mp4);;AVI (*.avi)")
        if p:
            self.save_path = p
            self.save_path_edit.setText(p)

    def on_start(self):
        src = self.source_combo.currentText()
        self.SLEEP_FRAMES = self.sleep_frames_spin.value()
        self.AWAKE_FRAMES = self.awake_frames_spin.value()
        self.reset_states()
        # Ensure model follows current selection
        preset = self.model_combo.currentText()
        desired = self._resolve_preset_path(preset)
        if desired and isinstance(getattr(self.pose_model, 'ckpt_path', None), str):
            try:
                cur = os.path.basename(getattr(self.pose_model, 'ckpt_path'))
            except Exception:
                cur = ""
        else:
            cur = ""
        if desired and (not cur or os.path.basename(desired) != cur):
            self._load_pose_model(desired)
        # Setup recording
        self.record_enabled = self.record_check.isChecked()
        if self.record_enabled:
            self.save_path = self.save_path_edit.text().strip() or None
            if not self.save_path:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.save_path = f"annotated_{ts}.mp4"
                self.save_path_edit.setText(self.save_path)
        # Reset writer; will lazy-init after first frame
        self._release_writer()

        if src == "Camera device":
            cam_id = int(self.dev_spin.value())
            self.cap = open_capture(cam_id, self.res_combo.currentText(), use_mjpg=self.mjpg_check.isChecked())
            if self.cap is None:
                self.append_log("Không mở được camera thiết bị.")
                return
            self.source_mode = "camera"
            self.timer.start(1)
        elif src == "RTSP/HTTP URL":
            url = self.url_edit.text().strip()
            if not url:
                self.append_log("URL trống.")
                return
            self.cap = cv2.VideoCapture(url)
            if not self.cap or not self.cap.isOpened():
                self.append_log("Không mở được luồng URL.")
                return
            self.source_mode = "camera"
            self.timer.start(1)
        elif src == "Video file":
            if not self.video_path:
                self.append_log("Chưa chọn video.")
                return
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap or not self.cap.isOpened():
                self.append_log("Không mở được video.")
                return
            self.source_mode = "video"
            self.timer.start(1)
        else:  # Image file
            if not self.image_path:
                self.append_log("Chưa chọn ảnh.")
                return
            img = cv2.imread(self.image_path)
            if img is None:
                self.append_log("Không đọc được ảnh.")
                return
            vis = self.process_frame_once(img)
            self.show_frame(vis)
            self.sb_left.setText("Image displayed")

    def on_stop(self):
        self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self._release_writer()
        self.sb_left.setText("Stopped")

    def append_log(self, s: str):
        self.log.appendleft(s)
        self.log_text.setPlainText("\n".join(list(self.log)))

    def reset_states(self):
        self.sleep_states.clear()
        self.awake_states.clear()
        self.sleep_status.clear()
        self.sleep_start_time.clear()
        self.max_sleep_duration.clear()
        self.tracker = SimpleTracker(iou_thr=0.35, max_age=25)
        self.angle_hist.clear()
        self.blinks = 0
        self.microsleeps = 0.0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.left_closed = False
        self.right_closed = False
        self.yawn_in_progress = False
        self.frame_idx = 0

    # Timer loop
    def on_timer(self):
        if self.cap is None:
            return
        t0 = time.time()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.source_mode == "video":
                self.on_stop()
            return
        fopt = self.flip_combo.currentText()
        if fopt == "h":
            frame = cv2.flip(frame, 1)
        elif fopt == "v":
            frame = cv2.flip(frame, 0)
        elif fopt == "180":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        vis = self.process_frame_once(frame)
        self.show_frame(vis)
        self.frame_idx += 1
        # FPS update
        t1 = time.time()
        fps_now = 1.0 / max(t1 - t0, 1e-6)
        self.ema_fps = fps_now if self.ema_fps is None else 0.8 * self.ema_fps + 0.2 * fps_now
        sleepy_num = sum(1 for v in self.sleep_status.values() if v in ("Ngủ gật", "Gục xuống bàn"))
        self.sb_left.setText(f"FPS: {self.ema_fps:.1f} | Sleepy: {sleepy_num}")
        if self.record_enabled:
            self.sb_right.setText(" REC ")
            # Style as pill based on theme
            if getattr(self, "theme", "light") == "light":
                self.sb_right.setStyleSheet("#recPill { background: #ef4444; color: #ffffff; border-radius: 10px; padding: 2px 8px; }")
            else:
                self.sb_right.setStyleSheet("#recPill { background: #b91c1c; color: #ffffff; border-radius: 10px; padding: 2px 8px; }")
        else:
            self.sb_right.setText("")
            self.sb_right.setStyleSheet("")

    def show_frame(self, vis: np.ndarray):
        if self.show_grid_check.isChecked():
            draw_grid(vis)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation  # type: ignore[attr-defined]
        ))
        # Recording write
        if self.record_enabled:
            self._write_frame(vis)

    # Core processing
    def process_frame_once(self, frame: np.ndarray) -> np.ndarray:
        res = self.pose_model(frame, imgsz=int(self.imgsz_spin.value()), conf=float(self.conf_spin.value()))
        r0 = res[0] if res and len(res) > 0 else None
        vis = r0.plot(line_width=2, conf=True) if r0 is not None else frame.copy()

        angle_thr = float(self.angle_spin.value())
        dropH_thr = float(self.dropH_spin.value())
        dropSW_thr = float(self.dropSW_spin.value())
        sleepy_count = 0

        if r0 is not None and hasattr(r0, "keypoints"):
            # Collect boxes, confidences, classes
            boxes = []
            confs = []
            classes = []
            if hasattr(r0, "boxes") and r0.boxes is not None:
                try:
                    boxes = r0.boxes.xyxy.cpu().numpy().tolist()
                except Exception:
                    boxes = []
                try:
                    confs = r0.boxes.conf.cpu().numpy().tolist()
                except Exception:
                    confs = []
                try:
                    classes = r0.boxes.cls.cpu().numpy().tolist()
                except Exception:
                    classes = []
            kps_all = list(r0.keypoints)
            order = list(range(len(kps_all)))
            if confs:
                order = list(np.argsort(-np.array(confs)))
            order = order[: self.max_people_spin.value()]
            # Build selected list
            selected = []
            for i in order:
                if i >= len(kps_all):
                    continue
                sel_box = boxes[i] if i < len(boxes) else None
                selected.append((i, sel_box))

            # Tracker update with available boxes
            dets = [b for (_, b) in selected if b is not None]
            id_to_sel: Dict[int, int] = {}
            if dets:
                assign = self.tracker.update(dets)  # track_id -> det_index
                # map det index back to selected index
                det_to_sel: Dict[int, int] = {}
                di = 0
                for si, (_, b) in enumerate(selected):
                    if b is not None:
                        det_to_sel[di] = si
                        di += 1
                for tid, di in assign.items():
                    si = det_to_sel.get(di, None)
                    if si is not None:
                        id_to_sel[tid] = si
            else:
                # fallback pseudo IDs when no boxes
                for si in range(len(selected)):
                    id_to_sel[-(si + 1)] = si

            # Iterate by stable track ids
            for tid, si in id_to_sel.items():
                i, box = selected[si]
                if i >= len(kps_all):
                    continue
                kp = kps_all[i]
                k = kp.xy[0].cpu().numpy()
                cls_id = int(classes[i]) if i < len(classes) else -1

                # bbox-aware classification
                state, ang, drop = classify_pose_bbox(
                    k,
                    vis.shape[0],
                    vis.shape[1],
                    tuple(box) if box else None,
                    angle_thr,
                    dropH_thr,
                    dropSW_thr,
                )
                # Optional override by class id if available
                if cls_id == 2:
                    state = "Gục xuống bàn"
                elif cls_id == 1:
                    state = "Ngủ gật"

                # Smooth angle per track id
                hq = self.angle_hist.setdefault(tid, deque(maxlen=8))
                hq.append(ang)
                if len(hq) >= 3:
                    ang = float(np.median(np.array(hq)))

                prev_sleep = self.sleep_states.get(tid, 0)
                prev_awake = self.awake_states.get(tid, 0)
                if state in ("Ngủ gật", "Gục xuống bàn"):
                    self.sleep_states[tid] = prev_sleep + 1
                    self.awake_states[tid] = 0
                else:
                    self.awake_states[tid] = prev_awake + 1
                    self.sleep_states[tid] = 0

                prev_status = self.sleep_status.get(tid, "Bình thường")
                now = time.time()
                eff_state = prev_status
                if prev_status in ("Ngủ gật", "Gục xuống bàn"):
                    if state not in ("Ngủ gật", "Gục xuống bàn") and self.awake_states[tid] >= self.AWAKE_FRAMES:
                        eff_state = "Bình thường"
                else:
                    if state in ("Ngủ gật", "Gục xuống bàn") and self.sleep_states[tid] >= self.SLEEP_FRAMES:
                        eff_state = state

                if prev_status != eff_state:
                    if eff_state in ("Ngủ gật", "Gục xuống bàn"):
                        self.sleep_start_time[tid] = now
                        self.append_log(f"[{time.strftime('%H:%M:%S')}] ID {tid}: {eff_state}")
                    elif prev_status in ("Ngủ gật", "Gục xuống bàn") and eff_state == "Bình thường":
                        if tid in self.sleep_start_time:
                            duration = now - self.sleep_start_time[tid]
                            self.max_sleep_duration[tid] = max(self.max_sleep_duration.get(tid, 0.0), duration)
                            self.append_log(f"[{time.strftime('%H:%M:%S')}] ID {tid} đã thức dậy. {duration:.1f}s")
                            del self.sleep_start_time[tid]
                self.sleep_status[tid] = eff_state

                color = (0, 255, 0)
                if eff_state == "Ngủ gật":
                    color = (0, 0, 255); sleepy_count += 1
                elif eff_state == "Gục xuống bàn":
                    color = (255, 0, 255); sleepy_count += 1
                label = f"ID {tid} — {eff_state} ({ang:.1f}°, {drop*100:.0f}%)"
                if box:
                    x1, y1, x2, y2 = list(map(int, box))
                    y_label = max(0, y1 - 26)
                    cv2.rectangle(vis, (x1, y_label), (x2, y_label + 24), color, -1)
                    vis = draw_text_unicode(vis, label, (x1 + 5, y_label + 3), color=(255, 255, 255), font_size=18)
                else:
                    nose = k[0]
                    vis = draw_text_unicode(
                        vis, label, (int(nose[0]), max(20, int(nose[1]) - 10)), color=color, font_size=22
                    )

        # Secondary eye/yawn pipeline (optional)
        sec_info = []
        if self.enable_eyes_check.isChecked():
            # Lazy init face mesh and models if checkbox is toggled at runtime
            if self.face_mesh is None and mp is not None:
                try:
                    self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                        min_detection_confidence=0.5, min_tracking_confidence=0.5
                    )
                except Exception:
                    self.face_mesh = None
            # Only run if we have FaceMesh and at least one of the detectors
            if self.face_mesh is not None and (self.eye_model is not None or self.yawn_model is not None):
                do_run = (self.frame_idx % max(1, self.secondary_interval) == 0)
                ih, iw = frame.shape[:2]
                if do_run:
                    try:
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        fm_res = self.face_mesh.process(image_rgb)
                        if hasattr(fm_res, "multi_face_landmarks") and fm_res.multi_face_landmarks:
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
                                mouth_roi = self._safe_crop(frame, x1, y1, x2, y3)
                                right_eye_roi = self._safe_crop(frame, x4, y4, x5, y5)
                                left_eye_roi = self._safe_crop(frame, x6a, y6a, x7a, y7a)

                                le = self._predict_eye(left_eye_roi)
                                re = self._predict_eye(right_eye_roi)
                                yaw = self._predict_yawn(mouth_roi)

                                prev_both_closed = (self.left_closed and self.right_closed)
                                if le == "Close Eye" and re == "Close Eye":
                                    if not prev_both_closed:
                                        self.blinks += 1
                                    self.left_closed = True
                                    self.right_closed = True
                                else:
                                    self.left_closed = (le == "Close Eye") if le is not None else False
                                    self.right_closed = (re == "Close Eye") if re is not None else False

                                now_t = time.time()
                                dt = max(now_t - getattr(self, "last_proc_time", now_t), 0.0)
                                self.last_proc_time = now_t
                                if self.left_closed and self.right_closed:
                                    self.microsleeps += dt
                                else:
                                    self.microsleeps = 0.0

                                if yaw == "Yawn":
                                    if not self.yawn_in_progress:
                                        self.yawns += 1
                                        self.yawn_in_progress = True
                                    self.yawn_duration += dt
                                else:
                                    self.yawn_in_progress = False
                                    self.yawn_duration = 0.0

                                sec_info.append(f"👁️ Blinks: {self.blinks}")
                                sec_info.append(f"💤 Microsleeps: {self.microsleeps:.1f}s")
                                sec_info.append(f"😮 Yawns: {self.yawns}")
                                sec_info.append(f"⏳ Yawn Dur: {self.yawn_duration:.1f}s")

                                if self.microsleeps >= 3.0 or self.yawn_duration >= 7.0:
                                    vis = draw_text_unicode(vis, "Cảnh báo: buồn ngủ!", (20, 90), color=(0, 0, 255), font_size=24)
                    except Exception:
                        pass
                else:
                    sec_info.append(f"👁️ Blinks: {self.blinks}")
                    sec_info.append(f"💤 Microsleeps: {self.microsleeps:.1f}s")
                    sec_info.append(f"😮 Yawns: {self.yawns}")
                    sec_info.append(f"⏳ Yawn Dur: {self.yawn_duration:.1f}s")

        # Optional: draw an info panel similar to CLI
        if sec_info:
            vw, vh = vis.shape[1], vis.shape[0]
            lines = sec_info[:6]
            line_h = 20
            pad = 8
            log_w = max(220, min(int(vw * 0.28), 300))
            info_h = 20 + len(lines) * line_h + pad
            info_x = vw - log_w - 12
            info_y = min(12 + 28 + 8 + 120, vh - info_h - 8)  # stack below potential logs area
            vis = draw_panel(vis, info_x, info_y, log_w, info_h, bg=(0, 30, 0))
            for k, line in enumerate(lines):
                vis = draw_text_unicode(vis, line, (info_x + 10, info_y + 12 + k * line_h), color=(200, 255, 200), font_size=18)

        if self.max_sleep_duration:
            max_time = max(self.max_sleep_duration.values())
            vis = draw_text_unicode(vis, f"Ngủ gật lâu nhất: {max_time:.1f}s", (20, 60), color=(255, 0, 0), font_size=22)

        # HUD
        hud = f"Sleepy: {sleepy_count}"
        vis = draw_text_unicode(vis, hud, (12, 10), color=(50, 255, 50), font_size=22)
        # Update stats label (right panel)
        self.stats_label.setText(
            f"Blinks: {self.blinks} | Microsleeps: {self.microsleeps:.1f}s | Yawns: {self.yawns} | YawnDur: {self.yawn_duration:.1f}s"
        )
        return vis

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    # Recording helpers
    def _ensure_writer(self, frame: np.ndarray):
        if not self.record_enabled:
            return
        if self.writer is not None:
            return
        if not self.save_path:
            return
        h, w = frame.shape[:2]
        self.last_frame_size = (w, h)
        fourcc = 0
        try:
            fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
            if callable(fourcc_fn):
                fourcc = int(fourcc_fn(*"mp4v"))  # type: ignore[arg-type]
        except Exception:
            fourcc = 0
        fps = 25.0
        self.writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))

    def _write_frame(self, frame: np.ndarray):
        try:
            self._ensure_writer(frame)
            if self.writer is not None and self.last_frame_size == (frame.shape[1], frame.shape[0]):
                self.writer.write(frame)
        except Exception:
            pass

    def _release_writer(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.writer = None

    # Snapshot
    def on_snapshot(self):
        if self.video_label.pixmap() is None:
            self.append_log("Chưa có khung hình để chụp.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.png"
        # Grab current frame by re-rendering latest capture if available
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                vis = self.process_frame_once(frame)
                cv2.imwrite(path, vis)
                self.append_log(f"Đã lưu ảnh: {path}")
                return
        # Fallback: try to save from displayed pixmap
        pm = self.video_label.pixmap()
        if pm is not None:
            pm.save(path)
            self.append_log(f"Đã lưu ảnh: {path}")


def launch_gui(args: argparse.Namespace):
    app = QApplication([])
    win = SleepyWindow(args)
    win.showMaximized()
    app.exec_()
