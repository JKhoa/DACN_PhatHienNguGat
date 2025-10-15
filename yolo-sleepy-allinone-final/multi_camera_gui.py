#!/usr/bin/env python3
"""
Multi-Camera GUI Widget
Widget for managing and displaying multiple camera feeds with YOLO detection
"""

import os
import time
import yaml
import threading
from typing import List, Dict, Optional
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QDialog, QFormLayout,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QFileDialog, QSplitter, QGroupBox, QTextEdit, QTabWidget
)

from camera_core import CameraConfig, CameraStream, generate_rtsp_url, connect_camera, SUPPORTED_BRANDS, start_threaded_capture, stop_threaded_capture
from utils.logging_ext import CsvEventLogger, send_alert
from models.fallback import resolve_pose_model_path
from detection_pipeline import RoundRobinDetector

import sys


class CameraConfigDialog(QDialog):
    """Dialog for adding/editing camera configuration"""
    
    def __init__(self, parent=None, config: Optional[CameraConfig] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera Configuration")
        self.setMinimumWidth(500)
        self.config = config
        
        layout = QFormLayout(self)
        
        # Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Ví dụ: Phòng học 101")
        if config:
            self.name_edit.setText(config.name)
        layout.addRow("Tên Camera:", self.name_edit)
        
        # Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["webcam", "ip"])
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        if config:
            self.type_combo.setCurrentText(config.type)
        layout.addRow("Loại:", self.type_combo)
        
        # Webcam source
        self.webcam_spin = QSpinBox()
        self.webcam_spin.setRange(0, 10)
        if config and config.type == "webcam":
            self.webcam_spin.setValue(config.source)
        layout.addRow("Webcam ID:", self.webcam_spin)
        
        # IP camera brand
        self.brand_combo = QComboBox()
        for brand in SUPPORTED_BRANDS:
            self.brand_combo.addItem(brand.title(), brand)
        if config and config.brand:
            index = self.brand_combo.findData(config.brand)
            if index >= 0:
                self.brand_combo.setCurrentIndex(index)
        layout.addRow("Brand:", self.brand_combo)
        
        # IP address
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("192.168.1.100")
        if config:
            self.ip_edit.setText(config.ip)
        layout.addRow("IP Address:", self.ip_edit)
        
        # Port
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(config.port if config else 554)
        layout.addRow("Port:", self.port_spin)
        
        # Username
        self.username_edit = QLineEdit()
        if config:
            self.username_edit.setText(config.username)
        layout.addRow("Username:", self.username_edit)
        
        # Password
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        if config:
            self.password_edit.setText(config.password)
        layout.addRow("Password:", self.password_edit)
        
        # Stream quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["main", "sub"])
        if config:
            self.quality_combo.setCurrentText(config.stream_quality)
        layout.addRow("Stream Quality:", self.quality_combo)
        
        # Enabled
        self.enabled_check = QCheckBox()
        self.enabled_check.setChecked(True if not config else config.enabled)
        layout.addRow("Enabled:", self.enabled_check)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_test = QPushButton("Test Connection")
        self.btn_test.clicked.connect(self.test_connection)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_test)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addRow(btn_layout)
        
        # Initial UI state
        self.on_type_changed(self.type_combo.currentText())
    
    def on_type_changed(self, cam_type: str):
        """Show/hide fields based on camera type"""
        is_webcam = (cam_type == "webcam")
        
        self.webcam_spin.setVisible(is_webcam)
        self.brand_combo.setVisible(not is_webcam)
        self.ip_edit.setVisible(not is_webcam)
        self.port_spin.setVisible(not is_webcam)
        self.username_edit.setVisible(not is_webcam)
        self.password_edit.setVisible(not is_webcam)
        self.quality_combo.setVisible(not is_webcam)
        self.btn_test.setVisible(not is_webcam)
    
    def test_connection(self):
        """Test camera connection"""
        config = self.get_config()
        if not config:
            QMessageBox.warning(self, "Error", "Please fill in all required fields")
            return
        
        QMessageBox.information(self, "Testing", "Testing connection... Please wait.")
        
        cap = connect_camera(config)
        if cap is None:
            QMessageBox.critical(self, "Failed", f"Cannot connect to camera {config.name}")
        else:
            QMessageBox.information(self, "Success", f"Successfully connected to {config.name}!")
            cap.release()
    
    def get_config(self) -> Optional[CameraConfig]:
        """Get camera configuration from form"""
        name = self.name_edit.text().strip()
        if not name:
            return None
        
        cam_type = self.type_combo.currentText()
        
        if cam_type == "webcam":
            return CameraConfig(
                name=name,
                type="webcam",
                source=self.webcam_spin.value(),
                enabled=self.enabled_check.isChecked()
            )
        else:
            ip = self.ip_edit.text().strip()
            if not ip:
                return None
            
            return CameraConfig(
                name=name,
                type="ip",
                source=None,
                brand=self.brand_combo.currentData(),
                username=self.username_edit.text().strip(),
                password=self.password_edit.text().strip(),
                ip=ip,
                port=self.port_spin.value(),
                stream_quality=self.quality_combo.currentText(),
                enabled=self.enabled_check.isChecked()
            )


class MultiCameraWidget(QWidget):
    """Widget for multi-camera monitoring"""
    
    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        
        self.model_path = model_path
        try:
            self.model = YOLO(model_path)
            self.model_error = ""
        except Exception as e:
            self.model = None  # type: ignore[assignment]
            self.model_error = f"Model load failed: {e}"
        
        self.streams: List[CameraStream] = []
        self.running = False
        self.display_mode = "grid"  # grid or single
        self.selected_index = 0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
        # Shared round-robin detector (max 10 FPS per camera)
        self.rr = RoundRobinDetector(self._infer_once, max_fps_per_cam=10)

        # Sleepy detection thresholds and hysteresis
        self.SLEEP_FRAMES = 15
        self.AWAKE_FRAMES = 5
        self.ANGLE_THR = 25.0
        self.DROP_H_THR = 0.12
        self.DROP_SW_THR = 0.35

        # Event logger (optional)
        try:
            self.event_logger = CsvEventLogger("sleepy_events.csv", flush_interval_s=60.0)
        except Exception:
            self.event_logger = None
        
        # Per-stream sleepy state stores
        self.sleep_states = {}
        self.awake_states = {}
        self.sleep_status = {}
        self.sleep_start_time = {}
        self.max_sleep_duration = {}
        
        self.empty_message = (
            "<div style='color:#ccc; font-size:16px; text-align:center; padding:40px;'>"
            "<h2 style='color:#eee;margin-bottom:12px;'>Chưa có camera nào</h2>"
            "Bấm <b>➕ Add Camera</b> để thêm.<br><br>"
            "Hỗ trợ: <b>Webcam</b> hoặc <b>IP Camera (RTSP)</b>.<br>"
            "Sau khi thêm xong, bấm <b>▶️ Start All</b> để bắt đầu stream.<br><br>"
            "Bạn có thể lưu cấu hình bằng <b>💾 Save Config</b> và tải lại bằng <b>📁 Load Config</b>."
            "</div>"
        )
        
        self._init_ui()
        # Force initial empty placeholder
        try:
            self._show_empty_state('no_streams')
        except Exception:
            pass

    def _infer_once(self, img):
        if self.model is None:
            return None
        try:
            return self.model(img, imgsz=int(self.imgsz_spin.value()), conf=0.5, verbose=False)
        except Exception:
            return None

    def _classify_pose_bbox(self, k, img_h, img_w, box_xyxy):
        if k is None or len(k) < 7:
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
        if drop_h_ratio > 0.22 or drop_sw_ratio > 0.65 or drop_bb_ratio > 0.70:
            return "Gục xuống bàn", float(angle_v), float(drop_h_ratio)
        if (
            angle_v > self.ANGLE_THR
            or drop_h_ratio > self.DROP_H_THR
            or drop_sw_ratio > self.DROP_SW_THR
            or drop_bb_ratio > 0.40
        ):
            return "Ngủ gật", float(angle_v), float(drop_h_ratio)
        return "Bình thường", float(angle_v), float(drop_h_ratio)
    
    def _init_ui(self):
        """Initialize UI with improved symmetry and alignment"""
        layout = QVBoxLayout(self)

        # Top controls
        controls = QHBoxLayout()
        self.btn_add = QPushButton("➕ Add Camera")
        self.btn_add.clicked.connect(self.add_camera)
        controls.addWidget(self.btn_add)

        # Help icon button
        self.btn_help = QPushButton("?")
        self.btn_help.setFixedSize(28, 28)
        self.btn_help.setToolTip("Hướng dẫn kết nối camera")
        self.btn_help.clicked.connect(self.show_camera_help)
        controls.addWidget(self.btn_help)
        
        self.btn_edit = QPushButton("✏️ Edit")
        self.btn_edit.clicked.connect(self.edit_camera)
        controls.addWidget(self.btn_edit)
        self.btn_remove = QPushButton("🗑️ Remove")
        self.btn_remove.clicked.connect(self.remove_camera)
        controls.addWidget(self.btn_remove)
        controls.addSpacing(16)
        self.btn_load = QPushButton("📁 Load Config")
        self.btn_load.clicked.connect(self.load_config)
        controls.addWidget(self.btn_load)
        self.btn_save = QPushButton("💾 Save Config")
        self.btn_save.clicked.connect(self.save_config)
        controls.addWidget(self.btn_save)
        controls.addSpacing(16)
        # Export stats
        self.btn_export = QPushButton("📊 Export Stats")
        self.btn_export.clicked.connect(self.export_stats)
        controls.addWidget(self.btn_export)
        controls.addSpacing(16)
        self.btn_start_all = QPushButton("▶️ Start All")
        self.btn_start_all.clicked.connect(self.start_all)
        controls.addWidget(self.btn_start_all)
        self.btn_stop_all = QPushButton("⏹️ Stop All")
        self.btn_stop_all.clicked.connect(self.stop_all)
        controls.addWidget(self.btn_stop_all)
        controls.addStretch()
        layout.addLayout(controls)

        # Main content: camera list + display (use QGridLayout for symmetry)
        main_grid = QGridLayout()
        main_grid.setContentsMargins(8, 8, 8, 8)
        main_grid.setSpacing(16)

        # Left: Camera list panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        lbl_list = QLabel("<b>Camera List</b>")
        lbl_list.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(lbl_list)
        self.camera_list = QListWidget()
        self.camera_list.setMinimumWidth(180)
        self.camera_list.setMaximumWidth(260)
        self.camera_list.setMinimumHeight(400)
        self.camera_list.currentRowChanged.connect(self.on_camera_selected)
        self.camera_list.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.camera_list.customContextMenuRequested.connect(self.on_camera_context_menu)
        left_layout.addWidget(self.camera_list, stretch=1)
        self.stats_label = QLabel("Total: 0 cameras\nActive: 0 cameras")
        self.stats_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 4px;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.stats_label)
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)
        main_grid.addWidget(left_panel, 0, 0)

        # Right: Display area panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        # Display controls
        display_controls = QHBoxLayout()
        display_controls.setSpacing(8)
        display_controls.addWidget(QLabel("Display Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Grid View", "Single View"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        display_controls.addWidget(self.mode_combo)
        # Toggle overlay
        self.overlay_check = QCheckBox("Overlay")
        self.overlay_check.setChecked(True)
        display_controls.addWidget(self.overlay_check)
        # Performance controls
        display_controls.addWidget(QLabel("imgsz:"))
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(256, 1280); self.imgsz_spin.setSingleStep(32); self.imgsz_spin.setValue(640)
        display_controls.addWidget(self.imgsz_spin)
        display_controls.addWidget(QLabel("Inf FPS:"))
        self.fps_per_cam_spin = QSpinBox(); self.fps_per_cam_spin.setRange(1, 30); self.fps_per_cam_spin.setValue(10)
        display_controls.addWidget(self.fps_per_cam_spin)
        # Threshold controls
        display_controls.addWidget(QLabel("Sleep frames:"))
        self.sleep_frames_spin = QSpinBox(); self.sleep_frames_spin.setRange(1, 90); self.sleep_frames_spin.setValue(self.SLEEP_FRAMES)
        display_controls.addWidget(self.sleep_frames_spin)
        display_controls.addWidget(QLabel("Awake frames:"))
        self.awake_frames_spin = QSpinBox(); self.awake_frames_spin.setRange(1, 60); self.awake_frames_spin.setValue(self.AWAKE_FRAMES)
        display_controls.addWidget(self.awake_frames_spin)
        display_controls.addWidget(QLabel("Angle:"))
        self.angle_spin = QDoubleSpinBox(); self.angle_spin.setRange(5, 80); self.angle_spin.setValue(self.ANGLE_THR)
        display_controls.addWidget(self.angle_spin)
        # Sound alert toggle if available
        self.sound_check = QCheckBox("Sound")
        self.sound_check.setChecked(True)
        display_controls.addWidget(self.sound_check)
        display_controls.addStretch()
        self.fps_label = QLabel("FPS: 0.0")
        display_controls.addWidget(self.fps_label)
        right_layout.addLayout(display_controls)
        # Display canvas
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner = ""
        if getattr(self, 'model_error', ""):
            banner = f"<div style='background:#dc2626;color:#fff;padding:6px;border-radius:6px;margin-bottom:6px;'>Model error: {self.model_error}. Running in no-inference mode.</div>"
        self.display_label.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        right_layout.addWidget(self.display_label, stretch=1)
        # Global stats panel
        self.global_stats = QLabel("Cameras: 0 | Active: 0 | Sleepy: 0 | Longest: 0.0s")
        self.global_stats.setStyleSheet("padding: 6px; background: #0f172a; color: #93c5fd; border-radius: 6px;")
        right_layout.addWidget(self.global_stats)
        right_panel.setMinimumWidth(600)
        main_grid.addWidget(right_panel, 0, 1)

        # Set column stretch for symmetry
        main_grid.setColumnStretch(0, 1)
        main_grid.setColumnStretch(1, 3)
        layout.addLayout(main_grid)
    
    def show_camera_help(self):
        """Show camera connection help dialog"""
        msg = (
            "<b>Hướng dẫn kết nối camera</b><br><br>"
            "<b>Nếu không tìm thấy camera:</b><br>"
            "- Kiểm tra camera đã bật và kết nối WiFi chưa?<br>"
            "- Camera và máy tính phải cùng mạng LAN.<br>"
            "<br><b>Cách tìm địa chỉ IP:</b><br>"
            "- Mở app camera trên điện thoại → Settings → Device Info<br>"
            "- Vào router (web interface) → DHCP Client List<br>"
            "- Dùng app IP Scanner (Fing, Advanced IP Scanner, ...)<br>"
            "<br><b>Test thủ công:</b><br>"
            "- Chọn loại camera phù hợp<br>"
            "- Nhập IP camera vào ô IP Address<br>"
            "- Xem kết quả port scanning nếu có<br>"
            "<br><i>Bạn muốn chạy thử tool này không? 😊</i>"
        )
        QMessageBox.information(self, "Hướng dẫn kết nối camera", msg)
    
    def add_camera(self):
        """Add new camera (with connection test)"""
        dialog = CameraConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            if not config:
                QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập đầy đủ thông tin camera!")
                return
            from camera_core import connect_camera
            cap = connect_camera(config)
            if cap is None:
                QMessageBox.critical(self, "Kết nối thất bại", f"Không thể kết nối tới camera: {config.name}.\nVui lòng kiểm tra lại thông tin hoặc kết nối mạng.")
                return
            else:
                cap.release()
                stream = CameraStream(config=config)
                self.streams.append(stream)
                self.update_camera_list()
                self.update_display()  # refresh UI state
                QMessageBox.information(self, "Thành công", f"Đã thêm và kết nối thành công camera: {config.name}")
    
    def edit_camera(self):
        """Edit selected camera"""
        row = self.camera_list.currentRow()
        if row < 0 or row >= len(self.streams):
            QMessageBox.warning(self, "Warning", "Please select a camera to edit")
            return
        
        stream = self.streams[row]
        dialog = CameraConfigDialog(self, stream.config)
        if dialog.exec_() == QDialog.Accepted:
            new_config = dialog.get_config()
            if new_config:
                # Stop stream if running
                if stream.running:
                    self.stop_stream(stream)
                
                # Update config
                stream.config = new_config
                self.update_camera_list()
    
    def remove_camera(self):
        """Remove selected camera"""
        row = self.camera_list.currentRow()
        if row < 0 or row >= len(self.streams):
            QMessageBox.warning(self, "Warning", "Please select a camera to remove")
            return
        
        reply = QMessageBox.question(
            self, "Confirm", "Are you sure you want to remove this camera?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            stream = self.streams[row]
            if stream.running:
                self.stop_stream(stream)
            self.streams.pop(row)
            self.update_camera_list()

    def on_camera_context_menu(self, pos):
        row = self.camera_list.currentRow()
        if row < 0 or row >= len(self.streams):
            return
        stream = self.streams[row]
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        act_restart = menu.addAction("Restart")
        act_toggle = menu.addAction("Disable" if stream.config.enabled else "Enable")
        act_remove = menu.addAction("Remove")
        if stream.config.type == 'ip':
            act_copy = menu.addAction("Copy RTSP URL")
        else:
            act_copy = None
        action = menu.exec_(self.camera_list.mapToGlobal(pos))
        if action == act_restart:
            if stream.running:
                self.stop_stream(stream)
            self.start_stream(stream)
        elif action == act_toggle:
            stream.config.enabled = not stream.config.enabled
            if not stream.config.enabled and stream.running:
                self.stop_stream(stream)
            self.update_camera_list()
        elif action == act_remove:
            self.remove_camera()
        elif act_copy and action == act_copy:
            try:
                url = generate_rtsp_url(stream.config)
                from PyQt5.QtWidgets import QApplication
                QApplication.clipboard().setText(url)
                QMessageBox.information(self, "Copied", "RTSP URL copied to clipboard")
            except Exception:
                pass
    
    def load_config(self):
        """Load cameras from YAML file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "YAML Files (*.yaml *.yml)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data or 'cameras' not in data:
                QMessageBox.warning(self, "Error", "Invalid configuration file")
                return
            
            # Stop all current streams
            self.stop_all()
            self.streams.clear()
            
            # Load cameras
            for cam_data in data['cameras']:
                config = CameraConfig(
                    name=cam_data.get('name', 'Unknown'),
                    type=cam_data.get('type', 'webcam'),
                    source=cam_data.get('source', 0),
                    brand=cam_data.get('brand', 'generic'),
                    username=cam_data.get('username', ''),
                    password=cam_data.get('password', ''),
                    ip=cam_data.get('ip', ''),
                    port=cam_data.get('port', 554),
                    stream_quality=cam_data.get('stream_quality', 'main'),
                    enabled=cam_data.get('enabled', True)
                )
                stream = CameraStream(config=config)
                self.streams.append(stream)
            
            self.update_camera_list()
            QMessageBox.information(self, "Success", f"Loaded {len(self.streams)} cameras")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")
    
    def save_config(self):
        """Save cameras to YAML file"""
        if not self.streams:
            QMessageBox.warning(self, "Warning", "No cameras to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "cameras.yaml", "YAML Files (*.yaml *.yml)"
        )
        
        if not filename:
            return
        
        try:
            cameras_data = []
            for stream in self.streams:
                config = stream.config
                cam_data = {
                    'name': config.name,
                    'type': config.type,
                    'enabled': config.enabled
                }
                
                if config.type == 'webcam':
                    cam_data['source'] = config.source
                else:
                    cam_data.update({
                        'brand': config.brand,
                        'ip': config.ip,
                        'port': config.port,
                        'username': config.username,
                        'password': config.password,
                        'stream_quality': config.stream_quality
                    })
                
                cameras_data.append(cam_data)
            
            data = {'cameras': cameras_data}
            
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            QMessageBox.information(self, "Success", "Configuration saved successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")
    
    def update_camera_list(self):
        """Update camera list widget"""
        self.camera_list.clear()
        
        for i, stream in enumerate(self.streams):
            status_icon = {
                "disconnected": "⚪",
                "connecting": "🟡",
                "connected": "🟢",
                "error": "🔴"
            }.get(stream.status, "⚪")
            
            item_text = f"{status_icon} {stream.config.name} ({stream.config.type})"
            if not stream.config.enabled:
                item_text += " [Disabled]"
            
            item = QListWidgetItem(item_text)
            # Thumbnail preview (if available)
            if stream.frame is not None:
                try:
                    thumb = cv2.resize(stream.frame, (120, 80))
                    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
                    item.setIcon(QPixmap.fromImage(qimg))
                except Exception:
                    pass
            self.camera_list.addItem(item)
        
        # Update stats
        active_count = sum(1 for s in self.streams if s.status == "connected")
        self.stats_label.setText(f"Total: {len(self.streams)} cameras\nActive: {active_count} cameras")
    
    def export_stats(self):
        """Export per-camera summary to CSV"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(self, "Export Stats", "camera_stats.csv", "CSV (*.csv)")
            if not path:
                return
            import csv, time
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(["camera","status","fps","detections","longest_sleep_s"]) 
                for s in self.streams:
                    w.writerow([
                        s.config.name,
                        s.status,
                        f"{getattr(s,'fps',0.0):.1f}",
                        int(getattr(s,'detection_count',0)),
                        f"{float(getattr(s,'longest_sleep',0.0)):.1f}",
                    ])
            QMessageBox.information(self, "Exported", f"Saved stats to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
    
    def start_all(self):
        """Start all enabled cameras"""
        if not self.streams:
            QMessageBox.warning(self, "Warning", "No cameras configured")
            return
        
        self.running = True
        
        for stream in self.streams:
            if stream.config.enabled and not stream.running:
                self.start_stream(stream)
        
        # Start shared detector when running
        self.rr.start()
        
        self.timer.start(30)  # Update display at ~30 FPS
        self.update_display()  # ensure UI updates
    
    def stop_all(self):
        """Stop all cameras"""
        self.running = False
        self.timer.stop()
        
        for stream in self.streams:
            if stream.running:
                self.stop_stream(stream)
        
        # Stop shared detector
        self.rr.stop()
        
        self.update_camera_list()
    
    def start_stream(self, stream: CameraStream):
        """Start a camera stream with threaded capture"""
        if stream.running:
            return
        stream.running = True
        # Connect camera if not already
        if stream.capture is None:
            stream.capture = connect_camera(stream.config)
        # Start threaded capture (high FPS, low latency)
        start_threaded_capture(stream, target_fps=30.0)
        # Register stream to RR detector
        try:
            self.rr.add_stream(
                stream.config.name,
                lambda s=stream: s.capture_thread.get_latest_frame() if s.capture_thread else None,
                lambda res, s=stream: setattr(s, 'detection_result', (res[0] if res else None))
            )
        except Exception:
            pass
    
    def stop_stream(self, stream: CameraStream):
        """Stop a camera stream and threaded capture"""
        stream.running = False
        stop_threaded_capture(stream)
        if stream.capture:
            try:
                stream.capture.release()
            except:
                pass
            stream.capture = None
        stream.status = "disconnected"
        stream.frame = None
    
    def capture_loop(self, stream: CameraStream):
        """Capture loop for a camera (runs in thread, now just for fallback/legacy)"""
        # This method is now legacy; main capture is handled by CameraCaptureThread and frame queue
        pass
    
    def _show_empty_state(self, reason: str):
        """Show placeholder when nothing to render"""
        tips = {
            'no_streams': self.empty_message,
            'no_frames': (
                "<div style='color:#bbb; font-size:15px; text-align:center; padding:40px;'>"
                "Đang chờ khung hình đầu tiên...<br><br>"
                "Nếu quá lâu:<br> - Kiểm tra kết nối IP/Webcam<br> - Thử Stop All rồi Start All<br> - Mở Edit để sửa thông số camera"
                "</div>"
            )
        }
        self.display_label.setText(tips.get(reason, self.empty_message))
        self.display_label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.fps_label.setText("FPS: 0.0")
    
    def update_display(self):
        """Update display canvas"""
        if not self.streams:
            self._show_empty_state('no_streams')
            return
        # Pull latest frames from threaded capture if needed
        for s in self.streams:
            try:
                need_refresh = s.frame is None
                if not need_refresh and s.capture_thread and s.frame_queue is not None:
                    # If queue has newer frames, refresh
                    need_refresh = not s.frame_queue.empty()
                if need_refresh and s.capture_thread:
                    latest = s.capture_thread.get_latest_frame()
                    if latest is not None:
                        s.frame = latest
                        s.frame_count += 1
                        s.status = "connected"
            except Exception:
                pass
        active_streams = [s for s in self.streams if s.frame is not None]
        
        if not active_streams:
            self._show_empty_state('no_frames')
            return
        
        # Update thresholds from UI
        self.SLEEP_FRAMES = self.sleep_frames_spin.value()
        self.AWAKE_FRAMES = self.awake_frames_spin.value()
        self.ANGLE_THR = float(self.angle_spin.value())

        if self.display_mode == "grid":
            self.display_grid(active_streams)
        else:
            self.display_single(active_streams)

        # Update global stats
        total = len(self.streams)
        active = sum(1 for s in self.streams if s.frame is not None)
        sleepy = sum(1 for s in self.streams if getattr(s, 'sleepy_count', 0) > 0)
        longest = 0.0
        for s in self.streams:
            longest = max(longest, float(getattr(s, 'longest_sleep', 0.0)))
        self.global_stats.setText(f"Cameras: {total} | Active: {active} | Sleepy: {sleepy} | Longest: {longest:.1f}s")
    
    def display_grid(self, streams: List[CameraStream]):
        """Display cameras in grid layout"""
        n = len(streams)
        if n == 0:
            self._show_empty_state('no_frames')
            return
        # Find first valid sample
        sample = None
        for s in streams:
            if s.frame is not None:
                sample = s.frame
                break
        if sample is None:
            self._show_empty_state('no_frames')
            return
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        h, w = sample.shape[:2]
        
        # Create grid canvas
        cell_h = 480 // rows
        cell_w = 640 // cols
        
        canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
        
        for i, stream in enumerate(streams):
            if stream.frame is None:
                continue
            
            row = i // cols
            col = i % cols
            
            # Resize frame
            frame = cv2.resize(stream.frame, (cell_w, cell_h))
            
            # Draw detection boxes if available
            if stream.detection_result:
                frame = stream.detection_result.plot(line_width=2)
                frame = cv2.resize(frame, (cell_w, cell_h))
            
            # Add label
            label = f"{stream.config.name} | {stream.fps:.1f} FPS"
            cv2.rectangle(frame, (0, 0), (cell_w, 25), (0, 0, 0), -1)
            cv2.putText(frame, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Connection badge
            status = stream.status.upper()
            if status == "CONNECTED":
                badge_col = (34, 197, 94)
            elif status in ("CONNECTING", "RECONNECT_WAIT"):
                badge_col = (245, 158, 11)
            elif status == "ERROR":
                badge_col = (220, 38, 38)
            else:
                badge_col = (100, 116, 139)
            cv2.circle(frame, (cell_w - 12, 12), 6, badge_col, -1)
            
            # Place in grid
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            canvas[y1:y2, x1:x2] = frame
        
        # Convert to QPixmap and display
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit label
        scaled = pixmap.scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.display_label.setPixmap(scaled)
        
        # Update FPS
        avg_fps = np.mean([s.fps for s in streams if s.fps > 0])
        self.fps_label.setText(f"FPS: {avg_fps:.1f}")
    
    def display_single(self, streams: List[CameraStream]):
        """Display single camera (fullscreen)"""
        if self.selected_index >= len(streams):
            self.selected_index = 0
        
        stream = streams[self.selected_index]
        
        if stream.frame is None:
            return
        
        frame = stream.frame.copy()
        
        # Draw detection
        if stream.detection_result:
            frame = stream.detection_result.plot(line_width=3)
        # Connection badge
        status = stream.status.upper()
        if status == "CONNECTED":
            badge_col = (34, 197, 94)
        elif status in ("CONNECTING", "RECONNECT_WAIT"):
            badge_col = (245, 158, 11)
        elif status == "ERROR":
            badge_col = (220, 38, 38)
        else:
            badge_col = (100, 116, 139)
        cv2.circle(frame, (24, 24), 10, badge_col, -1)
        
        # Add info overlay
        info = f"{stream.config.name} | {stream.fps:.1f} FPS | Detection: {stream.detection_count}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit
        scaled = pixmap.scaled(
            self.display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.display_label.setPixmap(scaled)
        
        self.fps_label.setText(f"FPS: {stream.fps:.1f}")
    
    def on_camera_selected(self, index: int):
        """Handle camera selection"""
        if index >= 0 and self.display_mode == "single":
            self.selected_index = index
    
    def on_mode_changed(self, mode: str):
        """Handle display mode change"""
        self.display_mode = "grid" if "Grid" in mode else "single"
    
    def closeEvent(self, a0):  # type: ignore[override]
        """Handle widget close"""
        self.stop_all()
        return super().closeEvent(a0)
