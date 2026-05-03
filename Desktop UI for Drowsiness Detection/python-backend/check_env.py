"""Môi trường pre-flight check cho hệ thống Phát hiện Ngủ gật.

Kiểm tra (không tự cài): Python version, Python deps từ requirements.txt,
Node/npm, node_modules, model weights, SQLite DB.
Exit code 0 = sẵn sàng chạy; != 0 = có thiếu sót cần fix.
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Windows console mặc định cp1252 — ép UTF-8 để in tiếng Việt có dấu
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"

MIN_PYTHON = (3, 9)

# Tên package pip → tên import (khi khác nhau)
IMPORT_NAMES = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "flask-cors": "flask_cors",
    "flask-socketio": "flask_socketio",
    "simple-websocket": "simple_websocket",
    "google-genai": "google.genai",
    "python-dotenv": "dotenv",
}

REQUIRED_WEIGHTS = [
    ("yolo11n.pt", "YOLO11 COCO - person detector + phone fallback"),
    ("drowsiness_cls.pt", "Drowsiness classifier (yolo11x-cls)"),
    ("phone_det.pt", "Phone detector (YOLOv8n)"),
]

OPTIONAL_WEIGHTS = [
    ("drowsiness_det.pt", "Drowsiness detector (Ultralytics) - optional, hybrid mode dùng cls thay"),
]


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def ok(self, msg: str) -> None:
        print(f"  [ OK ] {msg}")


def parse_requirement(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Loại bỏ version spec: "flask>=2.0.0,<=3.0.3" -> "flask"
    name = re.split(r"[<>=!~]", line, 1)[0]
    return name.strip()


def check_python(r: Report) -> None:
    print("\n[1/5] Python runtime")
    v = sys.version_info
    if (v.major, v.minor) < MIN_PYTHON:
        r.err(f"Python {v.major}.{v.minor} < yêu cầu tối thiểu {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    else:
        r.ok(f"Python {v.major}.{v.minor}.{v.micro}")


def check_python_deps(r: Report) -> None:
    print("\n[2/5] Python dependencies (requirements.txt)")
    req_path = BACKEND_DIR / "requirements.txt"
    if not req_path.exists():
        r.err(f"Không tìm thấy {req_path}")
        return

    with open(req_path, encoding="utf-8") as f:
        pkgs = [p for p in (parse_requirement(l) for l in f) if p]

    for pkg in pkgs:
        import_name = IMPORT_NAMES.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(import_name)
            r.ok(f"{pkg}")
        except ImportError as e:
            r.err(f"{pkg} (import {import_name!r} thất bại: {e})")


def check_node(r: Report) -> None:
    print("\n[3/5] Node.js + npm")
    for tool in ("node", "npm"):
        exe = shutil.which(tool)
        if not exe:
            r.err(f"Không tìm thấy {tool} trong PATH")
            continue
        try:
            out = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, timeout=10, shell=False
            )
            r.ok(f"{tool} {out.stdout.strip()}")
        except (subprocess.SubprocessError, OSError) as e:
            r.err(f"{tool} lỗi khi lấy version: {e}")

    nm = APP_DIR / "node_modules"
    if not nm.exists():
        r.err(f"Thiếu node_modules tại {nm} — chạy `npm install` ở {APP_DIR}")
    else:
        r.ok(f"node_modules tồn tại ({nm})")


def check_model_weights(r: Report) -> None:
    print("\n[4/5] Model weights (.pt)")
    if not MODELS_DIR.exists():
        r.err(f"Thiếu thư mục {MODELS_DIR}")
        return

    for fname, desc in REQUIRED_WEIGHTS:
        p = MODELS_DIR / fname
        if not p.exists():
            r.err(f"Thiếu {fname} — {desc}")
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            r.ok(f"{fname} ({size_mb:.1f} MB) — {desc}")

    for fname, desc in OPTIONAL_WEIGHTS:
        p = MODELS_DIR / fname
        if not p.exists():
            r.warn(f"Thiếu {fname} (optional) — {desc}")
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            r.ok(f"{fname} ({size_mb:.1f} MB) — {desc}")


def check_database(r: Report) -> None:
    print("\n[5/5] SQLite database")
    candidates = [
        BACKEND_DIR / "events.db",
        BACKEND_DIR / "drowsiness.db",
        BACKEND_DIR / "drowsiness_logs" / "events.db",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        r.warn(
            "Không tìm thấy file .db — sẽ tự tạo khi backend chạy lần đầu. "
            f"Các vị trí đã kiểm tra: {[str(p) for p in candidates]}"
        )
        return
    for p in found:
        size_kb = p.stat().st_size / 1024
        r.ok(f"{p.name} ({size_kb:.1f} KB) tại {p.parent}")


def main() -> int:
    print("=" * 60)
    print("PRE-FLIGHT CHECK — Hệ thống Phát hiện Ngủ gật")
    print(f"App dir: {APP_DIR}")
    print("=" * 60)

    r = Report()
    check_python(r)
    check_python_deps(r)
    check_node(r)
    check_model_weights(r)
    check_database(r)

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT: {len(r.errors)} lỗi, {len(r.warnings)} cảnh báo")
    print("=" * 60)
    if r.errors:
        print("\nSửa các [FAIL] trên rồi chạy lại. Gợi ý:")
        print(f"  - Python deps:  pip install -r {BACKEND_DIR / 'requirements.txt'}")
        print(f"  - Node deps:    cd \"{APP_DIR}\" && npm install")
        print(f"  - Model weights: python {BACKEND_DIR / 'download_models.py'}")
        return 1
    print("\nSẵn sàng chạy backend + frontend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
