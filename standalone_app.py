import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GUI_FILE = ROOT / "yolo-sleepy-allinone-final" / "gui_app.py"


def _default_model() -> str:
    local_model = ROOT / "yolo11n-pose.pt"
    if local_model.exists():
        return str(local_model)
    return "yolo11n-pose.pt"


def _load_gui_module():
    if not GUI_FILE.exists():
        raise FileNotFoundError(f"Cannot find GUI module: {GUI_FILE}")

    spec = importlib.util.spec_from_file_location("sleepy_gui_app", str(GUI_FILE))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load GUI module specification")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone desktop app for drowsiness detection")
    p.add_argument("--gui", action="store_true", help="Launch native PyQt desktop GUI")
    p.add_argument("--model", default=_default_model(), help="YOLO model weights path")
    p.add_argument("--cam", type=int, default=0, help="Camera device index")
    p.add_argument("--res", default="1280x720", help="Camera resolution, e.g. 1280x720")
    p.add_argument("--conf", type=float, default=0.4, help="Inference confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--flip", choices=["none", "h", "v", "180"], default="none", help="Display flip mode")
    p.add_argument("--mjpg", action="store_true", default=True, help="Prefer MJPG camera format")
    p.add_argument("--max-people", type=int, default=6, help="Maximum people to process")
    p.add_argument("--enable-eyes", action="store_true", default=True, help="Enable eye/yawn analysis")
    p.add_argument("--eye-weights", default=str(ROOT / "models" / "eye_open_close.pt"), help="Path to eye model")
    p.add_argument("--yawn-weights", default=str(ROOT / "models" / "yawn.pt"), help="Path to yawn model")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.gui:
        parser.error("Only GUI mode is available in this repository. Use --gui.")

    module = _load_gui_module()
    if not hasattr(module, "launch_gui"):
        raise RuntimeError("GUI module does not expose launch_gui(args)")

    module.launch_gui(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
