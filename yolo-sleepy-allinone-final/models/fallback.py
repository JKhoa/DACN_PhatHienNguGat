import os
from typing import Optional


def resolve_pose_model_path(preferred: Optional[str] = None) -> str:
    """Resolve a usable pose model path with graceful fallback.

    Order:
    1) preferred if provided
    2) local yolo11n-pose.pt or yolo11s-pose.pt (repo root)
    3) alias 'yolov8n-pose.pt' (Ultralytics auto-download)
    """
    if preferred:
        return preferred
    base_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
    v11n = os.path.join(repo_root, "yolo11n-pose.pt")
    v11s = os.path.join(repo_root, "yolo11s-pose.pt")
    if os.path.exists(v11n):
        return v11n
    if os.path.exists(v11s):
        return v11s
    return "yolov8n-pose.pt"



