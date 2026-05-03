# DACN PhatHienNguGat - 2026
"""Multi-task detectors for classroom student monitoring."""
from __future__ import annotations

from .multi_task_detector import (
    BehaviorEvent,
    BehaviorType,
    MultiTaskDetector,
)
from .hybrid_detector import HybridDetector
from .ensemble import Detection, DetectionResult, EnsembleDetector

__all__ = [
    "BehaviorEvent",
    "BehaviorType",
    "Detection",
    "DetectionResult",
    "EnsembleDetector",
    "HybridDetector",
    "MultiTaskDetector",
]
