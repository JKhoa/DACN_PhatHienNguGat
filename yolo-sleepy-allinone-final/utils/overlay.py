import cv2
from typing import Tuple


COLOR = {
    "CONNECTED": (34, 197, 94),
    "RECONNECTING": (245, 158, 11),
    "ERROR": (220, 38, 38),
    "IDLE": (100, 116, 139),
    "ACCENT": (37, 99, 235),
}


def draw_badge(img, text: str, tl: Tuple[int, int] = (6, 6), color=(37, 99, 235)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = tl
    cv2.rectangle(img, (x, y - h - 6), (x + w + 12, y + 6), color, -1)
    cv2.putText(img, text, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img



