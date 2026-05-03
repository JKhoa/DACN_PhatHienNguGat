# DACN PhatHienNguGat - 2026
"""Chuẩn hoá tên class EN → slug VN + ngưỡng confidence theo từng loại.

Whitelist: chỉ giữ các class thuộc nhóm ngủ gật / ngáp / mắt / điện thoại.
Mọi class khác (person, car, v.v.) bị DROP.

Slug (``class_name``) là không dấu, snake_case — ổn định cho log / DB.
``display_name`` là chuỗi tiếng Việt có dấu để render UI.

Example:
    >>> _map_any_name("Drowsy")
    ('ngu_gat', 'Ngủ gật')
    >>> _map_any_name("Phone (Object)")
    ('dien_thoai', 'Điện thoại')
    >>> _map_any_name("person")
    (None, None)
"""
from __future__ import annotations

import re

__all__ = [
    "ALLOWED_SLUGS",
    "CONF_FLOOR",
    "SEVERITY",
    "DISPLAY_NAME",
    "_map_any_name",
    "passes_floor",
]


# slug → display tiếng Việt có dấu
DISPLAY_NAME: dict[str, str] = {
    "ngu_gat": "Ngủ gật",
    "ngap": "Ngáp",
    "mat_nham": "Mắt nhắm",
    "tinh_tao": "Tỉnh táo",
    "dien_thoai": "Điện thoại",
    "bam_dien_thoai": "Bấm điện thoại",
}

# Mức độ cảnh báo: "danger" = đỏ + beep, "warn" = vàng, "info" = xanh.
SEVERITY: dict[str, str] = {
    "ngu_gat": "danger",
    "mat_nham": "danger",
    "ngap": "warn",
    "bam_dien_thoai": "warn",
    "dien_thoai": "warn",
    "tinh_tao": "info",
}

# Ngưỡng confidence tối thiểu theo slug — phòng false alarm.
CONF_FLOOR: dict[str, float] = {
    "ngu_gat": 0.30,
    "ngap": 0.30,
    "mat_nham": 0.30,
    "tinh_tao": 0.25,
    "dien_thoai": 0.40,
    "bam_dien_thoai": 0.40,
}

ALLOWED_SLUGS: frozenset[str] = frozenset(DISPLAY_NAME.keys())

# Pattern EN/vi-alias → slug. Key là regex (case-insensitive, match toàn từ).
_ALIAS_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(ngu[_\s-]?gat|ngu_gat)$"), "ngu_gat"),
    (re.compile(r"^(drowsy|drowsiness|sleepy|sleeping|asleep|fatigue)$"), "ngu_gat"),
    (re.compile(r"^(yawn|yawning|ngap)$"), "ngap"),
    (re.compile(r"^(eye[_\s-]?closed|closed[_\s-]?eyes?|eyes?[_\s-]?closed|mat[_\s-]?nham)$"), "mat_nham"),
    (re.compile(r"^(eye[_\s-]?open|open[_\s-]?eyes?|awake|alert|non[_\s-]?drowsy|tinh[_\s-]?tao)$"), "tinh_tao"),
    (re.compile(r"^(using[_\s-]?phone|phone[_\s-]?use|texting|bam[_\s-]?dien[_\s-]?thoai)$"), "bam_dien_thoai"),
    (re.compile(r"^(phone|mobile|cell[_\s-]?phone|cellphone|smartphone|dien[_\s-]?thoai)$"), "dien_thoai"),
)


def _normalize(name: str) -> str:
    """Chuẩn hoá: hạ chữ, strip, bỏ ngoặc parens, hợp nhất khoảng/underscore."""
    s = name.strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)       # bỏ "(Object)" v.v.
    s = re.sub(r"[^a-z0-9\s_-]", "", s)   # bỏ ký tự lạ
    s = re.sub(r"[\s-]+", "_", s).strip("_")
    return s


def _map_any_name(name: str | None) -> tuple[str | None, str | None]:
    """Ánh xạ tên class tuỳ ý → (slug, display_vn). Trả (None, None) nếu không thuộc whitelist."""
    if not name:
        return None, None
    norm = _normalize(name)
    if not norm:
        return None, None
    for pattern, slug in _ALIAS_RULES:
        if pattern.match(norm):
            return slug, DISPLAY_NAME[slug]
    return None, None


def passes_floor(slug: str, confidence: float) -> bool:
    """True nếu confidence vượt ngưỡng tối thiểu của slug."""
    return confidence >= CONF_FLOOR.get(slug, 0.30)
