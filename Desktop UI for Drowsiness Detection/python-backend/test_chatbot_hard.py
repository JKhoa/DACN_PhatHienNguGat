"""Battery test cho chatbot — kiểm định offline với DB hiện tại.

Mỗi câu hỏi: in intent được classify, số dòng trả về, SQL có chạy OK không,
summary tạo ra, cảnh báo khi intent bị hiểu sai / unknown rơi về AI fallback.

Chạy: python test_chatbot_hard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
import chatbot  # noqa: E402

# (câu hỏi, intent mong đợi — None nếu chấp nhận unknown)
CASES: list[tuple[str, str | None]] = [
    # Baseline — keyword thẳng, phải match
    ("Hôm nay có bao nhiêu ca ngủ gật?", "today_overview"),
    ("Phòng nào rủi ro cao nhất?", "high_risk_rooms"),
    ("So sánh tuần này với tuần trước", "weekly_compare"),
    ("Top học sinh ngủ nhiều nhất tháng này", "top_students"),
    ("Khung giờ nào dễ ngủ gật?", "peak_hours"),
    ("Ai đang ngủ gật ngay lúc này?", "active_now"),

    # Phức tạp — ambiguous / multi-keyword
    ("So sánh giữa phòng A101 và B202 tuần này",      "weekly_compare"),
    ("Tuần này phòng nào có nhiều học sinh ngủ gật nhất?", "weekly_compare"),
    ("Học sinh nào ngủ nhiều trong khung giờ buổi chiều?", "top_students"),
    ("Hôm nay có phòng nào rủi ro không?",             "today_overview"),

    # Trick — keyword va chạm
    ("Tháng này top phòng rủi ro",                      "high_risk_rooms"),
    ("Cao điểm giờ nào tuần này?",                      "peak_hours"),

    # Unknown — nên rơi về AI fallback
    ("Dự báo thời tiết ngày mai ra sao?",               None),
    ("Giải thích cho tôi thuật toán YOLO",              None),
]


def _snip(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    print("=" * 70)
    print("CHATBOT HARD TEST — DB:", chatbot._get_db_path())
    print("=" * 70)

    fails = 0
    ambiguous = 0
    for i, (q, expected) in enumerate(CASES, 1):
        res = chatbot.handle_question(q)
        got = res.get("intent", "?")
        ok = res.get("success", False)
        nrows = len(res.get("rows") or [])
        summary = _snip(res.get("summary_text", ""))

        tag = " OK "
        if not ok:
            tag, fails = "FAIL", fails + 1
        elif expected is not None and got != expected:
            tag, ambiguous = "MISS", ambiguous + 1
        elif expected is None and got != "unknown":
            tag, ambiguous = "?FB ", ambiguous + 1

        print(f"\n[{i:02d}] [{tag}] Q: {q}")
        print(f"     expected={expected}  got={got}  rows={nrows}")
        print(f"     summary: {summary}")

    print("\n" + "=" * 70)
    print(f"TỔNG KẾT: {len(CASES)} cases | {fails} FAIL | {ambiguous} MISS/fallback khác mong đợi")
    print("=" * 70)
    print("\nGhi chú:")
    print("  - MISS = classifier chọn intent khác kỳ vọng → cần thêm/điều chỉnh keyword.")
    print("  - ?FB  = câu lẽ ra unknown nhưng lại match 1 intent → keyword quá rộng.")
    print("  - FAIL = SQL chạy lỗi hoặc success=False → cần fix chatbot.py hoặc DB.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
