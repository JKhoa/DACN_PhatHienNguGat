# DACN PhatHienNguGat - 2026
"""Write behavior events to Obsidian-compatible Markdown files.

Mỗi học sinh có 1 file ``03_Reports/Students/<ten_hoc_sinh>.md`` trong vault
Obsidian dùng chung (``D:\\Claude_Code_Resources\\claude-obsidian``).

File dùng định dạng **Dataview inline fields** (``Key:: value``) để
``Dashboard_Lop_Hoc.md`` tổng hợp bảng xếp hạng.

Style guide: ``00_System/YOLO_Standard_Style`` trong Obsidian vault.
"""
from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Final

from detectors.multi_task_detector import BehaviorEvent, BehaviorType


DEFAULT_VAULT: Final[str] = r"D:\Claude_Code_Resources\claude-obsidian"
STUDENTS_SUBDIR: Final[str] = r"03_Reports\Students"

FIELD_BY_BEHAVIOR: Final[dict[BehaviorType, str]] = {
    BehaviorType.DROWSY: "Sleep_Count",
    BehaviorType.PHONE_USAGE: "Phone_Count",
    BehaviorType.DISTRACTED: "Distracted_Count",
}

_INLINE_FIELD_RE = re.compile(r"^(?P<key>[A-Za-z_]\w*)::\s*(?P<value>.*)$")


def _slugify(name: str) -> str:
    """Chuẩn hoá tên học sinh thành filename an toàn.

    Args:
        name (str): Tên gốc, có thể chứa dấu tiếng Việt hoặc khoảng trắng.

    Returns:
        (str): Tên file viết thường, gạch dưới, không dấu, không ký tự đặc biệt.

    Examples:
        >>> _slugify("Nguyễn Văn An")
        'nguyen_van_an'
    """
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^\w\s-]", "", ascii_name).strip().lower()
    return re.sub(r"[-\s]+", "_", ascii_name)


class StatsLogger:
    """Append behavior events to per-student Markdown files in Obsidian vault.

    Attributes:
        vault_path (Path): Root của Obsidian vault.
        students_dir (Path): Thư mục chứa file của từng học sinh.

    Methods:
        log_event: Ghi 1 BehaviorEvent vào file của học sinh tương ứng.

    Examples:
        >>> logger = StatsLogger()
        >>> logger.log_event(event, student_name="Nguyễn Văn An", camera_id="cam01")
    """

    def __init__(self, vault_path: str | os.PathLike[str] | None = None) -> None:
        base = Path(vault_path) if vault_path else Path(os.environ.get("OBSIDIAN_VAULT", DEFAULT_VAULT))
        self.vault_path = base
        self.students_dir = base / STUDENTS_SUBDIR
        self.students_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_event(
        self,
        event: BehaviorEvent,
        student_name: str,
        camera_id: str = "",
    ) -> Path:
        """Tăng counter và append 1 dòng timeline cho học sinh.

        Args:
            event (BehaviorEvent): Sự kiện từ MultiTaskDetector.
            student_name (str): Tên học sinh (sẽ được slugify).
            camera_id (str): ID camera phát hiện (optional).

        Returns:
            (Path): Đường dẫn file Markdown đã cập nhật.
        """
        slug = _slugify(student_name) or f"student_{event.track_id}"
        file_path = self.students_dir / f"{slug}.md"

        with self._lock:
            counters = self._read_counters(file_path, student_name, event.track_id)
            field = FIELD_BY_BEHAVIOR[event.behavior]
            counters[field] = counters.get(field, 0) + 1
            counters["Total_Events"] = counters.get("Total_Events", 0) + 1
            self._write(file_path, student_name, event, counters)

        return file_path

    def _read_counters(
        self, path: Path, student_name: str, track_id: int
    ) -> dict[str, int | str]:
        """Parse inline fields hiện có, hoặc khởi tạo nếu file mới."""
        if not path.exists():
            return {
                "Student_Name": student_name,
                "Track_ID": track_id,
                "Sleep_Count": 0,
                "Phone_Count": 0,
                "Distracted_Count": 0,
                "Total_Events": 0,
            }

        counters: dict[str, int | str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            m = _INLINE_FIELD_RE.match(line.strip())
            if not m:
                continue
            key, raw = m.group("key"), m.group("value").strip()
            counters[key] = int(raw) if raw.isdigit() else raw
        return counters

    def _write(
        self,
        path: Path,
        student_name: str,
        event: BehaviorEvent,
        counters: dict[str, int | str],
    ) -> None:
        """Viết lại file với counters mới + append 1 dòng timeline."""
        now = datetime.fromtimestamp(event.start_time)
        today = now.strftime("%Y-%m-%d")

        header = (
            f"---\n"
            f"title: {student_name}\n"
            f"type: student-report\n"
            f"tags: [student, classroom-monitoring]\n"
            f"last_event: {now.isoformat(timespec='seconds')}\n"
            f"---\n\n"
            f"# {student_name}\n\n"
            f"## Counters\n\n"
            f"Student_Name:: {student_name}\n"
            f"Track_ID:: {counters.get('Track_ID', event.track_id)}\n"
            f"Sleep_Count:: {counters.get('Sleep_Count', 0)}\n"
            f"Phone_Count:: {counters.get('Phone_Count', 0)}\n"
            f"Distracted_Count:: {counters.get('Distracted_Count', 0)}\n"
            f"Total_Events:: {counters.get('Total_Events', 0)}\n"
            f"Last_Date:: {today}\n\n"
            f"## Timeline\n\n"
        )

        timeline_line = (
            f"- `{now.strftime('%H:%M:%S')}` **{event.behavior.value}** "
            f"— duration {event.duration:.1f}s, conf {event.confidence:.2f}\n"
        )

        existing_timeline = ""
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if "## Timeline" in text:
                existing_timeline = text.split("## Timeline", 1)[1].lstrip("\n")
                existing_timeline = existing_timeline.lstrip()

        path.write_text(header + timeline_line + existing_timeline, encoding="utf-8")
