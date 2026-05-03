"""
Chatbot thống kê ngủ gật — Intent-based SQL executor
Nhận câu hỏi tiếng Việt, map sang SQL an toàn (SELECT only),
truy vấn SQLite, trả về kết quả có cấu trúc cho frontend.
"""

import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

try:
    from google import genai as _genai
    _GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '')
    _ai_client = _genai.Client(api_key=_GEMINI_KEY) if _GEMINI_KEY else None
except ImportError:
    _ai_client = None

# ── Đường dẫn database ───────────────────────────────────────────────────────

def _get_db_path() -> str:
    here = Path(__file__).parent
    return str(here / 'drowsiness_logs' / 'events.db')


def _open_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


# ── SQL safety guard ─────────────────────────────────────────────────────────
# Chỉ cho phép SELECT và WITH (CTE). Chặn mọi lệnh ghi/sửa/xóa.

_ALLOWED_RE = re.compile(r'^\s*(SELECT|WITH)\s', re.IGNORECASE)
_DANGEROUS_RE = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b',
    re.IGNORECASE,
)


def _is_safe_sql(sql: str) -> bool:
    return bool(_ALLOWED_RE.match(sql)) and not _DANGEROUS_RE.search(sql)


# ── Query runner (có timeout) ─────────────────────────────────────────────────

def _run_query(
    sql: str, timeout_sec: float = 5.0
) -> Tuple[List[str], List[List[Any]], Optional[str]]:
    """Chạy SQL trong thread riêng với timeout. Trả về (columns, rows, error)."""
    result: Dict[str, Any] = {}
    error_box: List[Optional[str]] = [None]

    def _worker():
        try:
            conn = _open_conn()
            conn.execute(f"PRAGMA busy_timeout = {int(timeout_sec * 1000)}")
            cur = conn.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            rows = [list(r) for r in cur.fetchall()]
            result['columns'] = columns
            result['rows'] = rows
            conn.close()
        except Exception as exc:
            error_box[0] = str(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_sec + 1.0)

    if t.is_alive():
        return [], [], 'Timeout: truy vấn mất quá nhiều thời gian (>5s)'
    if error_box[0]:
        return [], [], error_box[0]
    return result.get('columns', []), result.get('rows', []), None


# ── Định nghĩa intents ────────────────────────────────────────────────────────

INTENTS: List[Dict[str, Any]] = [
    # 1. Ai đang ngủ gật ngay lúc này?
    {
        'name': 'active_now',
        'keywords': [
            'đang ngủ', 'lúc này', 'hiện tại', 'ngay lúc',
            'đang gật', 'đang buồn', 'đang ngủ gật',
        ],
        'sql': """
            SELECT
                camera_id,
                COALESCE(camera_name, camera_id)                                AS camera_name,
                student_id,
                start_time,
                ROUND((julianday('now') - julianday(start_time)) * 86400, 2)   AS current_duration_sec
            FROM drowsy_events
            WHERE is_active = 1
            ORDER BY current_duration_sec DESC
        """,
        'chart': 'bar',
        'summary_template': 'Hiện có {n} học sinh đang ngủ gật.',
        'empty_text': 'Hiện không có học sinh nào đang ngủ gật.',
        'risk': 'high',
    },

    # 2. KPI tổng quan hôm nay
    {
        'name': 'today_overview',
        'keywords': [
            'hôm nay', 'today', 'tổng quan', 'kpi',
            'tổng hợp', 'báo cáo hôm', 'bảng tổng',
        ],
        'sql': """
            SELECT
                COUNT(*)                                                              AS total_events_today,
                COUNT(DISTINCT student_id)                                            AS unique_students,
                COUNT(DISTINCT camera_id)                                             AS active_cameras,
                ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                                      AS total_duration_sec,
                ROUND(AVG(CASE WHEN is_active = 0 THEN duration_seconds END), 2)      AS avg_duration_sec
            FROM drowsy_events
            WHERE date(start_time) = date('now')
        """,
        'chart': 'bar',
        'summary_template': (
            'Hôm nay: {total_events_today} lần ngủ gật '
            'từ {unique_students} học sinh trên {active_cameras} phòng. '
            'Tổng thời gian: {total_duration_sec}s.'
        ),
        'empty_text': 'Hôm nay chưa ghi nhận sự kiện ngủ gật nào.',
        'risk': 'none',
    },

    # 3. Phòng rủi ro cao nhất hôm nay
    {
        'name': 'high_risk_rooms',
        'keywords': [
            'phòng', 'rủi ro', 'nguy hiểm', 'nguy cơ',
            'room', 'lớp học', 'phòng nào',
        ],
        'sql': """
            SELECT
                camera_id,
                camera_name,
                total_events,
                unique_students,
                total_duration_sec,
                avg_duration_sec
            FROM v_daily_camera_stats
            WHERE event_date = date('now')
            ORDER BY total_duration_sec DESC, total_events DESC
            LIMIT 5
        """,
        'chart': 'bar',
        'summary_template': 'Top {n} phòng rủi ro cao nhất hôm nay.',
        'empty_text': 'Hôm nay chưa có dữ liệu phòng.',
        'risk': 'high',
    },

    # 4. So sánh tuần này với tuần trước
    {
        'name': 'weekly_compare',
        'keywords': [
            'tuần', 'so sánh', 'tuần này', 'tuần trước',
            'week', 'so với', 'so tuần',
        ],
        'sql': """
            WITH this_week AS (
                SELECT
                    camera_id,
                    COALESCE(camera_name, camera_id)                                    AS camera_name,
                    COUNT(*)                                                             AS events_this_week,
                    ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                                         AS dur_this_week
                FROM drowsy_events
                WHERE strftime('%Y-%W', start_time) = strftime('%Y-%W', 'now')
                GROUP BY camera_id, COALESCE(camera_name, camera_id)
            ),
            last_week AS (
                SELECT
                    camera_id,
                    COUNT(*)                                                             AS events_last_week,
                    ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                                         AS dur_last_week
                FROM drowsy_events
                WHERE strftime('%Y-%W', start_time) = strftime('%Y-%W', date('now', '-7 day'))
                GROUP BY camera_id
            )
            SELECT
                t.camera_id,
                t.camera_name,
                t.events_this_week,
                COALESCE(l.events_last_week, 0)                                         AS events_last_week,
                t.dur_this_week,
                COALESCE(l.dur_last_week, 0)                                            AS dur_last_week,
                (t.events_this_week - COALESCE(l.events_last_week, 0))                  AS delta_events,
                ROUND(t.dur_this_week - COALESCE(l.dur_last_week, 0), 2)                AS delta_duration_sec
            FROM this_week t
            LEFT JOIN last_week l ON t.camera_id = l.camera_id
            ORDER BY delta_duration_sec DESC
        """,
        'chart': 'bar',
        'summary_template': 'So sánh tuần này vs tuần trước ({n} phòng).',
        'empty_text': 'Chưa đủ dữ liệu để so sánh tuần.',
        'risk': 'none',
    },

    # 5. Top học sinh ngủ gật nhiều nhất tháng này
    {
        'name': 'top_students',
        'keywords': [
            'top học sinh', 'học sinh', 'ngủ nhiều', 'sinh viên',
            'nhiều nhất', 'tháng', 'top',
        ],
        'sql': """
            SELECT
                camera_id,
                student_id,
                COUNT(*)                                                              AS total_events,
                ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                                      AS total_duration_sec,
                ROUND(AVG(CASE WHEN is_active = 0 THEN duration_seconds END), 2)      AS avg_duration_sec
            FROM drowsy_events
            WHERE strftime('%Y-%m', start_time) = strftime('%Y-%m', 'now')
            GROUP BY camera_id, student_id
            ORDER BY total_duration_sec DESC, total_events DESC
            LIMIT 10
        """,
        'chart': 'bar',
        'summary_template': 'Top {n} học sinh ngủ gật nhiều nhất tháng này.',
        'empty_text': 'Tháng này chưa ghi nhận học sinh ngủ gật nào.',
        'risk': 'none',
    },

    # 6. Khung giờ dễ ngủ gật nhất
    {
        'name': 'peak_hours',
        'keywords': [
            'khung giờ', 'giờ', 'dễ ngủ', 'thời gian',
            'buổi', 'lúc mấy', 'cao điểm',
        ],
        'sql': """
            SELECT
                strftime('%H:00', start_time)                                         AS hour_of_day,
                COUNT(*)                                                               AS total_events,
                COUNT(DISTINCT student_id)                                             AS unique_students,
                ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                                       AS total_duration_sec
            FROM drowsy_events
            WHERE date(start_time) >= date('now', '-30 day')
            GROUP BY strftime('%H', start_time)
            ORDER BY total_events DESC
            LIMIT 8
        """,
        'chart': 'bar',
        'summary_template': 'Top {n} khung giờ ngủ gật nhiều nhất (30 ngày qua).',
        'empty_text': 'Chưa có dữ liệu trong 30 ngày gần đây.',
        'risk': 'none',
    },
]

# ── Intent classifier ─────────────────────────────────────────────────────────

def _classify(question: str) -> Optional[Dict[str, Any]]:
    """Keyword-based classifier. Ưu tiên intent xuất hiện trước trong danh sách."""
    q = question.lower().strip()
    for intent in INTENTS:
        for kw in intent['keywords']:
            if kw in q:
                return intent
    return None


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(
    intent: Dict[str, Any],
    columns: List[str],
    rows: List[List[Any]],
) -> str:
    if not rows:
        return intent['empty_text']

    tmpl = intent['summary_template']
    n = len(rows)

    # today_overview trả về 1 dòng KPI — format theo tên cột
    if intent['name'] == 'today_overview' and rows:
        vals = dict(zip(columns, rows[0]))
        vals = {k: (v if v is not None else 0) for k, v in vals.items()}
        try:
            return tmpl.format(**vals)
        except Exception:
            pass

    return tmpl.format(n=n)


def _risk_level(intent: Dict[str, Any], rows: List[List[Any]]) -> str:
    if not rows:
        return 'none'
    base = intent.get('risk', 'none')
    if base == 'high':
        return 'high' if len(rows) >= 3 else 'medium'
    return base


# ── AI helpers ────────────────────────────────────────────────────────────────

_INTENT_NAMES = [i['name'] for i in INTENTS]
_INTENT_KEYWORDS_HINT = ', '.join(
    f"{i['name']} ({', '.join(i['keywords'][:3])})" for i in INTENTS
)

_GEMINI_MODEL = 'gemini-2.0-flash'


def _ai_summary(question: str, intent_name: str, columns: List[str], rows: List[List[Any]]) -> Optional[str]:
    """Dùng Gemini sinh tóm tắt tự nhiên từ dữ liệu thực tế."""
    if not _ai_client or not rows:
        return None
    header = '\t'.join(columns)
    data_lines = '\n'.join('\t'.join(str(v) for v in r) for r in rows[:20])
    prompt = (
        f"Bạn là chuyên gia phân tích dữ liệu hệ thống giám sát học tập.\n"
        f"Người dùng hỏi: '{question}'\n"
        f"Báo cáo: {intent_name}\n"
        f"Dữ liệu thực tế:\n{header}\n{data_lines}\n\n"
        f"Hãy viết 1-2 câu nhận xét ngắn gọn, thông minh bằng tiếng Việt. "
        f"Nêu bật xu hướng hoặc vấn đề đáng chú ý nhất từ dữ liệu, tránh liệt kê máy móc."
    )
    try:
        resp = _ai_client.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
        return resp.text.strip()
    except Exception:
        return None


def _ai_handle_unknown(question: str) -> Dict[str, Any]:
    """Dùng Gemini xử lý câu hỏi ngoài keyword — gợi ý cách hỏi lại."""
    _fallback = (
        'Xin lỗi, tôi chưa hiểu câu hỏi này. '
        'Thử hỏi: "tổng quan hôm nay", "phòng rủi ro cao", '
        '"so sánh tuần", "top học sinh", "khung giờ dễ ngủ", "đang ngủ gật".'
    )
    # FIXED: Use _ai_client instead of _ai_model
    if not _ai_client:
        ai_text = _fallback
    else:
        prompt = (
            f"Bạn là trợ lý chatbot hệ thống giám sát ngủ gật trong phòng học.\n"
            f"Các loại báo cáo khả dụng: {_INTENT_KEYWORDS_HINT}\n\n"
            f"Câu hỏi của người dùng: {question}\n\n"
            f"Dựa trên các báo cáo hiện có, hãy gợi ý lịch sự cách đặt câu hỏi để người dùng "
            f"có được thông tin họ cần. Nếu hoàn toàn không liên quan, hãy trả lời rằng bạn "
            f"chuyên về phân tích dữ liệu ngủ gật. Tối đa 2 câu tiếng Việt."
        )
        try:
            resp = _ai_client.models.generate_content(model=_GEMINI_MODEL, contents=prompt)
            ai_text = resp.text.strip()
        except Exception:
            ai_text = _fallback
    return {
        'success': True,
        'question': question,
        'intent': 'unknown',
        'sql_used': '',
        'column_names': [],
        'rows': [],
        'summary_text': ai_text,
        'chart_suggestion': 'none',
        'risk_level': 'none',
    }


# ── Public entry point ────────────────────────────────────────────────────────

def handle_question(question: str) -> Dict[str, Any]:
    """
    Nhận câu hỏi tiếng Việt, trả về dict:
      success, question, intent, sql_used,
      column_names, rows, summary_text, chart_suggestion, risk_level
    """
    intent = _classify(question)

    # Unknown intent — thử AI
    if intent is None:
        return _ai_handle_unknown(question)

    # Database chưa tồn tại
    if not os.path.exists(_get_db_path()):
        return {
            'success': True,
            'question': question,
            'intent': intent['name'],
            'sql_used': '',
            'column_names': [],
            'rows': [],
            'summary_text': (
                'Cơ sở dữ liệu chưa có dữ liệu. '
                'Hãy khởi động camera và ghi nhận sự kiện trước.'
            ),
            'chart_suggestion': 'none',
            'risk_level': 'none',
        }

    sql = intent['sql'].strip()

    # Safety check (dự phòng — SQL trong INTENTS luôn là SELECT)
    if not _is_safe_sql(sql):
        return {
            'success': False,
            'question': question,
            'intent': intent['name'],
            'sql_used': sql,
            'column_names': [],
            'rows': [],
            'summary_text': 'Lỗi nội bộ: câu truy vấn không được phép.',
            'chart_suggestion': 'none',
            'risk_level': 'none',
        }

    columns, rows, error = _run_query(sql)

    if error:
        return {
            'success': False,
            'question': question,
            'intent': intent['name'],
            'sql_used': sql,
            'column_names': [],
            'rows': [],
            'summary_text': f'Lỗi truy vấn cơ sở dữ liệu: {error}',
            'chart_suggestion': 'none',
            'risk_level': 'none',
        }

    template_summary = _build_summary(intent, columns, rows)
    ai_text = _ai_summary(question, intent['name'], columns, rows) if rows else None

    return {
        'success': True,
        'question': question,
        'intent': intent['name'],
        'sql_used': sql,
        'column_names': columns,
        'rows': rows,
        'summary_text': ai_text or template_summary,
        'chart_suggestion': intent['chart'],
        'risk_level': _risk_level(intent, rows),
    }
