
"""
static_testing_solutions.py

Giải bài thực hành 6 – Các kỹ thuật kiểm thử tĩnh (Phần III) và cung cấp
bản sửa (refactor) + demo chạy thử. File này không sử dụng thư viện ngoài.

Cách chạy nhanh:
    python static_testing_solutions.py --demo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import sys
import itertools


# ---------------------------
# CÂU 1. HÀM TÍNH GIÁ SAU GIẢM
# ---------------------------

def calculate_price_fixed(price: float,
                          discount_percent: float,
                          is_summer_sale: bool = False,
                          seasonal_extra_off: float = 0.05,
                          round_to: Optional[int] = 2) -> float:
    """
    Tính giá cuối cùng sau chiết khấu.
    - discount_percent chấp nhận 0..1 (0%..100%) hoặc 0..100 (phần trăm).
    - Nếu nhập >1, hàm sẽ chuẩn hóa về thang 0..1 bằng cách chia 100.
    - Hợp lệ: price >= 0, 0 <= discount_percent <= 1 sau chuẩn hóa, 0 <= seasonal_extra_off < 1.

    Thêm:
    - Nếu is_summer_sale=True, giảm thêm seasonal_extra_off (mặc định 5%).
    - Có thể làm tròn kết quả với round_to chữ số thập phân (mặc định 2).

    Trả về:
    - Giá cuối cùng (float). Ném ValueError nếu input không hợp lệ.
    """
    # Kiểm tra kiểu/giá trị cơ bản
    if price < 0:
        raise ValueError("price phải >= 0")
    # Chuẩn hóa chiết khấu nếu người dùng nhập theo phần trăm (ví dụ 20)
    if discount_percent > 1:
        discount_percent = discount_percent / 100.0

    if not (0.0 <= discount_percent <= 1.0):
        raise ValueError("discount_percent sau chuẩn hóa phải trong [0, 1]")

    if not (0.0 <= seasonal_extra_off < 1.0):
        raise ValueError("seasonal_extra_off phải trong [0, 1)")

    # Tính giá sau giảm thông thường
    final_price = price * (1.0 - discount_percent)

    # Giảm thêm theo mùa nếu có
    if is_summer_sale:
        final_price *= (1.0 - seasonal_extra_off)

    # Không trả về số âm do lỗi làm tròn hay input biên
    final_price = max(0.0, final_price)

    if round_to is not None:
        return round(final_price, round_to)
    return final_price


# ---------------------------
# CÂU 2. ÁP DỤNG TIÊU CHUẨN & CHỈ SỐ MÃ
# ---------------------------

MAX_LOGIN_ATTEMPTS = 5  # Không dùng "số ma thuật"

def process_user_data(data: Dict[str, Any], user_role: str, is_active: bool) -> Dict[str, str]:
    """
    Bản refactor với tiêu chuẩn:
    - snake_case cho tên hàm/biến
    - loại bỏ lồng nhau sâu bằng early-return & tách nhánh rõ ràng
    - tách hằng số MAX_LOGIN_ATTEMPTS
    - độ dài < 20 LOC (không tính comment/trắng)
    """
    if user_role == "ADMIN":
        return _handle_admin(data, is_active)
    if user_role == "EDITOR":
        return _handle_editor(is_active)
    # Mặc định GUEST
    return _handle_guest(data)

def _handle_admin(data: Dict[str, Any], is_active: bool) -> Dict[str, str]:
    if not is_active:
        return {"status": "Inactive admin, no action"}
    # "Xử lý phức tạp" cho admin (minh họa)
    permissions = set(data.get("permissions", []))
    if "full_access" in permissions:
        # Có thể gán quyền hệ thống ở đây
        pass
    return {"status": "Processed"}

def _handle_editor(is_active: bool) -> Dict[str, str]:
    return {"status": "Processed editor data"} if is_active else {"status": "Inactive editor"}

def _handle_guest(data: Dict[str, Any]) -> Dict[str, str]:
    attempts = int(data.get("login_attempts", 0) or 0)
    return {"status": "Guest access allowed"} if attempts < MAX_LOGIN_ATTEMPTS else {"status": "Guest locked out"}


ORIGINAL_PROCESS_USER_DATA = """
# Process user data based on their role and status
def processUserData(data, userRole, is_active):
    if userRole == "ADMIN":
        if is_active:
            print("Processing admin data...")
            # Some complex processing for admin
            if 'permissions' in data:
                for p in data['permissions']:
                    if p == 'full_access':
                        print("Admin has full access.")
                        # Grant full system access
            result = {"status": "Processed"}
        else:
            result = {"status": "Inactive admin, no action"}
    elif userRole == "EDITOR":
        if is_active:
            result = {"status": "Processed editor data"}
        else:
            result = {"status": "Inactive editor"}
    else:
        # Default processing for GUEST
        if data['login_attempts'] < 5:
            result = {"status": "Guest access allowed"}
        else:
            result = {"status": "Guest locked out"}
    return result
""".strip("\n")

def compute_code_metrics(py_code: str) -> Tuple[int, int]:
    """
    Tính số dòng mã (LoC) không tính dòng trống/comment và độ sâu lồng nhau xấp xỉ
    dựa trên độ thụt đầu dòng bội số 4.
    """
    loc = 0
    max_depth = 0
    for raw in py_code.splitlines():
        line = raw.rstrip("\n")
        striped = line.strip()
        if not striped or striped.startswith("#"):
            continue
        loc += 1
        indent_spaces = len(line) - len(line.lstrip(" "))
        depth = indent_spaces // 4
        if depth > max_depth:
            max_depth = depth
    return loc, max_depth


# ---------------------------
# CÂU 3. QUY TRÌNH ĐÁNH GIÁ CHÍNH THỨC – BẢN SỬA LỚP AccountManager
# ---------------------------

@dataclass
class User:
    id: int
    username: str
    password_hash: str
    email: str
    last_login: str

class FakeDB:
    """DB giả lập để demo, hỗ trợ truy vấn tham số."""
    def __init__(self) -> None:
        self._users: Dict[int, User] = {}
        self._by_username: Dict[str, int] = {}

    def add_user(self, u: User) -> None:
        self._users[u.id] = u
        self._by_username[u.username] = u.id

    # Trả về True/False cho câu truy vấn xác thực
    def execute(self, query: str, params: Tuple[Any, ...]) -> bool:
        # Demo cực tối giản: nhận dạng câu truy vấn "SELECT 1 FROM users WHERE username = ? AND password_hash = ?"
        if "FROM users" in query and len(params) == 2:
            username, password_hash = params
            uid = self._by_username.get(str(username))
            if uid is None:
                return False
            user = self._users.get(uid)
            return bool(user and user.password_hash == password_hash)
        raise NotImplementedError("FakeDB.execute: query không được hỗ trợ trong demo.")

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        u = self._users.get(int(user_id))
        if not u:
            return None
        return {"id": u.id, "name": u.username, "email": u.email, "last_login": u.last_login}

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        uid = self._by_username.get(username)
        if uid is None:
            return None
        return self.get_user(uid)

class AccountManagerSecure:
    """
    Bản viết lại khắc phục các khiếm khuyết:
    - Dùng truy vấn tham số, không nối chuỗi => chống SQL Injection
    - Không lưu/truy vấn mật khẩu thuần: hash SHA-256 minh họa (demo)
    - Trả về bool rõ ràng cho tất cả nhánh
    - Xử lý lỗi/không tìm thấy trong get_user_profile
    - Loại bỏ "mã chết"
    """
    def __init__(self, db_connection: FakeDB) -> None:
        self.db = db_connection

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def validate_password(self, username: str, password: str) -> bool:
        # Ràng buộc tối thiểu về độ dài theo đề bài
        if not isinstance(password, str) or len(password) < 8:
            return False
        if not isinstance(username, str) or not username:
            return False

        pwd_hash = self._hash_password(password)
        query = "SELECT 1 FROM users WHERE username = ? AND password_hash = ?"
        try:
            return bool(self.db.execute(query, (username, pwd_hash)))
        except Exception:
            # Không để lỗi DB làm lộ trạng thái hệ thống, coi như xác thực thất bại
            return False

    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        try:
            user_data = self.db.get_user(user_id)
        except Exception:
            return None
        if not user_data:
            return None
        return {
            "name": user_data["name"],
            "email": user_data["email"],
            "last_login": user_data["last_login"],
        }


# ---------------------------
# DEMO / CLI
# ---------------------------

def _demo_cau1() -> None:
    print("== DEMO Câu 1: calculate_price_fixed ==")
    tests = [
        {"price": 100.0, "discount_percent": 20, "summer": False},  # 20% (theo %)
        {"price": 100.0, "discount_percent": 0.2, "summer": True},  # 20% (theo 0..1) + summer 5%
        {"price": 0.0, "discount_percent": 50, "summer": False},    # biên price=0
        {"price": 100.0, "discount_percent": 100, "summer": True},  # 100% + summer
    ]
    for i, t in enumerate(tests, 1):
        price = t["price"]
        d = t["discount_percent"]
        summer = t["summer"]
        try:
            out = calculate_price_fixed(price, d, is_summer_sale=summer)
        except Exception as e:
            out = f"Error: {e}"
        print(f"  TC{i}: price={price}, discount={d}, summer={summer} -> {out}")

def _demo_cau2_metrics() -> None:
    print("\n== DEMO Câu 2: Chỉ số mã cho bản gốc ==")
    loc, depth = compute_code_metrics(ORIGINAL_PROCESS_USER_DATA)
    print(f"  LOC (không tính comment/trắng): {loc}")
    print(f"  Độ sâu lồng nhau ước lượng:     {depth}")

def _demo_cau2_refactor() -> None:
    print("\n== DEMO Câu 2: Chạy bản refactor ==")
    print("  Admin active full_access ->", process_user_data({"permissions": ["full_access"]}, "ADMIN", True))
    print("  Admin inactive ->", process_user_data({}, "ADMIN", False))
    print("  Editor active ->", process_user_data({}, "EDITOR", True))
    print("  Guest attempts=3 ->", process_user_data({"login_attempts": 3}, "GUEST", True))
    print("  Guest attempts=7 ->", process_user_data({"login_attempts": 7}, "GUEST", True))

def _demo_cau3_secure() -> None:
    print("\n== DEMO Câu 3: AccountManagerSecure ==")
    # Setup DB giả
    db = FakeDB()
    # Tạo người dùng demo
    def _hash(p): return hashlib.sha256(p.encode()).hexdigest()
    db.add_user(User(id=1, username="alice", password_hash=_hash("password123"), email="a@example.com", last_login="2025-04-10"))
    db.add_user(User(id=2, username="bob", password_hash=_hash("s3cretP@ss"), email="b@example.com", last_login="2025-04-11"))

    am = AccountManagerSecure(db)
    print("  Đúng mật khẩu alice:", am.validate_password("alice", "password123"))
    print("  Sai mật khẩu alice:", am.validate_password("alice", "wrongpass"))
    print("  Mật khẩu quá ngắn:", am.validate_password("alice", "short"))
    print("  Hồ sơ user id 1:", am.get_user_profile(1))
    print("  Hồ sơ user id 99:", am.get_user_profile(99))

def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("Dùng: python static_testing_solutions.py --demo")
        return 0
    if argv[0] == "--demo":
        _demo_cau1()
        _demo_cau2_metrics()
        _demo_cau2_refactor()
        _demo_cau3_secure()
        return 0
    print("Tham số không hợp lệ. Dùng --demo để chạy minh họa.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
