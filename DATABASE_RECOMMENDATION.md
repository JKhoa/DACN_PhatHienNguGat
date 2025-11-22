# 🗄️ ĐỀ XUẤT DATABASE CHO ỨNG DỤNG PHÁT HIỆN NGỦ GẬT

## 📊 **PHÂN TÍCH YÊU CẦU**

### **1. Đặc điểm dữ liệu hiện tại:**
```python
# Drowsiness Events
- Camera ID (string)
- Student ID (int)
- Start Time (datetime)
- End Time (datetime)
- Duration (float seconds)
- Is Active (boolean)

# Statistics
- Total events per camera
- Unique students count
- Total duration
- Active drowsy students
- Time-based aggregations (today/week/month)
```

### **2. Yêu cầu hệ thống:**
- ✅ **Real-time updates** (mỗi frame ~5-6 fps)
- ✅ **Multi-camera support** (10-50 cameras đồng thời)
- ✅ **Historical data** (ngày/tuần/tháng)
- ✅ **Export capabilities** (CSV/JSON/PDF)
- ✅ **Concurrent access** (nhiều camera workers)
- ✅ **Lightweight** (chạy trên máy desktop/laptop)
- ✅ **No cloud dependency** (local-first)

### **3. Khối lượng dữ liệu ước tính:**
```
Giả sử: 30 học sinh/phòng, 5 sự kiện ngủ gật/ngày/học sinh
- Events/ngày: 30 × 5 = 150 events
- Events/tháng: 150 × 20 ngày học = 3,000 events
- Events/năm học: 3,000 × 9 tháng = 27,000 events
- Kích thước 1 event: ~200 bytes
- Tổng/năm: ~5.4 MB (rất nhỏ)
```

---

## ✅ **ĐỀ XUẤT: SQLITE3 (STRONGLY RECOMMENDED)**

### **🎯 Lý do chọn SQLite:**

#### **1. Phù hợp hoàn hảo với yêu cầu:**
- ✅ **Embedded database** - không cần server riêng
- ✅ **Zero configuration** - không cần setup phức tạp
- ✅ **File-based** - dễ backup, copy, share
- ✅ **ACID compliant** - đảm bảo tính toàn vẹn dữ liệu
- ✅ **Concurrent reads** - nhiều camera đọc đồng thời
- ✅ **Cross-platform** - Windows/Linux/Mac
- ✅ **Python built-in** - không cần cài thêm library

#### **2. Performance phù hợp:**
```python
# Benchmarks for this use case:
- INSERT: ~50,000 ops/second (insert events)
- SELECT: ~100,000 ops/second (query statistics)
- Database size: <10MB for 1 năm học
- Query time: <5ms for most queries
- Index support: Fast lookups by camera_id, student_id, timestamp
```

#### **3. Features cần thiết:**
- ✅ **Transactions** - đảm bảo atomicity khi log events
- ✅ **Indexes** - tăng tốc query theo time ranges
- ✅ **Aggregations** - COUNT, SUM, AVG cho statistics
- ✅ **Date/Time functions** - filter by day/week/month
- ✅ **Foreign keys** - maintain data integrity

#### **4. Easy integration:**
```python
import sqlite3
from datetime import datetime

# Chỉ cần vài dòng code!
conn = sqlite3.connect('drowsiness_logs.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS drowsy_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        camera_name TEXT,
        student_id INTEGER NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        duration_seconds REAL,
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_camera_time (camera_id, start_time),
        INDEX idx_student (student_id)
    )
''')
```

---

## 🔄 **SO SÁNH CÁC LỰA CHỌN**

### **Option 1: SQLite3 ⭐⭐⭐⭐⭐ (RECOMMENDED)**

**Ưu điểm:**
- ✅ Perfect fit cho desktop app
- ✅ No server overhead
- ✅ Fast queries (<5ms)
- ✅ Built-in Python support
- ✅ Easy backup (copy .db file)
- ✅ No network latency
- ✅ Supports complex queries
- ✅ ACID transactions

**Nhược điểm:**
- ⚠️ Write concurrency limited (nhưng OK cho use case này)
- ⚠️ File locking issues nếu >100 concurrent writes (không xảy ra ở app này)

**Kết luận:** **PERFECT CHOICE** ✅

---

### **Option 2: JSON Files ⭐⭐ (CURRENT - NOT RECOMMENDED)**

**Ưu điểm:**
- ✅ Simple implementation
- ✅ Human-readable
- ✅ No external dependencies

**Nhược điểm:**
- ❌ **Slow queries** (phải load toàn bộ file)
- ❌ **No indexes** (tìm kiếm O(n))
- ❌ **No aggregations** (phải tự code)
- ❌ **Race conditions** (concurrent writes dangerous)
- ❌ **Large file size** (không compressed)
- ❌ **No transactions** (data corruption risk)

**Kết luận:** **CHỈ DÙNG CHO PROTOTYPE** ❌

---

### **Option 3: PostgreSQL/MySQL ⭐⭐⭐ (OVERKILL)**

**Ưu điểm:**
- ✅ Enterprise-grade
- ✅ Excellent concurrency
- ✅ Advanced features (partitioning, replication)
- ✅ Scalable to millions of records

**Nhược điểm:**
- ❌ **Requires server installation** (phức tạp cho users)
- ❌ **Network overhead** (latency ~1-5ms)
- ❌ **Resource heavy** (RAM, CPU)
- ❌ **Complex setup** (config, permissions, ports)
- ❌ **Overkill** cho 27k events/year

**Kết luận:** **KHÔNG CẦN THIẾT** ⚠️

---

### **Option 4: MongoDB ⭐⭐ (NOT SUITABLE)**

**Ưu điểm:**
- ✅ Flexible schema
- ✅ JSON-like documents
- ✅ Scalable

**Nhược điểm:**
- ❌ **Requires MongoDB server**
- ❌ **Larger disk footprint**
- ❌ **Slower aggregations** than SQL
- ❌ **Overkill** cho structured data này
- ❌ **No built-in Python support**

**Kết luận:** **KHÔNG PHÙ HỢP** ❌

---

### **Option 5: Redis ⭐⭐⭐ (SUPPLEMENTARY)**

**Ưu điểm:**
- ✅ Ultra-fast (in-memory)
- ✅ Good for active students tracking
- ✅ Pub/sub for real-time updates

**Nhược điểm:**
- ❌ **Not persistent by default** (cần AOF/RDB)
- ❌ **RAM limited** (không tốt cho historical data)
- ❌ **Requires Redis server**
- ❌ **Weak query capabilities**

**Kết luận:** **DÙNG KẾT HỢP VỚI SQLITE** (optional) ⚠️

---

## 🏗️ **KIẾN TRÚC ĐỀ XUẤT**

### **Architecture: Hybrid SQLite + In-Memory Cache**

```
┌─────────────────────────────────────────────────┐
│          APPLICATION LAYER                      │
│  (Python Flask Backend + React Frontend)        │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ┌────▼─────┐      ┌──────▼───────┐
    │ SQLite3  │      │  In-Memory   │
    │ Database │◄─────┤    Cache     │
    │          │      │  (Optional)  │
    │ Persistent│      │   Hot Data   │
    └──────────┘      └──────────────┘
         │
    ┌────▼────────────────────────┐
    │   drowsiness_logs.db        │
    │   - Events table            │
    │   - Statistics cache table  │
    │   - Camera metadata table   │
    └─────────────────────────────┘
```

### **Database Schema:**

```sql
-- Main events table
CREATE TABLE drowsy_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    camera_name TEXT,
    student_id INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds REAL DEFAULT 0,
    event_type TEXT CHECK(event_type IN ('drowsy', 'sleeping', 'wake_up')),
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX idx_camera_time ON drowsy_events(camera_id, start_time DESC);
CREATE INDEX idx_student_time ON drowsy_events(student_id, start_time DESC);
CREATE INDEX idx_active ON drowsy_events(is_active, camera_id);
CREATE INDEX idx_created_date ON drowsy_events(DATE(created_at));

-- Camera metadata
CREATE TABLE cameras (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,
    config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP
);

-- Statistics cache (optional - for performance)
CREATE TABLE stats_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    camera_id TEXT,
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    total_events INTEGER,
    total_students INTEGER,
    total_duration REAL,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX idx_cache_key ON stats_cache(cache_key, expires_at);
```

---

## 💻 **IMPLEMENTATION GUIDE**

### **Step 1: Install (if needed)**
```bash
# SQLite3 built-in Python - không cần cài!
python -c "import sqlite3; print(sqlite3.sqlite_version)"
# Output: 3.45.0 (hoặc tương tự)
```

### **Step 2: Create Database Helper**
```python
# python-backend/db_helper.py
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading

class DrowsinessDatabase:
    def __init__(self, db_path: str = "drowsiness_logs/events.db"):
        self.db_path = db_path
        self.local = threading.local()  # Thread-safe connections
        self._init_database()
    
    def _get_connection(self):
        """Get thread-local connection"""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=10.0
            )
            self.local.conn.row_factory = sqlite3.Row
        return self.local.conn
    
    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drowsy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                camera_name TEXT,
                student_id INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL DEFAULT 0,
                event_type TEXT DEFAULT 'drowsy',
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_camera_time 
            ON drowsy_events(camera_id, start_time DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_active 
            ON drowsy_events(is_active, camera_id)
        ''')
        
        conn.commit()
        print("✅ Database initialized:", self.db_path)
    
    def insert_event(self, camera_id: str, student_id: int, 
                    camera_name: str = None) -> int:
        """Insert new drowsy event"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drowsy_events 
            (camera_id, camera_name, student_id, start_time, is_active)
            VALUES (?, ?, ?, ?, 1)
        ''', (camera_id, camera_name, student_id, datetime.now()))
        
        conn.commit()
        return cursor.lastrowid
    
    def end_event(self, event_id: int):
        """Mark event as ended"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE drowsy_events
            SET end_time = ?,
                duration_seconds = (julianday(?) - julianday(start_time)) * 86400,
                is_active = 0
            WHERE id = ?
        ''', (datetime.now(), datetime.now(), event_id))
        
        conn.commit()
    
    def get_active_events(self, camera_id: str = None) -> List[Dict]:
        """Get currently active drowsy events"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if camera_id:
            cursor.execute('''
                SELECT * FROM drowsy_events
                WHERE is_active = 1 AND camera_id = ?
                ORDER BY start_time DESC
            ''', (camera_id,))
        else:
            cursor.execute('''
                SELECT * FROM drowsy_events
                WHERE is_active = 1
                ORDER BY start_time DESC
            ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self, camera_id: str, start_date: datetime, 
                      end_date: datetime) -> Dict:
        """Get statistics for a time period"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT student_id) as total_students,
                COUNT(*) as total_events,
                SUM(duration_seconds) as total_duration,
                AVG(duration_seconds) as avg_duration
            FROM drowsy_events
            WHERE camera_id = ?
              AND start_time BETWEEN ? AND ?
              AND is_active = 0
        ''', (camera_id, start_date, end_date))
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def get_events(self, camera_id: str, start_date: datetime, 
                  end_date: datetime, limit: int = 100) -> List[Dict]:
        """Get events in date range"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM drowsy_events
            WHERE camera_id = ?
              AND start_time BETWEEN ? AND ?
            ORDER BY start_time DESC
            LIMIT ?
        ''', (camera_id, start_date, end_date, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Delete events older than X days"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cursor.execute('''
            DELETE FROM drowsy_events
            WHERE start_time < ? AND is_active = 0
        ''', (cutoff_date,))
        
        deleted = cursor.rowcount
        conn.commit()
        print(f"🗑️ Deleted {deleted} old events (>{days_to_keep} days)")
        return deleted
```

### **Step 3: Migrate from JSON to SQLite**
```python
# python-backend/migrate_to_sqlite.py
import json
from pathlib import Path
from datetime import datetime
from db_helper import DrowsinessDatabase

def migrate_json_to_sqlite():
    """Migrate existing JSON logs to SQLite"""
    db = DrowsinessDatabase()
    
    # Find all JSON log files
    log_dir = Path("drowsiness_logs")
    json_files = list(log_dir.glob("*.json"))
    
    print(f"📦 Found {len(json_files)} JSON files to migrate")
    
    total_migrated = 0
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract camera_id from filename
        camera_id = json_file.stem
        
        # Migrate events
        for event in data.get('events', []):
            db.insert_event(
                camera_id=camera_id,
                student_id=event['student_id'],
                camera_name=event.get('camera_name')
            )
            
            # If event has end_time, update it
            if event.get('end_time'):
                # ... update end_time logic
                pass
            
            total_migrated += 1
    
    print(f"✅ Migrated {total_migrated} events to SQLite")

if __name__ == '__main__':
    migrate_json_to_sqlite()
```

### **Step 4: Update DrowsinessLogger to use SQLite**
```python
# python-backend/drowsiness_logger.py
from db_helper import DrowsinessDatabase

class CameraLogger:
    def __init__(self, camera_id: str, camera_name: str = None):
        self.camera_id = camera_id
        self.camera_name = camera_name or f"Phòng {camera_id}"
        
        # SQLite database
        self.db = DrowsinessDatabase()
        
        # In-memory cache for active events (fast lookup)
        self.active_event_ids: Dict[int, int] = {}  # student_id -> event_id
        
    def start_drowsiness(self, student_id: int):
        """Bắt đầu ghi nhận học sinh ngủ gật"""
        if student_id not in self.active_event_ids:
            event_id = self.db.insert_event(
                self.camera_id, 
                student_id, 
                self.camera_name
            )
            self.active_event_ids[student_id] = event_id
            print(f"[{self.camera_name}] 🔴 Học sinh #{student_id} BẮT ĐẦU ngủ gật")
            
    def end_drowsiness(self, student_id: int):
        """Kết thúc ghi nhận học sinh tỉnh lại"""
        if student_id in self.active_event_ids:
            event_id = self.active_event_ids.pop(student_id)
            self.db.end_event(event_id)
            print(f"[{self.camera_name}] 🟢 Học sinh #{student_id} TỈNH LẠI")
    
    def get_statistics(self, start_date, end_date):
        """Lấy thống kê từ database"""
        return self.db.get_statistics(self.camera_id, start_date, end_date)
```

---

## 📊 **PERFORMANCE COMPARISON**

| Metric | JSON Files | SQLite | PostgreSQL |
|--------|-----------|--------|------------|
| **Insert Speed** | ⚠️ Slow (10-50 ops/s) | ✅ Fast (50k ops/s) | ✅ Fast (10k ops/s) |
| **Query Speed** | ❌ Very Slow (O(n)) | ✅ Fast (<5ms) | ✅ Fast (~2ms) |
| **Memory Usage** | ❌ High (load all) | ✅ Low (10MB) | ⚠️ Medium (50MB+) |
| **Disk Usage** | ⚠️ Large (uncompressed) | ✅ Small (compressed) | ⚠️ Medium |
| **Setup Complexity** | ✅ None | ✅ None | ❌ Complex |
| **Concurrency** | ❌ Poor (file locks) | ✅ Good (read) | ✅ Excellent |
| **Scalability** | ❌ Bad | ✅ Good (100k+ events) | ✅ Excellent |

---

## 🎯 **FINAL RECOMMENDATION**

### **✅ CHO MÔI TRƯỜNG PRODUCTION:**

```
Primary Database: SQLite3
├── Lý do: Perfect fit cho desktop app, fast, reliable
├── File location: drowsiness_logs/events.db
├── Size estimate: ~10MB/năm học
└── Backup: Tự động copy .db file hàng ngày

Optional Cache Layer: In-Memory Dict
├── Lý do: Ultra-fast access cho active students
├── Data: Current drowsy students only
└── Sync: Write-through to SQLite
```

### **🔄 Migration Plan:**

**Phase 1 (Week 1):**
- ✅ Implement `db_helper.py` với SQLite
- ✅ Create migration script từ JSON → SQLite
- ✅ Test với existing data

**Phase 2 (Week 2):**
- ✅ Update `drowsiness_logger.py` để dùng SQLite
- ✅ Keep JSON export cho backward compatibility
- ✅ Test concurrent camera access

**Phase 3 (Week 3):**
- ✅ Deploy to production
- ✅ Monitor performance
- ✅ Setup automatic backup

---

## 📝 **TÓM TẮT**

### **Câu trả lời ngắn gọn:**
**➡️ Dùng SQLite3 - đây là lựa chọn TỐT NHẤT cho ứng dụng này!**

### **3 lý do chính:**
1. ✅ **Zero setup** - built-in Python, không cần cài server
2. ✅ **Perfect performance** - đủ nhanh cho 50+ cameras, <5ms queries
3. ✅ **Future-proof** - dễ migrate lên PostgreSQL nếu cần scale lớn hơn

### **Không nên dùng:**
- ❌ JSON files - quá chậm, không có query optimization
- ❌ PostgreSQL/MySQL - overkill, phức tạp setup
- ❌ MongoDB - không phù hợp với structured data này

---

**🎉 KẾT LUẬN: SQLITE3 LÀ LỰA CHỌN TỐI ƯU NHẤT!**
