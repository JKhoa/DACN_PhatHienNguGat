"""
SQLite Database Helper for Drowsiness Detection System
Handles all database operations with thread-safe connections
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import threading
from pathlib import Path
import os


class DrowsinessDatabase:
    """Thread-safe SQLite database manager for drowsiness events"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Create drowsiness_logs directory if not exists
            log_dir = Path(__file__).parent / "drowsiness_logs"
            log_dir.mkdir(exist_ok=True)
            db_path = str(log_dir / "events.db")
        
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage for connections
        
        # Initialize database schema
        self._init_database()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection
        Creates new connection if doesn't exist for current thread
        """
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None  # Autocommit mode
            )
            # Return rows as dictionaries
            self.local.conn.row_factory = sqlite3.Row
            # WAL mode: cho phép reader và writer chạy song song, giảm "database locked"
            self.local.conn.execute("PRAGMA journal_mode=WAL")
            self.local.conn.execute("PRAGMA synchronous=NORMAL")
            self.local.conn.execute("PRAGMA busy_timeout=30000")

        return self.local.conn
    
    def _init_database(self):
        """Initialize database schema and indexes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create main events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drowsy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                camera_name TEXT,
                student_id INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL DEFAULT 0,
                event_type TEXT DEFAULT 'drowsy' CHECK(event_type IN ('drowsy', 'sleeping', 'wake_up')),
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_camera_time 
            ON drowsy_events(camera_id, start_time DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_student_time 
            ON drowsy_events(student_id, start_time DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_active 
            ON drowsy_events(is_active, camera_id) 
            WHERE is_active = 1
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_date 
            ON drowsy_events(DATE(created_at))
        ''')
        
        # Create camera metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP
            )
        ''')
        
        # ── Chatbot analytics views ──────────────────────────────────────────
        # v_daily_camera_stats: tổng hợp sự kiện theo ngày/phòng
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_daily_camera_stats AS
            SELECT
                date(start_time)                                         AS event_date,
                camera_id,
                COALESCE(camera_name, camera_id)                         AS camera_name,
                COUNT(*)                                                 AS total_events,
                COUNT(DISTINCT student_id)                               AS unique_students,
                ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                         AS total_duration_sec,
                ROUND(AVG(CASE WHEN is_active = 0 THEN duration_seconds END), 2)
                                                                         AS avg_duration_sec
            FROM drowsy_events
            GROUP BY date(start_time), camera_id, COALESCE(camera_name, camera_id)
        ''')

        # v_weekly_student_stats: top học sinh theo tuần
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS v_weekly_student_stats AS
            SELECT
                strftime('%Y-%W', start_time)                            AS year_week,
                camera_id,
                student_id,
                COUNT(*)                                                 AS total_events,
                ROUND(SUM(CASE WHEN is_active = 0 THEN duration_seconds ELSE 0 END), 2)
                                                                         AS total_duration_sec,
                ROUND(AVG(CASE WHEN is_active = 0 THEN duration_seconds END), 2)
                                                                         AS avg_duration_sec
            FROM drowsy_events
            GROUP BY strftime('%Y-%W', start_time), camera_id, student_id
        ''')

        conn.commit()

    def insert_event(self, camera_id: str, student_id: int, 
                    camera_name: str = None, event_type: str = 'drowsy') -> int:
        """
        Insert new drowsy event
        
        Args:
            camera_id: Camera/room identifier
            student_id: Student/person ID
            camera_name: Optional camera display name
            event_type: Type of event ('drowsy', 'sleeping')
            
        Returns:
            event_id: ID of inserted event
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drowsy_events 
            (camera_id, camera_name, student_id, start_time, event_type, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
        ''', (camera_id, camera_name, student_id, datetime.now(), event_type))
        
        event_id = cursor.lastrowid
        
        # Update camera last_active
        self._update_camera_activity(camera_id, camera_name)
        
        return event_id
    
    def end_event(self, event_id: int) -> bool:
        """
        Mark event as ended and calculate duration
        
        Args:
            event_id: ID of event to end
            
        Returns:
            True if updated, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now()
        
        cursor.execute('''
            UPDATE drowsy_events
            SET end_time = ?,
                duration_seconds = (julianday(?) - julianday(start_time)) * 86400,
                is_active = 0,
                updated_at = ?
            WHERE id = ? AND is_active = 1
        ''', (now, now, now, event_id))
        
        return cursor.rowcount > 0
    
    def end_event_by_student(self, camera_id: str, student_id: int) -> bool:
        """
        End active event for a specific student
        
        Args:
            camera_id: Camera identifier
            student_id: Student identifier
            
        Returns:
            True if event was ended, False if no active event
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now()
        
        cursor.execute('''
            UPDATE drowsy_events
            SET end_time = ?,
                duration_seconds = (julianday(?) - julianday(start_time)) * 86400,
                is_active = 0,
                updated_at = ?
            WHERE camera_id = ? AND student_id = ? AND is_active = 1
        ''', (now, now, now, camera_id, student_id))
        
        return cursor.rowcount > 0
    
    def get_active_event(self, camera_id: str, student_id: int) -> Optional[Dict]:
        """
        Get active event for a student in a camera
        
        Args:
            camera_id: Camera identifier
            student_id: Student identifier
            
        Returns:
            Event dict or None if no active event
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM drowsy_events
            WHERE camera_id = ? AND student_id = ? AND is_active = 1
            LIMIT 1
        ''', (camera_id, student_id))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_active_events(self, camera_id: str = None) -> List[Dict]:
        """
        Get all currently active drowsy events
        
        Args:
            camera_id: Optional camera filter
            
        Returns:
            List of active events with current duration
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if camera_id:
            cursor.execute('''
                SELECT *,
                       (julianday('now') - julianday(start_time)) * 86400 as current_duration
                FROM drowsy_events
                WHERE is_active = 1 AND camera_id = ?
                ORDER BY start_time DESC
            ''', (camera_id,))
        else:
            cursor.execute('''
                SELECT *,
                       (julianday('now') - julianday(start_time)) * 86400 as current_duration
                FROM drowsy_events
                WHERE is_active = 1
                ORDER BY start_time DESC
            ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self, camera_id: str, start_date: datetime, 
                      end_date: datetime) -> Dict:
        """
        Get statistics for a time period
        
        Args:
            camera_id: Camera identifier
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Statistics dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get completed events stats
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT student_id) as total_students,
                COUNT(*) as total_events,
                COALESCE(SUM(duration_seconds), 0) as total_duration,
                COALESCE(AVG(duration_seconds), 0) as avg_duration,
                COALESCE(MAX(duration_seconds), 0) as max_duration
            FROM drowsy_events
            WHERE camera_id = ?
              AND start_time BETWEEN ? AND ?
              AND is_active = 0
        ''', (camera_id, start_date, end_date))
        
        stats = dict(cursor.fetchone())
        
        # Get active events count
        cursor.execute('''
            SELECT COUNT(*) as active_count
            FROM drowsy_events
            WHERE camera_id = ? AND is_active = 1
        ''', (camera_id,))
        
        stats['currently_drowsy'] = cursor.fetchone()['active_count']
        
        return stats
    
    def get_events(self, camera_id: str = None, start_date: datetime = None, 
                  end_date: datetime = None, limit: int = 100, 
                  include_active: bool = True) -> List[Dict]:
        """
        Get events with optional filters
        
        Args:
            camera_id: Optional camera filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
            include_active: Include active (ongoing) events
            
        Returns:
            List of events
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM drowsy_events WHERE 1=1'
        params = []
        
        if camera_id:
            query += ' AND camera_id = ?'
            params.append(camera_id)
        
        if start_date:
            query += ' AND start_time >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND start_time <= ?'
            params.append(end_date)
        
        if not include_active:
            query += ' AND is_active = 0'
        
        query += ' ORDER BY start_time DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_student_summary(self, student_id: int, days: int = 7) -> Dict:
        """
        Get summary statistics for a specific student
        
        Args:
            student_id: Student identifier
            days: Number of days to look back
            
        Returns:
            Summary statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_events,
                COALESCE(SUM(duration_seconds), 0) as total_duration,
                COALESCE(AVG(duration_seconds), 0) as avg_duration,
                COALESCE(MAX(duration_seconds), 0) as max_duration,
                COUNT(DISTINCT DATE(start_time)) as days_with_events
            FROM drowsy_events
            WHERE student_id = ?
              AND start_time >= ?
              AND is_active = 0
        ''', (student_id, start_date))
        
        return dict(cursor.fetchone())
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Delete events older than specified days
        
        Args:
            days_to_keep: Keep events newer than this many days
            
        Returns:
            Number of deleted events
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cursor.execute('''
            DELETE FROM drowsy_events
            WHERE start_time < ? AND is_active = 0
        ''', (cutoff_date,))
        
        deleted = cursor.rowcount
        
        if deleted > 0:
            print(f"🗑️ Deleted {deleted} old events (>{days_to_keep} days)")
            
            # Vacuum to reclaim space
            cursor.execute('VACUUM')
        
        return deleted
    
    def _update_camera_activity(self, camera_id: str, camera_name: str = None):
        """Update camera last_active timestamp"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cameras (id, name, last_active)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_active = excluded.last_active,
                name = COALESCE(excluded.name, name)
        ''', (camera_id, camera_name, datetime.now()))
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total events
        cursor.execute('SELECT COUNT(*) as total FROM drowsy_events')
        total = cursor.fetchone()['total']
        
        # Active events
        cursor.execute('SELECT COUNT(*) as active FROM drowsy_events WHERE is_active = 1')
        active = cursor.fetchone()['active']
        
        # Unique students
        cursor.execute('SELECT COUNT(DISTINCT student_id) as students FROM drowsy_events')
        students = cursor.fetchone()['students']
        
        # Database file size
        file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        file_size_mb = file_size / (1024 * 1024)
        
        return {
            'total_events': total,
            'active_events': active,
            'unique_students': students,
            'database_size_mb': round(file_size_mb, 2),
            'database_path': self.db_path
        }
    
    def close(self):
        """Close database connection for current thread"""
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None


# Singleton instance
_db_instance = None

def get_database() -> DrowsinessDatabase:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DrowsinessDatabase()
    return _db_instance


if __name__ == '__main__':
    # Test database
    print("🧪 Testing SQLite Database...")
    
    db = DrowsinessDatabase()
    
    # Test insert
    event_id = db.insert_event('camera_1', 123, 'Test Camera')
    print(f"✅ Inserted event ID: {event_id}")
    
    # Test get active
    active = db.get_active_events('camera_1')
    print(f"✅ Active events: {len(active)}")
    
    # Test end event
    import time
    time.sleep(2)
    db.end_event(event_id)
    print(f"✅ Ended event ID: {event_id}")
    
    # Test statistics
    stats = db.get_database_stats()
    print(f"✅ Database stats: {stats}")
    
    print("\n🎉 Database test completed!")
