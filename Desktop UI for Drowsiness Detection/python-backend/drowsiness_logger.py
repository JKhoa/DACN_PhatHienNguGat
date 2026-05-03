"""
Drowsiness Logging System - Multi-Camera/Multi-Room
Ghi log chi tiết học sinh ngủ gật theo từng camera (phòng học)
Thống kê theo ngày, tuần, tháng
Database: SQLite3 for persistent storage
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading
from pathlib import Path

# Import SQLite database helper
try:
    from db_helper import DrowsinessDatabase, get_database
    USE_SQLITE = True
except ImportError:
    print("⚠️ db_helper not found, using in-memory storage only")
    USE_SQLITE = False


class DrowsinessEvent:
    """Một sự kiện ngủ gật của học sinh"""
    
    def __init__(self, camera_id: str, student_id: int, start_time: datetime):
        self.camera_id = camera_id
        self.student_id = student_id
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.duration_seconds: float = 0.0
        self.is_active = True
        
    def end_event(self):
        """Kết thúc sự kiện ngủ gật"""
        if self.is_active:
            self.end_time = datetime.now()
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
            self.is_active = False
            
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'camera_id': self.camera_id,
            'student_id': self.student_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': round(self.duration_seconds, 2),
            'is_active': self.is_active
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'DrowsinessEvent':
        """Create from dictionary"""
        event = DrowsinessEvent(
            camera_id=data['camera_id'],
            student_id=data['student_id'],
            start_time=datetime.fromisoformat(data['start_time'])
        )
        if data.get('end_time'):
            event.end_time = datetime.fromisoformat(data['end_time'])
        event.duration_seconds = data.get('duration_seconds', 0.0)
        event.is_active = data.get('is_active', False)
        return event


class CameraLogger:
    """Logger cho một camera/phòng học cụ thể - with SQLite backend"""
    
    def __init__(self, camera_id: str, camera_name: Optional[str] = None, use_db: bool = USE_SQLITE):
        self.camera_id = camera_id
        self.camera_name = camera_name or f"Phòng {camera_id}"
        self.use_db = use_db
        
        # SQLite database connection
        if self.use_db:
            self.db = get_database()
            print(f"✅ CameraLogger '{self.camera_name}' using SQLite database")
        else:
            self.db = None
            print(f"⚠️ CameraLogger '{self.camera_name}' using in-memory storage")
        
        # In-memory cache for active events (fast lookup by student_id -> event_id)
        self.active_event_ids: Dict[int, int] = {}
        
        # Fallback: In-memory storage if database disabled
        self.active_events: Dict[int, DrowsinessEvent] = {}
        self.all_events: List[DrowsinessEvent] = []
        
        # Statistics cache
        self._stats_cache = {}
        self._cache_timestamp = None
        
    def start_drowsiness(self, student_id: int):
        """Bắt đầu ghi nhận học sinh ngủ gật"""
        if self.use_db and self.db:
            # Check if already active in database
            existing = self.db.get_active_event(self.camera_id, student_id)
            if existing:
                # Already has active event
                return
            
            # Insert new event to database
            event_id = self.db.insert_event(
                camera_id=self.camera_id,
                student_id=student_id,
                camera_name=self.camera_name,
                event_type='drowsy'
            )
            self.active_event_ids[student_id] = event_id
            print(f"[{self.camera_name}] 🔴 Học sinh #{student_id} BẮT ĐẦU ngủ gật (DB ID: {event_id})")
            
        else:
            # Fallback to in-memory
            if student_id not in self.active_events:
                event = DrowsinessEvent(self.camera_id, student_id, datetime.now())
                self.active_events[student_id] = event
                print(f"[{self.camera_name}] Học sinh #{student_id} BẮT ĐẦU ngủ gật lúc {event.start_time.strftime('%H:%M:%S')}")
            
    def end_drowsiness(self, student_id: int):
        """Kết thúc ghi nhận học sinh tỉnh lại"""
        if self.use_db and self.db:
            # End event in database
            if student_id in self.active_event_ids:
                event_id = self.active_event_ids.pop(student_id)
                success = self.db.end_event(event_id)
                
                if success:
                    # Get ended event to show duration
                    events = self.db.get_events(
                        camera_id=self.camera_id,
                        limit=1,
                        include_active=False
                    )
                    if events:
                        duration = events[0]['duration_seconds']
                        minutes = int(duration // 60)
                        seconds = int(duration % 60)
                        print(f"[{self.camera_name}] 🟢 Học sinh #{student_id} TỈNH LẠI (Ngủ gật: {minutes}m {seconds}s)")
                
                # Clear stats cache
                self._cache_timestamp = None
        else:
            # Fallback to in-memory
            if student_id in self.active_events:
                event = self.active_events.pop(student_id)
                event.end_event()
                self.all_events.append(event)
                
                minutes = int(event.duration_seconds // 60)
                seconds = int(event.duration_seconds % 60)
                print(f"[{self.camera_name}] Học sinh #{student_id} TỈNH LẠI lúc {event.end_time.strftime('%H:%M:%S')} "
                      f"(Ngủ gật: {minutes}m {seconds}s)")
                
                # Clear stats cache
                self._cache_timestamp = None
            
    def update_student_state(self, student_id: int, is_drowsy: bool):
        """Cập nhật trạng thái học sinh (gọi mỗi frame)"""
        if is_drowsy:
            self.start_drowsiness(student_id)
        else:
            self.end_drowsiness(student_id)
            
    def get_active_drowsy_students(self) -> List[Dict]:
        """Lấy danh sách học sinh đang ngủ gật"""
        if self.use_db and self.db:
            # Get from database
            active_events = self.db.get_active_events(self.camera_id)
            result = []
            
            for event in active_events:
                duration = event.get('current_duration', 0)
                result.append({
                    'student_id': event['student_id'],
                    'start_time': event['start_time'],
                    'duration_seconds': round(duration, 2),
                    'duration_display': self._format_duration(duration)
                })
            
            return result
        else:
            # Fallback to in-memory
            result = []
            now = datetime.now()
            
            for student_id, event in self.active_events.items():
                duration = (now - event.start_time).total_seconds()
                result.append({
                    'student_id': student_id,
                    'start_time': event.start_time.isoformat(),
                    'duration_seconds': round(duration, 2),
                    'duration_display': self._format_duration(duration)
                })
            
            return result
    
    def get_events_in_range(self, start_date: datetime, end_date: datetime) -> List[DrowsinessEvent]:
        """Lấy các sự kiện trong khoảng thời gian"""
        events = []
        
        # Include active events
        for event in self.active_events.values():
            if start_date <= event.start_time <= end_date:
                events.append(event)
        
        # Include completed events
        for event in self.all_events:
            if start_date <= event.start_time <= end_date:
                events.append(event)
        
        return events
    
    def get_statistics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Thống kê chi tiết trong khoảng thời gian"""
        
        # Check cache (5 seconds validity)
        cache_key = f"{start_date.isoformat()}_{end_date.isoformat()}"
        if (self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).total_seconds() < 5 and
            cache_key in self._stats_cache):
            return self._stats_cache[cache_key]
        
        if self.use_db and self.db:
            # Get statistics from database
            db_stats = self.db.get_statistics(self.camera_id, start_date, end_date)
            
            stats = {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_drowsy_students': db_stats.get('total_students', 0),
                'currently_drowsy': db_stats.get('currently_drowsy', 0),
                'total_events': db_stats.get('total_events', 0),
                'total_duration_seconds': round(db_stats.get('total_duration', 0), 2),
                'total_duration_display': self._format_duration(db_stats.get('total_duration', 0)),
                'avg_duration_seconds': round(db_stats.get('avg_duration', 0), 2),
                'max_duration_seconds': round(db_stats.get('max_duration', 0), 2),
            }
        else:
            # Fallback to in-memory calculation
            events = self.get_events_in_range(start_date, end_date)
            
            # Count unique students
            unique_students = set(e.student_id for e in events)
            
            # Total drowsiness time
            total_duration = sum(e.duration_seconds for e in events if not e.is_active)
            
            # Currently drowsy students
            active_count = len(self.active_events)
            
            stats = {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_drowsy_students': len(unique_students),
                'currently_drowsy': active_count,
                'total_events': len(events),
                'total_duration_seconds': round(total_duration, 2),
                'total_duration_display': self._format_duration(total_duration),
            }
        
        # Cache result
        self._stats_cache[cache_key] = stats
        self._cache_timestamp = datetime.now()
        
        return stats
    
    def get_detailed_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Lấy log chi tiết các sự kiện ngủ gật"""
        events = self.get_events_in_range(start_date, end_date)
        
        # Sort by start time (newest first)
        events.sort(key=lambda e: e.start_time, reverse=True)
        
        result = []
        for event in events:
            result.append({
                'student_id': event.student_id,
                'start_time': event.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': event.end_time.strftime('%Y-%m-%d %H:%M:%S') if event.end_time else 'Đang ngủ',
                'duration_seconds': round(event.duration_seconds, 2),
                'duration_display': self._format_duration(event.duration_seconds) if not event.is_active else 'Đang ngủ',
                'is_active': event.is_active
            })
        
        return result
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration to human readable string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class MultiCameraLogger:
    """Quản lý logging cho nhiều camera/phòng học"""
    
    def __init__(self, log_dir: str = "drowsiness_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Camera loggers
        self.cameras: Dict[str, CameraLogger] = {}
        
        # Lock for thread safety — RLock because some public methods call
        # each other while holding the lock (e.g. get_all_cameras_stats → get_camera_stats).
        self.lock = threading.RLock()
        
        # Auto-save interval (seconds)
        self.autosave_interval = 300  # 5 minutes
        self._last_save_time = datetime.now()
        
    def register_camera(self, camera_id: str, camera_name: Optional[str] = None):
        """Đăng ký một camera/phòng học mới"""
        with self.lock:
            if camera_id not in self.cameras:
                self.cameras[camera_id] = CameraLogger(camera_id, camera_name)
                print(f"✅ Đăng ký camera: {camera_name or camera_id}")
                
    def update_student_state(self, camera_id: str, student_id: int, is_drowsy: bool):
        """Cập nhật trạng thái học sinh cho camera cụ thể"""
        with self.lock:
            if camera_id not in self.cameras:
                self.register_camera(camera_id)
            
            self.cameras[camera_id].update_student_state(student_id, is_drowsy)
            
            # Auto-save check
            if (datetime.now() - self._last_save_time).total_seconds() > self.autosave_interval:
                self._autosave()
                
    def get_camera_stats(self, camera_id: str, period: str = 'today') -> Dict:
        """Lấy thống kê cho một camera theo khoảng thời gian
        
        Args:
            camera_id: ID của camera
            period: 'today', 'week', 'month', hoặc custom 'YYYY-MM-DD_YYYY-MM-DD'
        """
        with self.lock:
            if camera_id not in self.cameras:
                return {'error': f'Camera {camera_id} not found'}
            
            start_date, end_date = self._parse_period(period)
            return self.cameras[camera_id].get_statistics(start_date, end_date)
    
    def get_all_cameras_stats(self, period: str = 'today') -> List[Dict]:
        """Lấy thống kê tất cả các camera"""
        with self.lock:
            results = []
            for camera_id in self.cameras:
                stats = self.get_camera_stats(camera_id, period)
                results.append(stats)
            return results
    
    def get_camera_events(self, camera_id: str, period: str = 'today') -> List[Dict]:
        """Lấy log chi tiết các sự kiện ngủ gật của camera"""
        with self.lock:
            if camera_id not in self.cameras:
                return []
            
            start_date, end_date = self._parse_period(period)
            return self.cameras[camera_id].get_detailed_events(start_date, end_date)
    
    def get_active_drowsy_all_cameras(self) -> Dict[str, List[Dict]]:
        """Lấy danh sách học sinh đang ngủ gật tất cả camera"""
        with self.lock:
            result = {}
            for camera_id, logger in self.cameras.items():
                active = logger.get_active_drowsy_students()
                if active:  # Only include cameras with drowsy students
                    result[camera_id] = active
            return result
    
    def get_summary_stats(self, period: str = 'today') -> Dict:
        """Thống kê tổng hợp tất cả các phòng, truy vấn trực tiếp từ DB"""
        from db_helper import get_database
        db = get_database()
        start_date, end_date = self._parse_period(period)
        
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Query total stats from DB
        conn = db._get_connection()
        cursor = conn.cursor()
        
        # 1. Tổng số sự kiện, số học sinh duy nhất, tổng thời gian
        cursor.execute('''
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT student_id) as total_students,
                SUM(duration_seconds) as total_duration,
                COUNT(DISTINCT camera_id) as total_cameras
            FROM drowsy_events
            WHERE start_time BETWEEN ? AND ?
        ''', (start_str, end_str))
        
        row = cursor.fetchone()
        total_events = row[0] or 0
        total_students = row[1] or 0
        total_duration = row[2] or 0
        total_cameras_db = row[3] or 0
        
        # 2. Lấy thống kê từng camera
        cursor.execute('''
            SELECT 
                camera_id, 
                camera_name,
                COUNT(*) as events,
                COUNT(DISTINCT student_id) as students,
                SUM(duration_seconds) as duration
            FROM drowsy_events
            WHERE start_time BETWEEN ? AND ?
            GROUP BY camera_id
        ''', (start_str, end_str))
        
        camera_stats = []
        for r in cursor.fetchall():
            camera_stats.append({
                'camera_id': r[0],
                'camera_name': r[1],
                'total_events': r[2],
                'total_drowsy_students': r[3],
                'total_duration_seconds': round(r[4] or 0, 2),
                'total_duration_display': CameraLogger._format_duration(r[4] or 0),
                'currently_drowsy': 0 # DB query only gives historical
            })

        # Add currently drowsy from memory if available
        currently_drowsy = 0
        with self.lock:
            for cam_id, logger in self.cameras.items():
                active_count = len(logger.active_events)
                currently_drowsy += active_count
                # Update memory stats in the list if cam exists
                for cs in camera_stats:
                    if cs['camera_id'] == cam_id:
                        cs['currently_drowsy'] = active_count

        return {
            'period': period,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_cameras': max(len(self.cameras), total_cameras_db),
            'total_drowsy_students_unique': total_students,
            'total_events': total_events,
            'total_duration_seconds': round(total_duration, 2),
            'total_duration_display': CameraLogger._format_duration(total_duration),
            'currently_drowsy_all_cameras': currently_drowsy,
            'cameras': camera_stats
        }
    
    @staticmethod
    def _parse_period(period: str) -> Tuple[datetime, datetime]:
        """Parse period string to date range"""
        now = datetime.now()
        
        if period == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            
        elif period == 'week':
            # Start of week (Monday)
            start_date = now - timedelta(days=now.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            
        elif period == 'month':
            # Start of month
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            
        elif '_' in period:
            # Custom range: 'YYYY-MM-DD_YYYY-MM-DD'
            try:
                start_str, end_str = period.split('_')
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            except Exception as e:
                print(f"Invalid period format: {period}, using today. Error: {e}")
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            # Default to today
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return start_date, end_date
    
    def _autosave(self):
        """Tự động lưu logs (chạy trong thread-safe context)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.log_dir / f"autosave_{timestamp}.json"
            self.save_to_file(str(filename))
            self._last_save_time = datetime.now()
            print(f"💾 Auto-saved logs to {filename}")
        except Exception as e:
            print(f"❌ Auto-save failed: {e}")
    
    def save_to_file(self, filepath: Optional[str] = None):
        """Lưu tất cả logs ra file JSON"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.log_dir / f"drowsiness_log_{timestamp}.json"
        
        with self.lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'cameras': {}
            }
            
            for camera_id, logger in self.cameras.items():
                camera_data = {
                    'camera_id': camera_id,
                    'camera_name': logger.camera_name,
                    'active_events': [e.to_dict() for e in logger.active_events.values()],
                    'all_events': [e.to_dict() for e in logger.all_events]
                }
                data['cameras'][camera_id] = camera_data
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved logs to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load logs từ file JSON"""
        with self.lock:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for camera_id, camera_data in data.get('cameras', {}).items():
                logger = CameraLogger(camera_id, camera_data.get('camera_name'))
                
                # Load active events
                for event_dict in camera_data.get('active_events', []):
                    event = DrowsinessEvent.from_dict(event_dict)
                    logger.active_events[event.student_id] = event
                
                # Load completed events
                for event_dict in camera_data.get('all_events', []):
                    event = DrowsinessEvent.from_dict(event_dict)
                    logger.all_events.append(event)
                
                self.cameras[camera_id] = logger
            
            print(f"✅ Loaded logs from {filepath}")


# Global instance (thread-safe double-checked locking)
_global_logger: Optional[MultiCameraLogger] = None
_global_logger_lock = threading.Lock()


def get_global_logger() -> MultiCameraLogger:
    """Get or create global logger instance (thread-safe)."""
    global _global_logger
    if _global_logger is None:                 # fast path — no lock
        with _global_logger_lock:
            if _global_logger is None:         # safe path — under lock
                _global_logger = MultiCameraLogger()
    return _global_logger


def init_logger(log_dir: str = "drowsiness_logs") -> MultiCameraLogger:
    """Initialize global logger with custom directory"""
    global _global_logger
    _global_logger = MultiCameraLogger(log_dir)
    return _global_logger


# Example usage
if __name__ == "__main__":
    # Test the logging system
    logger = MultiCameraLogger()
    
    # Register cameras
    logger.register_camera("camera_1", "Phòng 101 - Toán")
    logger.register_camera("camera_2", "Phòng 102 - Văn")
    
    # Simulate some drowsiness events
    print("\n=== Simulating drowsiness events ===")
    
    # Camera 1: Student 5 starts drowsy
    logger.update_student_state("camera_1", 5, True)
    time.sleep(2)
    
    # Camera 2: Student 10 starts drowsy
    logger.update_student_state("camera_2", 10, True)
    time.sleep(1)
    
    # Camera 1: Student 5 wakes up
    logger.update_student_state("camera_1", 5, False)
    
    # Camera 1: Student 8 starts drowsy
    logger.update_student_state("camera_1", 8, True)
    time.sleep(1)
    
    # Get statistics
    print("\n=== Today's Statistics ===")
    summary = logger.get_summary_stats('today')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n=== Camera 1 Events ===")
    events = logger.get_camera_events("camera_1", 'today')
    print(json.dumps(events, indent=2, ensure_ascii=False))
    
    print("\n=== Active Drowsy Students ===")
    active = logger.get_active_drowsy_all_cameras()
    print(json.dumps(active, indent=2, ensure_ascii=False))
    
    # Save to file
    logger.save_to_file("test_log.json")
