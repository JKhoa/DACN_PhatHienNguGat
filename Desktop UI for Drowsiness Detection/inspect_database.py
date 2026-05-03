"""
SQLite Database Inspector - Kiểm tra và hiển thị dữ liệu trong database
"""

import sys
from pathlib import Path

# Add python-backend to path
sys.path.insert(0, str(Path(__file__).parent / 'python-backend'))

from db_helper import DrowsinessDatabase
from datetime import datetime, timedelta
from tabulate import tabulate


def inspect_database():
    """Kiểm tra và hiển thị nội dung database"""
    
    print("=" * 80)
    print("🔍 SQLITE DATABASE INSPECTOR - Drowsiness Detection System")
    print("=" * 80)
    print()
    
    # Initialize database
    db = DrowsinessDatabase()
    
    # 1. Database Overview
    print("📊 DATABASE OVERVIEW:")
    print("-" * 80)
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print()
    
    # 2. Active Events
    print("🔴 ACTIVE EVENTS (Currently Drowsy):")
    print("-" * 80)
    active_events = db.get_active_events()
    
    if active_events:
        table_data = []
        for event in active_events:
            duration = event.get('current_duration', 0)
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            table_data.append([
                event['id'],
                event['camera_id'],
                f"#{event['student_id']}",
                event['start_time'],
                f"{minutes}m {seconds}s"
            ])
        
        headers = ['ID', 'Camera', 'Student', 'Start Time', 'Duration']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        print("  ✅ No active drowsy events")
    print()
    
    # 3. Recent Events (Last 10)
    print("📋 RECENT EVENTS (Last 10):")
    print("-" * 80)
    recent_events = db.get_events(limit=10, include_active=False)
    
    if recent_events:
        table_data = []
        for event in recent_events:
            duration = event.get('duration_seconds', 0)
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            table_data.append([
                event['id'],
                event['camera_name'] or event['camera_id'],
                f"#{event['student_id']}",
                event['start_time'],
                event['end_time'] or 'Active',
                f"{minutes}m {seconds}s" if not event['is_active'] else 'Ongoing'
            ])
        
        headers = ['ID', 'Camera', 'Student', 'Start', 'End', 'Duration']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        print("  📭 No events in database yet")
    print()
    
    # 4. Today's Statistics
    print("📅 TODAY'S STATISTICS:")
    print("-" * 80)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = datetime.now()
    
    # Get all cameras
    all_events = db.get_events(start_date=today_start, end_date=today_end, limit=1000)
    cameras = set(e['camera_id'] for e in all_events)
    
    if cameras:
        for camera_id in sorted(cameras):
            stats = db.get_statistics(camera_id, today_start, today_end)
            
            print(f"\n  🎥 Camera: {camera_id}")
            print(f"     • Total Events: {stats['total_events']}")
            print(f"     • Unique Students: {stats['total_students']}")
            print(f"     • Currently Drowsy: {stats['currently_drowsy']}")
            
            total_duration = stats['total_duration']
            if total_duration > 0:
                hours = int(total_duration // 3600)
                minutes = int((total_duration % 3600) // 60)
                seconds = int(total_duration % 60)
                print(f"     • Total Duration: {hours}h {minutes}m {seconds}s")
                
                avg_duration = stats.get('avg_duration', 0)
                avg_min = int(avg_duration // 60)
                avg_sec = int(avg_duration % 60)
                print(f"     • Average Duration: {avg_min}m {avg_sec}s")
    else:
        print("  📭 No events today")
    print()
    
    # 5. Database File Info
    print("💾 DATABASE FILE:")
    print("-" * 80)
    print(f"  • Location: {stats.get('database_path', 'N/A')}")
    print(f"  • Size: {stats.get('database_size_mb', 0)} MB")
    print()
    
    print("=" * 80)
    print("✅ Database inspection completed!")
    print("=" * 80)


def main():
    """Main function"""
    try:
        inspect_database()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
