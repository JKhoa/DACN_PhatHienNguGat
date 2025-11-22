"""
Script to seed sample drowsiness events into the database for testing Dashboard and Charts
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db_helper import get_database

def seed_sample_data():
    """Insert sample drowsiness events for testing"""
    
    db = get_database()
    
    print("🌱 Seeding sample data into database...")
    print("=" * 80)
    
    # Camera IDs
    cameras = [
        ('camera_1', 'Camera Phòng A101'),
        ('camera_2', 'Camera Phòng B202'),
        ('camera_3', 'Camera Phòng C303')
    ]
    
    # Student IDs
    student_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Time ranges
    now = datetime.now()
    
    # Generate events for the past 7 days
    total_events = 0
    
    for days_ago in range(7):
        date = now - timedelta(days=days_ago)
        
        # Random number of events per day (5-15)
        num_events = random.randint(5, 15)
        
        for i in range(num_events):
            # Random camera
            camera_id, camera_name = random.choice(cameras)
            
            # Random student
            student_id = random.choice(student_ids)
            
            # Random start time during the day (8 AM - 5 PM)
            hour = random.randint(8, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            start_time = date.replace(hour=hour, minute=minute, second=second)
            
            # Random duration (5 seconds to 5 minutes)
            duration_seconds = random.uniform(5, 300)
            end_time = start_time + timedelta(seconds=duration_seconds)
            
            # Insert event
            event_id = db.insert_event(
                camera_id=camera_id,
                student_id=student_id,
                camera_name=camera_name,
                event_type='drowsy'
            )
            
            # End the event immediately with calculated duration
            cursor = db._get_connection().cursor()
            cursor.execute('''
                UPDATE drowsy_events 
                SET end_time = ?, 
                    duration_seconds = ?,
                    is_active = 0
                WHERE id = ?
            ''', (end_time.isoformat(), duration_seconds, event_id))
            db._get_connection().commit()
            
            total_events += 1
            
            if total_events % 10 == 0:
                print(f"✅ Inserted {total_events} events...")
    
    print("=" * 80)
    print(f"🎉 Successfully seeded {total_events} sample events!")
    
    # Show statistics
    print("\n📊 Database Statistics:")
    stats = db.get_database_stats()
    print(f"  • Total Events: {stats['total_events']}")
    print(f"  • Active Events: {stats['active_events']}")
    print(f"  • Unique Students: {stats['unique_students']}")
    print(f"  • Database Size: {stats['database_size_mb']:.2f} MB")
    
    # Show recent events
    print("\n📋 Recent 5 Events:")
    recent = db.get_events(limit=5)
    for event in recent:
        duration_min = int(event['duration_seconds'] // 60)
        duration_sec = int(event['duration_seconds'] % 60)
        print(f"  • [{event['camera_name']}] Student #{event['student_id']}: {duration_min}m {duration_sec}s")
    
    print("\n✅ Sample data ready! Refresh Dashboard/Charts to see data.")

if __name__ == '__main__':
    seed_sample_data()
