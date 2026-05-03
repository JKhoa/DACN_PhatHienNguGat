#!/usr/bin/env python3
"""
Minimal test server để kiểm tra API logging hoạt động
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import time

# Import logging system
try:
    from drowsiness_logger import MultiCameraLogger, get_global_logger, init_logger
    LOGGER_AVAILABLE = True
    print("✅ Drowsiness logger imported successfully")
except ImportError as e:
    LOGGER_AVAILABLE = False
    print(f"❌ Drowsiness logger import failed: {e}")

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return jsonify({'message': 'Drowsiness Logging Test Server', 'timestamp': time.time()})

@app.route('/api/test')
def test_api():
    return jsonify({'success': True, 'message': 'API working!'})

@app.route('/api/drowsiness/active')
def get_active_drowsy_students():
    """Lấy danh sách học sinh đang ngủ gật (real-time)"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Drowsiness logger not available'
            }), 500
        
        logger = get_global_logger()
        active_students = logger.get_active_drowsy_all_cameras()
        
        return jsonify({
            'success': True,
            'active_drowsy_students': active_students
        })
    except Exception as e:
        print(f"Error getting active drowsy students: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/drowsiness/summary')
def get_drowsiness_summary():
    """Lấy thống kê tổng hợp tất cả camera"""
    try:
        period = request.args.get('period', 'today')
        
        if not LOGGER_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Drowsiness logger not available'
            }), 500
        
        logger = get_global_logger()
        summary = logger.get_summary_stats(period)
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        print(f"Error getting drowsiness summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/drowsiness/test-log')
def test_logging():
    """Test logging system"""
    try:
        if not LOGGER_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Drowsiness logger not available'
            }), 500
        
        logger = get_global_logger()
        
        # Register a test camera
        logger.register_camera("test_camera", "Test Camera - Phòng Test")
        
        # Simulate some events
        logger.update_student_state("test_camera", 1, True)  # Student 1 starts drowsy
        time.sleep(1)
        logger.update_student_state("test_camera", 2, True)  # Student 2 starts drowsy  
        time.sleep(1)
        logger.update_student_state("test_camera", 1, False) # Student 1 wakes up
        
        # Get active students
        active = logger.get_active_drowsy_all_cameras()
        summary = logger.get_summary_stats('today')
        
        return jsonify({
            'success': True,
            'message': 'Test completed',
            'active_students': active,
            'summary': summary
        })
    except Exception as e:
        print(f"Error in test logging: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Drowsiness Logging Test Server...")
    
    # Initialize logger
    if LOGGER_AVAILABLE:
        try:
            init_logger('test_logs')
            print("✅ Logger initialized")
        except Exception as e:
            print(f"❌ Logger initialization failed: {e}")
    
    print("🔗 Server URL: http://127.0.0.1:5001")
    print("📝 Test endpoints:")
    print("   - GET  / (basic test)")
    print("   - GET  /api/test (API test)")
    print("   - GET  /api/drowsiness/active (active students)")
    print("   - GET  /api/drowsiness/summary (summary stats)")
    print("   - GET  /api/drowsiness/test-log (create test data)")
    
    app.run(host='127.0.0.1', port=5001, debug=True)