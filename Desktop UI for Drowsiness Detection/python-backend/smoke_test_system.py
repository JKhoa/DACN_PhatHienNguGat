import os
import sys
import time
import sqlite3
import logging
from pathlib import Path

# Cấu hình logging để theo dõi quá trình test
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_1_environment():
    logging.info("--- TEST 1: KIỂM TRA MÔI TRƯỜNG ---")
    try:
        import cv2
        import torch
        import ultralytics
        from flask_socketio import SocketIO
        logging.info(f"✅ OpenCV: {cv2.__version__}")
        logging.info(f"✅ Torch: {torch.__version__}")
        logging.info(f"✅ YOLO (Ultralytics): {ultralytics.__version__}")
        return True
    except ImportError as e:
        logging.error(f"❌ Thiếu thư viện: {e}")
        return False

def test_2_yolo_model():
    logging.info("\n--- TEST 2: KIỂM TRA MÔI HÌNH YOLO ---")
    try:
        from ultralytics import YOLO
        # Thử tải mô hình mặc định
        model = YOLO("yolo11n-pose.pt")
        logging.info("✅ Tải mô hình YOLOv11n-pose thành công.")
        return True
    except Exception as e:
        logging.error(f"❌ Lỗi tải mô hình YOLO: {e}")
        return False

def test_3_database():
    logging.info("\n--- TEST 3: KIỂM TRA CƠ SỞ DỮ LIỆU ---")
    db_path = Path(__file__).parent / 'drowsiness_logs' / 'events.db'
    try:
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logging.info(f"✅ Kết nối Database thành công. Các bảng hiện có: {[t[0] for t in tables]}")
        conn.close()
        return True
    except Exception as e:
        logging.error(f"❌ Lỗi Database: {e}")
        return False

def test_4_gemini_ai():
    logging.info("\n--- TEST 4: KIỂM TRA CHATBOT GEMINI ---")
    from google import genai
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key or not api_key.startswith('AIza'):
        logging.error("❌ API Key không hợp lệ hoặc thiếu trong tệp .env")
        return False
        
    client = genai.Client(api_key=api_key)
    try:
        # Thử một câu hỏi đơn giản
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents="Kiểm tra kết nối hệ thống. Trả lời ngắn gọn 'OK' nếu bạn nhận được tin nhắn này."
        )
        logging.info(f"✅ Gemini phản hồi: {response.text.strip()}")
        return True
    except Exception as e:
        logging.error(f"❌ Lỗi kết nối Gemini: {e}")
        return False

if __name__ == "__main__":
    logging.info("🚀 BẮT ĐẦU SMOKE TEST HỆ THỐNG PHÁT HIỆN NGỦ GẬT\n")
    results = []
    results.append(test_1_environment())
    results.append(test_2_yolo_model())
    results.append(test_3_database())
    results.append(test_4_gemini_ai())
    
    logging.info("\n" + "="*40)
    if all(results):
        logging.info("🎉 TẤT CẢ CÁC BÀI TEST ĐỀU VƯỢT QUA! HỆ THỐNG ĐÃ SẴN SÀNG.")
    else:
        logging.error("⚠️ CẢNH BÁO: CÓ MỘT SỐ VẤN ĐỀ CẦN KIỂM TRA LẠI.")
    logging.info("="*40)
