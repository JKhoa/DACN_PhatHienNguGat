import os
from google import genai
from dotenv import load_dotenv
from pathlib import Path

# Load API key from .env
load_dotenv(Path(__file__).parent / '.env')
api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    print("❌ Lỗi: Không tìm thấy API Key trong tệp .env")
    exit(1)

client = genai.Client(api_key=api_key)

try:
    # Test connection by listing models
    models = client.models.list()
    print("✅ Kết nối thành công! API Key của bạn đang hoạt động bình thường.")
    print("🤖 Chatbot hiện đã có thể sử dụng mô hình 'gemini-2.0-flash'.")
except Exception as e:
    print(f"❌ Lỗi kết nối: {e}")
    print("Có thể khóa này bị giới hạn quyền truy cập hoặc sai định dạng.")
