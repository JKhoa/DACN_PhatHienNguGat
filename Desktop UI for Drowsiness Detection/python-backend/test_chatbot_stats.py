import sys
from pathlib import Path

# Add the directory to sys.path
sys.path.append(str(Path(__file__).parent))

from chatbot import handle_question
import json

questions = [
    "tổng quan hôm nay",
    "phòng rủi ro cao nhất",
    "top học sinh ngủ gật",
    "khung giờ dễ ngủ gật nhất"
]

for q in questions:
    print(f"\nQuestion: {q}")
    result = handle_question(q)
    print(f"Intent: {result['intent']}")
    print(f"Summary: {result['summary_text']}")
    if result['rows']:
        print(f"Data rows: {len(result['rows'])}")
