"""
Test độ chính xác phát hiện ngủ gật với các tư thế khác nhau
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

class DrowsinessAccuracyTester:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """Khởi tạo YOLO model và các ngưỡng"""
        print("=" * 80)
        print("🧪 DROWSINESS DETECTION ACCURACY TEST SUITE")
        print("=" * 80)
        
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Ngưỡng phát hiện ngủ gật (từ logic thực tế)
        self.EAR_THRESHOLD = 0.25  # Eye Aspect Ratio
        self.HEAD_TILT_THRESHOLD = 20  # degrees
        self.MOUTH_OPEN_THRESHOLD = 0.6
        
        # Các tư thế test
        self.test_cases = {
            'normal': {
                'description': '👤 Ngồi thẳng, mắt mở, tập trung',
                'expected': 'NOT_DROWSY',
                'ear_range': (0.25, 0.35),
                'head_tilt_range': (-10, 10),
                'mouth_ratio_range': (0.0, 0.3)
            },
            'eyes_closed': {
                'description': '😴 Mắt nhắm (ngủ gật rõ ràng)',
                'expected': 'DROWSY',
                'ear_range': (0.0, 0.20),
                'head_tilt_range': (-10, 10),
                'mouth_ratio_range': (0.0, 0.3)
            },
            'head_down': {
                'description': '🙇 Đầu cúi xuống (mệt mỏi)',
                'expected': 'DROWSY',
                'ear_range': (0.15, 0.30),
                'head_tilt_range': (20, 45),
                'mouth_ratio_range': (0.0, 0.4)
            },
            'head_tilted': {
                'description': '😪 Đầu nghiêng sang bên',
                'expected': 'DROWSY',
                'ear_range': (0.15, 0.25),
                'head_tilt_range': (15, 40),
                'mouth_ratio_range': (0.0, 0.3)
            },
            'mouth_open': {
                'description': '🥱 Há miệng (ngáp/buồn ngủ)',
                'expected': 'DROWSY',
                'ear_range': (0.20, 0.30),
                'head_tilt_range': (-10, 15),
                'mouth_ratio_range': (0.6, 1.2)
            },
            'half_closed_eyes': {
                'description': '😑 Mắt mở một nửa (buồn ngủ)',
                'expected': 'DROWSY',
                'ear_range': (0.15, 0.24),
                'head_tilt_range': (-5, 10),
                'mouth_ratio_range': (0.0, 0.4)
            },
            'head_back': {
                'description': '😵 Đầu ngả ra sau',
                'expected': 'DROWSY',
                'ear_range': (0.10, 0.25),
                'head_tilt_range': (-45, -20),
                'mouth_ratio_range': (0.4, 0.8)
            },
            'looking_down': {
                'description': '📱 Nhìn xuống (có thể ngủ hoặc xem điện thoại)',
                'expected': 'DROWSY',
                'ear_range': (0.20, 0.30),
                'head_tilt_range': (15, 35),
                'mouth_ratio_range': (0.0, 0.3)
            },
            'reading': {
                'description': '📖 Đọc sách (tư thế bình thường)',
                'expected': 'NOT_DROWSY',
                'ear_range': (0.25, 0.35),
                'head_tilt_range': (5, 20),
                'mouth_ratio_range': (0.0, 0.3)
            },
            'extreme_fatigue': {
                'description': '😩 Cực kỳ mệt mỏi (nhiều dấu hiệu)',
                'expected': 'DROWSY',
                'ear_range': (0.05, 0.18),
                'head_tilt_range': (25, 50),
                'mouth_ratio_range': (0.5, 1.0)
            }
        }
    
    def calculate_ear(self, eye_landmarks):
        """Tính Eye Aspect Ratio"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Khoảng cách dọc
        vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Khoảng cách ngang
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def calculate_head_tilt(self, nose, left_eye, right_eye):
        """Tính góc nghiêng đầu"""
        eye_center = (left_eye + right_eye) / 2
        dx = nose[0] - eye_center[0]
        dy = nose[1] - eye_center[1]
        
        angle = np.degrees(np.arctan2(dy, dx)) - 90
        return abs(angle)
    
    def calculate_mouth_ratio(self, mouth_landmarks):
        """Tính tỷ lệ há miệng"""
        if len(mouth_landmarks) < 4:
            return 0.0
        
        vertical = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[3])
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[2])
        
        if horizontal == 0:
            return 0.0
        
        return vertical / horizontal
    
    def detect_drowsiness(self, keypoints):
        """
        Phát hiện ngủ gật dựa trên keypoints
        Returns: (is_drowsy, confidence, details)
        """
        if keypoints is None or len(keypoints) < 17:
            return False, 0.0, "No person detected"
        
        # Lấy keypoints quan trọng
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        left_ear_point = keypoints[3]
        right_ear_point = keypoints[4]
        
        # Tính các chỉ số
        # Mắt trái (landmarks 1, 2 và các điểm xung quanh)
        left_eye_landmarks = np.array([
            keypoints[1], keypoints[2], keypoints[1], 
            keypoints[1], keypoints[2], keypoints[2]
        ])
        left_ear = self.calculate_ear(left_eye_landmarks)
        
        # Mắt phải
        right_eye_landmarks = np.array([
            keypoints[2], keypoints[1], keypoints[2],
            keypoints[2], keypoints[1], keypoints[1]
        ])
        right_ear = self.calculate_ear(right_eye_landmarks)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Góc nghiêng đầu
        head_tilt = self.calculate_head_tilt(nose, left_eye, right_eye)
        
        # Miệng (giả định từ vị trí mũi và mắt)
        mouth_landmarks = np.array([
            keypoints[0], keypoints[1], keypoints[0], keypoints[2]
        ])
        mouth_ratio = self.calculate_mouth_ratio(mouth_landmarks)
        
        # Phát hiện ngủ gật
        drowsy_signals = []
        confidence = 0.0
        
        if avg_ear < self.EAR_THRESHOLD:
            drowsy_signals.append(f"Eyes closing (EAR={avg_ear:.3f})")
            confidence += 0.4
        
        if head_tilt > self.HEAD_TILT_THRESHOLD:
            drowsy_signals.append(f"Head tilted ({head_tilt:.1f}°)")
            confidence += 0.3
        
        if mouth_ratio > self.MOUTH_OPEN_THRESHOLD:
            drowsy_signals.append(f"Mouth open (ratio={mouth_ratio:.3f})")
            confidence += 0.3
        
        is_drowsy = len(drowsy_signals) >= 1
        
        details = {
            'ear': avg_ear,
            'head_tilt': head_tilt,
            'mouth_ratio': mouth_ratio,
            'signals': drowsy_signals
        }
        
        return is_drowsy, confidence, details
    
    def generate_synthetic_keypoints(self, test_case_params):
        """Tạo keypoints giả lập theo tham số test case"""
        # Tạo 17 keypoints YOLO pose
        keypoints = np.zeros((17, 2), dtype=np.float32)
        
        # Vị trí cơ bản (giả định người ở giữa frame 640x480)
        center_x, center_y = 320, 240
        
        # 0: Nose
        keypoints[0] = [center_x, center_y]
        
        # 1-2: Eyes (điều chỉnh theo EAR)
        ear_target = np.mean(test_case_params['ear_range'])
        eye_offset_y = int(30 * (1 - ear_target / 0.3))  # Mắt nhắm -> offset nhỏ
        
        keypoints[1] = [center_x - 30, center_y - 40 + eye_offset_y]  # Left eye
        keypoints[2] = [center_x + 30, center_y - 40 + eye_offset_y]  # Right eye
        
        # 3-4: Ears
        keypoints[3] = [center_x - 60, center_y - 30]  # Left ear
        keypoints[4] = [center_x + 60, center_y - 30]  # Right ear
        
        # 5-6: Shoulders (điều chỉnh theo head tilt)
        head_tilt_target = np.mean(test_case_params['head_tilt_range'])
        tilt_offset = int(head_tilt_target * 2)
        
        keypoints[5] = [center_x - 80, center_y + 60 + tilt_offset]  # Left shoulder
        keypoints[6] = [center_x + 80, center_y + 60 - tilt_offset]  # Right shoulder
        
        # 7-10: Elbows and wrists
        keypoints[7] = [center_x - 100, center_y + 120]  # Left elbow
        keypoints[8] = [center_x + 100, center_y + 120]  # Right elbow
        keypoints[9] = [center_x - 110, center_y + 180]  # Left wrist
        keypoints[10] = [center_x + 110, center_y + 180]  # Right wrist
        
        # 11-16: Hips, knees, ankles
        keypoints[11] = [center_x - 50, center_y + 160]  # Left hip
        keypoints[12] = [center_x + 50, center_y + 160]  # Right hip
        keypoints[13] = [center_x - 55, center_y + 260]  # Left knee
        keypoints[14] = [center_x + 55, center_y + 260]  # Right knee
        keypoints[15] = [center_x - 50, center_y + 360]  # Left ankle
        keypoints[16] = [center_x + 50, center_y + 360]  # Right ankle
        
        return keypoints
    
    def run_test_suite(self):
        """Chạy toàn bộ test cases"""
        print("\n" + "=" * 80)
        print("🔬 STARTING ACCURACY TEST SUITE")
        print("=" * 80)
        
        results = {
            'total': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for test_name, test_params in self.test_cases.items():
            print(f"\n{'─' * 80}")
            print(f"📋 Test Case: {test_name.upper()}")
            print(f"📝 Description: {test_params['description']}")
            print(f"🎯 Expected: {test_params['expected']}")
            
            # Tạo keypoints giả lập
            keypoints = self.generate_synthetic_keypoints(test_params)
            
            # Chạy detection
            is_drowsy, confidence, details = self.detect_drowsiness(keypoints)
            
            # So sánh kết quả
            actual = 'DROWSY' if is_drowsy else 'NOT_DROWSY'
            passed = (actual == test_params['expected'])
            
            print(f"\n📊 Detection Results:")
            print(f"   • Eye Aspect Ratio: {details['ear']:.3f} (threshold: {self.EAR_THRESHOLD})")
            print(f"   • Head Tilt Angle: {details['head_tilt']:.1f}° (threshold: {self.HEAD_TILT_THRESHOLD}°)")
            print(f"   • Mouth Open Ratio: {details['mouth_ratio']:.3f} (threshold: {self.MOUTH_OPEN_THRESHOLD})")
            
            if details['signals']:
                print(f"\n⚠️  Drowsy Signals Detected:")
                for signal in details['signals']:
                    print(f"   • {signal}")
            else:
                print(f"\n✅ No drowsy signals detected")
            
            print(f"\n🔍 Verdict:")
            print(f"   • Actual: {actual}")
            print(f"   • Expected: {test_params['expected']}")
            print(f"   • Confidence: {confidence:.2%}")
            
            if passed:
                print(f"   • Result: ✅ PASS")
                results['passed'] += 1
            else:
                print(f"   • Result: ❌ FAIL")
                results['failed'] += 1
            
            results['details'].append({
                'test': test_name,
                'expected': test_params['expected'],
                'actual': actual,
                'passed': passed,
                'confidence': float(confidence),
                'ear': float(details['ear']),
                'head_tilt': float(details['head_tilt']),
                'mouth_ratio': float(details['mouth_ratio'])
            })
        
        # Tổng kết
        self.print_summary(results)
        return results
    
    def print_summary(self, results):
        """In tổng kết kết quả"""
        print("\n" + "=" * 80)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 80)
        
        accuracy = (results['passed'] / results['total']) * 100 if results['total'] > 0 else 0
        
        print(f"\n📈 Overall Statistics:")
        print(f"   • Total Tests: {results['total']}")
        print(f"   • Passed: {results['passed']} ✅")
        print(f"   • Failed: {results['failed']} ❌")
        print(f"   • Accuracy: {accuracy:.1f}%")
        
        # Bảng chi tiết
        print(f"\n📋 Detailed Results:")
        print(f"{'Test Case':<25} {'Expected':<15} {'Actual':<15} {'Confidence':<12} {'Status':<8}")
        print("─" * 80)
        
        for detail in results['details']:
            status = '✅ PASS' if detail['passed'] else '❌ FAIL'
            print(f"{detail['test']:<25} {detail['expected']:<15} {detail['actual']:<15} {detail['confidence']:<11.2%} {status}")
        
        # Đánh giá
        print(f"\n🎯 Performance Rating:")
        if accuracy >= 90:
            rating = "🌟 EXCELLENT - Model rất chính xác!"
        elif accuracy >= 75:
            rating = "👍 GOOD - Model hoạt động tốt"
        elif accuracy >= 60:
            rating = "⚠️  ACCEPTABLE - Cần cải thiện"
        else:
            rating = "❌ POOR - Cần kiểm tra lại logic"
        
        print(f"   {rating}")
        
        # Khuyến nghị
        print(f"\n💡 Recommendations:")
        if results['failed'] > 0:
            print(f"   • Xem xét lại các test cases thất bại")
            print(f"   • Điều chỉnh ngưỡng (thresholds) nếu cần")
            print(f"   • Kiểm tra logic phát hiện keypoints")
        
        failed_tests = [d for d in results['details'] if not d['passed']]
        if failed_tests:
            print(f"\n❌ Failed Tests Details:")
            for test in failed_tests:
                print(f"   • {test['test']}: Expected {test['expected']}, got {test['actual']}")
                print(f"     - EAR: {test['ear']:.3f}, Head Tilt: {test['head_tilt']:.1f}°, Mouth: {test['mouth_ratio']:.3f}")
        
        print("\n" + "=" * 80)

def main():
    """Main function"""
    try:
        tester = DrowsinessAccuracyTester()
        results = tester.run_test_suite()
        
        # Export results to file
        import json
        output_file = 'test_drowsiness_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
