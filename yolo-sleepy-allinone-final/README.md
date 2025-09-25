# Hệ thống phát hiện ngủ gật YOLO — Phiên bản tổng hợp

## Cách chạy nhanh
```bash
python -m venv .venv
# Windows (cửa sổ lệnh)
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
- Chạy demo OpenCV toàn màn hình: `python sleepy_demo.py`LO Pose — All-in-one HUD Build

## Chạy nhanh
```bash
python -m venv .venv
# Windows
.\.venv\Scriptsctivate
pip install -r requirements.txt
streamlit run app.py
```
- Demo OpenCV fullscreen: `python sleepy_demo.py`

## Tính năng chính
- **Giao diện HUD tương lai** (màu cyan neon, nền lưới).
- Ứng dụngw bao gồm:
  - Webcam/Video với lựa chọn chỉ số camera
  - Điều chỉnh độ phân giải + định dạng MJPG  
  - Kích thước ảnh, độ dày đường, lật ảnh, làm sắc nét, chiều rộng hiển thị
  - **Thanh trượt ngưỡng độ tin cậy**
  - FPS/Độ trễ hiển thị gọn trong 1 dòng (dòng trạng thái) — có thể ẩn bằng cách comment dòng `status_placeholder.markdown(...)`
- Demo OpenCV toàn màn hình + letterbox, sử dụng phím ESC/Q/M để điều khiển

## Huấn luyện mô hình YOLO Pose
Sắp xếp dữ liệu theo cấu trúc sau:
```
datasets/sleepy_pose/
 ├─ images/{train,val}   # Thư mục chứa ảnh huấn luyện và kiểm tra
 └─ labels/{train,val}   # Thư mục chứa nhãn định dạng YOLO Pose
```
Lệnh huấn luyện:
```bash
yolo task=pose mode=train model=yolo11n-pose.pt data=datasets/sleepy_pose/sleepy.yaml epochs=100 imgsz=640 batch=16 device=0
```
Giải thích tham số:
- `task=pose`: Nhiệm vụ phát hiện tư thế
- `mode=train`: Chế độ huấn luyện  
- `epochs=100`: Số lượng epochs huấn luyện
- `imgsz=640`: Kích thước ảnh đầu vào
- `batch=16`: Kích thước batch
- `device=0`: Sử dụng GPU (hoặc 'cpu' cho CPU)

Sử dụng file trọng số `runs/pose/train/weights/best.pt` cho ứng dụng và demo.
