# Sleepy YOLO Pose — All-in-one HUD Build

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
- Giao diện **futuristic HUD** (neon cyan, grid background).
- Streamlit app với:
  - Webcam/Video, Camera index
  - Ép độ phân giải + MJPG
  - imgsz, line_width, flip, sharpen, render width
  - **Confidence threshold slider**
  - FPS/Latency gọn 1 dòng (status line) — comment dòng `status_placeholder.markdown(...)` để ẩn.
- OpenCV demo fullscreen + letterbox, phím ESC/Q/M.

## Huấn luyện YOLO Pose
Đặt dữ liệu theo:
```
datasets/sleepy_pose/
 ├─ images/{train,val}
 └─ labels/{train,val}   # YOLO Pose format
```
Train:
```
yolo task=pose mode=train model=yolo11n-pose.pt data=datasets/sleepy_pose/sleepy.yaml      epochs=100 imgsz=640 batch=16 device=0
```
Dùng `runs/pose/train/weights/best.pt` cho app và demo.
