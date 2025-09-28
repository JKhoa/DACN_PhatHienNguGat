# 😴 Hệ Thống Phát Hiện Ngủ Gật YOLO - Tổng Hợp Multi-Model# Hệ thống phát hiện ngủ gật YOLO — Phiên bản tổng hợp



> 🚀 **All-in-one Sleepy Detection System** với YOLOv5, YOLOv8, và YOLOv11 | GUI hiện đại + HUD tương lai## Cách chạy nhanh

```bash

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)python -m venv .venv

[![YOLO](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-green.svg)](https://github.com/ultralytics/ultralytics)# Windows (cửa sổ lệnh)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io).\.venv\Scripts\activate

[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org)pip install -r requirements.txt

streamlit run app.py

## 📋 Mục Lục```

- Chạy demo OpenCV toàn màn hình: `python sleepy_demo.py`LO Pose — All-in-one HUD Build

- [🎯 Tính năng chính](#-tính-năng-chính)

- [🚀 Cài đặt nhanh](#-cài-đặt-nhanh)## Chạy nhanh

- [🖥️ Chạy ứng dụng](#️-chạy-ứng-dụng)```bash

- [🤖 Models có sẵn](#-models-có-sẵn)python -m venv .venv

- [🎮 Demo modes](#-demo-modes)# Windows

- [⚙️ Tính năng GUI](#️-tính-năng-gui).\.venv\Scriptsctivate

- [🔧 Training tùy chỉnh](#-training-tùy-chỉnh)pip install -r requirements.txt

- [📁 Cấu trúc project](#-cấu-trúc-project)streamlit run app.py

- [🔍 Troubleshooting](#-troubleshooting)```

- Demo OpenCV fullscreen: `python sleepy_demo.py`

## 🎯 Tính năng chính

## Tính năng chính

### 🌟 **Multi-Model Support**- **Giao diện HUD tương lai** (màu cyan neon, nền lưới).

- ✅ **YOLOv11** (1000 epochs) - Độ chính xác cao nhất- Ứng dụngw bao gồm:

- ✅ **YOLOv8** (59 epochs) - Cân bằng tốc độ/chính xác    - Webcam/Video với lựa chọn chỉ số camera

- ✅ **YOLOv5** (50 epochs) - Tối ưu hiệu năng  - Điều chỉnh độ phân giải + định dạng MJPG  

  - Kích thước ảnh, độ dày đường, lật ảnh, làm sắc nét, chiều rộng hiển thị

### 🎨 **Giao diện đa dạng**  - **Thanh trượt ngưỡng độ tin cậy**

- **GUI App** - Giao diện người dùng thân thiện  - FPS/Độ trễ hiển thị gọn trong 1 dòng (dòng trạng thái) — có thể ẩn bằng cách comment dòng `status_placeholder.markdown(...)`

- **HUD Demo** - Màn hình fullscreen phong cách tương lai- Demo OpenCV toàn màn hình + letterbox, sử dụng phím ESC/Q/M để điều khiển

- **Streamlit Web** - Chạy trên trình duyệt

- **Standalone** - Chạy độc lập không cần GUI## Huấn luyện mô hình YOLO Pose

Sắp xếp dữ liệu theo cấu trúc sau:

### 🎪 **Tính năng nâng cao**```

- 📹 **Real-time detection** từ webcam hoặc video filedatasets/sleepy_pose/

- 🎛️ **Adjustable confidence threshold** - Điều chỉnh độ nhạy ├─ images/{train,val}   # Thư mục chứa ảnh huấn luyện và kiểm tra

- 📊 **FPS monitoring** - Hiển thị hiệu năng real-time └─ labels/{train,val}   # Thư mục chứa nhãn định dạng YOLO Pose

- 🎯 **Multi-person detection** - Phát hiện nhiều người cùng lúc```

- 🎨 **Customizable UI** - Tùy chỉnh màu sắc và hiển thịLệnh huấn luyện:

- 💾 **Model switching** - Chuyển đổi model linh hoạt```bash

yolo task=pose mode=train model=yolo11n-pose.pt data=datasets/sleepy_pose/sleepy.yaml epochs=100 imgsz=640 batch=16 device=0

## 🚀 Cài đặt nhanh```

Giải thích tham số:

### 1️⃣ **Clone Repository**- `task=pose`: Nhiệm vụ phát hiện tư thế

```bash- `mode=train`: Chế độ huấn luyện  

git clone https://github.com/JKhoa/DACN_PhatHienNguGat.git- `epochs=100`: Số lượng epochs huấn luyện

cd DACN_PhatHienNguGat/yolo-sleepy-allinone-final- `imgsz=640`: Kích thước ảnh đầu vào

```- `batch=16`: Kích thước batch

- `device=0`: Sử dụng GPU (hoặc 'cpu' cho CPU)

### 2️⃣ **Tạo Virtual Environment**

```bashSử dụng file trọng số `runs/pose/train/weights/best.pt` cho ứng dụng và demo.

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux  
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ **Cài đặt Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Verify Installation**
```bash
python -c "import ultralytics; print('✅ Ultralytics OK')"
python -c "import streamlit; print('✅ Streamlit OK')"
python -c "import cv2; print('✅ OpenCV OK')"
```

## 🖥️ Chạy ứng dụng

### 🎨 **GUI App (Recommended)**
```bash
python gui_app.py
```
**Tính năng GUI:**
- 🎛️ Chọn model (YOLOv5/v8/v11)
- 📹 Chọn camera hoặc video file
- 🎚️ Điều chỉnh confidence threshold
- 📊 Monitor FPS real-time
- 💾 Save/load settings
- 🎨 Dark/Light theme

### 🌐 **Web App (Streamlit)**  
```bash
streamlit run app.py
```
Mở trình duyệt: `http://localhost:8501`

### 🎮 **HUD Demo (Fullscreen)**
```bash
python sleepy_demo.py
```
**Phím điều khiển:**
- `ESC` hoặc `Q` - Thoát
- `M` - Toggle thông tin hiển thị
- `SPACE` - Pause/Resume
- `C` - Chuyển camera

### ⚡ **Standalone App**
```bash
python standalone_app.py
```

## 🤖 Models có sẵn

| Model | Epochs | Accuracy | Speed | Size | Recommended Use |
|-------|--------|----------|-------|------|-----------------|
| **YOLOv11** | 1000 | 🏆 **Cao nhất** | ⚡ Nhanh | 5.9MB | Production, Accuracy critical |
| **YOLOv8** | 59 | 👍 Tốt | ⚡⚡ Rất nhanh | 19.3MB | Balanced, General use |  
| **YOLOv5** | 50 | ✅ Ổn định | ⚡⚡⚡ Siêu nhanh | 5.3MB | Real-time, Edge devices |

### 📊 **Performance Comparison**
```bash
# Chạy benchmark tất cả models
python tools/benchmark_models.py
```

## 🎮 Demo modes

### 🎯 **Basic Detection**
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov11_1000ep_best.pt')

# Detect on webcam
results = model(source=0, show=True, conf=0.5)
```

### 🎨 **Custom GUI Detection**  
```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov11_1000ep_best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    
    cv2.imshow('Sleepy Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## ⚙️ Tính năng GUI

### 🎛️ **Control Panel**
- **Model Selection**: Dropdown chọn YOLOv5/v8/v11
- **Source Selection**: Webcam, Video file, hoặc Image
- **Confidence Slider**: Điều chỉnh từ 0.1 đến 0.9
- **Resolution Settings**: 480p, 720p, 1080p
- **FPS Limit**: Giới hạn FPS để tiết kiệm tài nguyên

### 🎨 **Display Options**
- **Theme**: Dark mode / Light mode
- **Colors**: Tùy chỉnh màu bounding box
- **Info Display**: Ẩn/hiện thông tin FPS, confidence
- **Fullscreen Mode**: Chế độ toàn màn hình
- **Recording**: Ghi lại video output

### 📊 **Statistics Panel**
- **Real-time FPS**: Hiển thị FPS hiện tại
- **Detection Count**: Số lượng phát hiện
- **Processing Time**: Thời gian xử lý frame
- **Model Info**: Thông tin model đang sử dụng

## 🔧 Training tùy chỉnh

### 📁 **Chuẩn bị Dataset**
```
datasets/sleepy_pose/
├── images/
│   ├── train/          # Ảnh training
│   └── val/            # Ảnh validation
├── labels/  
│   ├── train/          # Labels training (YOLO format)
│   └── val/            # Labels validation
└── sleepy.yaml         # Dataset config
```

### 🏋️ **Training Scripts**

#### **YOLOv11 (High Accuracy)**
```bash
python tools/train_v11_1000_epochs.py
```

#### **YOLOv8 (Balanced)**  
```bash
python tools/train_v8_optimized.py
```

#### **YOLOv5 (Fast Training)**
```bash
python tools/train_yolov5_50_epochs.py
```

### ⚙️ **Custom Training Parameters**
```python
# Example: Custom YOLOv11 training
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
results = model.train(
    data='datasets/sleepy_pose/sleepy.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    patience=50,
    device='cpu'  # hoặc 'cuda' nếu có GPU
)
```

## 📁 Cấu trúc project

```
yolo-sleepy-allinone-final/
├── 📱 GUI Applications
│   ├── gui_app.py              # Main GUI application
│   ├── app.py                  # Streamlit web app  
│   ├── sleepy_demo.py          # HUD fullscreen demo
│   └── standalone_app.py       # Standalone detection
│
├── 🤖 Trained Models  
│   ├── yolov11_1000ep_best.pt  # YOLOv11 (1000 epochs)
│   ├── yolov5_50ep_best.pt     # YOLOv5 (50 epochs)  
│   └── runs/pose-train/        # Training outputs
│
├── 🛠️ Tools & Scripts
│   ├── tools/
│   │   ├── benchmark_models.py      # Model comparison
│   │   ├── train_yolov5_50_epochs.py
│   │   ├── train_v11_1000_epochs.py
│   │   └── monitor_training.py
│
├── 📊 Datasets
│   └── datasets/sleepy_pose/   # Training dataset
│
├── 📋 Configuration  
│   ├── requirements.txt        # Python dependencies
│   ├── README.md              # This file
│   └── .gitignore            # Git ignore rules
│
└── 📈 Results & Backups
    ├── model_backups_*/       # Model backups
    └── training_results_*/    # Training logs
```

## 🔍 Troubleshooting

### ❗ **Common Issues**

#### **1. Model not found**
```bash
# Đảm bảo models có trong thư mục
ls *.pt
# Nếu không có, download từ Ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

#### **2. Camera not working**
```python
# Test camera
import cv2
cap = cv2.VideoCapture(0)  # Thử camera index khác: 1, 2, 3...
print(f"Camera opened: {cap.isOpened()}")
```

#### **3. Slow performance**
- ✅ Giảm resolution: 480p thay vì 1080p
- ✅ Tăng confidence threshold: 0.7 thay vì 0.3
- ✅ Sử dụng YOLOv5 thay vì YOLOv11
- ✅ Limit FPS: 15 FPS thay vì 30 FPS

#### **4. Out of memory**
```python
# Giảm batch size khi training
batch=4  # thay vì batch=16
imgsz=416  # thay vì imgsz=640
```

### 🆘 **Getting Help**

1. **Check logs**: Xem console output cho error details
2. **Update packages**: `pip install -U ultralytics opencv-python`
3. **Verify GPU**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Test basic YOLO**: `yolo predict model=yolo11n.pt source=0`

### 📞 **Support**
- 🐛 **Issues**: [GitHub Issues](https://github.com/JKhoa/DACN_PhatHienNguGat/issues)
- 📧 **Contact**: Tạo issue trên GitHub với chi tiết lỗi
- 📖 **Docs**: [Ultralytics Documentation](https://docs.ultralytics.com)

## 🎉 **Quick Start Examples**

### 🚀 **5-Minute Setup**
```bash
# Clone + Setup + Run
git clone https://github.com/JKhoa/DACN_PhatHienNguGat.git
cd DACN_PhatHienNguGat/yolo-sleepy-allinone-final
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
python gui_app.py  # Chạy GUI app ngay!
```

### 🎯 **One-Line Detection**  
```bash
# Detect ngay với webcam
python -c "from ultralytics import YOLO; YOLO('yolov11_1000ep_best.pt')(source=0, show=True)"
```

---

## 🏆 **Model Performance Summary**

| Metric | YOLOv11 (1000ep) | YOLOv8 (59ep) | YOLOv5 (50ep) |
|--------|------------------|---------------|---------------|
| **Box mAP@50** | 🥇 **0.892** | 🥈 0.743 | 🥉 0.681 |
| **Pose mAP@50** | 🥇 **0.845** | 🥈 0.698 | 🥉 0.612 |
| **Inference Speed** | ⚡ 23ms | ⚡⚡ 18ms | ⚡⚡⚡ 15ms |
| **Model Size** | 💾 5.9MB | 💾 19.3MB | 💾 5.3MB |
| **Training Time** | 🕐 48h | 🕐 3.2h | 🕐 1.5h |
| **Best For** | 🎯 Production | ⚖️ Balanced | 🚀 Real-time |

### 🎯 **Recommendation**
- **🏆 Production**: YOLOv11 (Highest accuracy)
- **⚖️ Development**: YOLOv8 (Good balance) 
- **🚀 Demo/Edge**: YOLOv5 (Fastest inference)

---

## 🚀 **Ready to use?**

```bash
# Bắt đầu ngay với GUI app
python gui_app.py

# Hoặc web app
streamlit run app.py

# Hoặc fullscreen demo
python sleepy_demo.py
```

**🎉 Happy Detecting! 😴**