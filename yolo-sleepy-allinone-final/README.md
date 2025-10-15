# 😴 Hệ Thống Phát Hiện Ngủ Gật YOLO - Tổng Hợp Multi-Model

> 🚀 **All-in-one Sleepy Detection System** với YOLOv5, YOLOv8, và YOLOv11 | GUI hiện đại + HUD tương lai

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org)

## Cách chạy nhanh

```bash
python -m venv .venv
# Windows (cửa sổ lệnh)
.\.venv\Scripts\activate
pip install -r requirements.txt
python standalone_app_copy.py
```

- Chạy demo OpenCV toàn màn hình: `python sleepy_demo.py`

## 📋 Mục Lục

- [🎯 Tính năng chính](#-tính-năng-chính)
- [🚀 Cài đặt nhanh](#-cài-đặt-nhanh)
- [🖥️ Chạy ứng dụng](#️-chạy-ứng-dụng)
- [🤖 Models có sẵn](#-models-có-sẵn)
- [🎮 Demo modes](#-demo-modes)
- [⚙️ Tính năng GUI](#️-tính-năng-gui)
- [🔧 Training tùy chỉnh](#-training-tùy-chỉnh)
- [📁 Cấu trúc project](#-cấu-trúc-project)
- [🔍 Troubleshooting](#-troubleshooting)

## 🎯 Tính năng chính

### 🌟 **Multi-Model Support**
- ✅ **YOLOv11** (1000 epochs) - Độ chính xác cao nhất
- ✅ **YOLOv8** (59 epochs) - Cân bằng tốc độ/chính xác
- ✅ **YOLOv5** (50 epochs) - Tối ưu hiệu năng

### 🎨 **Giao diện đa dạng**
- **GUI App** - Giao diện người dùng thân thiện với **Multi-Camera Tab** tích hợp
- **HUD Demo** - Màn hình fullscreen phong cách tương lai
- **Multi-Camera CLI** - 🆕 Giám sát không giới hạn camera từ command line
- **Standalone Copy** - Phiên bản độc lập có thể tùy chỉnh
- **Standalone** - Chạy độc lập không cần GUI

### 🎪 **Tính năng nâng cao**
- 📹 **Real-time detection** từ webcam, video file hoặc **IP camera**
- 🎥 **Multi-camera monitoring** - Giám sát không giới hạn camera cùng lúc
- 📷 **IP Camera support** - Kết nối với camera IMOU, Hikvision, Dahua
- 🎛️ **Adjustable confidence threshold** - Điều chỉnh độ nhạy
- 📊 **FPS monitoring** - Hiển thị hiệu năng real-time
- 🎯 **Multi-person detection** - Phát hiện nhiều người cùng lúc
- 🎨 **Customizable UI** - Tùy chỉnh màu sắc và hiển thị
- 💾 **Model switching** - Chuyển đổi model linh hoạt

## 🚀 Cài đặt nhanh

### 1️⃣ **Clone Repository**
```bash
git clone https://github.com/JKhoa/DACN_PhatHienNguGat.git
cd DACN_PhatHienNguGat/yolo-sleepy-allinone-final
```

### 2️⃣ **Tạo Virtual Environment**
```bash
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
python -c "import cv2; print('✅ OpenCV OK')"
python -c "import numpy; print('✅ NumPy OK')"

# Test multi-camera integration
python test_multi_camera_integration.py
```

## 🖥️ Chạy ứng dụng

### 🎨 **GUI App (Recommended)**
```bash
python gui_app.py
```
**Tính năng GUI:**
- 🎛️ Chọn model (YOLOv5/v8/v11)
- 📹 Chọn camera hoặc video file
- � **Multi-Camera Tab** - Giám sát nhiều camera cùng lúc (NEW!)
- �🎚️ Điều chỉnh confidence threshold
- 📊 Monitor FPS real-time
- 💾 Save/load settings & camera configs
- 🎨 Dark/Light theme

**Multi-Camera trong GUI:**
1. Mở GUI: `python gui_app.py`
2. Click tab **"📹 Multi-Camera"**
3. Click **"➕ Add Camera"** để thêm webcam hoặc IP camera
4. Click **"▶️ Start All"** để bắt đầu giám sát
5. Switch giữa **Grid View** (mosaic) và **Single View** (fullscreen)
6. Save/Load config bằng YAML cho dễ dàng quản lý

👉 **Chi tiết**: Xem [docs/MULTI_CAMERA_GUI_GUIDE.md](docs/MULTI_CAMERA_GUI_GUIDE.md)
- 🎥 **📹 Multi-Camera Tab** - Giám sát nhiều camera trong cùng một giao diện

**Multi-Camera trong GUI:**
1. Mở tab "📹 Multi-Camera" trong GUI
2. Click "➕ Add Camera" để thêm camera
3. Chọn loại: Webcam hoặc IP Camera
4. Với IP Camera: nhập brand, IP, username, password
5. Click "Test Connection" để kiểm tra
6. Click "▶️ Start All" để bắt đầu giám sát
7. Chọn "Grid View" để xem tất cả cùng lúc hoặc "Single View" để xem từng camera
8. Lưu cấu hình: "💾 Save Config" để sử dụng lại sau

### 🎮 **HUD Demo (Fullscreen)**
```bash
python sleepy_demo.py
```
**Phím điều khiển:**
- `ESC` hoặc `Q` - Thoát
- `M` - Toggle thông tin hiển thị
- `SPACE` - Pause/Resume
- `C` - Chuyển camera

### 🎥 **Multi-Camera CLI Mode (Advanced)**
```bash
# Setup cameras
copy cameras.sample.yaml cameras.yaml
# Edit cameras.yaml với thông tin camera của bạn

# Run multi-camera app
python multi_camera_app.py --config cameras.yaml

# Grid view (mosaic)
python multi_camera_app.py --config cameras.yaml --view grid

# Single view (từng camera)
python multi_camera_app.py --config cameras.yaml --view single

# CLI mode (no GUI, for servers)
python multi_camera_app.py --config cameras.yaml --mode cli

# Performance tuning
python multi_camera_app.py --config cameras.yaml --stride 2 --max-fps 15
```
**Tính năng:**
- 🎯 **Không giới hạn camera** - Giám sát bao nhiêu camera cũng được
- 🖥️ **Dynamic grid layout** - Tự động tính toán bố cục (2x2, 3x3, 4x4...)
- 🔄 **Auto-reconnect** - Tự động kết nối lại khi camera mất kết nối
- 📊 **Per-camera stats** - FPS và detection count cho từng camera
- ⚡ **Multi-threading** - Xử lý song song nhiều camera
- 🎨 **Multiple views** - Grid, Single, HUD modes

**📖 Xem chi tiết**: [docs/MULTI_CAMERA_GUIDE.md](docs/MULTI_CAMERA_GUIDE.md)

### ⚡ **Standalone App (Editable Version)**
```bash
python standalone_app_copy.py
```
**Lưu ý:** Sử dụng `standalone_app_copy.py` thay vì `standalone_app.py` để có thể tùy chỉnh mà không ảnh hưởng file gốc.

### 📋 **Standalone App (Original)**
```bash
python standalone_app.py
```

### 📷 **IP Camera Support - 15+ Thương Hiệu**
```bash
# IMOU Ranger
python standalone_app.py --ip-camera --ip 192.168.1.100 \
  --username admin --password 123456 --camera-brand imou

# Hikvision DS series
python standalone_app.py --ip-camera --ip 192.168.1.101 \
  --username admin --password abcd --camera-brand hikvision

# TP-Link Tapo C200/C210
python standalone_app.py --ip-camera --ip 192.168.1.102 \
  --username admin --password tapopass --camera-brand tapo

# Xiaomi Mi Home Security
python standalone_app.py --ip-camera --ip 192.168.1.103 \
  --username admin --password xiaomipass --camera-brand xiaomi

# Reolink RLC series
python standalone_app.py --ip-camera --ip 192.168.1.104 \
  --username admin --password reopass --camera-brand reolink

# Foscam FI/R series
python standalone_app.py --ip-camera --ip 192.168.1.105 \
  --username admin --password foscampass --camera-brand foscam

# Axis Professional
python standalone_app.py --ip-camera --ip 192.168.1.106 \
  --username root --password axispass --camera-brand axis

# Bosch Security
python standalone_app.py --ip-camera --ip 192.168.1.107 \
  --username service --password boschpass --camera-brand bosch

# Camera khác (Generic RTSP)
python standalone_app.py --ip-camera --ip 192.168.1.108 \
  --username admin --password genericpass --camera-brand generic

# Test camera trước khi sử dụng
python test_ip_camera.py --ip 192.168.1.100 --username admin --password 123456 --brand imou

# Demo test nhiều camera cùng lúc
python demo_multi_camera.py
```

**📖 Hỗ trợ đầy đủ:** 
- 🏠 **Gia đình**: IMOU, TP-Link Tapo, Xiaomi, Reolink, Foscam
- 🏢 **Doanh nghiệp**: Hikvision, Dahua, Axis, Bosch, Sony, Panasonic, Vivotek
- 🌐 **Khác**: D-Link, Netgear Arlo, ONVIF, Generic RTSP

**📋 Chi tiết setup**: Xem [CAMERA_SUPPORT_EXTENDED.md](CAMERA_SUPPORT_EXTENDED.md) cho 15+ thương hiệu

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
│   ├── gui_app.py              # Main GUI application (với Multi-Camera Tab)
│   ├── multi_camera_gui.py     # 🆕 Multi-camera widget cho GUI
│   ├── camera_core.py          # 🆕 Core camera classes (shared)
│   ├── sleepy_demo.py          # HUD fullscreen demo
│   ├── multi_camera_app.py     # Multi-camera CLI monitoring
│   ├── standalone_app.py       # Standalone detection (original)
│   └── standalone_app_copy.py  # Standalone detection (editable copy)
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
│   │   ├── test_ip_camera.py        # Test single camera
│   │   └── demo_multi_camera.py     # Test multiple cameras
│
├── 📊 Datasets
│   └── datasets/sleepy_pose/   # Training dataset
│
├── 📋 Configuration  
│   ├── requirements.txt        # Python dependencies
│   ├── cameras.sample.yaml     # Multi-camera config sample
│   ├── cameras.yaml           # Your camera config (create from sample)
│   ├── README.md              # This file
│   └── .gitignore            # Git ignore rules
│
├── 📖 Documentation
│   ├── docs/
│   │   ├── MULTI_CAMERA_GUIDE.md    # Multi-camera CLI guide
│   │   ├── MULTI_CAMERA_QUICKSTART.md # Quick start
│   │   ├── IP_CAMERA_GUIDE.md       # IP camera setup
│   │   └── CAMERA_SUPPORT_EXTENDED.md # 15+ camera brands
```
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
python standalone_app_copy.py  # Chạy standalone app ngay!
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
# Bắt đầu ngay với standalone app (có thể tùy chỉnh)
python standalone_app_copy.py

# Hoặc GUI app
python gui_app.py

# Hoặc fullscreen demo
python sleepy_demo.py
```

**🎉 Happy Detecting! 😴**