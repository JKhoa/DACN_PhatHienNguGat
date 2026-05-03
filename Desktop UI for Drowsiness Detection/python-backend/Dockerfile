# Sử dụng Python 3.11 slim để giảm dung lượng ảnh
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn backend vào container
COPY . .

# Expose cổng 5000 cho Flask-SocketIO
EXPOSE 5000

# Lệnh chạy server (sử dụng server_with_tracking_backup.py làm bản chuẩn nhất)
CMD ["python", "server_with_tracking_backup.py"]
