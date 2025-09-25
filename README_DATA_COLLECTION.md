# 📊 Hướng Dẫn Thu Thập Dữ Liệu - Sleepy Detection Dataset

## 🎯 Mục Tiêu
Mở rộng dataset từ **~180 ảnh** hiện tại lên **300-400 ảnh** để cải thiện độ chính xác của model.

## 🛠️ Công Cụ Có Sẵn

### 1. `download_images.py` - Tải Ảnh Từ URL
**Mục đích**: Tải xuống ảnh từ các nguồn miễn phí (Pexels, Unsplash)

**Cách sử dụng**:
```bash
# Chỉnh sửa SAMPLE_URLS trong file trước
python download_images.py
```

**Chuẩn bị**:
1. Mở `download_images.py`
2. Thay thế `SAMPLE_URLS = []` bằng danh sách URL thực tế:
```python
SAMPLE_URLS = [
    "https://images.pexels.com/photos/9489900/pexels-photo-9489900.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940",
    "https://images.unsplash.com/photo-1234567890?auto=format&fit=crop&w=1000&q=80",
    # ... thêm URL khác
]
```

### 2. `collect_data.py` - Công Cụ Tổng Hợp
**Mục đích**: Xử lý tất cả các tác vụ thu thập và xử lý dữ liệu

**Các lệnh**:
```bash
# Tải ảnh từ URLs (chạy download_images.py)
python collect_data.py --download

# Trích xuất frame từ video
python collect_data.py --video "path/to/video.mp4" --fps 2 --max-frames 50

# Sao chép ảnh về data_raw chính
python collect_data.py --copy

# Chạy auto-labeling
python collect_data.py --auto-label

# Xem thống kê
python collect_data.py --stats

# Chạy toàn bộ quy trình (khuyến nghị)
python collect_data.py --full-pipeline
```

## 📋 Quy Trình Thu Thập Khuyến Nghị

### Bước 1: Chuẩn Bị URLs
1. Tham khảo `docs/DATA_COLLECTION_GUIDE.md` để biết danh sách nguồn
2. Thu thập URL trực tiếp đến file ảnh từ:
   - **Pexels**: Tìm "sleeping", "tired", "student sleeping"
   - **Unsplash**: Tìm "sleepy", "nap", "exhausted" 
   - **Pixabay**: Tìm "sleep", "drowsy", "yawn"

3. Cập nhật `SAMPLE_URLS` trong `download_images.py`

### Bước 2: Thu Thập Tự Động
```bash
# Chạy toàn bộ quy trình
python collect_data.py --full-pipeline
```

Quy trình này sẽ:
1. 🔽 Tải ảnh từ URLs
2. 📁 Sao chép về data_raw
3. 🏷️ Tạo labels tự động
4. 📊 Hiển thị thống kê

### Bước 3: Kiểm Tra Kết Quả
```bash
# Xem chi tiết thống kê
python collect_data.py --stats
```

## 📁 Cấu Trúc Thư Mục

```
data_raw/
├── cap_000000.jpg          # Ảnh gốc
├── cap_000001.jpg
├── downloaded_image_001.jpg # Ảnh từ URLs
├── downloaded_image_002.jpg
├── pexels_*.jpg            # Ảnh từ Pexels
├── unsplash_*.jpg          # Ảnh từ Unsplash
├── video_*.jpg             # Frame từ video
├── downloaded_images/      # Thư mục tạm
├── pexels_images/          # Thư mục tạm
├── unsplash_images/        # Thư mục tạm
└── video_frames/           # Thư mục tạm
```

## 🎥 Thu Thập Từ Video

### Chuẩn Bị Video
1. Tìm video có nội dung:
   - Học sinh ngủ gật trong lớp
   - Người làm việc mệt mỏi
   - Video công sở, thư viện

2. Tải video về máy

### Trích Xuất Frame
```bash
# Trích xuất 2 frame/giây, tối đa 50 frame
python collect_data.py --video "path/to/video.mp4" --fps 2 --max-frames 50

# Sau đó sao chép và xử lý
python collect_data.py --copy --auto-label
```

## 🏷️ Auto-Labeling

### Nguyên Lý
- Sử dụng model đã train để tạo label tự động
- Chỉ giữ lại những detection có confidence cao
- Tạo format YOLO (.txt) cho training

### Kiểm Tra Chất Lượng Labels
```bash
# Xem thống kê labels
python collect_data.py --stats

# Kiểm tra visual trong thư mục
yolo-sleepy-allinone-final/datasets/sleepy_pose/train/
```

## ⚖️ Tuân Thủ Bản Quyền

### Nguồn An Toàn
- ✅ **Pexels**: License miễn phí thương mại
- ✅ **Unsplash**: License miễn phí thương mại  
- ✅ **Pixabay**: License miễn phí thương mại

### Không Sử Dụng
- ❌ Google Images (có bản quyền)
- ❌ Ảnh có watermark
- ❌ Ảnh từ social media cá nhân

## 📊 Theo Dõi Tiến Độ

### Mục Tiêu
- **Hiện tại**: ~180 ảnh
- **Mục tiêu**: 300-400 ảnh 
- **Cần thêm**: 120-220 ảnh

### Kiểm Tra
```bash
python collect_data.py --stats
```

Sẽ hiển thị:
- Tổng số ảnh theo nguồn
- Số labels đã tạo
- Tiến độ đạt mục tiêu

## 🚨 Xử Lý Sự Cố

### Lỗi Download
```
❌ HTTP 403/404: URL không hợp lệ
```
**Giải pháp**: Lấy URL trực tiếp từ nút download, không phải URL trang

### Lỗi Auto-Labeling
```
Auto-label script not found
```
**Giải pháp**: 
```bash
cd yolo-sleepy-allinone-final/tools/
python auto_label_pose.py --help
```

### File Không Được Tạo
```bash
# Kiểm tra quyền thư mục
ls -la data_raw/

# Tạo thư mục thủ công nếu cần
mkdir -p data_raw/downloaded_images
```

## 🎯 Tips Hiệu Quả

1. **Batch Processing**: Thu thập 20-30 ảnh một lần, kiểm tra chất lượng
2. **Đa dạng hóa**: Mix ảnh từ nhiều nguồn khác nhau
3. **Kiểm tra định kỳ**: Chạy `--stats` sau mỗi batch
4. **Backup**: Sao lưu data_raw trước khi thử nghiệm lớn

## 📞 Hỗ Trợ

- Xem chi tiết: `docs/DATA_COLLECTION_GUIDE.md`
- Hướng dẫn training: `yolo-sleepy-allinone-final/tools/README_TRAINING.md`
- Log tiến độ: `docs/PROGRESS.md`