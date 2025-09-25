# Hướng dẫn Thu thập Dữ liệu - Học sinh Ngủ gật

## 📋 Tổng quan
Tài liệu này hướng dẫn thu thập và xử lý hình ảnh/video từ các nguồn miễn phí để bổ sung vào dataset `data_raw` cho việc huấn luyện mô hình phát hiện ngủ gật.

## 🎯 Mục tiêu thu thập
- **Đa dạng tư thế**: ngủ gật nhẹ, gục xuống bàn, ngả nghiêng
- **Đa dạng môi trường**: lớp học, thư viện, bàn làm việc
- **Đa dạng đối tượng**: nam/nữ, độ tuổi khác nhau, dân tộc khác nhau
- **Đa dạng điều kiện**: ánh sáng tự nhiên/nhân tạo, góc camera khác nhau

## 📂 Nguồn dữ liệu được đề xuất

### 🖼️ Hình ảnh chất lượng cao từ Pexels:

1. **Photo of a Schoolgirl Sleeping**
   - URL: https://www.pexels.com/photo/photo-of-a-schoolgirl-sleeping-9489900/
   - Đặc điểm: Nữ sinh ngủ trong lớp học
   - Ưu điểm: Môi trường lớp học thực tế

2. **Tired female student sleeping on books in light room**
   - URL: https://www.pexels.com/photo/tired-female-student-sleeping-on-books-in-light-room-7034472/
   - Đặc điểm: Gục xuống trên sách vở
   - Ưu điểm: Tư thế gục xuống bàn điển hình

3. **Student Sleeping on his Desk**
   - URL: https://www.pexels.com/photo/student-sleeping-on-his-desk-6683966/
   - Đặc điểm: Nam sinh ngủ trên bàn
   - Ưu điểm: Góc chụp phù hợp cho pose detection

4. **A Kid Sleeping on White Desk with Textbooks**
   - URL: https://www.pexels.com/photo/a-kid-sleeping-on-white-desk-with-textbooks-8423127/
   - Đặc điểm: Trẻ em ngủ trên bàn học
   - Ưu điểm: Đa dạng độ tuổi

5. **Exhausted female student sleeping at table in library**
   - URL: https://www.pexels.com/photo/exhausted-female-student-sleeping-at-table-in-library-3808119/
   - Đặc điểm: Môi trường thư viện
   - Ưu điểm: Ánh sáng khác biệt

### 🎬 Video từ Pexels:

1. **Student Sleeping in Classroom**
   - URL: https://www.pexels.com/video/student-sleeping-in-classroom-6672265/
   - Đặc điểm: Video trong lớp học
   - Ưu điểm: Có thể trích xuất nhiều frame

2. **Exhausted Student Sleeping on a Bench**
   - URL: https://www.pexels.com/video/exhausted-student-sleeping-on-a-bench-6083402/
   - Đặc điểm: Video ngoài trời
   - Ưu điểm: Môi trường khác biệt

### 📷 Hình ảnh từ Unsplash:

1. **School girl sleeping in class (foreground)**
   - URL: https://unsplash.com/photos/school-girl-sleeping-on-class-teenagers-students-sitting-in-the-classroom-focus-is-on-foreground-abEsPl3RWDc
   - Đặc điểm: Focus rõ nét ở foreground

2. **Tired and bored school girl/boy sleeping at desk**
   - URLs: 
     - https://unsplash.com/photos/a-small-tired-and-bored-school-girl-sitting-at-the-desk-in-classroom-sleeping-D6psBQOI62A
     - https://unsplash.com/photos/a-small-tired-and-bored-school-boy-sitting-at-the-desk-in-classroom-sleeping-Lw0ifWVh6sI
   - Đặc điểm: Cặp ảnh nam/nữ tương tự

## 🛠️ Hướng dẫn tải xuống

### Cách 1: Tải thủ công
```bash
# Tạo thư mục cho dữ liệu mới
mkdir data_raw/pexels_images
mkdir data_raw/unsplash_images
mkdir data_raw/video_frames
```

1. **Với Pexels**:
   - Truy cập link → Click "Free Download" → Chọn kích thước "Large" hoặc "Original"
   - Lưu vào `data_raw/pexels_images/`

2. **Với Unsplash**:
   - Truy cập link → Click "Download" → Chọn kích thước lớn nhất
   - Lưu vào `data_raw/unsplash_images/`

3. **Với Video**:
   - Tải video về → Sử dụng script trích xuất frame

### Cách 2: Script tự động (đề xuất)
```python
# Tạo file download_dataset.py
import requests
import os
from urllib.parse import urlparse

def download_image(url, filename):
    """Tải ảnh từ URL"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

# Danh sách URL cần tải (cập nhật URL thực tế từ Pexels/Unsplash)
image_urls = [
    # Thêm URL trực tiếp đến file ảnh (không phải trang web)
    # Ví dụ: "https://images.pexels.com/photos/9489900/pexels-photo-9489900.jpeg"
]

# Tải xuống
for i, url in enumerate(image_urls):
    filename = f"data_raw/downloaded_image_{i:03d}.jpg"
    if download_image(url, filename):
        print(f"Downloaded: {filename}")
```

## 🎞️ Trích xuất frame từ video

```python
# Tạo file extract_frames.py
import cv2
import os

def extract_frames(video_path, output_dir, fps=2):
    """Trích xuất frame từ video"""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Lấy fps frame/giây
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            filename = f"{output_dir}/frame_{saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Sử dụng
extract_frames("downloaded_video.mp4", "data_raw/video_frames", fps=2)
```

## 🏷️ Quy trình gán nhãn

1. **Sao chép ảnh mới vào data_raw**:
```bash
# Di chuyển tất cả ảnh mới vào data_raw
cp pexels_images/* ../data_raw/
cp unsplash_images/* ../data_raw/
cp video_frames/* ../data_raw/
```

2. **Chạy auto-labeling**:
```bash
cd yolo-sleepy-allinone-final/tools
python auto_label_pose.py --source "../../data_raw" --out "../datasets/sleepy_pose" --val-ratio 0.2
```

3. **Kiểm tra và hiệu chỉnh nhãn thủ công**:
   - Duyệt qua các file `.txt` trong `datasets/sleepy_pose/train/labels`
   - Sửa lại class label nếu cần:
     - `0`: binhthuong (bình thường)
     - `1`: ngugat (ngủ gật) 
     - `2`: gucxuongban (gục xuống bàn)

## 📊 Mục tiêu số lượng

- **Hiện tại**: ~25 ảnh → 64 annotations
- **Mục tiêu**: ~100-150 ảnh mới từ nguồn miễn phí
- **Kỳ vọng**: ~300-400 annotations tổng cộng
- **Phân bố mong muốn**:
  - binhthuong: 40%
  - ngugat: 35% 
  - gucxuongban: 25%

## ⚖️ Lưu ý pháp lý

### Giấy phép sử dụng:
- **Pexels License**: Miễn phí thương mại, không cần attribution
- **Unsplash License**: Miễn phí thương mại, khuyến khích attribution  
- **Pixabay License**: Miễn phí thương mại, không cần attribution

### Attribution đề xuất:
```
# Trong file README.md hoặc báo cáo
## Dataset Attribution
Some images in this dataset are sourced from:
- Pexels (https://www.pexels.com) - Pexels License
- Unsplash (https://unsplash.com) - Unsplash License
- Pixabay (https://pixabay.com) - Pixabay License

All images are used under their respective free licenses for educational/research purposes.
```

## 🔄 Quy trình tự động hoá

### Script tổng hợp (recommended):
```bash
# Tạo file collect_and_process.py
#!/usr/bin/env python3

def main():
    print("=== Dataset Collection & Processing Pipeline ===")
    
    # 1. Tạo thư mục
    create_directories()
    
    # 2. Tải xuống (nếu có URL trực tiếp)
    download_images()
    
    # 3. Trích xuất video frames
    process_videos()
    
    # 4. Sao chép vào data_raw
    copy_to_data_raw()
    
    # 5. Chạy auto-labeling
    run_auto_labeling()
    
    # 6. Báo cáo kết quả
    report_results()

if __name__ == "__main__":
    main()
```

## 🎯 Tiếp theo

1. **Tải xuống 20-30 ảnh chất lượng cao nhất** từ danh sách trên
2. **Chạy auto-labeling** với dữ liệu mới
3. **Huấn luyện lại model** với dataset mở rộng
4. **So sánh performance** trước và sau khi bổ sung dữ liệu
5. **Tối ưu hóa** dataset dựa trên kết quả

---
*Ghi chú: Vì không thể tải tự động, việc thu thập dữ liệu cần thực hiện thủ công theo hướng dẫn trên.*