# Ví dụ URLs cho download_images.py

## Cách lấy URL từ Pexels

1. Truy cập: https://www.pexels.com/search/sleeping%20student/
2. Chọn ảnh phù hợp (VD: https://www.pexels.com/photo/photo-of-a-schoolgirl-sleeping-9489900/)
3. Click "Free Download"
4. Chọn kích thước "Large" hoặc "Original"
5. Right-click ảnh preview → "Copy image address"

**URL ví dụ**: 
```
https://images.pexels.com/photos/9489900/pexels-photo-9489900.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940
```

## Cách lấy URL từ Unsplash

1. Truy cập: https://unsplash.com/s/photos/sleeping-tired
2. Chọn ảnh (VD: ảnh người ngủ gật)
3. Click "Download" 
4. Chọn kích thước lớn nhất
5. Right-click ảnh → "Copy image address"

**URL ví dụ**:
```
https://images.unsplash.com/photo-1234567890?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80
```

## Template cho SAMPLE_URLS

Thay thế nội dung này vào `download_images.py`:

```python
SAMPLE_URLS = [
    # Pexels - Sleeping students
    "https://images.pexels.com/photos/9489900/pexels-photo-9489900.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940",
    "URL_PEXELS_2",
    "URL_PEXELS_3",
    
    # Unsplash - Tired workers
    "URL_UNSPLASH_1", 
    "URL_UNSPLASH_2",
    
    # Pixabay - Drowsy people
    "URL_PIXABAY_1",
    "URL_PIXABAY_2",
]
```

## Keywords để tìm ảnh

### Pexels
- "sleeping student"
- "tired student" 
- "drowsy classroom"
- "napping office"
- "exhausted worker"

### Unsplash
- "sleepy person"
- "tired at desk"
- "yawning"
- "fatigue"
- "dozin off"

### Pixabay  
- "sleep desk"
- "tired office"
- "student sleeping"
- "drowsy"
- "nap time"

## Lưu ý quan trọng

1. **URL phải trực tiếp**: Kết thúc bằng .jpg, .jpeg, .png
2. **Không dùng URL trang web**: Chỉ dùng URL ảnh 
3. **Test URL**: Paste vào browser phải hiện ảnh trực tiếp
4. **Respect rate limits**: Không download quá nhanh

## Quy trình chuẩn

1. Thu thập 10-20 URLs từ mỗi nguồn
2. Cập nhật `SAMPLE_URLS` trong `download_images.py`
3. Chạy: `python download_images.py`
4. Kiểm tra kết quả trong `data_raw/downloaded_images/`
5. Nếu OK, chạy: `python collect_data.py --full-pipeline`