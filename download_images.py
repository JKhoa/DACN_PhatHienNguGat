#!/usr/bin/env python3
"""
Script tải xuống hình ảnh từ các nguồn miễn phí đã được chỉ định
Chú ý: Script này chỉ hoạt động với URL trực tiếp đến file ảnh, không phải trang web
"""

import os
import requests
import time
from pathlib import Path

# URLs hình ảnh từ Pexels - Sleeping/Tired Students
# Tập hợp từ các trang search và links được cung cấp

PEXELS_IMAGE_URLS = [
    # Batch 1: Từ danh sách ban đầu
    "https://images.pexels.com/photos/7034472/pexels-photo-7034472.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/6683966/pexels-photo-6683966.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/8423127/pexels-photo-8423127.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/9158508/pexels-photo-9158508.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/3808119/pexels-photo-3808119.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/6281902/pexels-photo-6281902.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/6281886/pexels-photo-6281886.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/7683901/pexels-photo-7683901.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/9159055/pexels-photo-9159055.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/9489900/pexels-photo-9489900.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    
    # Batch 2: Từ Pexels search "sleeping student"
    "https://images.pexels.com/photos/6148379/pexels-photo-6148379.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/5669619/pexels-photo-5669619.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145190/pexels-photo-4145190.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145066/pexels-photo-4145066.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/3807755/pexels-photo-3807755.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    
    # Batch 3: Từ Pexels search "tired student"
    "https://images.pexels.com/photos/3184418/pexels-photo-3184418.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/3184292/pexels-photo-3184292.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145017/pexels-photo-4145017.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/3807760/pexels-photo-3807760.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4226142/pexels-photo-4226142.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    
    # Batch 4: Từ Pexels search "sleep desk"
    "https://images.pexels.com/photos/5212317/pexels-photo-5212317.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145203/pexels-photo-4145203.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145115/pexels-photo-4145115.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/3807943/pexels-photo-3807943.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145156/pexels-photo-4145156.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    
    # Batch 5: Thêm từ "drowsy", "yawn", "fatigue"
    "https://images.pexels.com/photos/4226771/pexels-photo-4226771.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145951/pexels-photo-4145951.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/3184339/pexels-photo-3184339.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4226764/pexels-photo-4226764.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4145029/pexels-photo-4145029.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
]

# URLs video từ Pexels - sẽ được xử lý riêng
PEXELS_VIDEO_URLS = [
    "https://www.pexels.com/video/student-sleeping-in-classroom-6672265/",
    "https://www.pexels.com/video/exhausted-student-sleeping-on-a-bench-6083402/",
    "https://www.pexels.com/video/a-student-sleeping-on-a-desk-9489688/",
    "https://www.pexels.com/video/student-sleeping-in-classroom-6672269/",
    "https://www.pexels.com/video/woman-sleeping-over-her-studies-8086497/",
    "https://www.pexels.com/video/a-woman-throwing-a-crumpled-paper-to-her-sleeping-friend-9489827/",
    "https://www.pexels.com/video/a-young-man-sleeping-on-his-desk-7949756/",
    "https://www.pexels.com/video/a-student-sleeping-while-studying-8569881/",
    "https://www.pexels.com/video/student-sleeping-at-library-7949795/",
    "https://www.pexels.com/video/a-boy-sleeping-on-books-in-classroom-8086400/",
    "https://www.pexels.com/video/a-girl-sleeping-on-her-desk-8469803/",
    "https://www.pexels.com/video/a-female-student-sleeping-during-class-9523085/",
    "https://www.pexels.com/video/a-student-yawning-in-classroom-8469702/",
    "https://www.pexels.com/video/young-woman-sleeping-in-classroom-9523101/",
    "https://www.pexels.com/video/student-resting-on-his-books-8469721/",
    "https://www.pexels.com/video/a-woman-sleeping-on-the-table-8569920/",
    "https://www.pexels.com/video/a-student-napping-at-his-desk-8569855/",
    "https://www.pexels.com/video/a-woman-falling-asleep-while-studying-8569938/",
    "https://www.pexels.com/video/a-man-sleeping-on-his-open-books-8569999/",
    "https://www.pexels.com/video/tired-student-sleeping-during-lecture-8570010/",
]

# Kết hợp URLs để tương thích với code cũ
SAMPLE_URLS = PEXELS_IMAGE_URLS

def get_direct_image_url_instructions():
    """Hướng dẫn lấy URL trực tiếp từ Pexels và Unsplash"""
    instructions = """
    === HƯỚNG DẪN LẤY URL TRỰC TIẾP ===
    
    ** PEXELS **
    1. Mở link ảnh Pexels (ví dụ: https://www.pexels.com/photo/photo-of-a-schoolgirl-sleeping-9489900/)
    2. Click nút "Free Download"
    3. Chọn kích thước "Large" hoặc "Original"  
    4. Click chuột phải vào ảnh preview → "Copy image address"
    5. Paste URL vào danh sách SAMPLE_URLS
    
    ** UNSPLASH **
    1. Mở link ảnh Unsplash
    2. Click nút "Download" 
    3. Chọn kích thước lớn nhất
    4. Click chuột phải vào ảnh preview → "Copy image address"
    5. Paste URL vào danh sách SAMPLE_URLS
    
    ** LƯU Ý **
    - URL phải kết thúc bằng .jpg, .jpeg, .png
    - Không sử dụng URL trang web, chỉ dùng URL trực tiếp đến file ảnh
    - Test URL bằng cách paste vào trình duyệt - phải hiển thị ảnh trực tiếp
    """
    return instructions

def download_image(url, filename, retries=3, delay=1):
    """Tải ảnh từ URL với retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(retries):
        try:
            print(f"Downloading {filename} (attempt {attempt + 1}/{retries})...")
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Kiểm tra kích thước file
                file_size = os.path.getsize(filename)
                if file_size > 1024:  # > 1KB
                    print(f"✅ Downloaded: {filename} ({file_size:,} bytes)")
                    return True
                else:
                    print(f"❌ File too small: {filename} ({file_size} bytes)")
                    os.remove(filename)
                    
            else:
                print(f"❌ HTTP {response.status_code}: {url}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if attempt < retries - 1:
            print(f"⏳ Waiting {delay}s before retry...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    return False

def main():
    """Main function"""
    print("Image Downloader for Sleepy Detection Dataset")
    print("=" * 50)
    
    # Kiểm tra danh sách URL
    if not SAMPLE_URLS:
        print("⚠️  No URLs configured!")
        print(get_direct_image_url_instructions())
        return
    
    # Tạo thư mục
    download_dir = Path("data_raw/downloaded_images")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Download directory: {download_dir}")
    print(f"🔗 URLs to download: {len(SAMPLE_URLS)}")
    print()
    
    # Tải xuống
    success_count = 0
    
    for i, url in enumerate(SAMPLE_URLS, 1):
        # Tạo tên file
        filename = download_dir / f"image_{i:03d}.jpg"
        
        # Tải xuống
        if download_image(url, filename):
            success_count += 1
        
        print()  # Dòng trống giữa các lần tải
    
    # Báo cáo kết quả
    print("=" * 50)
    print(f"📊 Results: {success_count}/{len(SAMPLE_URLS)} images downloaded successfully")
    
    if success_count > 0:
        print(f"✅ Images saved to: {download_dir}")
        print()
        print("🚀 Next steps:")
        print(f"1. Copy images to data_raw: python collect_data.py --copy")
        print(f"2. Run auto-labeling: python collect_data.py --auto-label") 
        print(f"3. Check results: python collect_data.py --stats")
    else:
        print("❌ No images downloaded. Please check URLs and try again.")
        print()
        print(get_direct_image_url_instructions())

if __name__ == "__main__":
    main()