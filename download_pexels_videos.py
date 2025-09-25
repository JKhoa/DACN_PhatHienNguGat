#!/usr/bin/env python3
"""
Script tải xuống và xử lý video từ Pexels
Sử dụng Pexels API để lấy video download URLs
"""

import os
import re
import requests
import time
from pathlib import Path

# Danh sách URLs video Pexels
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

def extract_video_id(pexels_url):
    """Trích xuất video ID từ Pexels URL"""
    match = re.search(r'/video/.*?-(\d+)/?$', pexels_url)
    if match:
        return match.group(1)
    return None

def get_pexels_video_download_url(video_id, quality='hd'):
    """Lấy direct download URL từ Pexels video ID"""
    # Pexels không cung cấp API miễn phí cho video downloads
    # Chúng ta sẽ sử dụng cách khác - scrape từ trang web
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        video_page_url = f"https://www.pexels.com/video/{video_id}/"
        response = requests.get(video_page_url, headers=headers)
        
        if response.status_code == 200:
            content = response.text
            # Tìm video download URL trong HTML
            # Pexels thường có pattern như này:
            # "download":"https://vod-progressive.akamaized.net/..."
            
            download_patterns = [
                r'"download":"(https://[^"]*\.mp4[^"]*)"',
                r'"link":"(https://[^"]*\.mp4[^"]*)"',
                r'href="(https://[^"]*\.mp4[^"]*)"[^>]*download',
            ]
            
            for pattern in download_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Lấy URL đầu tiên tìm thấy
                    download_url = matches[0].replace('\\/', '/')
                    return download_url
                    
    except Exception as e:
        print(f"Error getting download URL for video {video_id}: {e}")
    
    return None

def download_video(url, filename, retries=3):
    """Tải video từ URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(retries):
        try:
            print(f"Downloading {filename} (attempt {attempt + 1}/{retries})...")
            
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                
                # Kiểm tra kích thước file
                if total_size > 100000:  # > 100KB
                    print(f"✅ Downloaded: {filename} ({total_size:,} bytes)")
                    return True
                else:
                    print(f"❌ File too small: {filename} ({total_size} bytes)")
                    os.remove(filename)
            else:
                print(f"❌ HTTP {response.status_code}: {url}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if attempt < retries - 1:
            print(f"⏳ Waiting before retry...")
            time.sleep(2)
    
    return False

def main():
    """Main function - download videos"""
    print("🎥 Pexels Video Downloader for Sleepy Detection Dataset")
    print("=" * 60)
    
    # Tạo thư mục
    download_dir = Path("data_raw/pexels_videos")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Download directory: {download_dir}")
    print(f"🔗 Videos to download: {len(PEXELS_VIDEO_URLS)}")
    print()
    
    success_count = 0
    
    for i, video_url in enumerate(PEXELS_VIDEO_URLS, 1):
        print(f"\n[{i}/{len(PEXELS_VIDEO_URLS)}] Processing: {video_url}")
        
        # Trích xuất video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"❌ Cannot extract video ID from URL")
            continue
            
        print(f"📹 Video ID: {video_id}")
        
        # Lấy download URL
        download_url = get_pexels_video_download_url(video_id)
        if not download_url:
            print(f"❌ Cannot get download URL")
            continue
            
        print(f"🔗 Download URL found")
        
        # Tạo tên file
        filename = download_dir / f"pexels_video_{video_id}.mp4"
        
        # Tải xuống
        if download_video(download_url, filename):
            success_count += 1
        
        # Delay giữa các lần tải
        time.sleep(2)
    
    # Báo cáo kết quả
    print("\n" + "=" * 60)
    print(f"📊 Results: {success_count}/{len(PEXELS_VIDEO_URLS)} videos downloaded")
    
    if success_count > 0:
        print(f"✅ Videos saved to: {download_dir}")
        print()
        print("🚀 Next steps:")
        print(f"1. Extract frames: python collect_data.py --video \"{download_dir}/pexels_video_*.mp4\"")
        print(f"2. Process all: python collect_data.py --full-pipeline")
    else:
        print("❌ No videos downloaded.")
        print("💡 Tip: Pexels video scraping can be tricky. Consider manual download:")
        print("   1. Visit each video URL")
        print("   2. Click 'Free Download'")
        print("   3. Save to data_raw/pexels_videos/")

if __name__ == "__main__":
    main()