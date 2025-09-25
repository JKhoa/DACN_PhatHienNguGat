#!/usr/bin/env python3
"""
Script thu thập tự động hình ảnh và video về học sinh ngủ gật
Sử dụng nhiều nguồn: Pexels, Unsplash, Pixabay
"""

import os
import re
import requests
import time
import json
from pathlib import Path
import urllib.parse

class AutoDataCollector:
    def __init__(self):
        self.base_dir = Path("data_raw")
        self.auto_collected_dir = self.base_dir / "auto_collected"
        self.auto_collected_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo thư mục phân loại
        self.pexels_dir = self.auto_collected_dir / "pexels"
        self.unsplash_dir = self.auto_collected_dir / "unsplash"
        self.pixabay_dir = self.auto_collected_dir / "pixabay"
        
        for dir_path in [self.pexels_dir, self.unsplash_dir, self.pixabay_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def get_pexels_search_results(self, query, per_page=30):
        """Tìm kiếm ảnh từ Pexels qua scraping"""
        print(f"🔍 Searching Pexels for: '{query}'")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # URL tìm kiếm Pexels
        search_url = f"https://www.pexels.com/search/{urllib.parse.quote(query)}/"
        
        try:
            response = requests.get(search_url, headers=headers)
            if response.status_code == 200:
                content = response.text
                
                # Tìm các URL ảnh trong HTML
                # Pexels sử dụng pattern như: "srcSet":"url1 1x, url2 2x"
                pattern = r'"src":"(https://images\.pexels\.com/photos/\d+/[^"]*\.jpeg[^"]*)"'
                matches = re.findall(pattern, content)
                
                # Lọc và làm sạch URLs
                image_urls = []
                for match in matches[:per_page]:
                    # Thêm parameters để lấy ảnh chất lượng cao
                    if '?' not in match:
                        match += "?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
                    elif 'w=' not in match:
                        match += "&w=1260&h=750&dpr=2"
                    
                    image_urls.append(match)
                
                print(f"✅ Found {len(image_urls)} images for '{query}'")
                return list(set(image_urls))  # Remove duplicates
                
        except Exception as e:
            print(f"❌ Error searching Pexels: {e}")
        
        return []
    
    def get_unsplash_search_results(self, query, per_page=20):
        """Tìm kiếm ảnh từ Unsplash"""
        print(f"🔍 Searching Unsplash for: '{query}'")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        search_url = f"https://unsplash.com/s/photos/{urllib.parse.quote(query)}"
        
        try:
            response = requests.get(search_url, headers=headers)
            if response.status_code == 200:
                content = response.text
                
                # Tìm URLs ảnh Unsplash
                pattern = r'"regular":"(https://images\.unsplash\.com/photo-[^"]*)"'
                matches = re.findall(pattern, content)
                
                image_urls = []
                for match in matches[:per_page]:
                    # Thêm parameters cho ảnh chất lượng cao
                    if '?' not in match:
                        match += "?auto=format&fit=crop&w=1000&q=80"
                    image_urls.append(match)
                
                print(f"✅ Found {len(image_urls)} images for '{query}'")
                return list(set(image_urls))
                
        except Exception as e:
            print(f"❌ Error searching Unsplash: {e}")
        
        return []
    
    def download_image(self, url, filepath, source="unknown"):
        """Download ảnh với retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        for attempt in range(3):
            try:
                print(f"📥 Downloading from {source}: {filepath.name} (attempt {attempt + 1}/3)")
                
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Kiểm tra kích thước file
                    file_size = filepath.stat().st_size
                    if file_size > 10000:  # > 10KB
                        print(f"✅ Downloaded: {filepath.name} ({file_size:,} bytes)")
                        return True
                    else:
                        print(f"❌ File too small: {filepath.name}")
                        filepath.unlink()
                else:
                    print(f"❌ HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
            if attempt < 2:
                time.sleep(2)
        
        return False
    
    def collect_from_source(self, source, queries, max_per_query=15):
        """Thu thập từ một nguồn cụ thể"""
        print(f"\n{'='*50}")
        print(f"🎯 Collecting from {source.upper()}")
        print(f"{'='*50}")
        
        if source == "pexels":
            search_func = self.get_pexels_search_results
            output_dir = self.pexels_dir
        elif source == "unsplash":
            search_func = self.get_unsplash_search_results
            output_dir = self.unsplash_dir
        else:
            print(f"❌ Unknown source: {source}")
            return 0
        
        total_downloaded = 0
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Tìm kiếm URLs
            image_urls = search_func(query, per_page=max_per_query)
            
            if not image_urls:
                print(f"⚠️ No images found for '{query}'")
                continue
            
            # Download images
            query_downloaded = 0
            for i, url in enumerate(image_urls[:max_per_query]):
                filename = output_dir / f"{source}_{query.replace(' ', '_')}_{i+1:03d}.jpg"
                
                if self.download_image(url, filename, source):
                    query_downloaded += 1
                    total_downloaded += 1
                
                # Rate limiting
                time.sleep(1)
            
            print(f"📊 Downloaded {query_downloaded} images for '{query}'")
            
            # Delay between queries
            time.sleep(2)
        
        print(f"\n✅ Total downloaded from {source}: {total_downloaded}")
        return total_downloaded
    
    def auto_collect_comprehensive(self):
        """Thu thập toàn diện từ nhiều nguồn"""
        print("🚀 Starting Comprehensive Auto Collection")
        print("=" * 60)
        
        # Keywords để tìm kiếm
        sleep_keywords = [
            "student sleeping",
            "tired student", 
            "sleepy classroom",
            "sleep desk",
            "student nap",
            "drowsy student",
            "exhausted student",
            "sleepy person",
            "sleep at work",
            "tired at desk",
            "napping office",
            "sleeping library"
        ]
        
        total_collected = 0
        
        # Thu thập từ Pexels
        pexels_count = self.collect_from_source("pexels", sleep_keywords[:8], max_per_query=12)
        total_collected += pexels_count
        
        # Thu thập từ Unsplash  
        unsplash_count = self.collect_from_source("unsplash", sleep_keywords[4:], max_per_query=10)
        total_collected += unsplash_count
        
        print(f"\n🎉 COLLECTION COMPLETE!")
        print(f"📊 Total images collected: {total_collected}")
        print(f"  - Pexels: {pexels_count}")
        print(f"  - Unsplash: {unsplash_count}")
        
        return total_collected
    
    def copy_to_data_raw(self):
        """Sao chép tất cả ảnh đã thu thập về data_raw"""
        print(f"\n📁 Copying collected images to data_raw...")
        
        copied = 0
        
        # Copy từ tất cả thư mục nguồn
        for source_dir in [self.pexels_dir, self.unsplash_dir, self.pixabay_dir]:
            for img_file in source_dir.glob("*.jpg"):
                dest = self.base_dir / f"auto_{img_file.name}"
                
                if not dest.exists():
                    import shutil
                    shutil.copy2(img_file, dest)
                    copied += 1
        
        print(f"✅ Copied {copied} new images to data_raw")
        return copied

def main():
    """Main function"""
    print("[AUTO] Data Collector for Sleepy Detection")
    print("Collecting images from multiple free sources...")
    print("=" * 60)
    
    collector = AutoDataCollector()
    
    # Thu thập tự động
    total_images = collector.auto_collect_comprehensive()
    
    if total_images > 0:
        # Sao chép về data_raw
        copied = collector.copy_to_data_raw()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"📥 Images collected: {total_images}")
        print(f"📁 Images copied to data_raw: {copied}")
        print(f"📍 Auto-collected images saved in: {collector.auto_collected_dir}")
        
        print(f"\n🚀 Next steps:")
        print(f"1. Run auto-labeling: python collect_data.py --auto-label")
        print(f"2. Check stats: python collect_data.py --stats")
        print(f"3. Train model: cd yolo-sleepy-allinone-final/tools && python train_pose.py")
        
    else:
        print("\n❌ No images collected. Please check internet connection and try again.")

if __name__ == "__main__":
    main()