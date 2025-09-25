#!/usr/bin/env python3
"""
Auto Data Collector for Sleepy Detection Dataset
Fixed version without Unicode/emoji issues
"""

import os
import time
import requests
from pathlib import Path
import random
from urllib.parse import quote, urljoin
import shutil

class AutoDataCollector:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / "auto_collected_data"
        self.data_raw_dir = self.base_dir / "data_raw"
        self.max_images_per_keyword = 8
        self.max_images_per_source = 30
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.data_raw_dir.mkdir(exist_ok=True)
        
        # Headers to avoid bot detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def search_pexels(self, query, max_images=8):
        """Search Pexels for images"""
        print(f"[SEARCH] Pexels for: '{query}'")
        
        try:
            # Pexels search URL (free images)
            search_url = f"https://www.pexels.com/search/{quote(query)}/"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                # Extract image URLs from HTML
                import re
                
                # Find image URLs in the page
                img_patterns = [
                    r'https://images\.pexels\.com/photos/\d+/[^"?]+\.jpeg\?[^"]*',
                    r'https://images\.pexels\.com/photos/\d+/[^"?]+\.jpg\?[^"]*'
                ]
                
                image_urls = []
                for pattern in img_patterns:
                    urls = re.findall(pattern, response.text)
                    image_urls.extend(urls)
                
                # Remove duplicates and limit
                image_urls = list(set(image_urls))[:max_images]
                
                print(f"[OK] Found {len(image_urls)} images for '{query}'")
                return image_urls
                
        except Exception as e:
            print(f"[ERROR] Pexels search failed: {e}")
            
        return []

    def search_unsplash(self, query, max_images=8):
        """Search Unsplash for images"""
        print(f"[SEARCH] Unsplash for: '{query}'")
        
        try:
            # Unsplash search URL (free images)
            search_url = f"https://unsplash.com/s/photos/{quote(query)}"
            
            response = requests.get(search_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                import re
                
                # Find image URLs in the page
                img_patterns = [
                    r'https://images\.unsplash\.com/photo-[^"?]+\?[^"]*',
                    r'https://plus\.unsplash\.com/premium_photo-[^"?]+\?[^"]*'
                ]
                
                image_urls = []
                for pattern in img_patterns:
                    urls = re.findall(pattern, response.text)
                    image_urls.extend(urls)
                
                # Remove duplicates and limit
                image_urls = list(set(image_urls))[:max_images]
                
                print(f"[OK] Found {len(image_urls)} images for '{query}'")
                return image_urls
                
        except Exception as e:
            print(f"[ERROR] Unsplash search failed: {e}")
            
        return []

    def download_image(self, url, filepath):
        """Download a single image"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15, stream=True)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Check file size
                file_size = filepath.stat().st_size
                if file_size > 5000:  # At least 5KB
                    print(f"[OK] Downloaded: {filepath.name} ({file_size:,} bytes)")
                    return True
                else:
                    print(f"[SKIP] File too small: {filepath.name}")
                    filepath.unlink()  # Delete small file
                    return False
            else:
                print(f"[ERROR] HTTP {response.status_code}")
                
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            
        return False

    def collect_from_source(self, source, keywords):
        """Collect images from a specific source"""
        print(f"[TARGET] Collecting from {source.upper()}")
        
        source_dir = self.output_dir / source
        source_dir.mkdir(exist_ok=True)
        
        total_downloaded = 0
        
        for query in keywords:
            if source == "pexels":
                image_urls = self.search_pexels(query, self.max_images_per_keyword)
            elif source == "unsplash":
                image_urls = self.search_unsplash(query, self.max_images_per_keyword)
            else:
                print(f"[ERROR] Unknown source: {source}")
                continue
            
            if not image_urls:
                continue
                
            print(f"\n[QUERY] '{query}'")
            
            query_downloaded = 0
            for i, url in enumerate(image_urls):
                if total_downloaded >= self.max_images_per_source:
                    break
                    
                # Generate filename
                safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_query = safe_query.replace(' ', '_')
                filename = f"{source}_{safe_query}_{i+1:02d}.jpg"
                filepath = source_dir / filename
                
                if filepath.exists():
                    print(f"[SKIP] Already exists: {filename}")
                    continue
                
                if self.download_image(url, filepath):
                    query_downloaded += 1
                    total_downloaded += 1
                
                # Rate limiting
                time.sleep(random.uniform(0.5, 1.5))
            
            print(f"[STATS] Downloaded {query_downloaded} images for '{query}'")
            
            if total_downloaded >= self.max_images_per_source:
                break
        
        print(f"\n[OK] Total downloaded from {source}: {total_downloaded}")
        return total_downloaded

    def auto_collect_comprehensive(self):
        """Comprehensive automatic collection"""
        
        # Keywords for searching
        keywords = [
            "sleeping students", "student nap", "student tired",
            "classroom sleeping", "sleepy student", "student desk sleep",
            "student drowsy", "tired student class", "student fatigue",
            "sleeping lecture", "dozing student", "student exhausted"
        ]
        
        print(f"[INFO] Keywords: {len(keywords)}")
        print(f"[INFO] Target per keyword: {self.max_images_per_keyword}")
        print(f"[INFO] Max per source: {self.max_images_per_source}")
        print("=" * 50)
        
        # Collect from multiple sources
        sources = ["pexels", "unsplash"]
        total_collected = 0
        
        for source in sources:
            collected = self.collect_from_source(source, keywords)
            total_collected += collected
            
            # Pause between sources
            time.sleep(2)
        
        print(f"\n[STATS] Total images collected: {total_collected}")
        return total_collected

    def copy_to_data_raw(self):
        """Copy collected images to data_raw directory"""
        
        print(f"\n[COPY] Moving images to data_raw...")
        
        copied = 0
        
        # Process all collected images
        for source_dir in self.output_dir.iterdir():
            if source_dir.is_dir():
                for img_file in source_dir.glob("*.jpg"):
                    # Generate new filename for data_raw
                    new_name = f"auto_{img_file.stem}_{copied+1:03d}.jpg"
                    dest_path = self.data_raw_dir / new_name
                    
                    if not dest_path.exists():
                        shutil.copy2(img_file, dest_path)
                        copied += 1
                        print(f"[COPY] {new_name}")
        
        print(f"[OK] Copied {copied} new images to data_raw")
        return copied

def main():
    """Main function"""
    print("[AUTO] Data Collector for Sleepy Detection")
    print("Collecting images from multiple free sources...")
    print("=" * 60)
    
    collector = AutoDataCollector()
    
    # Auto collection
    total_images = collector.auto_collect_comprehensive()
    
    if total_images > 0:
        # Copy to data_raw
        copied = collector.copy_to_data_raw()
        
        print(f"\n[FINAL] RESULTS:")
        print(f"[FINAL] Total collected: {total_images}")
        print(f"[FINAL] Images copied to data_raw: {copied}")
        
        if copied > 0:
            print("\n[NEXT] Next steps:")
            print("  1. Run auto-labeling: python collect_data.py --auto-label")
            print("  2. Train model: cd yolo-sleepy-allinone-final/tools && python train_pose.py")
            print("  3. Test: python standalone_app.py")
    else:
        print("\n[ERROR] No images collected. Check internet connection.")

if __name__ == "__main__":
    main()