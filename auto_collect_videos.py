#!/usr/bin/env python3
"""
Script thu thập video tự động từ các nguồn miễn phí
"""

import os
import re
import requests
import time
import cv2
from pathlib import Path
import urllib.parse

class VideoCollector:
    def __init__(self):
        self.base_dir = Path("data_raw")
        self.video_dir = self.base_dir / "auto_videos"
        self.frames_dir = self.base_dir / "auto_video_frames"
        
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
    
    def search_pexels_videos(self, query, max_results=10):
        """Tìm kiếm video từ Pexels"""
        print(f"🎬 Searching Pexels videos for: '{query}'")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # URLs video cụ thể từ Pexels về sleeping students
        predefined_videos = [
            "https://www.pexels.com/video/student-sleeping-in-classroom-6672265/",
            "https://www.pexels.com/video/exhausted-student-sleeping-on-a-bench-6083402/", 
            "https://www.pexels.com/video/a-student-sleeping-on-a-desk-9489688/",
            "https://www.pexels.com/video/student-sleeping-in-classroom-6672269/",
            "https://www.pexels.com/video/woman-sleeping-over-her-studies-8086497/",
            "https://www.pexels.com/video/a-young-man-sleeping-on-his-desk-7949756/",
            "https://www.pexels.com/video/a-student-sleeping-while-studying-8569881/",
            "https://www.pexels.com/video/student-sleeping-at-library-7949795/",
            "https://www.pexels.com/video/a-boy-sleeping-on-books-in-classroom-8086400/",
            "https://www.pexels.com/video/a-girl-sleeping-on-her-desk-8469803/",
            "https://www.pexels.com/video/a-female-student-sleeping-during-class-9523085/",
            "https://www.pexels.com/video/a-student-yawning-in-classroom-8469702/",
            "https://www.pexels.com/video/young-woman-sleeping-in-classroom-9523101/",
            "https://www.pexels.com/video/student-resting-on-his-books-8469721/",
            "https://www.pexels.com/video/a-woman-sleeping-on-the-table-8569920/"
        ]
        
        return predefined_videos[:max_results]
    
    def get_video_download_url(self, pexels_video_url):
        """Lấy direct download URL từ Pexels video page"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(pexels_video_url, headers=headers)
            if response.status_code == 200:
                content = response.text
                
                # Tìm download URL trong HTML
                patterns = [
                    r'"download_url":"([^"]*\.mp4[^"]*)"',
                    r'"link":"([^"]*\.mp4[^"]*)"', 
                    r'href="([^"]*\.mp4[^"]*)"[^>]*download',
                    r'"url":"([^"]*\.mp4[^"]*)"'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Decode escaped characters
                        download_url = matches[0].replace('\\/', '/')
                        if download_url.startswith('http'):
                            return download_url
                
                # Alternative: look for video sources
                video_pattern = r'<source[^>]*src="([^"]*\.mp4[^"]*)"'
                matches = re.findall(video_pattern, content)
                if matches:
                    return matches[0]
                    
        except Exception as e:
            print(f"❌ Error getting download URL: {e}")
        
        return None
    
    def download_video(self, url, filepath):
        """Download video với progress tracking"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            print(f"📥 Downloading: {filepath.name}")
            
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress indicator
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r⏳ Progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
                
                print()  # New line after progress
                
                # Check file size
                file_size = filepath.stat().st_size
                if file_size > 500000:  # > 500KB
                    print(f"✅ Downloaded: {filepath.name} ({file_size:,} bytes)")
                    return True
                else:
                    print(f"❌ File too small: {filepath.name}")
                    filepath.unlink()
            else:
                print(f"❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Download error: {e}")
        
        return False
    
    def extract_frames_from_video(self, video_path, fps=2, max_frames=30):
        """Trích xuất frames từ video"""
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return 0
        
        print(f"🎞️ Extracting frames from: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return 0
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        video_name = video_path.stem
        frame_count = 0
        saved_count = 0
        
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = self.frames_dir / f"{video_name}_frame_{saved_count:03d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
                print(f"📸 Saved frame {saved_count}/{max_frames}")
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {saved_count} frames from {video_path.name}")
        return saved_count
    
    def collect_videos_auto(self):
        """Thu thập video tự động"""
        print("🎬 Starting Auto Video Collection")
        print("=" * 50)
        
        queries = [
            "student sleeping",
            "tired student classroom", 
            "sleepy desk",
            "student nap"
        ]
        
        total_videos = 0
        total_frames = 0
        
        for query in queries:
            video_urls = self.search_pexels_videos(query, max_results=4)
            
            print(f"\n🔍 Processing {len(video_urls)} videos for '{query}'")
            
            for i, video_url in enumerate(video_urls):
                # Extract video ID
                video_id_match = re.search(r'-(\d+)/?$', video_url)
                if not video_id_match:
                    continue
                
                video_id = video_id_match.group(1)
                video_filename = self.video_dir / f"pexels_{video_id}.mp4"
                
                # Skip if already downloaded
                if video_filename.exists():
                    print(f"⏭️ Already exists: {video_filename.name}")
                    continue
                
                print(f"\n[{i+1}/{len(video_urls)}] Processing video ID: {video_id}")
                
                # Get download URL
                download_url = self.get_video_download_url(video_url)
                if not download_url:
                    print(f"❌ Cannot get download URL for {video_url}")
                    continue
                
                # Download video
                if self.download_video(download_url, video_filename):
                    total_videos += 1
                    
                    # Extract frames
                    frames_extracted = self.extract_frames_from_video(video_filename, fps=2, max_frames=25)
                    total_frames += frames_extracted
                
                # Rate limiting
                time.sleep(3)
            
            # Delay between queries
            time.sleep(5)
        
        return total_videos, total_frames

def main():
    """Main function"""
    print("🎬 Auto Video Collector for Sleepy Detection")
    print("=" * 50)
    
    collector = VideoCollector()
    
    try:
        videos_count, frames_count = collector.collect_videos_auto()
        
        print(f"\n🎉 VIDEO COLLECTION COMPLETE!")
        print(f"📊 Results:")
        print(f"  - Videos downloaded: {videos_count}")
        print(f"  - Frames extracted: {frames_count}")
        print(f"📁 Videos saved in: {collector.video_dir}")
        print(f"📁 Frames saved in: {collector.frames_dir}")
        
        if frames_count > 0:
            print(f"\n🚀 Next steps:")
            print(f"1. Copy frames to data_raw:")
            print(f"   python auto_copy_frames.py")
            print(f"2. Run auto-labeling: python collect_data.py --auto-label")
            print(f"3. Check stats: python collect_data.py --stats")
        
    except Exception as e:
        print(f"❌ Error in video collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()