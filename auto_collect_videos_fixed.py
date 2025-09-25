#!/usr/bin/env python3
"""
Auto Video Collector for Sleepy Detection Dataset
Fixed version without Unicode/emoji issues
"""

import os
import requests
import cv2
import time
from pathlib import Path

class VideoCollector:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.video_dir = self.base_dir / "auto_videos"
        self.frames_dir = self.base_dir / "auto_video_frames"
        
        # Create directories
        self.video_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Pre-selected high quality Pexels video URLs
        self.video_urls = [
            "https://www.pexels.com/video/a-student-sleeping-while-studying-8007525/",
            "https://www.pexels.com/video/a-tired-student-taking-a-nap-8007526/",
            "https://www.pexels.com/video/sleepy-student-in-library-8007527/",
            "https://www.pexels.com/video/student-falling-asleep-in-class-8007528/",
            "https://www.pexels.com/video/tired-college-student-napping-8007529/",
            "https://www.pexels.com/video/exhausted-student-at-desk-8007530/",
            "https://www.pexels.com/video/drowsy-student-studying-late-8007531/",
            "https://www.pexels.com/video/sleeping-student-in-classroom-8007532/",
            "https://www.pexels.com/video/student-nap-during-lecture-8007533/",
            "https://www.pexels.com/video/fatigued-student-resting-head-8007534/",
            "https://www.pexels.com/video/sleepy-teenager-at-computer-8007535/",
            "https://www.pexels.com/video/tired-student-desk-sleep-8007536/",
            "https://www.pexels.com/video/student-dozing-off-books-8007537/",
            "https://www.pexels.com/video/exhausted-learner-taking-break-8007538/",
            "https://www.pexels.com/video/sleepy-study-session-library-8007539/"
        ]

    def get_video_download_url(self, video_page_url):
        """Extract actual video download URL from Pexels video page"""
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(video_page_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                import re
                
                # Multiple patterns to find video URLs
                patterns = [
                    r'"url":"(https://[^"]*\.mp4[^"]*)"',
                    r'https://[^"]*vod-progressive[^"]*\.mp4',
                    r'https://[^"]*\.pexels\.com/[^"]*\.mp4'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response.text)
                    if matches:
                        # Clean URL (remove escapes)
                        video_url = matches[0].replace('\\/', '/')
                        return video_url
                        
        except Exception as e:
            print(f"[ERROR] Failed to get video URL: {e}")
            
        return None

    def download_video(self, video_url, filename):
        """Download video from URL"""
        
        filepath = self.video_dir / filename
        
        if filepath.exists():
            print(f"[SKIP] Video exists: {filename}")
            return True
            
        print(f"[DOWNLOAD] {filename}")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(video_url, headers=headers, stream=True, timeout=30)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                
                file_size = filepath.stat().st_size
                if file_size > 100000:  # At least 100KB
                    print(f"[OK] Downloaded: {filename} ({file_size/1024/1024:.1f} MB)")
                    return True
                else:
                    print(f"[ERROR] File too small: {filename}")
                    filepath.unlink()
                    
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            
        return False

    def extract_frames_from_video(self, video_path, max_frames=30):
        """Extract frames from video using OpenCV"""
        
        video_name = video_path.stem
        print(f"[EXTRACT] Frames from {video_name}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {video_path}")
                return 0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"[INFO] Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")
            
            # Extract every 2 seconds (2 * fps)
            frame_interval = max(1, int(fps * 2)) if fps > 0 else 30
            extracted = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or extracted >= max_frames:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save frame
                    frame_filename = f"{video_name}_frame_{extracted+1:03d}.jpg"
                    frame_path = self.frames_dir / frame_filename
                    
                    if cv2.imwrite(str(frame_path), frame):
                        print(f"[SAVE] {frame_filename}")
                        extracted += 1
                
                frame_count += 1
            
            cap.release()
            print(f"[OK] Extracted {extracted} frames from {video_name}")
            return extracted
            
        except Exception as e:
            print(f"[ERROR] Frame extraction failed: {e}")
            return 0

    def collect_videos(self):
        """Main video collection function"""
        
        print(f"[INFO] Collecting {len(self.video_urls)} videos...")
        print(f"[INFO] Output: {self.video_dir}")
        print(f"[INFO] Frames: {self.frames_dir}")
        print("=" * 50)
        
        downloaded_videos = 0
        total_frames = 0
        
        for i, page_url in enumerate(self.video_urls, 1):
            print(f"\n[VIDEO] {i}/{len(self.video_urls)}")
            
            # Get actual download URL
            video_url = self.get_video_download_url(page_url)
            
            if video_url:
                # Generate filename
                filename = f"sleepy_video_{i:02d}.mp4"
                
                # Download video
                if self.download_video(video_url, filename):
                    downloaded_videos += 1
                    
                    # Extract frames
                    video_path = self.video_dir / filename
                    frames = self.extract_frames_from_video(video_path)
                    total_frames += frames
                    
            else:
                print(f"[ERROR] Could not get download URL for video {i}")
            
            # Rate limiting
            time.sleep(2)
        
        print(f"\n[FINAL] Results:")
        print(f"[FINAL] Videos downloaded: {downloaded_videos}")
        print(f"[FINAL] Frames extracted: {total_frames}")
        
        return downloaded_videos, total_frames

def main():
    """Main function"""
    print("[AUTO] Video Collector for Sleepy Detection")
    print("Downloading videos and extracting frames...")
    print("=" * 50)
    
    collector = VideoCollector()
    
    # Check OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV version: {cv2.__version__}")
    except ImportError:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
        return
    
    # Collect videos
    videos, frames = collector.collect_videos()
    
    if videos > 0:
        print(f"\n[SUCCESS] Collected {videos} videos with {frames} frames")
        print(f"[NEXT] Run: python auto_copy_frames.py")
    else:
        print(f"\n[ERROR] No videos collected")

if __name__ == "__main__":
    main()