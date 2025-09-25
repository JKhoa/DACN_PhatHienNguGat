#!/usr/bin/env python3
"""
Script hỗ trợ thu thập và xử lý dữ liệu từ các nguồn miễn phí
Sử dụng: python collect_data.py [options]
"""

import os
import cv2
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse
import shutil

class DataCollector:
    def __init__(self, base_dir="data_raw"):
        self.base_dir = Path(base_dir)
        self.pexels_dir = self.base_dir / "pexels_images"
        self.unsplash_dir = self.base_dir / "unsplash_images" 
        self.video_frames_dir = self.base_dir / "video_frames"
        
        # Tạo thư mục nếu chưa có
        for dir_path in [self.pexels_dir, self.unsplash_dir, self.video_frames_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_image(self, url, filename, timeout=30):
        """Tải ảnh từ URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, stream=True, headers=headers, timeout=timeout)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"HTTP {response.status_code} for {url}")
                
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        return False
    
    def extract_frames_from_video(self, video_path, output_dir, fps=2, max_frames=50):
        """Trích xuất frame từ video"""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return 0
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return 0
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps))
        
        # Tạo tên file dựa trên tên video
        video_name = Path(video_path).stem
        frame_count = 0
        saved_count = 0
        
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                filename = output_dir / f"{video_name}_frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved_count += 1
                print(f"Saved {video_name} frame {saved_count}/{max_frames}")
                
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames from {video_path}")
        return saved_count
    
    def process_all_videos(self, video_dir, fps=2, max_frames_per_video=25):
        """Xử lý tất cả video trong thư mục"""
        video_dir = Path(video_dir)
        if not video_dir.exists():
            print(f"Video directory not found: {video_dir}")
            return 0
            
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            
        if not video_files:
            print(f"No video files found in {video_dir}")
            return 0
            
        print(f"🎥 Found {len(video_files)} video files")
        total_frames = 0
        
        for video_file in video_files:
            print(f"\n📹 Processing: {video_file.name}")
            frames_extracted = self.extract_frames_from_video(
                video_file, 
                self.video_frames_dir, 
                fps=fps, 
                max_frames=max_frames_per_video
            )
            total_frames += frames_extracted
            
        print(f"\n[OK] Total frames extracted: {total_frames} from {len(video_files)} videos")
        return total_frames
    
    def copy_to_data_raw(self):
        """Sao chép tất cả ảnh về data_raw chính"""
        total_copied = 0
        
        # Sao chép từ downloaded_images (từ download_images.py)
        downloaded_dir = Path("data_raw/downloaded_images")
        if downloaded_dir.exists():
            for img_file in downloaded_dir.glob("*.jpg"):
                dest = self.base_dir / f"downloaded_{img_file.name}"
                if not dest.exists():  # Chỉ copy nếu chưa có
                    shutil.copy2(img_file, dest)
                    total_copied += 1
        
        # Sao chép từ pexels_images
        for img_file in self.pexels_dir.glob("*.jpg"):
            dest = self.base_dir / f"pexels_{img_file.name}"
            if not dest.exists():
                shutil.copy2(img_file, dest)
                total_copied += 1
            
        # Sao chép từ unsplash_images  
        for img_file in self.unsplash_dir.glob("*.jpg"):
            dest = self.base_dir / f"unsplash_{img_file.name}"
            if not dest.exists():
                shutil.copy2(img_file, dest)
                total_copied += 1
            
        # Sao chép từ video_frames
        for img_file in self.video_frames_dir.glob("*.jpg"):
            dest = self.base_dir / f"video_{img_file.name}"
            if not dest.exists():
                shutil.copy2(img_file, dest)
                total_copied += 1
            
        print(f"Copied {total_copied} new images to {self.base_dir}")
        return total_copied
    
    def download_from_urls(self):
        """Chạy script download_images.py để tải hình ảnh"""
        script_path = Path("download_images.py")
        
        if not script_path.exists():
            print(f"Download script not found: {script_path}")
            print("Please create download_images.py with your image URLs")
            return False
            
        import subprocess
        try:
            result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
            print("Download output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            # Sao chép ảnh đã tải về data_raw
            downloaded_dir = Path("data_raw/downloaded_images")
            if downloaded_dir.exists():
                copied = 0
                for img_file in downloaded_dir.glob("*.jpg"):
                    dest = self.base_dir / f"downloaded_{img_file.name}"
                    shutil.copy2(img_file, dest)
                    copied += 1
                print(f"Copied {copied} downloaded images to {self.base_dir}")
                
            return result.returncode == 0
        except Exception as e:
            print(f"Error running download script: {e}")
            return False
    
    def run_auto_labeling(self):
        """Chạy auto-labeling script"""
        script_path = Path("yolo-sleepy-allinone-final/tools/auto_label_pose.py")
        
        if not script_path.exists():
            print(f"Auto-label script not found at {script_path}")
            return False
            
        import subprocess
        cmd = [
            "python", str(script_path),
            "--source", str(self.base_dir),
            "--out", "yolo-sleepy-allinone-final/datasets/sleepy_pose",
            "--val-ratio", "0.2"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print("Auto-labeling output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running auto-labeling: {e}")
            return False
    
    def report_stats(self):
        """Báo cáo thống kê dữ liệu"""
        # Đếm ảnh theo nguồn
        total_images = len(list(self.base_dir.glob("*.jpg")))
        original_count = len([f for f in self.base_dir.glob("cap_*.jpg")])
        downloaded_count = len([f for f in self.base_dir.glob("downloaded_*.jpg")])
        pexels_count = len([f for f in self.base_dir.glob("pexels_*.jpg")])
        unsplash_count = len([f for f in self.base_dir.glob("unsplash_*.jpg")])
        video_count = len([f for f in self.base_dir.glob("video_*.jpg")])
        
        # Ảnh trong thư mục con
        sub_pexels = len(list(self.pexels_dir.glob("*.jpg")))
        sub_unsplash = len(list(self.unsplash_dir.glob("*.jpg")))
        sub_video = len(list(self.video_frames_dir.glob("*.jpg")))
        
        # Ảnh đã tải xuống từ download_images.py
        downloaded_dir = Path("data_raw/downloaded_images")
        raw_downloaded = len(list(downloaded_dir.glob("*.jpg"))) if downloaded_dir.exists() else 0
        
        print(f"\n[STATS] === Dataset Statistics ===")
        print(f"Total images in data_raw: {total_images}")
        print(f"  - Original (cap_*): {original_count}")
        print(f"  - Downloaded URLs: {downloaded_count}")
        print(f"  - Pexels: {pexels_count}")
        print(f"  - Unsplash: {unsplash_count}")
        print(f"  - Video frames: {video_count}")
        
        print(f"\nImages in subdirectories:")
        print(f"  - pexels_images/: {sub_pexels}")
        print(f"  - unsplash_images/: {sub_unsplash}")
        print(f"  - video_frames/: {sub_video}")
        print(f"  - downloaded_images/: {raw_downloaded}")
        
        # Kiểm tra labels nếu có
        label_dir = Path("yolo-sleepy-allinone-final/datasets/sleepy_pose/train/labels")
        if label_dir.exists():
            label_count = len(list(label_dir.glob("*.txt")))
            print(f"\n[LABELS] Generated labels: {label_count}")
            
        val_label_dir = Path("yolo-sleepy-allinone-final/datasets/sleepy_pose/val/labels")
        if val_label_dir.exists():
            val_label_count = len(list(val_label_dir.glob("*.txt")))
            print(f"Validation labels: {val_label_count}")
            
        print(f"\n📈 Progress towards 300-400 images goal: {total_images}/300-400")

def main():
    parser = argparse.ArgumentParser(description="Data collection and processing tool")
    parser.add_argument("--download", action="store_true", help="Download images from URLs")
    parser.add_argument("--download-videos", action="store_true", help="Download videos using download_pexels_videos.py")
    parser.add_argument("--video", help="Path to video file for frame extraction")
    parser.add_argument("--video-dir", help="Path to directory containing videos")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum frames to extract per video")
    parser.add_argument("--copy", action="store_true", help="Copy images to data_raw")
    parser.add_argument("--auto-label", action="store_true", help="Run auto-labeling")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline: download → videos → copy → auto-label → stats")
    parser.add_argument("--pexels-pipeline", action="store_true", help="Complete Pexels pipeline: download images + videos + process")
    
    args = parser.parse_args()
    
    collector = DataCollector()
    
    if args.download:
        print("📸 Downloading images from URLs...")
        collector.download_from_urls()
    
    if args.download_videos:
        print("🎥 Downloading videos from Pexels...")
        import subprocess
        try:
            result = subprocess.run(["python", "download_pexels_videos.py"], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Error running video downloader: {e}")
    
    if args.video:
        print(f"🎬 Extracting frames from: {args.video}")
        collector.extract_frames_from_video(
            args.video, 
            collector.video_frames_dir,
            fps=args.fps,
            max_frames=args.max_frames
        )
    
    if args.video_dir:
        print(f"🎬 Processing all videos in: {args.video_dir}")
        collector.process_all_videos(
            args.video_dir,
            fps=args.fps,
            max_frames_per_video=args.max_frames
        )
    
    if args.copy:
        print("[COPY] Copying images to data_raw...")
        collector.copy_to_data_raw()
    
    if args.auto_label:
        print("[LABEL] Running auto-labeling...")
        collector.run_auto_labeling()
        
    if args.pexels_pipeline:
        print("[RUN] Running complete Pexels data collection pipeline...")
        
        print("\n1️⃣ Downloading Pexels images...")
        collector.download_from_urls()
        
        print("\n2️⃣ Downloading Pexels videos...")
        import subprocess
        try:
            result = subprocess.run(["python", "download_pexels_videos.py"], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Error downloading videos: {e}")
        
        print("\n3️⃣ Extracting frames from videos...")
        video_dir = Path("data_raw/pexels_videos")
        if video_dir.exists():
            collector.process_all_videos(video_dir, fps=args.fps, max_frames_per_video=20)
        
        print("\n4️⃣ Copying all images to data_raw...")
        collector.copy_to_data_raw()
        
        print("\n5️⃣ Running auto-labeling...")
        collector.run_auto_labeling()
        
        print("\n6️⃣ Final statistics...")
        collector.report_stats()
        
        print("\n🎉 Pexels pipeline completed!")
        
    if args.full_pipeline:
        print("[RUN] Running full data collection pipeline...")
        print("\n1️⃣ Downloading images...")
        collector.download_from_urls()
        
        print("\n2️⃣ Copying images to data_raw...")
        collector.copy_to_data_raw()
        
        print("\n3️⃣ Running auto-labeling...")
        collector.run_auto_labeling()
        
        print("\n4️⃣ Final statistics...")
        collector.report_stats()
        
        print("\n[OK] Full pipeline completed!")
    
    if args.stats or not any([args.download, args.download_videos, args.video, args.video_dir, args.copy, args.auto_label, args.full_pipeline, args.pexels_pipeline]):
        collector.report_stats()

if __name__ == "__main__":
    main()