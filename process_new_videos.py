#!/usr/bin/env python3
"""
Process new videos and images for multi-person sleepy detection
Extract frames from videos and prepare dataset for training
"""

import cv2
import os
from pathlib import Path
import shutil

class VideoFrameProcessor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_raw_dir = self.base_dir / "data_raw"
        self.video_frames_dir = self.data_raw_dir / "video_frames"
        self.extracted_frames_dir = self.data_raw_dir / "extracted_frames"
        
        # Create extracted frames directory
        self.extracted_frames_dir.mkdir(exist_ok=True)
    
    def extract_frames_from_video(self, video_path, prefix, max_frames=20):
        """Extract frames from video with multi-person focus"""
        print(f"[EXTRACT] Processing video: {video_path.name}")
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {video_path}")
                return 0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"[INFO] Video stats: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")
            
            # Extract every 1 second for better coverage
            frame_interval = max(1, int(fps)) if fps > 0 else 30
            extracted = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or extracted >= max_frames:
                    break
                
                # Extract every interval frames
                if frame_count % frame_interval == 0:
                    frame_filename = f"{prefix}_frame_{extracted+1:03d}.jpg"
                    frame_path = self.extracted_frames_dir / frame_filename
                    
                    if cv2.imwrite(str(frame_path), frame):
                        print(f"[SAVE] {frame_filename}")
                        extracted += 1
                
                frame_count += 1
            
            cap.release()
            print(f"[OK] Extracted {extracted} frames from {video_path.name}")
            return extracted
            
        except Exception as e:
            print(f"[ERROR] Frame extraction failed: {e}")
            return 0
    
    def process_all_videos(self):
        """Process all videos in video_frames directory"""
        video_files = list(self.video_frames_dir.glob("*.mp4"))
        
        if not video_files:
            print("[INFO] No video files found in video_frames directory")
            return 0
        
        print(f"[INFO] Found {len(video_files)} video files to process")
        
        total_frames = 0
        for i, video_file in enumerate(video_files, 1):
            prefix = f"custom_video_{i:02d}"
            frames = self.extract_frames_from_video(video_file, prefix)
            total_frames += frames
        
        print(f"[FINAL] Total frames extracted: {total_frames}")
        return total_frames
    
    def copy_new_images_to_data_raw(self):
        """Copy newly extracted frames to data_raw for training"""
        copied = 0
        
        # Copy extracted frames
        for frame_file in self.extracted_frames_dir.glob("*.jpg"):
            dest_path = self.data_raw_dir / frame_file.name
            
            if not dest_path.exists():
                try:
                    shutil.copy2(frame_file, dest_path)
                    print(f"[COPY] {frame_file.name}")
                    copied += 1
                except Exception as e:
                    print(f"[ERROR] Failed to copy {frame_file.name}: {e}")
        
        print(f"[OK] Copied {copied} new frames to data_raw")
        return copied
    
    def count_current_dataset(self):
        """Count current images in data_raw"""
        jpg_files = list(self.data_raw_dir.glob("*.jpg"))
        print(f"[STATS] Current dataset: {len(jpg_files)} images")
        return len(jpg_files)

def main():
    """Main processing function"""
    print("=== NEW VIDEO & IMAGE PROCESSOR FOR MULTI-PERSON DETECTION ===")
    
    processor = VideoFrameProcessor()
    
    # Count current dataset
    initial_count = processor.count_current_dataset()
    
    # Process videos
    frames_extracted = processor.process_all_videos()
    
    # Copy frames to data_raw
    frames_copied = processor.copy_new_images_to_data_raw()
    
    # Final count
    final_count = processor.count_current_dataset()
    
    print(f"\n[FINAL] PROCESSING COMPLETE:")
    print(f"[FINAL] Frames extracted from videos: {frames_extracted}")
    print(f"[FINAL] Frames copied to dataset: {frames_copied}")
    print(f"[FINAL] Dataset growth: {initial_count} -> {final_count} (+{final_count-initial_count})")
    
    if frames_extracted > 0:
        print(f"\n[NEXT] Next steps:")
        print(f"  1. Run auto-labeling: python collect_data.py --auto-label")
        print(f"  2. Update YOLO config for multi-person detection")
        print(f"  3. Train models: YOLOv5, YOLOv8, YOLOv11")
        print(f"  4. Update application for multi-person detection")

if __name__ == "__main__":
    main()