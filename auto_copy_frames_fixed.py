#!/usr/bin/env python3
"""
Copy video frames to data_raw directory
Fixed version without Unicode issues
"""

from pathlib import Path
import shutil

def copy_video_frames():
    """Copy extracted video frames to data_raw"""
    
    base_dir = Path(__file__).parent
    frames_dir = base_dir / "auto_video_frames"
    data_raw_dir = base_dir / "data_raw"
    
    # Create data_raw if not exists
    data_raw_dir.mkdir(exist_ok=True)
    
    if not frames_dir.exists():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return 0
    
    # Get all frame files
    frame_files = list(frames_dir.glob("*.jpg"))
    
    if not frame_files:
        print(f"[INFO] No frames found in {frames_dir}")
        return 0
    
    print(f"[INFO] Found {len(frame_files)} frames to copy")
    
    copied = 0
    
    for frame_file in frame_files:
        # Generate new name for data_raw
        new_name = f"video_frame_{copied+1:03d}.jpg"
        dest_path = data_raw_dir / new_name
        
        # Copy if not exists
        if not dest_path.exists():
            try:
                shutil.copy2(frame_file, dest_path)
                print(f"[COPY] {new_name}")
                copied += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy {frame_file.name}: {e}")
        else:
            print(f"[SKIP] {new_name} already exists")
    
    print(f"[OK] Copied {copied} frames to data_raw")
    return copied

if __name__ == "__main__":
    copy_video_frames()