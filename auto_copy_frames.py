#!/usr/bin/env python3
"""
Helper script để copy video frames vào data_raw
"""

import shutil
from pathlib import Path

def copy_video_frames():
    """Copy video frames từ auto_video_frames vào data_raw"""
    frames_dir = Path("data_raw/auto_video_frames")
    data_raw = Path("data_raw")
    
    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        return 0
    
    copied = 0
    for frame_file in frames_dir.glob("*.jpg"):
        dest = data_raw / f"video_{frame_file.name}"
        if not dest.exists():
            shutil.copy2(frame_file, dest)
            copied += 1
            print(f"📁 Copied: {frame_file.name} → {dest.name}")
    
    print(f"\n✅ Copied {copied} video frames to data_raw")
    return copied

if __name__ == "__main__":
    copy_video_frames()