#!/usr/bin/env python3
"""
Script tổng hợp thu thập dữ liệu tự động hoàn chỉnh
Chạy tất cả: Images + Videos + Processing + Auto-labeling
"""

import subprocess
import time
from pathlib import Path

def run_script(script_name, description):
    """Chạy script và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"[RUN] {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ["python", script_name], 
            capture_output=True, 
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("[WARNING] Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully")
            return True
        else:
            print(f"[ERROR] {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] Error running {script_name}: {e}")
        return False

def get_dataset_stats():
    """Lấy thống kê dataset hiện tại"""
    try:
        result = subprocess.run(
            ["python", "collect_data.py", "--stats"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            # Parse thống kê từ output
            output = result.stdout
            
            # Tìm số lượng ảnh
            import re
            total_match = re.search(r'Total images in data_raw: (\d+)', output)
            labels_match = re.search(r'Generated labels: (\d+)', output)
            
            total_images = int(total_match.group(1)) if total_match else 0
            total_labels = int(labels_match.group(1)) if labels_match else 0
            
            return total_images, total_labels
        
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    return 0, 0

def main():
    """Main comprehensive collection pipeline"""
    print("[AUTO] COMPREHENSIVE DATA COLLECTION PIPELINE")
    print("=" * 60)
    print("[TARGET] Collect 100+ new images and videos of sleeping students")
    print("[SOURCES] Pexels, Unsplash, Video extraction")
    print("[PROCESS] Download -> Extract -> Copy -> Auto-label -> Stats")
    print("=" * 60)
    
    # Lấy stats ban đầu
    initial_images, initial_labels = get_dataset_stats()
    print(f"[INITIAL] DATASET:")
    print(f"   - Images: {initial_images}")
    print(f"   - Labels: {initial_labels}")
    
    start_time = time.time()
    success_count = 0
    
    # 1. Thu thập ảnh tự động
    if run_script("auto_collect_data_fixed.py", "STEP 1: Auto Image Collection"):
        success_count += 1
    
    time.sleep(2)  # Brief pause
    
    # 2. Thu thập video tự động  
    if run_script("auto_collect_videos_fixed.py", "STEP 2: Auto Video Collection"):
        success_count += 1
    
    time.sleep(2)
    
    # 3. Copy video frames
    if run_script("auto_copy_frames_fixed.py", "STEP 3: Copy Video Frames"):
        success_count += 1
    
    time.sleep(2)
    
    # 4. Chạy auto-labeling
    print(f"\n{'='*60}")
    print(f"[RUN] STEP 4: Auto-Labeling")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ["python", "collect_data.py", "--auto-label"], 
            capture_output=True, 
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        if result.returncode == 0:
            print("[OK] Auto-labeling completed")
            success_count += 1
    except Exception as e:
        print(f"[ERROR] Auto-labeling error: {e}")
    
    # 5. Thống kê cuối cùng
    print(f"\n{'='*60}")
    print(f"[STATS] STEP 5: Final Statistics")
    print(f"{'='*60}")
    
    final_images, final_labels = get_dataset_stats()
    
    # Tính toán kết quả
    elapsed_time = time.time() - start_time
    images_added = final_images - initial_images
    labels_added = final_labels - initial_labels
    
    print(f"\n[FINAL] COMPREHENSIVE COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"[TIME] Total time: {elapsed_time/60:.1f} minutes")
    print(f"[RESULTS] Summary:")
    print(f"   - Steps completed: {success_count}/5")
    print(f"   - Images added: {images_added}")
    print(f"   - Labels added: {labels_added}")
    print(f"   - Final dataset: {final_images} images, {final_labels} labels")
    print(f"   - Progress to goal: {final_images}/400 ({final_images/400*100:.1f}%)")
    
    if success_count >= 4:
        print(f"\n[SUCCESS] Dataset successfully expanded")
        print(f"[NEXT] Next steps:")
        print(f"   1. Review new images in data_raw/")
        print(f"   2. Train model: cd yolo-sleepy-allinone-final/tools && python train_pose.py")
        print(f"   3. Test application: python standalone_app.py")
    else:
        print(f"\n[WARNING] PARTIAL SUCCESS - Some steps failed")
        print(f"[TIP] Try running individual scripts manually if needed")

if __name__ == "__main__":
    main()