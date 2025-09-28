#!/usr/bin/env python3
"""
Training Monitor cho YOLOv5 50 epochs
Real-time tracking của training progress
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def monitor_training_progress():
    """Monitor real-time training progress"""
    
    base_dir = Path(__file__).parent.parent
    project_dir = base_dir / "runs" / "pose-train" / "sleepy_pose_v5n_50ep_optimal"
    
    print("🔍 YOLOv5 TRAINING MONITOR")
    print("=" * 60)
    print(f"📁 Monitoring: {project_dir}")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    last_epoch = 0
    start_time = time.time()
    
    while True:
        try:
            # Check if training is running
            results_csv = project_dir / "results.csv"
            
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                current_epoch = len(df)
                
                if current_epoch > last_epoch:
                    last_row = df.iloc[-1]
                    
                    # Progress info
                    elapsed = time.time() - start_time
                    elapsed_hours = elapsed / 3600
                    epochs_per_hour = current_epoch / elapsed_hours if elapsed_hours > 0 else 0
                    remaining_epochs = 50 - current_epoch
                    eta_hours = remaining_epochs / epochs_per_hour if epochs_per_hour > 0 else 0
                    
                    # Metrics
                    box_loss = last_row.get('train/box_loss', 'N/A')
                    pose_loss = last_row.get('train/pose_loss', 'N/A')
                    val_box_loss = last_row.get('val/box_loss', 'N/A')
                    val_pose_loss = last_row.get('val/pose_loss', 'N/A')
                    
                    print(f"\n⚡ EPOCH {current_epoch}/50 COMPLETED!")
                    print(f"📊 Train - Box: {box_loss:.4f}, Pose: {pose_loss:.4f}")
                    print(f"🧪 Val - Box: {val_box_loss:.4f}, Pose: {val_pose_loss:.4f}")
                    print(f"⏱️  Elapsed: {elapsed_hours:.2f}h | ETA: {eta_hours:.2f}h")
                    print(f"📈 Speed: {epochs_per_hour:.1f} epochs/hour")
                    print("-" * 60)
                    
                    last_epoch = current_epoch
                    
                    # Check if training completed
                    if current_epoch >= 50:
                        print("🎉 TRAINING COMPLETED!")
                        break
            else:
                print(f"⏳ Waiting for training to start... ({datetime.now().strftime('%H:%M:%S')})")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⚠️ Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(10)
    
    # Final summary
    results_csv = project_dir / "results.csv"
    if results_csv.exists():
        print("\n📋 FINAL SUMMARY")
        print("=" * 60)
        df = pd.read_csv(results_csv)
        final_row = df.iloc[-1]
        
        print(f"🎯 Final Epoch: {len(df)}/50")
        print(f"📦 Final Box Loss: {final_row.get('train/box_loss', 'N/A'):.4f}")
        print(f"🤸 Final Pose Loss: {final_row.get('train/pose_loss', 'N/A'):.4f}")
        print(f"⏱️  Total Time: {(time.time() - start_time)/3600:.2f} hours")
        
        # Check for model files
        weights_dir = project_dir / "weights"
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024*1024)
                print(f"💾 Best Model: {size_mb:.1f}MB")
            
            if last_pt.exists():
                size_mb = last_pt.stat().st_size / (1024*1024)
                print(f"💾 Last Model: {size_mb:.1f}MB")
    
    print("🏁 MONITOR FINISHED!")

if __name__ == "__main__":
    monitor_training_progress()