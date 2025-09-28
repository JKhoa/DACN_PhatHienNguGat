#!/usr/bin/env python3
"""
Enhanced Multi-Person Display for Sleepy Detection App
Improvements for better multi-person tracking visualization
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def draw_enhanced_multi_person_display(
    frame: np.ndarray, 
    track_data: Dict[int, Dict], 
    sleep_status: Dict[int, str],
    sleep_start_time: Dict[int, float],
    max_sleep_duration: Dict[int, float],
    current_time: float
) -> np.ndarray:
    """
    Enhanced display for multi-person sleepy detection
    
    Args:
        frame: Input frame
        track_data: Dictionary with track_id -> {'bbox': [x1,y1,x2,y2], 'keypoints': numpy array}
        sleep_status: Dictionary with track_id -> status string
        sleep_start_time: Dictionary with track_id -> sleep start timestamp
        max_sleep_duration: Dictionary with track_id -> max sleep duration
        current_time: Current timestamp
    
    Returns:
        Enhanced frame with multi-person display
    """
    vis = frame.copy()
    height, width = vis.shape[:2]
    
    # Color scheme for different states
    colors = {
        "Bình thường": (0, 255, 0),     # Green
        "Ngủ gật": (0, 0, 255),         # Red  
        "Gục xuống bàn": (255, 0, 255)  # Magenta
    }
    
    # Enhanced panel for statistics
    panel_height = 120
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background
    
    # Count people by status
    status_counts = defaultdict(int)
    total_people = len(track_data)
    
    for tid, data in track_data.items():
        status = sleep_status.get(tid, "Bình thường")
        status_counts[status] += 1
        
        # Draw person tracking info
        bbox = data.get('bbox')
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate sleep duration if sleeping
            sleep_duration = 0
            if tid in sleep_start_time:
                sleep_duration = current_time - sleep_start_time[tid]
            
            # Enhanced label with more info
            max_duration = max_sleep_duration.get(tid, 0)
            if sleep_duration > 0:
                label = f"ID-{tid:02d}: {status} ({sleep_duration:.1f}s)"
            elif max_duration > 0:
                label = f"ID-{tid:02d}: {status} (Max: {max_duration:.1f}s)"
            else:
                label = f"ID-{tid:02d}: {status}"
            
            # Color based on status
            color = colors.get(status, (255, 255, 255))
            
            # Draw bounding box with thicker line for sleeping people
            thickness = 3 if status != "Bình thường" else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw enhanced label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_bg_y1 = max(0, y1 - label_size[1] - 10)
            label_bg_y2 = label_bg_y1 + label_size[1] + 8
            
            cv2.rectangle(vis, (x1, label_bg_y1), (x1 + label_size[0] + 10, label_bg_y2), color, -1)
            cv2.putText(vis, label, (x1 + 5, label_bg_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw sleep progress bar for sleeping people
            if sleep_duration > 0:
                bar_width = x2 - x1
                bar_height = 6
                bar_y = y2 + 5
                
                # Background bar
                cv2.rectangle(vis, (x1, bar_y), (x2, bar_y + bar_height), (100, 100, 100), -1)
                
                # Progress bar (fill based on duration, max at 30s)
                progress = min(sleep_duration / 30.0, 1.0)
                bar_fill_width = int(bar_width * progress)
                progress_color = (0, 255, 255) if sleep_duration < 10 else (0, 165, 255) if sleep_duration < 20 else (0, 0, 255)
                
                if bar_fill_width > 0:
                    cv2.rectangle(vis, (x1, bar_y), (x1 + bar_fill_width, bar_y + bar_height), progress_color, -1)
    
    # Enhanced statistics panel
    y_offset = 25
    
    # Title
    cv2.putText(panel, f"Multi-Person Sleepy Detection - Total: {total_people} people", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    # Status breakdown
    status_text = f"Awake: {status_counts['Bình thường']} | "
    status_text += f"Sleepy: {status_counts['Ngủ gật']} | "
    status_text += f"Head Down: {status_counts['Gục xuống bàn']}"
    
    cv2.putText(panel, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    
    # Alert status
    sleepy_total = status_counts['Ngủ gật'] + status_counts['Gục xuống bàn']
    if sleepy_total > 0:
        alert_text = f"⚠️ ALERT: {sleepy_total} person(s) detected sleeping!"
        alert_color = (0, 0, 255) if sleepy_total > 1 else (0, 165, 255)
        cv2.putText(panel, alert_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
    else:
        cv2.putText(panel, "✅ All people are awake", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Combine panel with main frame
    vis_enhanced = np.vstack([panel, vis])
    
    return vis_enhanced

def draw_person_id_circles(frame: np.ndarray, track_data: Dict[int, Dict], sleep_status: Dict[int, str]) -> np.ndarray:
    """
    Draw small circles with person IDs for easier tracking
    """
    vis = frame.copy()
    
    for tid, data in track_data.items():
        bbox = data.get('bbox')
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw ID circle at top-right of bounding box
            circle_x = x2 - 15
            circle_y = y1 + 15
            
            status = sleep_status.get(tid, "Bình thường")
            
            if status == "Bình thường":
                circle_color = (0, 255, 0)  # Green
            elif status == "Ngủ gật":
                circle_color = (0, 0, 255)  # Red
            else:  # "Gục xuống bàn"
                circle_color = (255, 0, 255)  # Magenta
            
            # Draw circle background
            cv2.circle(vis, (circle_x, circle_y), 12, circle_color, -1)
            cv2.circle(vis, (circle_x, circle_y), 12, (255, 255, 255), 2)
            
            # Draw ID number
            cv2.putText(vis, str(tid), (circle_x - 6, circle_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis

# Test the enhanced display functions
if __name__ == "__main__":
    print("🎨 Enhanced Multi-Person Display Module for Sleepy Detection")
    print("Features:")
    print("- Enhanced bounding boxes with color-coded status")
    print("- Sleep duration progress bars")
    print("- Multi-person statistics panel")
    print("- Person ID circles for easy tracking")
    print("- Alert system for sleeping people")
    print("✅ Module ready for integration!")