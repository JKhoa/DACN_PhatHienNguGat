"""
🎨 Enhanced UX/UI Decorators for Sleepy Detection Apps
=====================================================

Decorator-based system to enhance existing applications without modifying source code.
Provides modern UI/UX improvements, data flow enhancements, and advanced features.

Author: Enhanced by AI Assistant
Date: September 30, 2025
"""

import functools
import time
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================================
# 📊 Data Models & Configuration
# ============================================================================

class AppMode(Enum):
    STANDALONE = "standalone"
    GUI = "gui"
    ENHANCED = "enhanced"
    WEB = "web"

@dataclass
class AppConfig:
    """Configuration class for enhanced app settings"""
    theme: str = "dark"
    language: str = "vi"
    auto_save: bool = True
    performance_mode: str = "balanced"  # fast, balanced, quality
    ui_scale: float = 1.0
    show_fps: bool = True
    show_confidence: bool = True
    enable_alerts: bool = True
    alert_sound: bool = True
    video_codec: str = "mp4v"
    output_dir: str = "outputs"
    log_level: str = "INFO"

@dataclass
class DetectionResult:
    """Enhanced detection result with metadata"""
    timestamp: float
    frame_id: int
    detections: List[Dict]
    fps: float
    processing_time: float
    confidence_avg: float
    sleepy_status: bool
    person_count: int
    metadata: Dict[str, Any]

# ============================================================================
# 🎯 Core Decorators
# ============================================================================

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor performance metrics"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Log performance
        logging.info(f"🚀 {func.__name__} executed in {execution_time:.3f}s")
        
        # Add performance data to result if possible
        if hasattr(result, '__dict__'):
            result.execution_time = execution_time
        
        return result
    return wrapper

def error_handler(fallback_result=None, show_dialog=True):
    """Decorator for enhanced error handling with UI feedback"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                logging.error(error_msg, exc_info=True)
                
                if show_dialog:
                    try:
                        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
                    except:
                        print(error_msg)  # Fallback to console
                
                return fallback_result
        return wrapper
    return decorator

def data_preprocessor(preprocessing_steps: List[str] = None):
    """Decorator to preprocess input data"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply preprocessing steps
            if preprocessing_steps:
                for step in preprocessing_steps:
                    if step == "normalize":
                        # Normalize image data if present
                        args = tuple(_normalize_image(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
                    elif step == "resize":
                        # Resize images to standard size
                        args = tuple(_resize_image(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
                    elif step == "enhance":
                        # Enhance image quality
                        args = tuple(_enhance_image(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def result_enhancer(enhance_options: Dict[str, bool] = None):
    """Decorator to enhance detection results"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if enhance_options:
                # Add timestamp
                if enhance_options.get("add_timestamp", True):
                    if hasattr(result, '__dict__'):
                        result.timestamp = time.time()
                
                # Add confidence statistics
                if enhance_options.get("add_stats", True):
                    result = _add_statistics(result)
                
                # Add visualization enhancements
                if enhance_options.get("enhance_viz", True):
                    result = _enhance_visualization(result)
            
            return result
        return wrapper
    return decorator

def ui_theme_adapter(theme: str = "dark"):
    """Decorator to apply UI theme enhancements"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply theme before function execution
            _apply_theme_settings(theme)
            
            result = func(*args, **kwargs)
            
            # Apply theme to result if it's a UI component
            if hasattr(result, 'configure'):
                _apply_theme_to_widget(result, theme)
            
            return result
        return wrapper
    return decorator

def auto_save_decorator(save_interval: int = 300):  # 5 minutes
    """Decorator to automatically save app state"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Setup auto-save timer
            def auto_save():
                while True:
                    time.sleep(save_interval)
                    try:
                        _save_app_state(args, kwargs)
                    except Exception as e:
                        logging.warning(f"Auto-save failed: {e}")
            
            # Start auto-save thread
            save_thread = threading.Thread(target=auto_save, daemon=True)
            save_thread.start()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# 🎨 UI Enhancement Decorators
# ============================================================================

def modern_ui_wrapper(title: str = "Enhanced Sleepy Detection", icon: str = None):
    """Decorator to create modern UI wrapper around existing apps"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create enhanced UI container
            enhanced_window = _create_modern_window(title, icon)
            
            # Add enhanced controls
            control_panel = _create_control_panel(enhanced_window)
            
            # Add status bar
            status_bar = _create_status_bar(enhanced_window)
            
            # Execute original function with enhanced context
            kwargs['enhanced_ui'] = {
                'window': enhanced_window,
                'controls': control_panel,
                'status': status_bar
            }
            
            result = func(*args, **kwargs)
            
            # Apply final UI enhancements
            _finalize_ui_enhancements(enhanced_window)
            
            return result
        return wrapper
    return decorator

def progress_tracker(show_progress=True):
    """Decorator to track and display progress"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if show_progress:
                progress_window = _create_progress_window(func.__name__)
                
                # Update progress during execution
                def update_progress(value):
                    if progress_window:
                        progress_window.update_progress(value)
                
                kwargs['progress_callback'] = update_progress
            
            result = func(*args, **kwargs)
            
            if show_progress and 'progress_window' in locals():
                progress_window.destroy()
            
            return result
        return wrapper
    return decorator

# ============================================================================
# 🔧 Helper Functions
# ============================================================================

def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image values to 0-1 range"""
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    return img

def _resize_image(img: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """Resize image to target size"""
    return cv2.resize(img, target_size)

def _enhance_image(img: np.ndarray) -> np.ndarray:
    """Apply image enhancement techniques"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return img

def _add_statistics(result):
    """Add statistical information to results"""
    # This would be implemented based on the specific result type
    return result

def _enhance_visualization(result):
    """Enhance visualization of results"""
    # This would add visual improvements to the detection results
    return result

def _apply_theme_settings(theme: str):
    """Apply global theme settings"""
    themes = {
        "dark": {
            "bg": "#2b2b2b",
            "fg": "#ffffff",
            "select_bg": "#404040",
            "select_fg": "#00ff00"
        },
        "light": {
            "bg": "#ffffff",
            "fg": "#000000",
            "select_bg": "#0078d4",
            "select_fg": "#ffffff"
        },
        "blue": {
            "bg": "#1e3a8a",
            "fg": "#ffffff",
            "select_bg": "#3b82f6",
            "select_fg": "#ffffff"
        }
    }
    
    # Apply theme (this would be expanded for full implementation)
    current_theme = themes.get(theme, themes["dark"])
    return current_theme

def _apply_theme_to_widget(widget, theme: str):
    """Apply theme to specific widget"""
    theme_settings = _apply_theme_settings(theme)
    try:
        widget.configure(
            bg=theme_settings["bg"],
            fg=theme_settings["fg"]
        )
    except:
        pass  # Not all widgets support all options

def _save_app_state(args, kwargs):
    """Save current application state"""
    state = {
        "timestamp": datetime.now().isoformat(),
        "args": str(args),  # Simplified serialization
        "kwargs_keys": list(kwargs.keys())
    }
    
    # Save to file
    save_path = Path("outputs/app_state.json")
    save_path.parent.mkdir(exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(state, f, indent=2)

def _create_modern_window(title: str, icon: str = None):
    """Create modern-styled window"""
    window = tk.Tk()
    window.title(f"🎯 {title}")
    window.geometry("1200x800")
    
    # Modern styling
    style = ttk.Style()
    style.theme_use('clam')
    
    return window

def _create_control_panel(parent):
    """Create enhanced control panel"""
    frame = ttk.LabelFrame(parent, text="🎛️ Enhanced Controls", padding=10)
    frame.pack(fill="x", padx=10, pady=5)
    
    # Add various controls
    ttk.Button(frame, text="🎨 Change Theme").pack(side="left", padx=5)
    ttk.Button(frame, text="⚙️ Settings").pack(side="left", padx=5)
    ttk.Button(frame, text="💾 Export Results").pack(side="left", padx=5)
    ttk.Button(frame, text="📊 Statistics").pack(side="left", padx=5)
    
    return frame

def _create_status_bar(parent):
    """Create enhanced status bar"""
    frame = ttk.Frame(parent)
    frame.pack(fill="x", side="bottom")
    
    # Status labels
    ttk.Label(frame, text="🟢 Ready").pack(side="left", padx=5)
    ttk.Label(frame, text="FPS: 0").pack(side="right", padx=5)
    ttk.Label(frame, text="Detections: 0").pack(side="right", padx=5)
    
    return frame

def _finalize_ui_enhancements(window):
    """Apply final UI enhancements"""
    # Add modern styling, animations, etc.
    pass

def _create_progress_window(task_name: str):
    """Create progress tracking window"""
    progress_win = tk.Toplevel()
    progress_win.title(f"Processing: {task_name}")
    progress_win.geometry("400x100")
    
    ttk.Label(progress_win, text=f"Processing {task_name}...").pack(pady=10)
    progress_bar = ttk.Progressbar(progress_win, mode='indeterminate')
    progress_bar.pack(fill="x", padx=20, pady=10)
    progress_bar.start()
    
    class ProgressWindow:
        def __init__(self, window, bar):
            self.window = window
            self.bar = bar
        
        def update_progress(self, value):
            self.bar['value'] = value
        
        def destroy(self):
            self.window.destroy()
    
    return ProgressWindow(progress_win, progress_bar)

# ============================================================================
# 🚀 Enhanced App Class
# ============================================================================

class EnhancedAppWrapper:
    """Main wrapper class for enhanced applications"""
    
    def __init__(self, config: AppConfig = None):
        self.config = config or AppConfig()
        self.setup_logging()
        self.results_queue = queue.Queue()
        self.is_running = False
    
    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/enhanced_app.log'),
                logging.StreamHandler()
            ]
        )
    
    @performance_monitor
    @error_handler(show_dialog=True)
    def wrap_existing_app(self, app_module, app_function, *args, **kwargs):
        """Wrap existing app with enhancements"""
        logging.info(f"🚀 Launching enhanced version of {app_module}.{app_function}")
        
        # Import the module dynamically
        import importlib
        module = importlib.import_module(app_module)
        func = getattr(module, app_function)
        
        # Apply enhancements
        enhanced_func = self._apply_all_decorators(func)
        
        # Execute with enhanced context
        return enhanced_func(*args, **kwargs)
    
    def _apply_all_decorators(self, func):
        """Apply all enhancement decorators to a function"""
        # Chain decorators
        enhanced_func = performance_monitor(func)
        enhanced_func = error_handler()(enhanced_func)
        enhanced_func = data_preprocessor(["normalize", "enhance"])(enhanced_func)
        enhanced_func = result_enhancer({
            "add_timestamp": True,
            "add_stats": True,
            "enhance_viz": True
        })(enhanced_func)
        enhanced_func = ui_theme_adapter(self.config.theme)(enhanced_func)
        enhanced_func = auto_save_decorator()(enhanced_func)
        enhanced_func = modern_ui_wrapper(
            "🎯 Enhanced Sleepy Detection System"
        )(enhanced_func)
        
        return enhanced_func
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self, path: str = "outputs/config.json"):
        """Save configuration to file"""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def load_config(self, path: str = "outputs/config.json"):
        """Load configuration from file"""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
                self.config = AppConfig(**config_dict)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {path}")

if __name__ == "__main__":
    # Example usage
    print("🎨 Enhanced UX/UI Decorators System")
    print("=" * 50)
    print("This module provides decorators to enhance existing apps without modifying source code.")
    print("\nKey Features:")
    print("✅ Performance monitoring")
    print("✅ Enhanced error handling")
    print("✅ Data preprocessing")
    print("✅ Result enhancement")
    print("✅ Modern UI themes")
    print("✅ Auto-save functionality")
    print("✅ Progress tracking")