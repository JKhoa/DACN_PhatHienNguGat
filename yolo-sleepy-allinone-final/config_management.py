"""
⚙️ Enhanced Configuration Management System
==========================================

Advanced configuration system with GUI for the enhanced sleepy detection apps.
Provides real-time configuration updates, profile management, and settings persistence.

Features:
- 🎛️ GUI configuration interface
- 📁 Profile management (save/load/share)
- 🔄 Real-time updates without restart
- 🎨 Theme and UI customization
- 📊 Performance tuning
- 💾 Auto-save and backup
- 🔧 Advanced developer settings

Author: Enhanced by AI Assistant
Date: September 30, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import threading

from enhanced_app_decorators import AppConfig

@dataclass
class UIConfig:
    """UI-specific configuration"""
    window_width: int = 1200
    window_height: int = 800
    font_family: str = "Segoe UI"
    font_size: int = 10
    icon_size: int = 16
    animation_enabled: bool = True
    transparency: float = 1.0
    always_on_top: bool = False
    remember_position: bool = True
    last_x: int = 100
    last_y: int = 100

@dataclass
class DetectionConfig:
    """Detection-specific configuration"""
    model_weights: str = "yolo11n-pose.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 5
    enable_tracking: bool = True
    tracking_threshold: float = 0.3
    sleepy_sensitivity: float = 0.8
    alert_cooldown: int = 5
    save_detections: bool = True
    detection_history_size: int = 100

@dataclass
class PerformanceConfig:
    """Performance-specific configuration"""
    target_fps: float = 30.0
    max_cpu_usage: float = 80.0
    enable_gpu: bool = True
    batch_size: int = 1
    num_threads: int = 4
    memory_limit_mb: int = 2048
    cache_size: int = 100
    optimization_level: str = "balanced"  # fast, balanced, quality
    enable_profiling: bool = False

class ConfigurationManager:
    """Advanced configuration management system"""
    
    def __init__(self, config_dir: str = "outputs/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration objects
        self.app_config = AppConfig()
        self.ui_config = UIConfig()
        self.detection_config = DetectionConfig()
        self.performance_config = PerformanceConfig()
        
        # Settings
        self.config_file = self.config_dir / "main_config.json"
        self.profiles_dir = self.config_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        # Callbacks for real-time updates
        self.update_callbacks = []
        
        # Load existing configuration
        self.load_config()
        
        logging.info("⚙️ Configuration Manager initialized")
    
    def add_update_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for configuration updates"""
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable[[str, Any], None]):
        """Remove update callback"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def _notify_update(self, key: str, value: Any):
        """Notify all callbacks of configuration update"""
        for callback in self.update_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                logging.warning(f"Callback error: {e}")
    
    def update_config(self, section: str, key: str, value: Any, save: bool = True):
        """Update configuration value"""
        config_obj = getattr(self, f"{section}_config", None)
        
        if config_obj and hasattr(config_obj, key):
            setattr(config_obj, key, value)
            
            # Notify callbacks
            self._notify_update(f"{section}.{key}", value)
            
            if save:
                self.save_config()
            
            logging.info(f"Config updated: {section}.{key} = {value}")
            return True
        
        logging.warning(f"Invalid config key: {section}.{key}")
        return False
    
    def get_config(self, section: str, key: str = None):
        """Get configuration value(s)"""
        config_obj = getattr(self, f"{section}_config", None)
        
        if not config_obj:
            return None
        
        if key:
            return getattr(config_obj, key, None)
        else:
            return config_obj
    
    def save_config(self, filename: str = None):
        """Save configuration to file"""
        if not filename:
            filename = self.config_file
        
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "app": asdict(self.app_config),
            "ui": asdict(self.ui_config),
            "detection": asdict(self.detection_config),
            "performance": asdict(self.performance_config)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logging.info(f"💾 Configuration saved to {filename}")
            return True
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False
    
    def load_config(self, filename: str = None):
        """Load configuration from file"""
        if not filename:
            filename = self.config_file
        
        if not Path(filename).exists():
            logging.info("No existing config found, using defaults")
            return True
        
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration objects
            if "app" in config_data:
                self.app_config = AppConfig(**config_data["app"])
            
            if "ui" in config_data:
                self.ui_config = UIConfig(**config_data["ui"])
            
            if "detection" in config_data:
                self.detection_config = DetectionConfig(**config_data["detection"])
            
            if "performance" in config_data:
                self.performance_config = PerformanceConfig(**config_data["performance"])
            
            logging.info(f"📂 Configuration loaded from {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return False
    
    def save_profile(self, profile_name: str):
        """Save current configuration as profile"""
        profile_file = self.profiles_dir / f"{profile_name}.json"
        
        profile_data = {
            "name": profile_name,
            "created": datetime.now().isoformat(),
            "description": f"Custom profile: {profile_name}",
            "config": {
                "app": asdict(self.app_config),
                "ui": asdict(self.ui_config),
                "detection": asdict(self.detection_config),
                "performance": asdict(self.performance_config)
            }
        }
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            logging.info(f"💾 Profile '{profile_name}' saved")
            return True
        except Exception as e:
            logging.error(f"Failed to save profile: {e}")
            return False
    
    def load_profile(self, profile_name: str):
        """Load configuration profile"""
        profile_file = self.profiles_dir / f"{profile_name}.json"
        
        if not profile_file.exists():
            logging.error(f"Profile not found: {profile_name}")
            return False
        
        try:
            with open(profile_file, 'r') as f:
                profile_data = json.load(f)
            
            config = profile_data["config"]
            
            # Update configurations
            if "app" in config:
                self.app_config = AppConfig(**config["app"])
            if "ui" in config:
                self.ui_config = UIConfig(**config["ui"])
            if "detection" in config:
                self.detection_config = DetectionConfig(**config["detection"])
            if "performance" in config:
                self.performance_config = PerformanceConfig(**config["performance"])
            
            # Notify all callbacks
            for section in ["app", "ui", "detection", "performance"]:
                config_obj = getattr(self, f"{section}_config")
                for key, value in asdict(config_obj).items():
                    self._notify_update(f"{section}.{key}", value)
            
            logging.info(f"📂 Profile '{profile_name}' loaded")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load profile: {e}")
            return False
    
    def get_profiles(self):
        """Get list of available profiles"""
        profiles = []
        
        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                
                profiles.append({
                    "name": profile_data.get("name", profile_file.stem),
                    "description": profile_data.get("description", ""),
                    "created": profile_data.get("created", ""),
                    "file": str(profile_file)
                })
            except Exception as e:
                logging.warning(f"Failed to read profile {profile_file}: {e}")
        
        return profiles
    
    def delete_profile(self, profile_name: str):
        """Delete configuration profile"""
        profile_file = self.profiles_dir / f"{profile_name}.json"
        
        try:
            if profile_file.exists():
                profile_file.unlink()
                logging.info(f"🗑️ Profile '{profile_name}' deleted")
                return True
        except Exception as e:
            logging.error(f"Failed to delete profile: {e}")
        
        return False
    
    def reset_to_defaults(self):
        """Reset all configurations to defaults"""
        self.app_config = AppConfig()
        self.ui_config = UIConfig()
        self.detection_config = DetectionConfig()
        self.performance_config = PerformanceConfig()
        
        # Notify callbacks
        for section in ["app", "ui", "detection", "performance"]:
            config_obj = getattr(self, f"{section}_config")
            for key, value in asdict(config_obj).items():
                self._notify_update(f"{section}.{key}", value)
        
        logging.info("🔄 Configuration reset to defaults")

class ConfigurationGUI:
    """GUI for configuration management"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.root = None
        self.variables = {}
        self.setup_gui()
    
    def setup_gui(self):
        """Create configuration GUI"""
        self.root = tk.Toplevel()
        self.root.title("⚙️ Enhanced App Configuration")
        self.root.geometry("800x600")
        self.root.configure(bg="#2d2d2d")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_app_tab()
        self.create_ui_tab()
        self.create_detection_tab()
        self.create_performance_tab()
        self.create_profiles_tab()
        
        # Create buttons frame
        self.create_buttons_frame()
        
        # Load current values
        self.load_current_values()
        
        # Setup auto-update
        self.config_manager.add_update_callback(self.on_config_update)
    
    def create_app_tab(self):
        """Create application settings tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🎯 Application")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame, bg="#2d2d2d")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # App settings
        self.create_setting_group(scrollable_frame, "🎨 Theme Settings", [
            ("theme", "Theme", "combobox", ["dark", "light", "blue", "green"]),
            ("language", "Language", "combobox", ["vi", "en"]),
            ("ui_scale", "UI Scale", "scale", (0.5, 2.0, 0.1))
        ], "app")
        
        self.create_setting_group(scrollable_frame, "📊 Display Options", [
            ("show_fps", "Show FPS", "checkbox"),
            ("show_confidence", "Show Confidence", "checkbox"),
            ("enable_alerts", "Enable Alerts", "checkbox"),
            ("alert_sound", "Alert Sound", "checkbox")
        ], "app")
        
        self.create_setting_group(scrollable_frame, "💾 Data Management", [
            ("auto_save", "Auto Save", "checkbox"),
            ("output_dir", "Output Directory", "entry"),
            ("log_level", "Log Level", "combobox", ["DEBUG", "INFO", "WARNING", "ERROR"])
        ], "app")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_ui_tab(self):
        """Create UI settings tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🎨 Interface")
        
        self.create_setting_group(frame, "🖼️ Window Settings", [
            ("window_width", "Window Width", "entry"),
            ("window_height", "Window Height", "entry"),
            ("always_on_top", "Always on Top", "checkbox"),
            ("remember_position", "Remember Position", "checkbox"),
            ("transparency", "Transparency", "scale", (0.3, 1.0, 0.1))
        ], "ui")
        
        self.create_setting_group(frame, "🔤 Font Settings", [
            ("font_family", "Font Family", "entry"),
            ("font_size", "Font Size", "spinbox", (8, 24)),
            ("icon_size", "Icon Size", "spinbox", (12, 32))
        ], "ui")
        
        self.create_setting_group(frame, "✨ Effects", [
            ("animation_enabled", "Enable Animations", "checkbox")
        ], "ui")
    
    def create_detection_tab(self):
        """Create detection settings tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🎯 Detection")
        
        self.create_setting_group(frame, "🤖 Model Settings", [
            ("model_weights", "Model Weights", "file"),
            ("confidence_threshold", "Confidence Threshold", "scale", (0.1, 1.0, 0.05)),
            ("iou_threshold", "IoU Threshold", "scale", (0.1, 1.0, 0.05)),
            ("max_detections", "Max Detections", "spinbox", (1, 20))
        ], "detection")
        
        self.create_setting_group(frame, "👁️ Sleepy Detection", [
            ("sleepy_sensitivity", "Sleepy Sensitivity", "scale", (0.1, 1.0, 0.1)),
            ("alert_cooldown", "Alert Cooldown (seconds)", "spinbox", (1, 60)),
            ("detection_history_size", "Detection History Size", "spinbox", (10, 1000))
        ], "detection")
        
        self.create_setting_group(frame, "🔍 Tracking", [
            ("enable_tracking", "Enable Tracking", "checkbox"),
            ("tracking_threshold", "Tracking Threshold", "scale", (0.1, 1.0, 0.05)),
            ("save_detections", "Save Detections", "checkbox")
        ], "detection")
    
    def create_performance_tab(self):
        """Create performance settings tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚡ Performance")
        
        self.create_setting_group(frame, "🎯 Performance Targets", [
            ("target_fps", "Target FPS", "scale", (5.0, 60.0, 1.0)),
            ("max_cpu_usage", "Max CPU Usage (%)", "scale", (20.0, 100.0, 5.0)),
            ("optimization_level", "Optimization Level", "combobox", ["fast", "balanced", "quality"])
        ], "performance")
        
        self.create_setting_group(frame, "🖥️ Hardware Settings", [
            ("enable_gpu", "Enable GPU", "checkbox"),
            ("num_threads", "Number of Threads", "spinbox", (1, 16)),
            ("memory_limit_mb", "Memory Limit (MB)", "entry"),
            ("batch_size", "Batch Size", "spinbox", (1, 32))
        ], "performance")
        
        self.create_setting_group(frame, "📊 Monitoring", [
            ("cache_size", "Cache Size", "spinbox", (10, 1000)),
            ("enable_profiling", "Enable Profiling", "checkbox")
        ], "performance")
    
    def create_profiles_tab(self):
        """Create profiles management tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📁 Profiles")
        
        # Profiles list
        list_frame = ttk.LabelFrame(frame, text="💾 Saved Profiles", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Profiles listbox with scrollbar
        list_container = tk.Frame(list_frame)
        list_container.pack(fill="both", expand=True)
        
        self.profiles_listbox = tk.Listbox(list_container, height=10)
        profiles_scrollbar = tk.Scrollbar(list_container, orient="vertical")
        
        self.profiles_listbox.configure(yscrollcommand=profiles_scrollbar.set)
        profiles_scrollbar.configure(command=self.profiles_listbox.yview)
        
        self.profiles_listbox.pack(side="left", fill="both", expand=True)
        profiles_scrollbar.pack(side="right", fill="y")
        
        # Profile buttons
        buttons_frame = tk.Frame(list_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        ttk.Button(buttons_frame, text="📂 Load Profile", 
                  command=self.load_selected_profile).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="💾 Save Current", 
                  command=self.save_current_profile).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="🗑️ Delete", 
                  command=self.delete_selected_profile).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="🔄 Refresh", 
                  command=self.refresh_profiles).pack(side="left", padx=5)
        
        # Load profiles
        self.refresh_profiles()
    
    def create_setting_group(self, parent, title, settings, section):
        """Create a group of settings"""
        group_frame = ttk.LabelFrame(parent, text=title, padding=10)
        group_frame.pack(fill="x", padx=10, pady=5)
        
        for setting in settings:
            self.create_setting_widget(group_frame, setting, section)
    
    def create_setting_widget(self, parent, setting, section):
        """Create individual setting widget"""
        if len(setting) == 3:
            key, label, widget_type = setting
            options = None
        else:
            key, label, widget_type, options = setting
        
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        # Label
        tk.Label(frame, text=f"{label}:", width=20, anchor="w").pack(side="left")
        
        var_key = f"{section}.{key}"
        
        if widget_type == "entry":
            var = tk.StringVar()
            widget = tk.Entry(frame, textvariable=var, width=30)
            widget.pack(side="left", padx=5)
            
        elif widget_type == "checkbox":
            var = tk.BooleanVar()
            widget = tk.Checkbutton(frame, variable=var)
            widget.pack(side="left", padx=5)
            
        elif widget_type == "combobox":
            var = tk.StringVar()
            widget = ttk.Combobox(frame, textvariable=var, values=options, 
                                state="readonly", width=25)
            widget.pack(side="left", padx=5)
            
        elif widget_type == "scale":
            var = tk.DoubleVar()
            min_val, max_val, resolution = options
            widget = tk.Scale(frame, variable=var, from_=min_val, to=max_val,
                            resolution=resolution, orient="horizontal", length=200)
            widget.pack(side="left", padx=5)
            
        elif widget_type == "spinbox":
            var = tk.IntVar()
            if isinstance(options, tuple):
                min_val, max_val = options
                widget = tk.Spinbox(frame, textvariable=var, from_=min_val, 
                                  to=max_val, width=10)
            else:
                widget = tk.Spinbox(frame, textvariable=var, width=10)
            widget.pack(side="left", padx=5)
            
        elif widget_type == "file":
            var = tk.StringVar()
            widget = tk.Entry(frame, textvariable=var, width=25)
            widget.pack(side="left", padx=5)
            
            def browse_file():
                filename = filedialog.askopenfilename(
                    title=f"Select {label}",
                    filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
                )
                if filename:
                    var.set(filename)
            
            tk.Button(frame, text="📁", command=browse_file).pack(side="left", padx=2)
        
        # Store variable reference
        self.variables[var_key] = var
        
        # Bind update callback
        def on_change(*args):
            value = var.get()
            self.config_manager.update_config(section, key, value)
        
        var.trace("w", on_change)
    
    def create_buttons_frame(self):
        """Create main buttons frame"""
        buttons_frame = tk.Frame(self.root, bg="#2d2d2d")
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(buttons_frame, text="💾 Save", command=self.save_config,
                 bg="#4CAF50", fg="white", padx=20).pack(side="left", padx=5)
        
        tk.Button(buttons_frame, text="🔄 Reset", command=self.reset_config,
                 bg="#FF9800", fg="white", padx=20).pack(side="left", padx=5)
        
        tk.Button(buttons_frame, text="📂 Load", command=self.load_config,
                 bg="#2196F3", fg="white", padx=20).pack(side="left", padx=5)
        
        tk.Button(buttons_frame, text="❌ Close", command=self.close_window,
                 bg="#F44336", fg="white", padx=20).pack(side="right", padx=5)
    
    def load_current_values(self):
        """Load current configuration values into GUI"""
        for section in ["app", "ui", "detection", "performance"]:
            config_obj = getattr(self.config_manager, f"{section}_config")
            for key, value in asdict(config_obj).items():
                var_key = f"{section}.{key}"
                if var_key in self.variables:
                    self.variables[var_key].set(value)
    
    def on_config_update(self, key: str, value: Any):
        """Handle configuration update callback"""
        if key in self.variables:
            self.variables[key].set(value)
    
    def save_config(self):
        """Save current configuration"""
        if self.config_manager.save_config():
            messagebox.showinfo("Success", "Configuration saved successfully!")
    
    def load_config(self):
        """Load configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.config_manager.config_dir
        )
        
        if filename:
            if self.config_manager.load_config(filename):
                self.load_current_values()
                messagebox.showinfo("Success", "Configuration loaded successfully!")
    
    def reset_config(self):
        """Reset configuration to defaults"""
        if messagebox.askyesno("Confirm Reset", "Reset all settings to defaults?"):
            self.config_manager.reset_to_defaults()
            self.load_current_values()
    
    def save_current_profile(self):
        """Save current configuration as profile"""
        def save_profile():
            profile_name = profile_name_entry.get().strip()
            if profile_name:
                if self.config_manager.save_profile(profile_name):
                    messagebox.showinfo("Success", f"Profile '{profile_name}' saved!")
                    profile_window.destroy()
                    self.refresh_profiles()
                else:
                    messagebox.showerror("Error", "Failed to save profile")
        
        # Create profile name dialog
        profile_window = tk.Toplevel(self.root)
        profile_window.title("💾 Save Profile")
        profile_window.geometry("300x150")
        profile_window.transient(self.root)
        
        tk.Label(profile_window, text="Profile Name:").pack(pady=10)
        profile_name_entry = tk.Entry(profile_window, width=30)
        profile_name_entry.pack(pady=5)
        profile_name_entry.focus()
        
        tk.Button(profile_window, text="💾 Save", command=save_profile).pack(pady=10)
    
    def load_selected_profile(self):
        """Load selected profile"""
        selection = self.profiles_listbox.curselection()
        if selection:
            profile_name = self.profiles_listbox.get(selection[0]).split(" - ")[0]
            if self.config_manager.load_profile(profile_name):
                self.load_current_values()
                messagebox.showinfo("Success", f"Profile '{profile_name}' loaded!")
    
    def delete_selected_profile(self):
        """Delete selected profile"""
        selection = self.profiles_listbox.curselection()
        if selection:
            profile_name = self.profiles_listbox.get(selection[0]).split(" - ")[0]
            if messagebox.askyesno("Confirm Delete", f"Delete profile '{profile_name}'?"):
                if self.config_manager.delete_profile(profile_name):
                    messagebox.showinfo("Success", f"Profile '{profile_name}' deleted!")
                    self.refresh_profiles()
    
    def refresh_profiles(self):
        """Refresh profiles list"""
        self.profiles_listbox.delete(0, tk.END)
        
        profiles = self.config_manager.get_profiles()
        for profile in profiles:
            display_text = f"{profile['name']} - {profile['description']}"
            self.profiles_listbox.insert(tk.END, display_text)
    
    def close_window(self):
        """Close configuration window"""
        self.config_manager.remove_update_callback(self.on_config_update)
        self.root.destroy()
    
    def show(self):
        """Show configuration window"""
        self.root.deiconify()
        self.root.lift()
    
    def hide(self):
        """Hide configuration window"""
        self.root.withdraw()

# ============================================================================
# 🧪 Testing and Demo
# ============================================================================

def main():
    """Demo/test function"""
    import tkinter as tk
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Create main window
    root = tk.Tk()
    root.title("🎯 Configuration Demo")
    root.geometry("400x300")
    
    # Demo callback
    def on_config_change(key, value):
        print(f"Config changed: {key} = {value}")
    
    config_manager.add_update_callback(on_config_change)
    
    # Create GUI
    config_gui = ConfigurationGUI(config_manager)
    
    # Show button
    tk.Button(root, text="⚙️ Open Configuration", 
             command=config_gui.show,
             font=("Segoe UI", 12),
             padx=20, pady=10).pack(expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()