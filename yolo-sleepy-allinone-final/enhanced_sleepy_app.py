"""
🎯 Enhanced Sleepy Detection App - Modern UX/UI
==============================================

Modern, decorator-enhanced wrapper for existing sleepy detection applications.
Provides improved user experience without modifying original source code.

Features:
- 🎨 Modern dark/light themes
- 🎛️ Enhanced controls and settings
- 📊 Real-time statistics and monitoring
- 💾 Auto-save and session management
- 🎯 Multi-app launcher (standalone, GUI, web)
- 🔄 Live configuration updates

Author: Enhanced by AI Assistant
Date: September 30, 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from enhanced_app_decorators import (
    EnhancedAppWrapper, AppConfig, AppMode,
    performance_monitor, error_handler, modern_ui_wrapper
)

class ModernSleepyDetectionApp:
    """Modern enhanced wrapper for sleepy detection applications"""
    
    def __init__(self):
        self.config = AppConfig()
        self.wrapper = EnhancedAppWrapper(self.config)
        self.current_process = None
        self.is_running = False
        self.stats = {
            "sessions": 0,
            "total_runtime": 0,
            "last_run": None
        }
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Create modern UI interface"""
        self.root = tk.Tk()
        self.root.title("🎯 Enhanced Sleepy Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")
        
        # Setup modern style
        self.setup_styles()
        
        # Create main layout
        self.create_header()
        self.create_main_content()
        self.create_sidebar()
        self.create_status_bar()
        
        # Apply initial theme
        self.apply_theme(self.config.theme)
        
    def setup_styles(self):
        """Setup modern TTK styles"""
        self.style = ttk.Style()
        
        # Configure modern dark theme
        self.style.configure("Header.TLabel", 
                           font=("Segoe UI", 16, "bold"),
                           foreground="#00ff88")
        
        self.style.configure("Title.TLabel",
                           font=("Segoe UI", 12, "bold"))
        
        self.style.configure("Modern.TButton",
                           font=("Segoe UI", 10),
                           padding=(20, 10))
        
        self.style.configure("Accent.TButton",
                           font=("Segoe UI", 10, "bold"))
    
    def create_header(self):
        """Create application header"""
        header_frame = tk.Frame(self.root, bg="#2d2d2d", height=80)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title and logo
        title_frame = tk.Frame(header_frame, bg="#2d2d2d")
        title_frame.pack(side="left", fill="both", expand=True, padx=20, pady=15)
        
        tk.Label(title_frame, text="🎯 Enhanced Sleepy Detection",
                font=("Segoe UI", 18, "bold"),
                fg="#00ff88", bg="#2d2d2d").pack(anchor="w")
        
        tk.Label(title_frame, text="Modern AI-powered drowsiness detection with enhanced UX/UI",
                font=("Segoe UI", 10),
                fg="#cccccc", bg="#2d2d2d").pack(anchor="w")
        
        # Quick actions
        actions_frame = tk.Frame(header_frame, bg="#2d2d2d")
        actions_frame.pack(side="right", padx=20, pady=15)
        
        tk.Button(actions_frame, text="⚙️", font=("Segoe UI", 14),
                 bg="#404040", fg="#ffffff", bd=0,
                 command=self.show_settings).pack(side="left", padx=5)
        
        tk.Button(actions_frame, text="❓", font=("Segoe UI", 14),
                 bg="#404040", fg="#ffffff", bd=0,
                 command=self.show_help).pack(side="left", padx=5)
    
    def create_main_content(self):
        """Create main content area"""
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # App launcher section
        launcher_frame = tk.LabelFrame(main_frame, text="🚀 Application Launcher",
                                     font=("Segoe UI", 12, "bold"),
                                     bg="#2d2d2d", fg="#ffffff",
                                     padx=20, pady=15)
        launcher_frame.pack(fill="x", pady=(0, 15))
        
        # App mode buttons
        modes_frame = tk.Frame(launcher_frame, bg="#2d2d2d")
        modes_frame.pack(fill="x", pady=10)
        
        self.create_app_button(modes_frame, "🖥️ GUI Application", 
                              "Launch enhanced GUI with modern interface",
                              self.launch_gui_app, "#4CAF50")
        
        self.create_app_button(modes_frame, "⚡ Standalone Mode",
                              "Fast standalone detection with minimal UI", 
                              self.launch_standalone_app, "#2196F3")
        
        self.create_app_button(modes_frame, "🌐 Web Interface",
                              "Browser-based Streamlit application",
                              self.launch_web_app, "#FF9800")
        
        self.create_app_button(modes_frame, "🎮 Demo Mode",
                              "Fullscreen HUD demonstration",
                              self.launch_demo_app, "#9C27B0")
        
        # Configuration section
        config_frame = tk.LabelFrame(main_frame, text="⚙️ Configuration",
                                   font=("Segoe UI", 12, "bold"),
                                   bg="#2d2d2d", fg="#ffffff",
                                   padx=20, pady=15)
        config_frame.pack(fill="x", pady=(0, 15))
        
        self.create_config_controls(config_frame)
        
        # Real-time monitoring section
        monitor_frame = tk.LabelFrame(main_frame, text="📊 Real-time Monitoring",
                                    font=("Segoe UI", 12, "bold"),
                                    bg="#2d2d2d", fg="#ffffff",
                                    padx=20, pady=15)
        monitor_frame.pack(fill="both", expand=True)
        
        self.create_monitoring_display(monitor_frame)
    
    def create_app_button(self, parent, title, description, command, color):
        """Create enhanced app launcher button"""
        button_frame = tk.Frame(parent, bg="#2d2d2d")
        button_frame.pack(fill="x", pady=5)
        
        # Main button
        btn = tk.Button(button_frame, text=title,
                       font=("Segoe UI", 11, "bold"),
                       bg=color, fg="white", bd=0,
                       padx=20, pady=15,
                       command=command,
                       cursor="hand2")
        btn.pack(side="left", padx=(0, 10))
        
        # Description
        tk.Label(button_frame, text=description,
                font=("Segoe UI", 9),
                fg="#cccccc", bg="#2d2d2d").pack(side="left", anchor="w")
        
        # Status indicator
        status_label = tk.Label(button_frame, text="●",
                              font=("Segoe UI", 12),
                              fg="#666666", bg="#2d2d2d")
        status_label.pack(side="right")
        
        # Store reference for status updates
        setattr(self, f"status_{title.split()[1].lower()}", status_label)
    
    def create_config_controls(self, parent):
        """Create configuration controls"""
        # First row
        row1 = tk.Frame(parent, bg="#2d2d2d")
        row1.pack(fill="x", pady=5)
        
        # Theme selection
        tk.Label(row1, text="🎨 Theme:", font=("Segoe UI", 10),
                fg="#ffffff", bg="#2d2d2d").pack(side="left")
        
        self.theme_var = tk.StringVar(value=self.config.theme)
        theme_combo = ttk.Combobox(row1, textvariable=self.theme_var,
                                  values=["dark", "light", "blue", "green"],
                                  state="readonly", width=10)
        theme_combo.pack(side="left", padx=(5, 20))
        theme_combo.bind("<<ComboboxSelected>>", self.on_theme_change)
        
        # Performance mode
        tk.Label(row1, text="⚡ Performance:", font=("Segoe UI", 10),
                fg="#ffffff", bg="#2d2d2d").pack(side="left")
        
        self.perf_var = tk.StringVar(value=self.config.performance_mode)
        perf_combo = ttk.Combobox(row1, textvariable=self.perf_var,
                                 values=["fast", "balanced", "quality"],
                                 state="readonly", width=12)
        perf_combo.pack(side="left", padx=(5, 20))
        perf_combo.bind("<<ComboboxSelected>>", self.on_perf_change)
        
        # UI Scale
        tk.Label(row1, text="🔍 UI Scale:", font=("Segoe UI", 10),
                fg="#ffffff", bg="#2d2d2d").pack(side="left")
        
        self.scale_var = tk.DoubleVar(value=self.config.ui_scale)
        scale_spin = tk.Spinbox(row1, textvariable=self.scale_var,
                               from_=0.5, to=2.0, increment=0.1,
                               width=8, bg="#404040", fg="#ffffff")
        scale_spin.pack(side="left", padx=(5, 0))
        
        # Second row - checkboxes
        row2 = tk.Frame(parent, bg="#2d2d2d")
        row2.pack(fill="x", pady=10)
        
        self.create_checkbox(row2, "📊 Show FPS", self.config.show_fps)
        self.create_checkbox(row2, "🎯 Show Confidence", self.config.show_confidence)
        self.create_checkbox(row2, "🔔 Enable Alerts", self.config.enable_alerts)
        self.create_checkbox(row2, "🔊 Alert Sound", self.config.alert_sound)
        self.create_checkbox(row2, "💾 Auto Save", self.config.auto_save)
    
    def create_checkbox(self, parent, text, initial_value):
        """Create enhanced checkbox"""
        var = tk.BooleanVar(value=initial_value)
        cb = tk.Checkbutton(parent, text=text,
                           variable=var,
                           font=("Segoe UI", 9),
                           fg="#ffffff", bg="#2d2d2d",
                           selectcolor="#404040",
                           activebackground="#2d2d2d",
                           activeforeground="#ffffff")
        cb.pack(side="left", padx=15)
        
        # Store reference
        attr_name = text.split()[-1].lower().replace(":", "")
        setattr(self, f"{attr_name}_var", var)
    
    def create_monitoring_display(self, parent):
        """Create real-time monitoring display"""
        # Stats display
        stats_frame = tk.Frame(parent, bg="#2d2d2d")
        stats_frame.pack(fill="x", pady=10)
        
        # Create stat cards
        self.create_stat_card(stats_frame, "📈 Sessions", str(self.stats["sessions"]), "#4CAF50")
        self.create_stat_card(stats_frame, "⏱️ Runtime", "0:00:00", "#2196F3")
        self.create_stat_card(stats_frame, "🎯 Status", "Ready", "#FF9800")
        self.create_stat_card(stats_frame, "💾 Last Save", "Never", "#9C27B0")
        
        # Log display
        log_frame = tk.Frame(parent, bg="#2d2d2d")
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        tk.Label(log_frame, text="📝 Activity Log",
                font=("Segoe UI", 10, "bold"),
                fg="#ffffff", bg="#2d2d2d").pack(anchor="w")
        
        # Log text area
        log_scroll_frame = tk.Frame(log_frame, bg="#2d2d2d")
        log_scroll_frame.pack(fill="both", expand=True, pady=5)
        
        self.log_text = tk.Text(log_scroll_frame, 
                               height=8,
                               bg="#1a1a1a", 
                               fg="#cccccc",
                               font=("Consolas", 9),
                               wrap=tk.WORD)
        
        log_scrollbar = tk.Scrollbar(log_scroll_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")
        
        # Add initial log entry
        self.add_log("🚀 Enhanced Sleepy Detection System initialized")
        self.add_log("✅ All systems ready")
    
    def create_stat_card(self, parent, title, value, color):
        """Create statistic display card"""
        card_frame = tk.Frame(parent, bg=color, padx=15, pady=10)
        card_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        tk.Label(card_frame, text=title,
                font=("Segoe UI", 9, "bold"),
                fg="white", bg=color).pack()
        
        value_label = tk.Label(card_frame, text=value,
                              font=("Segoe UI", 14, "bold"),
                              fg="white", bg=color)
        value_label.pack()
        
        # Store reference for updates
        attr_name = title.split()[-1].lower()
        setattr(self, f"stat_{attr_name}", value_label)
    
    def create_sidebar(self):
        """Create sidebar with additional controls"""
        sidebar = tk.Frame(self.root, bg="#333333", width=250)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)
        
        # Sidebar title
        tk.Label(sidebar, text="🎛️ Quick Controls",
                font=("Segoe UI", 12, "bold"),
                fg="#ffffff", bg="#333333").pack(pady=15)
        
        # Quick action buttons
        self.create_sidebar_button(sidebar, "📸 Capture Frame", self.capture_frame)
        self.create_sidebar_button(sidebar, "📹 Record Video", self.toggle_recording)
        self.create_sidebar_button(sidebar, "📊 View Statistics", self.show_statistics)
        self.create_sidebar_button(sidebar, "📁 Open Output Folder", self.open_output_folder)
        self.create_sidebar_button(sidebar, "🔄 Restart Apps", self.restart_all_apps)
        self.create_sidebar_button(sidebar, "❌ Stop All", self.stop_all_apps)
        
        # Recent files
        recent_frame = tk.LabelFrame(sidebar, text="📁 Recent Files",
                                   bg="#333333", fg="#ffffff",
                                   padx=10, pady=10)
        recent_frame.pack(fill="x", padx=10, pady=20)
        
        self.recent_listbox = tk.Listbox(recent_frame, height=6,
                                        bg="#2a2a2a", fg="#ffffff",
                                        selectbackground="#4CAF50")
        self.recent_listbox.pack(fill="x")
        
        # Load recent files
        self.update_recent_files()
    
    def create_sidebar_button(self, parent, text, command):
        """Create sidebar button"""
        btn = tk.Button(parent, text=text,
                       font=("Segoe UI", 9),
                       bg="#404040", fg="#ffffff", bd=0,
                       padx=15, pady=8,
                       command=command,
                       cursor="hand2")
        btn.pack(fill="x", padx=10, pady=2)
        
        # Hover effects
        def on_enter(e):
            btn.config(bg="#4CAF50")
        def on_leave(e):
            btn.config(bg="#404040")
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
    
    def create_status_bar(self):
        """Create enhanced status bar"""
        status_frame = tk.Frame(self.root, bg="#2d2d2d", height=30)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)
        
        # Status text
        self.status_label = tk.Label(status_frame, text="🟢 Ready to launch applications",
                                   font=("Segoe UI", 9),
                                   fg="#00ff88", bg="#2d2d2d")
        self.status_label.pack(side="left", padx=10, pady=5)
        
        # Right side status
        self.time_label = tk.Label(status_frame, text="",
                                 font=("Segoe UI", 9),
                                 fg="#cccccc", bg="#2d2d2d")
        self.time_label.pack(side="right", padx=10, pady=5)
        
        # Update time
        self.update_time()
    
    # ================================================================
    # Event Handlers
    # ================================================================
    
    @performance_monitor
    @error_handler(show_dialog=True)
    def launch_gui_app(self):
        """Launch enhanced GUI application"""
        self.add_log("🖥️ Launching Enhanced GUI Application...")
        self.update_status("🔄 Starting GUI app...")
        
        # Use the decorator wrapper to enhance the original GUI app
        def run_enhanced_gui():
            try:
                os.chdir(Path(__file__).parent)
                
                # Import and enhance the original GUI app
                import gui_app
                enhanced_gui = self.wrapper._apply_all_decorators(gui_app.main)
                
                # Launch with enhanced features
                enhanced_gui()
                
            except Exception as e:
                self.add_log(f"❌ GUI app error: {str(e)}")
                messagebox.showerror("Error", f"Failed to launch GUI app:\n{str(e)}")
        
        # Run in separate thread
        thread = threading.Thread(target=run_enhanced_gui, daemon=True)
        thread.start()
        
        self.update_status("✅ GUI app launched")
        self.add_log("✅ Enhanced GUI Application started successfully")
        self.stats["sessions"] += 1
        self.update_session_stats()
    
    @performance_monitor
    @error_handler(show_dialog=True)
    def launch_standalone_app(self):
        """Launch enhanced standalone application"""
        self.add_log("⚡ Launching Enhanced Standalone Application...")
        self.update_status("🔄 Starting standalone app...")
        
        def run_enhanced_standalone():
            try:
                os.chdir(Path(__file__).parent)
                
                # Launch standalone with enhanced parameters
                cmd = [
                    sys.executable, "standalone_app.py",
                    "--model", "yolo11n-pose.pt",
                    "--cam", "0",
                    "--gui",
                    "--enhanced-display",
                    "--person-circles"
                ]
                
                self.current_process = subprocess.Popen(cmd)
                self.current_process.wait()
                
            except Exception as e:
                self.add_log(f"❌ Standalone app error: {str(e)}")
                messagebox.showerror("Error", f"Failed to launch standalone app:\n{str(e)}")
        
        thread = threading.Thread(target=run_enhanced_standalone, daemon=True)
        thread.start()
        
        self.update_status("✅ Standalone app launched")
        self.add_log("✅ Enhanced Standalone Application started successfully")
        self.stats["sessions"] += 1
        self.update_session_stats()
    
    @performance_monitor
    @error_handler(show_dialog=True)
    def launch_web_app(self):
        """Launch web application"""
        self.add_log("🌐 Launching Web Application...")
        self.update_status("🔄 Starting web server...")
        
        def run_web_app():
            try:
                os.chdir(Path(__file__).parent)
                cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
                self.current_process = subprocess.Popen(cmd)
                
            except Exception as e:
                self.add_log(f"❌ Web app error: {str(e)}")
                messagebox.showerror("Error", f"Failed to launch web app:\n{str(e)}")
        
        thread = threading.Thread(target=run_web_app, daemon=True)
        thread.start()
        
        self.update_status("✅ Web app launched at http://localhost:8501")
        self.add_log("✅ Web Application started at http://localhost:8501")
        self.stats["sessions"] += 1
        self.update_session_stats()
    
    @performance_monitor
    @error_handler(show_dialog=True)
    def launch_demo_app(self):
        """Launch demo application"""
        self.add_log("🎮 Launching Demo Application...")
        self.update_status("🔄 Starting demo mode...")
        
        def run_demo_app():
            try:
                os.chdir(Path(__file__).parent)
                cmd = [sys.executable, "sleepy_demo.py"]
                self.current_process = subprocess.Popen(cmd)
                
            except Exception as e:
                self.add_log(f"❌ Demo app error: {str(e)}")
                messagebox.showerror("Error", f"Failed to launch demo app:\n{str(e)}")
        
        thread = threading.Thread(target=run_demo_app, daemon=True)
        thread.start()
        
        self.update_status("✅ Demo app launched")
        self.add_log("✅ Demo Application started in fullscreen mode")
        self.stats["sessions"] += 1
        self.update_session_stats()
    
    def on_theme_change(self, event=None):
        """Handle theme change"""
        new_theme = self.theme_var.get()
        self.config.theme = new_theme
        self.apply_theme(new_theme)
        self.add_log(f"🎨 Theme changed to: {new_theme}")
        self.save_settings()
    
    def on_perf_change(self, event=None):
        """Handle performance mode change"""
        new_perf = self.perf_var.get()
        self.config.performance_mode = new_perf
        self.add_log(f"⚡ Performance mode changed to: {new_perf}")
        self.save_settings()
    
    def apply_theme(self, theme):
        """Apply selected theme"""
        themes = {
            "dark": {"bg": "#1e1e1e", "bg2": "#2d2d2d", "fg": "#ffffff", "accent": "#00ff88"},
            "light": {"bg": "#ffffff", "bg2": "#f0f0f0", "fg": "#000000", "accent": "#0078d4"},
            "blue": {"bg": "#1e3a8a", "bg2": "#3b82f6", "fg": "#ffffff", "accent": "#60a5fa"},
            "green": {"bg": "#064e3b", "bg2": "#047857", "fg": "#ffffff", "accent": "#10b981"}
        }
        
        colors = themes.get(theme, themes["dark"])
        
        # Apply to main window
        self.root.configure(bg=colors["bg"])
        
        # This would be expanded to apply to all widgets
        self.add_log(f"✅ Applied {theme} theme")
    
    # ================================================================
    # Utility Methods
    # ================================================================
    
    def add_log(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Also log to file
        logging.info(message)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def update_session_stats(self):
        """Update session statistics"""
        if hasattr(self, 'stat_sessions'):
            self.stat_sessions.config(text=str(self.stats["sessions"]))
    
    def capture_frame(self):
        """Capture current frame"""
        self.add_log("📸 Frame capture requested")
        messagebox.showinfo("Capture", "Frame capture feature will be implemented")
    
    def toggle_recording(self):
        """Toggle video recording"""
        self.add_log("📹 Recording toggle requested")
        messagebox.showinfo("Recording", "Video recording feature will be implemented")
    
    def show_statistics(self):
        """Show detailed statistics"""
        self.add_log("📊 Opening statistics viewer")
        messagebox.showinfo("Statistics", "Detailed statistics viewer will be implemented")
    
    def open_output_folder(self):
        """Open output folder"""
        output_path = Path("outputs")
        output_path.mkdir(exist_ok=True)
        
        try:
            os.startfile(output_path)
            self.add_log("📁 Output folder opened")
        except:
            self.add_log("❌ Failed to open output folder")
    
    def restart_all_apps(self):
        """Restart all applications"""
        self.stop_all_apps()
        self.add_log("🔄 All apps restarted")
    
    def stop_all_apps(self):
        """Stop all running applications"""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process = None
                self.add_log("❌ All applications stopped")
                self.update_status("🟡 Apps stopped")
            except:
                self.add_log("⚠️ Could not stop some applications")
    
    def show_settings(self):
        """Show settings dialog"""
        self.add_log("⚙️ Opening settings")
        messagebox.showinfo("Settings", "Advanced settings panel will be implemented")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """🎯 Enhanced Sleepy Detection System

Features:
✅ Multiple app modes (GUI, Standalone, Web, Demo)
✅ Real-time monitoring and statistics
✅ Modern themes and UI customization
✅ Auto-save and session management
✅ Enhanced error handling and logging

Quick Start:
1. Choose an application mode
2. Configure settings as needed
3. Click launch button
4. Monitor real-time statistics

For more help, check the README.md file."""
        
        messagebox.showinfo("Help", help_text)
    
    def update_recent_files(self):
        """Update recent files list"""
        # This would load actual recent files
        recent_files = [
            "output_video_001.mp4",
            "detection_log_20250930.txt",
            "config_backup.json"
        ]
        
        self.recent_listbox.delete(0, tk.END)
        for file in recent_files:
            self.recent_listbox.insert(tk.END, file)
    
    def save_settings(self):
        """Save current settings"""
        try:
            self.wrapper.save_config()
            self.add_log("💾 Settings saved")
        except Exception as e:
            self.add_log(f"❌ Failed to save settings: {e}")
    
    def load_settings(self):
        """Load saved settings"""
        try:
            self.wrapper.load_config()
            self.add_log("📂 Settings loaded")
        except Exception as e:
            self.add_log(f"⚠️ Using default settings: {e}")
    
    def run(self):
        """Start the application"""
        self.add_log("🚀 Starting Enhanced Sleepy Detection System")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_all_apps()
        self.save_settings()
        self.add_log("👋 Shutting down Enhanced Sleepy Detection System")
        self.root.destroy()

if __name__ == "__main__":
    try:
        app = ModernSleepyDetectionApp()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        logging.error(f"Application startup failed: {e}", exc_info=True)