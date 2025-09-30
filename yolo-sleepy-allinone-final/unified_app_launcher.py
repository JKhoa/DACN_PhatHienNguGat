"""
🚀 Unified Enhanced App Launcher
==============================

Main launcher application that integrates all enhanced features and decorators.
Provides a unified interface to launch any sleepy detection app with enhancements.

Features:
- 🎯 Multi-app launcher (GUI, Standalone, Web, Demo)
- 🎨 Decorator-based enhancements without modifying source code
- ⚙️ Real-time configuration management
- 📊 Performance monitoring and analytics
- 🎛️ Modern UI with theme support
- 💾 Session management and auto-save

Author: Enhanced by AI Assistant
Date: September 30, 2025
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path
import threading
import time
import json
from typing import Dict, Any, Optional, Callable
import logging

# Import our enhancement modules
from enhanced_app_decorators import (
    EnhancedAppWrapper, AppConfig, AppMode,
    performance_monitor, error_handler, modern_ui_wrapper,
    ui_theme_adapter, auto_save_decorator
)

from data_flow_decorators import (
    input_validator, output_validator, smart_cache,
    data_pipeline, result_aggregator, adaptive_processing,
    sleepy_detection_enhancer, pose_quality_filter
)

class UnifiedAppLauncher:
    """Unified launcher for all enhanced sleepy detection applications"""
    
    def __init__(self):
        self.config = AppConfig()
        self.wrapper = EnhancedAppWrapper(self.config)
        self.running_processes = {}
        self.enhancement_profiles = self._load_enhancement_profiles()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup enhanced logging system"""
        log_dir = Path("outputs/logs")
        log_dir.mkdir(exist_ok=True, parents=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'unified_launcher.log'),
                logging.StreamHandler()
            ]
        )
        
        logging.info("🚀 Unified Enhanced App Launcher initialized")
    
    def _load_enhancement_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined enhancement profiles"""
        return {
            "performance": {
                "description": "Optimized for maximum performance and speed",
                "decorators": ["smart_cache", "adaptive_processing", "batch_processor"],
                "config_overrides": {
                    "performance_mode": "fast",
                    "ui_scale": 0.8
                }
            },
            "quality": {
                "description": "Optimized for maximum detection quality",
                "decorators": ["data_pipeline", "result_aggregator", "sleepy_detection_enhancer"],
                "config_overrides": {
                    "performance_mode": "quality",
                    "show_confidence": True
                }
            },
            "balanced": {
                "description": "Balanced performance and quality",
                "decorators": ["smart_cache", "data_pipeline", "adaptive_processing"],
                "config_overrides": {
                    "performance_mode": "balanced"
                }
            },
            "debug": {
                "description": "Maximum logging and debugging features",
                "decorators": ["data_logger", "input_validator", "output_validator"],
                "config_overrides": {
                    "log_level": "DEBUG",
                    "show_fps": True,
                    "show_confidence": True
                }
            },
            "production": {
                "description": "Production-ready with error handling and monitoring",
                "decorators": ["error_handler", "performance_monitor", "auto_save_decorator"],
                "config_overrides": {
                    "auto_save": True,
                    "enable_alerts": True
                }
            }
        }
    
    def get_available_apps(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available applications"""
        base_dir = Path(__file__).parent
        
        apps = {
            "gui": {
                "name": "Enhanced GUI Application",
                "description": "Modern GUI with advanced controls and real-time monitoring",
                "file": "gui_app.py",
                "icon": "🖥️",
                "launch_function": "main",
                "supports_enhancements": True
            },
            "standalone": {
                "name": "Enhanced Standalone Application", 
                "description": "Fast standalone detection with minimal overhead",
                "file": "standalone_app.py",
                "icon": "⚡",
                "launch_function": "main",
                "supports_enhancements": True
            },
            "web": {
                "name": "Web Application (Streamlit)",
                "description": "Browser-based interface with real-time streaming",
                "file": "app.py",
                "icon": "🌐",
                "launch_function": None,  # Uses streamlit run
                "supports_enhancements": False
            },
            "demo": {
                "name": "Fullscreen Demo",
                "description": "Immersive fullscreen demonstration mode",
                "file": "sleepy_demo.py", 
                "icon": "🎮",
                "launch_function": "main",
                "supports_enhancements": True
            },
            "enhanced_ui": {
                "name": "Modern Enhanced UI",
                "description": "Our custom enhanced UI with all modern features",
                "file": "enhanced_sleepy_app.py",
                "icon": "🎯",
                "launch_function": "main",
                "supports_enhancements": True
            }
        }
        
        # Check which apps actually exist
        available_apps = {}
        for app_id, app_info in apps.items():
            app_file = base_dir / app_info["file"]
            if app_file.exists():
                available_apps[app_id] = app_info
                available_apps[app_id]["path"] = str(app_file)
        
        return available_apps
    
    @performance_monitor
    @error_handler(show_dialog=False)
    def launch_app(self, app_id: str, enhancement_profile: str = "balanced", 
                   custom_args: Dict[str, Any] = None) -> bool:
        """Launch application with specified enhancements"""
        apps = self.get_available_apps()
        
        if app_id not in apps:
            logging.error(f"Application '{app_id}' not found")
            return False
        
        app_info = apps[app_id]
        logging.info(f"🚀 Launching {app_info['name']} with {enhancement_profile} profile")
        
        # Apply enhancement profile
        self._apply_enhancement_profile(enhancement_profile)
        
        try:
            if app_info["supports_enhancements"]:
                return self._launch_enhanced_app(app_info, custom_args)
            else:
                return self._launch_standard_app(app_info, custom_args)
        except Exception as e:
            logging.error(f"Failed to launch {app_id}: {e}")
            return False
    
    def _apply_enhancement_profile(self, profile_name: str):
        """Apply enhancement profile configuration"""
        if profile_name not in self.enhancement_profiles:
            logging.warning(f"Enhancement profile '{profile_name}' not found, using balanced")
            profile_name = "balanced"
        
        profile = self.enhancement_profiles[profile_name]
        
        # Apply configuration overrides
        if "config_overrides" in profile:
            for key, value in profile["config_overrides"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logging.info(f"Applied config: {key} = {value}")
    
    def _launch_enhanced_app(self, app_info: Dict[str, Any], custom_args: Dict[str, Any] = None) -> bool:
        """Launch app with full enhancement decorators"""
        try:
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(
                app_info["file"].replace(".py", ""), 
                app_info["path"]
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the main function
            launch_func = getattr(module, app_info.get("launch_function", "main"), None)
            
            if not launch_func:
                # Try alternative approach - execute as script
                return self._launch_as_script(app_info, custom_args)
            
            # Apply enhancement decorators
            enhanced_func = self._apply_enhancement_decorators(launch_func, app_info)
            
            # Launch in separate thread
            def run_enhanced():
                try:
                    if custom_args:
                        enhanced_func(**custom_args)
                    else:
                        enhanced_func()
                except Exception as e:
                    logging.error(f"Enhanced app execution error: {e}")
            
            thread = threading.Thread(target=run_enhanced, daemon=True)
            thread.start()
            
            # Store process info
            self.running_processes[app_info["name"]] = {
                "thread": thread,
                "start_time": time.time(),
                "app_info": app_info
            }
            
            logging.info(f"✅ Successfully launched enhanced {app_info['name']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to launch enhanced app: {e}")
            return False
    
    def _launch_as_script(self, app_info: Dict[str, Any], custom_args: Dict[str, Any] = None) -> bool:
        """Launch app as external script with enhancements"""
        try:
            cmd = [sys.executable, app_info["path"]]
            
            # Add custom arguments
            if custom_args:
                for key, value in custom_args.items():
                    cmd.extend([f"--{key}", str(value)])
            
            # Launch process
            process = subprocess.Popen(
                cmd,
                cwd=Path(app_info["path"]).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Store process info
            self.running_processes[app_info["name"]] = {
                "process": process,
                "start_time": time.time(),
                "app_info": app_info
            }
            
            logging.info(f"✅ Successfully launched {app_info['name']} as script")
            return True
            
        except Exception as e:
            logging.error(f"Failed to launch script: {e}")
            return False
    
    def _launch_standard_app(self, app_info: Dict[str, Any], custom_args: Dict[str, Any] = None) -> bool:
        """Launch app without enhancements (for web apps, etc.)"""
        try:
            if app_info["file"] == "app.py":  # Streamlit app
                cmd = [sys.executable, "-m", "streamlit", "run", app_info["path"]]
            else:
                cmd = [sys.executable, app_info["path"]]
            
            # Add custom arguments
            if custom_args:
                for key, value in custom_args.items():
                    cmd.extend([f"--{key}", str(value)])
            
            process = subprocess.Popen(cmd, cwd=Path(app_info["path"]).parent)
            
            # Store process info
            self.running_processes[app_info["name"]] = {
                "process": process,
                "start_time": time.time(),
                "app_info": app_info
            }
            
            logging.info(f"✅ Successfully launched standard {app_info['name']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to launch standard app: {e}")
            return False
    
    def _apply_enhancement_decorators(self, func: Callable, app_info: Dict[str, Any]) -> Callable:
        """Apply enhancement decorators to function"""
        enhanced_func = func
        
        # Apply decorators based on app type and current profile
        decorators_to_apply = [
            performance_monitor,
            error_handler(show_dialog=False),
            ui_theme_adapter(self.config.theme),
        ]
        
        # Add sleepy detection specific enhancements
        if "sleepy" in app_info["name"].lower() or "pose" in app_info["name"].lower():
            decorators_to_apply.extend([
                sleepy_detection_enhancer(sensitivity=0.8),
                pose_quality_filter(min_keypoints=10)
            ])
        
        # Add performance enhancements based on mode
        if self.config.performance_mode == "fast":
            decorators_to_apply.extend([
                smart_cache(cache_size=50, ttl_seconds=60),
                adaptive_processing(performance_target=45.0)
            ])
        elif self.config.performance_mode == "quality":
            decorators_to_apply.extend([
                data_pipeline(steps=["denoise", "sharpen", "contrast"]),
                result_aggregator(window_size=15, method="average")
            ])
        
        # Apply auto-save if enabled
        if self.config.auto_save:
            decorators_to_apply.append(auto_save_decorator(save_interval=300))
        
        # Apply all decorators
        for decorator in decorators_to_apply:
            try:
                enhanced_func = decorator(enhanced_func)
            except Exception as e:
                logging.warning(f"Failed to apply decorator {decorator}: {e}")
        
        return enhanced_func
    
    def get_running_apps(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently running apps"""
        running = {}
        
        for name, info in self.running_processes.items():
            runtime = time.time() - info["start_time"]
            
            status = "running"
            if "process" in info:
                if info["process"].poll() is not None:
                    status = "stopped"
            elif "thread" in info:
                if not info["thread"].is_alive():
                    status = "stopped"
            
            running[name] = {
                "status": status,
                "runtime": runtime,
                "app_info": info["app_info"]
            }
        
        return running
    
    def stop_app(self, app_name: str) -> bool:
        """Stop specific running application"""
        if app_name not in self.running_processes:
            return False
        
        try:
            info = self.running_processes[app_name]
            
            if "process" in info:
                info["process"].terminate()
            elif "thread" in info:
                # Note: Python threads cannot be forcefully terminated
                logging.warning(f"Cannot force-stop thread for {app_name}")
            
            del self.running_processes[app_name]
            logging.info(f"🛑 Stopped {app_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop {app_name}: {e}")
            return False
    
    def stop_all_apps(self):
        """Stop all running applications"""
        app_names = list(self.running_processes.keys())
        for app_name in app_names:
            self.stop_app(app_name)
    
    def get_enhancement_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get available enhancement profiles"""
        return self.enhancement_profiles
    
    def save_session(self, filename: str = None):
        """Save current session configuration"""
        if not filename:
            filename = f"session_{int(time.time())}.json"
        
        session_data = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "running_apps": self.get_running_apps(),
            "enhancement_profiles": self.enhancement_profiles
        }
        
        sessions_dir = Path("outputs/sessions")
        sessions_dir.mkdir(exist_ok=True, parents=True)
        
        session_file = sessions_dir / filename
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logging.info(f"💾 Session saved to {session_file}")
        return str(session_file)
    
    def load_session(self, filename: str):
        """Load session configuration"""
        session_file = Path(filename)
        
        if not session_file.exists():
            logging.error(f"Session file not found: {filename}")
            return False
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Restore configuration
            for key, value in session_data["config"].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logging.info(f"📂 Session loaded from {session_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load session: {e}")
            return False

# ============================================================================
# 🎯 Command Line Interface
# ============================================================================

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🚀 Unified Enhanced Sleepy Detection App Launcher"
    )
    
    parser.add_argument("--app", choices=["gui", "standalone", "web", "demo", "enhanced_ui"],
                       default="enhanced_ui", help="Application to launch")
    parser.add_argument("--profile", choices=["performance", "quality", "balanced", "debug", "production"],
                       default="balanced", help="Enhancement profile to use")
    parser.add_argument("--list-apps", action="store_true", help="List available applications")
    parser.add_argument("--list-profiles", action="store_true", help="List enhancement profiles")
    parser.add_argument("--config", help="Custom configuration file")
    parser.add_argument("--model", help="Model file to use")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--theme", choices=["dark", "light", "blue", "green"], 
                       default="dark", help="UI theme")
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = UnifiedAppLauncher()
    
    # Apply theme
    launcher.config.theme = args.theme
    
    # Load custom config if provided
    if args.config:
        launcher.load_session(args.config)
    
    # Handle list commands
    if args.list_apps:
        print("\n🎯 Available Applications:")
        print("=" * 50)
        apps = launcher.get_available_apps()
        for app_id, app_info in apps.items():
            print(f"{app_info['icon']} {app_id}: {app_info['name']}")
            print(f"   {app_info['description']}")
        return
    
    if args.list_profiles:
        print("\n⚙️ Enhancement Profiles:")
        print("=" * 50)
        profiles = launcher.get_enhancement_profiles()
        for profile_id, profile_info in profiles.items():
            print(f"🎛️ {profile_id}: {profile_info['description']}")
            print(f"   Decorators: {', '.join(profile_info['decorators'])}")
        return
    
    # Launch application
    custom_args = {}
    if args.model:
        custom_args["model"] = args.model
    if args.cam is not None:
        custom_args["cam"] = args.cam
    
    print(f"\n🚀 Launching {args.app} with {args.profile} profile...")
    success = launcher.launch_app(args.app, args.profile, custom_args)
    
    if success:
        print("✅ Application launched successfully!")
        print("\nPress Ctrl+C to exit...")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all applications...")
            launcher.stop_all_apps()
            print("👋 Goodbye!")
    else:
        print("❌ Failed to launch application")
        sys.exit(1)

if __name__ == "__main__":
    main()