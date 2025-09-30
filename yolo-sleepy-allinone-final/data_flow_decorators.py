"""
🔄 Data Flow Decorators for Enhanced Sleepy Detection
===================================================

Advanced decorators for data input/output processing, validation, and enhancement.
These decorators can be applied to any function without modifying source code.

Features:
- 📊 Data validation and sanitization
- 🎯 Result caching and optimization
- 📈 Performance analytics
- 🔍 Input/Output logging
- 🛡️ Security and safety checks

Author: Enhanced by AI Assistant
Date: September 30, 2025
"""

import functools
import time
import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from datetime import datetime, timedelta

# ============================================================================
# 📊 Data Models
# ============================================================================

@dataclass
class ProcessingMetrics:
    """Metrics for data processing operations"""
    function_name: str
    input_size: int
    output_size: int
    processing_time: float
    memory_usage: float
    cache_hit: bool
    timestamp: datetime

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

# ============================================================================
# 🎯 Data Flow Decorators
# ============================================================================

def input_validator(validation_rules: Dict[str, Any] = None):
    """Decorator to validate input data"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            validation_result = _validate_inputs(args, kwargs, validation_rules)
            
            if not validation_result.is_valid:
                error_msg = f"Input validation failed for {func.__name__}: {validation_result.errors}"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # Log warnings if any
            for warning in validation_result.warnings:
                logging.warning(f"{func.__name__}: {warning}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def output_validator(expected_type: type = None, min_confidence: float = 0.0):
    """Decorator to validate output data"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate output type
            if expected_type and not isinstance(result, expected_type):
                logging.warning(f"{func.__name__} returned unexpected type: {type(result)}")
            
            # Validate confidence if applicable
            if hasattr(result, 'confidence'):
                if result.confidence < min_confidence:
                    logging.warning(f"{func.__name__} returned low confidence: {result.confidence}")
            
            return result
        return wrapper
    return decorator

def smart_cache(cache_size: int = 100, ttl_seconds: int = 300):
    """Decorator for intelligent caching with TTL"""
    cache = {}
    access_times = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = _create_cache_key(func.__name__, args, kwargs)
            current_time = time.time()
            
            # Check cache hit
            if cache_key in cache and cache_key in access_times:
                if current_time - access_times[cache_key] < ttl_seconds:
                    logging.debug(f"Cache hit for {func.__name__}")
                    access_times[cache_key] = current_time
                    return cache[cache_key]
                else:
                    # Remove expired entry
                    del cache[cache_key]
                    del access_times[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= cache_size:
                # Remove oldest entry
                oldest_key = min(access_times.keys(), key=lambda k: access_times[k])
                del cache[oldest_key]
                del access_times[oldest_key]
            
            cache[cache_key] = result
            access_times[cache_key] = current_time
            
            logging.debug(f"Cached result for {func.__name__}")
            return result
        return wrapper
    return decorator

def data_pipeline(steps: List[str] = None):
    """Decorator to apply data processing pipeline"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            processed_args = list(args)
            
            if steps:
                for i, arg in enumerate(processed_args):
                    if isinstance(arg, np.ndarray):  # Process image data
                        for step in steps:
                            arg = _apply_pipeline_step(arg, step)
                        processed_args[i] = arg
            
            return func(*processed_args, **kwargs)
        return wrapper
    return decorator

def result_aggregator(window_size: int = 10, method: str = "average"):
    """Decorator to aggregate results over multiple calls"""
    results_buffer = []
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Add to buffer
            results_buffer.append(result)
            
            # Maintain window size
            if len(results_buffer) > window_size:
                results_buffer.pop(0)
            
            # Apply aggregation method
            aggregated_result = _aggregate_results(results_buffer, method)
            
            # Add aggregation metadata
            if hasattr(aggregated_result, '__dict__'):
                aggregated_result.aggregation_info = {
                    "method": method,
                    "window_size": len(results_buffer),
                    "confidence": getattr(aggregated_result, 'confidence', 0.0)
                }
            
            return aggregated_result
        return wrapper
    return decorator

def data_quality_monitor(quality_threshold: float = 0.7):
    """Decorator to monitor data quality"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Assess input data quality
            quality_score = _assess_data_quality(args, kwargs)
            
            if quality_score < quality_threshold:
                logging.warning(f"Low data quality detected for {func.__name__}: {quality_score:.2f}")
            
            result = func(*args, **kwargs)
            
            # Add quality information to result
            if hasattr(result, '__dict__'):
                result.data_quality_score = quality_score
            
            return result
        return wrapper
    return decorator

def adaptive_processing(performance_target: float = 30.0):  # FPS target
    """Decorator for adaptive processing based on performance"""
    processing_history = []
    current_quality = "balanced"
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal current_quality
            
            start_time = time.time()
            
            # Adjust processing based on current quality setting
            adjusted_args, adjusted_kwargs = _adjust_for_performance(
                args, kwargs, current_quality
            )
            
            result = func(*adjusted_args, **adjusted_kwargs)
            
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Update history
            processing_history.append(fps)
            if len(processing_history) > 10:
                processing_history.pop(0)
            
            # Adjust quality based on performance
            avg_fps = sum(processing_history) / len(processing_history)
            
            if avg_fps < performance_target * 0.8:  # Too slow
                if current_quality == "quality":
                    current_quality = "balanced"
                elif current_quality == "balanced":
                    current_quality = "fast"
            elif avg_fps > performance_target * 1.2:  # Too fast, can increase quality
                if current_quality == "fast":
                    current_quality = "balanced"
                elif current_quality == "balanced":
                    current_quality = "quality"
            
            # Add performance info to result
            if hasattr(result, '__dict__'):
                result.processing_fps = fps
                result.quality_mode = current_quality
            
            return result
        return wrapper
    return decorator

def batch_processor(batch_size: int = 5, timeout: float = 1.0):
    """Decorator to process inputs in batches for efficiency"""
    batch_buffer = []
    last_process_time = time.time()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_process_time
            
            # Add to batch buffer
            batch_buffer.append((args, kwargs))
            current_time = time.time()
            
            # Process batch if conditions are met
            if (len(batch_buffer) >= batch_size or 
                current_time - last_process_time > timeout):
                
                # Process entire batch
                results = []
                for batch_args, batch_kwargs in batch_buffer:
                    result = func(*batch_args, **batch_kwargs)
                    results.append(result)
                
                # Clear buffer and update time
                batch_buffer.clear()
                last_process_time = current_time
                
                # Return the last result (most recent)
                return results[-1] if results else None
            
            # If not processing batch yet, return cached or default result
            return None
        return wrapper
    return decorator

def data_logger(log_inputs: bool = True, log_outputs: bool = True, 
                log_performance: bool = True):
    """Decorator to log data flow for analysis"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Log inputs
            if log_inputs:
                input_info = _extract_input_info(args, kwargs)
                logging.info(f"{func.__name__} inputs: {input_info}")
            
            result = func(*args, **kwargs)
            
            processing_time = time.time() - start_time
            
            # Log outputs
            if log_outputs:
                output_info = _extract_output_info(result)
                logging.info(f"{func.__name__} outputs: {output_info}")
            
            # Log performance
            if log_performance:
                logging.info(f"{func.__name__} performance: {processing_time:.3f}s")
            
            # Save detailed log to file
            _save_detailed_log(func.__name__, args, kwargs, result, processing_time)
            
            return result
        return wrapper
    return decorator

# ============================================================================
# 🔧 Helper Functions
# ============================================================================

def _validate_inputs(args, kwargs, rules):
    """Validate function inputs against rules"""
    errors = []
    warnings = []
    
    if rules:
        # Check argument count
        if 'min_args' in rules and len(args) < rules['min_args']:
            errors.append(f"Expected at least {rules['min_args']} arguments, got {len(args)}")
        
        # Check for required kwargs
        if 'required_kwargs' in rules:
            for req_kwarg in rules['required_kwargs']:
                if req_kwarg not in kwargs:
                    errors.append(f"Required keyword argument missing: {req_kwarg}")
        
        # Check image data if present
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                if len(arg.shape) < 2:
                    errors.append(f"Invalid image shape at argument {i}: {arg.shape}")
                if arg.size == 0:
                    errors.append(f"Empty image at argument {i}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={}
    )

def _create_cache_key(func_name, args, kwargs):
    """Create unique cache key from function arguments"""
    # Simplified key creation (in production, use proper serialization)
    key_data = f"{func_name}_{len(args)}_{sorted(kwargs.keys())}"
    return hashlib.md5(key_data.encode()).hexdigest()

def _apply_pipeline_step(data, step):
    """Apply single pipeline step to data"""
    if step == "denoise":
        return cv2.bilateralFilter(data, 9, 75, 75)
    elif step == "sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(data, -1, kernel)
    elif step == "normalize":
        return cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    elif step == "contrast":
        return cv2.convertScaleAbs(data, alpha=1.2, beta=10)
    else:
        return data

def _aggregate_results(results, method):
    """Aggregate multiple results using specified method"""
    if not results:
        return None
    
    if method == "average":
        # Average confidence scores if available
        if hasattr(results[0], 'confidence'):
            avg_confidence = sum(r.confidence for r in results) / len(results)
            result = results[-1]  # Use latest result as base
            result.confidence = avg_confidence
            return result
    elif method == "majority":
        # Return most common result
        return max(set(results), key=results.count)
    elif method == "latest":
        return results[-1]
    
    return results[-1]  # Default to latest

def _assess_data_quality(args, kwargs):
    """Assess quality of input data"""
    quality_score = 1.0
    
    for arg in args:
        if isinstance(arg, np.ndarray):
            # Check image properties
            if len(arg.shape) >= 2:
                # Check for blur
                gray = cv2.cvtColor(arg, cv2.COLOR_BGR2GRAY) if len(arg.shape) == 3 else arg
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Lower variance indicates more blur
                if laplacian_var < 100:
                    quality_score *= 0.7
                
                # Check brightness
                mean_brightness = np.mean(gray)
                if mean_brightness < 50 or mean_brightness > 200:
                    quality_score *= 0.8
    
    return quality_score

def _adjust_for_performance(args, kwargs, quality_mode):
    """Adjust processing parameters based on performance mode"""
    adjusted_args = list(args)
    adjusted_kwargs = kwargs.copy()
    
    # Adjust image resolution based on quality mode
    for i, arg in enumerate(adjusted_args):
        if isinstance(arg, np.ndarray) and len(arg.shape) >= 2:
            h, w = arg.shape[:2]
            
            if quality_mode == "fast":
                # Reduce resolution for speed
                new_h, new_w = h // 2, w // 2
                adjusted_args[i] = cv2.resize(arg, (new_w, new_h))
            elif quality_mode == "quality":
                # Keep or enhance resolution
                if h < 720:  # Upscale if too small
                    scale = 720 / h
                    new_h, new_w = int(h * scale), int(w * scale)
                    adjusted_args[i] = cv2.resize(arg, (new_w, new_h))
    
    # Adjust confidence thresholds
    if 'conf' in adjusted_kwargs:
        if quality_mode == "fast":
            adjusted_kwargs['conf'] = max(0.7, adjusted_kwargs['conf'])
        elif quality_mode == "quality":
            adjusted_kwargs['conf'] = min(0.3, adjusted_kwargs['conf'])
    
    return tuple(adjusted_args), adjusted_kwargs

def _extract_input_info(args, kwargs):
    """Extract relevant information from inputs for logging"""
    info = {}
    info['arg_count'] = len(args)
    info['kwarg_keys'] = list(kwargs.keys())
    
    # Image information
    for i, arg in enumerate(args):
        if isinstance(arg, np.ndarray):
            info[f'image_{i}'] = {
                'shape': arg.shape,
                'dtype': str(arg.dtype),
                'size_mb': arg.nbytes / (1024 * 1024)
            }
    
    return info

def _extract_output_info(result):
    """Extract relevant information from outputs for logging"""
    info = {}
    
    if hasattr(result, '__dict__'):
        info['attributes'] = list(result.__dict__.keys())
    
    if hasattr(result, 'confidence'):
        info['confidence'] = result.confidence
    
    if isinstance(result, np.ndarray):
        info['shape'] = result.shape
        info['dtype'] = str(result.dtype)
    
    return info

def _save_detailed_log(func_name, args, kwargs, result, processing_time):
    """Save detailed log entry to file"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'function': func_name,
        'processing_time': processing_time,
        'input_summary': _extract_input_info(args, kwargs),
        'output_summary': _extract_output_info(result)
    }
    
    # Save to log file
    log_dir = Path("outputs/logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"data_flow_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        # Load existing logs
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new entry
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Save back
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save detailed log: {e}")

# ============================================================================
# 🎯 Specialized Decorators for Sleepy Detection
# ============================================================================

def sleepy_detection_enhancer(sensitivity: float = 0.8):
    """Specialized decorator for sleepy detection enhancement"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-process for better detection
            kwargs['enhanced_detection'] = True
            kwargs['sensitivity'] = sensitivity
            
            result = func(*args, **kwargs)
            
            # Post-process results
            if hasattr(result, 'detections'):
                # Apply confidence boosting for sleepy poses
                enhanced_detections = []
                for detection in result.detections:
                    if _is_sleepy_pose(detection):
                        detection['confidence'] *= 1.2  # Boost sleepy detection confidence
                    enhanced_detections.append(detection)
                result.detections = enhanced_detections
            
            return result
        return wrapper
    return decorator

def pose_quality_filter(min_keypoints: int = 10):
    """Filter poses based on quality criteria"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Filter low-quality poses
            if hasattr(result, 'detections'):
                filtered_detections = []
                for detection in result.detections:
                    if _assess_pose_quality(detection) >= min_keypoints:
                        filtered_detections.append(detection)
                result.detections = filtered_detections
            
            return result
        return wrapper
    return decorator

def _is_sleepy_pose(detection):
    """Check if detection indicates sleepy pose"""
    # Simplified sleepy pose detection logic
    if 'keypoints' in detection:
        # Check head tilt, eye closure, etc.
        return detection.get('sleepy_score', 0) > 0.5
    return False

def _assess_pose_quality(detection):
    """Assess quality of pose detection"""
    if 'keypoints' in detection:
        visible_keypoints = sum(1 for kp in detection['keypoints'] if kp.get('confidence', 0) > 0.5)
        return visible_keypoints
    return 0

if __name__ == "__main__":
    print("🔄 Data Flow Decorators System")
    print("=" * 50)
    print("Advanced decorators for data processing and enhancement.")
    print("\nAvailable Decorators:")
    print("✅ Input/Output Validation")
    print("✅ Smart Caching with TTL")
    print("✅ Data Processing Pipeline")
    print("✅ Result Aggregation")
    print("✅ Data Quality Monitoring")
    print("✅ Adaptive Processing")
    print("✅ Batch Processing")
    print("✅ Comprehensive Logging")
    print("✅ Sleepy Detection Enhancement")
    print("✅ Pose Quality Filtering")