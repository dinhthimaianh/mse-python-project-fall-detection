# app/utils/helpers.py
import os
import json
import numpy as np
import cv2
from PIL import Image
import logging
import time
import hashlib
import psutil
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import yaml

def setup_directories(base_path: str = ".") -> Dict[str, str]:
    """Create necessary directories for the application"""
    directories = {
        'logs': os.path.join(base_path, 'logs'),
        'models': os.path.join(base_path, 'models'),
        'config': os.path.join(base_path, 'config'),
        'data': os.path.join(base_path, 'data'),
        'temp': os.path.join(base_path, 'temp')
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        
    return directories

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file with environment variable substitution"""
    try:
        with open(config_path, 'r') as file:
            content = file.read()
            
        # Replace environment variables
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        content = re.sub(r'\$\{([^}:]+)(?::([^}]*))?\}', replace_env_var, content)
        
        return yaml.safe_load(content)
        
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return {}

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """Save data to JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
        
    except Exception as e:
        logging.error(f"Failed to save JSON to {filepath}: {e}")
        return False

def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {filepath}: {e}")
        return None

def resize_image_keep_aspect(image: Union[Image.Image, np.ndarray], 
                            target_size: Tuple[int, int]) -> Union[Image.Image, np.ndarray]:
    """Resize image while keeping aspect ratio"""
    if isinstance(image, np.ndarray):
        # OpenCV format
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to target size
        if new_w != target_w or new_h != target_h:
            # Calculate padding
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            
            if len(image.shape) == 3:
                padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            else:
                padded = np.zeros((target_h, target_w), dtype=image.dtype)
                padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
                
            return padded
        
        return resized
        
    else:
        # PIL format
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / image.width, target_h / image.height)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        
        # Resize
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Pad to target size
        if new_w != target_w or new_h != target_h:
            padded = Image.new(image.mode, (target_w, target_h), (0, 0, 0))
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            padded.paste(resized, (pad_w, pad_h))
            return padded
        
        return resized

def calculate_fps(timestamps: List[float], window_size: int = 10) -> float:
    """Calculate FPS from list of timestamps"""
    if len(timestamps) < 2:
        return 0.0
    
    # Use last 'window_size' timestamps
    recent_timestamps = timestamps[-window_size:]
    
    if len(recent_timestamps) < 2:
        return 0.0
    
    time_span = recent_timestamps[-1] - recent_timestamps[0]
    if time_span <= 0:
        return 0.0
    
    return (len(recent_timestamps) - 1) / time_span

def get_system_metrics() -> Dict[str, float]:
    """Get current system performance metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_usage': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
        
    except Exception as e:
        logging.error(f"Failed to get system metrics: {e}")
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'memory_available_gb': 0.0,
            'disk_usage': 0.0,
            'disk_free_gb': 0.0
        }

def generate_unique_id(prefix: str = "") -> str:
    """Generate unique ID based on timestamp and random hash"""
    timestamp = str(int(time.time() * 1000))
    random_str = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    return f"{prefix}{timestamp}_{random_str}" if prefix else f"{timestamp}_{random_str}"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def validate_camera_device(device_id: int) -> bool:
    """Check if camera device is available"""
    try:
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except:
        return False

def get_available_cameras() -> List[int]:
    """Get list of available camera device IDs"""
    available_cameras = []
    
    # Test first 10 camera indices
    for i in range(10):
        if validate_camera_device(i):
            available_cameras.append(i)
    
    return available_cameras

def normalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Normalize keypoints to 0-1 range"""
    normalized = keypoints.copy()
    
    if len(normalized.shape) == 2:
        # Format: [[y, x], [y, x], ...]
        normalized[:, 0] = normalized[:, 0] / image_height  # y coordinates
        normalized[:, 1] = normalized[:, 1] / image_width   # x coordinates
    elif len(normalized.shape) == 3:
        # Format: [[[y, x]], [[y, x]], ...]
        normalized[:, :, 0] = normalized[:, :, 0] / image_height
        normalized[:, :, 1] = normalized[:, :, 1] / image_width
    
    return normalized

def denormalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Denormalize keypoints from 0-1 range to pixel coordinates"""
    denormalized = keypoints.copy()
    
    if len(denormalized.shape) == 2:
        denormalized[:, 0] = denormalized[:, 0] * image_height
        denormalized[:, 1] = denormalized[:, 1] * image_width
    elif len(denormalized.shape) == 3:
        denormalized[:, :, 0] = denormalized[:, :, 0] * image_height
        denormalized[:, :, 1] = denormalized[:, :, 1] * image_width
    
    return denormalized

def draw_pose_on_image(image: np.ndarray, keypoints: np.ndarray, 
                      confidences: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
    """Draw pose keypoints and connections on image"""
    try:
        # Create a copy to avoid modifying original
        img_with_pose = image.copy()
        
        # Define colors for different body parts
        colors = {
            'head': (255, 0, 0),      # Red
            'torso': (0, 255, 0),     # Green
            'arms': (0, 0, 255),      # Blue
            'legs': (255, 255, 0)     # Yellow
        }
        
        # Define connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head
            (5, 6), (5, 7), (7, 9),          # left arm
            (6, 8), (8, 10),                 # right arm
            (5, 11), (6, 12),                # shoulders to hips
            (11, 12),                        # hip connection
            (11, 13), (13, 15),              # left leg
            (12, 14), (14, 16)               # right leg
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                confidences[start_idx] > confidence_threshold and 
                confidences[end_idx] > confidence_threshold):
                
                start_point = (int(keypoints[start_idx][1]), int(keypoints[start_idx][0]))
                end_point = (int(keypoints[end_idx][1]), int(keypoints[end_idx][0]))
                
                cv2.line(img_with_pose, start_point, end_point, (255, 255, 255), 2)
        
        # Draw keypoints
        for i, (keypoint, confidence) in enumerate(zip(keypoints, confidences)):
            if confidence > confidence_threshold:
                x, y = int(keypoint[1]), int(keypoint[0])
                
                # Choose color based on body part
                if i <= 4:  # head
                    color = colors['head']
                elif i <= 6:  # shoulders
                    color = colors['torso']
                elif i <= 10:  # arms
                    color = colors['arms']
                elif i <= 12:  # hips
                    color = colors['torso']
                else:  # legs
                    color = colors['legs']
                
                cv2.circle(img_with_pose, (x, y), 4, color, -1)
                cv2.circle(img_with_pose, (x, y), 6, (255, 255, 255), 2)
        
        return img_with_pose
        
    except Exception as e:
        logging.error(f"Failed to draw pose on image: {e}")
        return image

def create_grid_image(images: List[np.ndarray], titles: List[str] = None, 
                     grid_size: Tuple[int, int] = None) -> np.ndarray:
    """Create a grid image from multiple images"""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    # Get image dimensions (assume all images are same size)
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    # Create grid
    grid_h = rows * h
    grid_w = cols * w
    
    if channels == 3:
        grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    else:
        grid_image = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    # Place images in grid
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        start_y = row * h
        end_y = start_y + h
        start_x = col * w
        end_x = start_x + w
        
        if channels == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif channels == 1 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        grid_image[start_y:end_y, start_x:end_x] = img
        
        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(grid_image, titles[i], 
                       (start_x + 10, start_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return grid_image

class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logging.debug(f"{self.name} took {duration:.3f} seconds")
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

class RollingAverage:
    """Calculate rolling average of values"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def add(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    @property
    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    def reset(self):
        self.values.clear()