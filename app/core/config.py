# app/core/config.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import time
@dataclass
class CameraConfig:
    '''Configuration for camera settings
    @dataclass: Decorator tự động tạo __init__, __repr__, __eq__ methods
    device_id: Camera nào sẽ được sử dụng (0, 1, 2... hoặc đường dẫn video file)
    resolution: Kích thước frame camera capture
    fps: Tốc độ camera (30fps = mượt, nhưng tốn CPU)
    buffer_size: Số frame giữ trong buffer để tránh lag
    capture_interval: Thời gian delay giữa các lần xử lý (để giảm CPU load)
    '''
    # Camera sẽ capture với 30fps nhưng chỉ process 5fps (mỗi 0.2s)
    # Điều này giúp giảm tải CPU while vẫn đảm bảo real-time
    device_id: int = 0                    # Camera ID (0=webcam mặc định)
    resolution: tuple = (640, 480)       # Độ phân giải camera (width, height)
    fps: int = 30                        # Frame per second của camera
    buffer_size: int = 3                 # Số frame buffer trong memory
    capture_interval: float = 0.2        # Khoảng thời gian giữa 2 lần capture (giây)
                                        # 0.2s = 5 FPS thực tế để xử lý
                                        # # 5 FPS instead of 2 FPS

@dataclass
class DetectionConfig:
    model_path: str = "models/posenet_mobilenet_v1.tflite"  # Đường dẫn model AI
    confidence_threshold: float = 0.15   # Ngưỡng tin cậy cho pose
    fall_threshold: Dict[str, float] = None # Các ngưỡng cho fall detection
    model_name: str = "mobilenet"  # Support model_name
    def __post_init__(self):
        '''Set default fall detection thresholds if not provided
        Giải thích các threshold:

            height_ratio: 0.3:

            Người đứng: height_ratio ≈ 0.8-1.0
            Người nằm: height_ratio ≈ 0.2-0.4
            Ngưỡng 0.3 = nếu < 0.3 thì được coi là đã nằm
            angle_threshold: 60:

            Người đứng: góc ≈ 0-20°
            Người ngã: góc ≈ 45-90°
            Ngưỡng 60° = nếu > 60° thì được coi là ngã
            velocity_threshold: 0.5:

            Di chuyển bình thường: velocity < 0.3
            Té ngã: velocity > 0.5 (di chuyển nhanh đột ngột)
            duration_threshold: 2.0:

            Phải duy trì trạng thái "fall" trong 2 giây để xác nhận
            Tránh false positive từ động tác ngồi/cúi nhanh

        '''
        
        if self.fall_threshold is None:
            self.fall_threshold = {
                "height_ratio": 0.3,        # Tỷ lệ chiều cao (người nằm vs đứng)
                "angle_threshold": 60,      # Góc nghiêng cơ thể (độ)
                "velocity_threshold": 0.5,  # Tốc độ di chuyển
                "duration_threshold": 2.0   # Thời gian duy trì trạng thái fall
            }

@dataclass
class NotificationConfig:
    sms_enabled: bool = False
    telegram_enabled: bool = True
    email_enabled: bool = True
    cooldown_period: int = 30  # seconds # Tránh spam notifications (30 giây)
     # Email settings
    email: dict =None
    
    # Telegram settings  
    telegram: dict = None

    # Emergency contacts
    emergency_contacts: list = None
    
    def __post_init__(self):
        if self.emergency_contacts is None:
            self.emergency_contacts = []
        if self.email is None:
            self.email = {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "anhdtm0897@gmail.com",
                "sender_password": "kwucalgywwnuroas",  # Gmail App Password
                "use_tls": True
                }
    
        
        if self.telegram is None:
            self.telegram = {
                'enabled': True,
                'bot_token': '',
                'chat_ids': []
            }
    def to_dict(self) -> Dict[str, Any]:
        """Convert NotificationConfig to dictionary"""
        return {
            'email_enabled': self.email_enabled,
            'telegram_enabled': self.telegram_enabled,
            'cooldown_period': self.cooldown_period,
            'email': self.email,
            'telegram': self.telegram,
            'emergency_contacts': self.emergency_contacts
        }
@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    url: str = "sqlite:///fall_detection.db"
    echo: bool = False

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True

@dataclass
class CloudServerConfig:
    '''Configuration for cloud server settings
    Giải thích các trường:

        base_url: URL của cloud server
        timeout: Thời gian chờ khi gửi request (giây)
        system_id: ID duy nhất của hệ thống (để phân biệt các thiết bị)
        # stats: Thống kê gửi request thành công/thất bại
    '''
    
    
    cloud_service_url: str = "https://cloud.example.com/api"
    api_key: str = ""
    system_id: str = f'system_{int(time.time())}'  # Unique ID for this system
    timeout: int = 10  # seconds
    health_interval: int = 300  # seconds (5 minutes)
    def to_dict(self) -> Dict[str, Any]:
        """Convert NotificationConfig to dictionary"""
        return {
            'cloud_service_url': self.cloud_service_url,
            'api_key': self.api_key,    
            'system_id': self.system_id,
            'timeout': self.timeout,
            'health_interval': self.health_interval
        }

class Config:
    '''Singleton class to manage application configuration
    Giải thích workflow:

        Try load YAML: Đọc file config.yaml
        Parse sections: Chia thành các section (camera, detection, etc.)
        Create objects: Tạo dataclass objects với YAML data
        Fallback: Nếu lỗi, sử dụng default values
'''
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        print(f"Loading configuration from {self.config_path}")
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            # Khởi tạo các config objects từ YAML data
            self.camera = CameraConfig(**config_data.get('camera', {}))
            self.detection = DetectionConfig(**config_data.get('detection', {}))
            self.notification = NotificationConfig(**config_data.get('notification', {}))
            self.database = DatabaseConfig(**config_data.get('database', {}))
            self.api = APIConfig(**config_data.get('api', {}))
            self.cloud_server = CloudServerConfig(**config_data.get('cloud_server', {}))
            
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using defaults.")
            self._load_defaults()
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.camera = CameraConfig()
        self.detection = DetectionConfig()
        self.notification = NotificationConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()

# Singleton instance
config = Config()
# Giải thích:

# Tạo 1 instance duy nhất của Config
# Toàn bộ app sẽ dùng chung instance này
# Import: from app.core.config import config