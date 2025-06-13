# run_local.py (Enhanced & Complete Version)
"""
Fall Detection System - Local Production Runner
Run the complete fall detection system on local machine
"""

import os
import sys
import time
import signal
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import io

# Add app to Python path
sys.path.append(str(Path(__file__).parent))

# Import all components
from app.services.display_services import DisplayService
import app.core.config as config_
from app.core.database import DatabaseManager
from app.services.camera_service import MultiCameraManager, FrameData
from app.services.notification_service import NotificationService, AlertEvent
from app.services.api_service import APIService
from app.utils.helpers import setup_directories, get_system_metrics, Timer
from app.core.logger import setup_logging
from app.models.pipeline_fall_detector import ProductionPipelineFallDetector
from app.models.h5_fall_detector_realtime import H5FallDetectorRealtime
from app.models.h5_fall_detector import H5FallDetector
class FallDetectionApp:
    """Main Fall Detection Application"""
    
    def __init__(self):
        # Setup logging first
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = config_.config
        
        # Create directories
        setup_directories()
        
        # Initialize component variables
        self.db_manager = None
        self.fall_detector = None
        self.camera_manager = None
        self.notification_service = None
        self.api_service = None
        self.display_service = None
        
        # System state
        self.running = False
        self.start_time = None
        
        # Fall detection state tracking (Enhanced cooldown)
        self.last_fall_detection_time = 0.0
        self.fall_detection_cooldown = 5.0  # 5 seconds between fall detections
        self.consecutive_fall_detections = 0
        self.max_consecutive_falls = 3  # Max consecutive falls before forcing cooldown
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'falls_detected': 0,
            'notifications_sent': 0,
            'notifications_suppressed': 0,
            'false_alarms': 0,
            'system_errors': 0,
            'uptime_seconds': 0
        }
        
        # Performance tracking
        self.performance = {
            'avg_processing_time': 0.0,
            'avg_fps': 0.0,
            'detection_latency': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        
        self.logger.info(" Fall Detection Application initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info(" Initializing system components...")

            # 1. Initialize database
            self.logger.info("   Setting up database...")
            self.db_manager = DatabaseManager(self.config.database.url)
            
            # 2. Initialize display service
            self.logger.info("   Setting up display service...")
            self.display_service = DisplayService(enable_display=True)
              # 3. Initialize fall detector
            self.logger.info("   Loading fall detection model...")
            if not os.path.exists(self.config.detection.model_path):
                self.logger.error(f" Model file not found: {self.config.detection.model_path}")
                self.logger.info("Please check model path in config")
                return False
            
            # Determine model type based on file extension
            model_path = self.config.detection.model_path
            if model_path.endswith('.h5'):
                # Use H5 detector (like fall_detection_realtime.py)
                self.logger.info("   Using H5 model detector...")
                self.fall_detector = H5FallDetectorRealtime(
                    model_path=model_path,
                    confidence_threshold=self.config.detection.confidence_threshold
                )
            elif model_path.endswith('.tflite'):
                # Use TFLite pipeline detector
                self.logger.info("   Using TFLite pipeline detector...")
                # Enhanced pipeline configuration
                pipeline_config = {
                    'temporal_window': 3,
                    'confidence_boost_factor': 1.2,
                    'false_positive_threshold': 2,
                    'edgetpu_model_path': None  # Set if you have EdgeTPU model
                }
                
                self.fall_detector = ProductionPipelineFallDetector(
                    model_path=model_path,
                    model_name="mobilenet",  # or "movenet"
                    confidence_threshold=self.config.detection.confidence_threshold,
                    config=pipeline_config
                )
            else:
                self.logger.error(f" Unsupported model format: {model_path}")
                self.logger.info("Supported formats: .h5, .tflite")
                return False
            
            # 4. Initialize notification service
            self.logger.info("   Setting up notification service...")
            # Convert config to dict if it's an object
            if hasattr(self.config.notification, 'to_dict'):
                notification_config = self.config.notification.to_dict()
            else:
                notification_config = self.config.notification
            
            self.notification_service = NotificationService(notification_config)
            
            # 5. Initialize camera manager
            self.logger.info("   Setting up camera system...")
            self.camera_manager = MultiCameraManager()
            
            # Add primary camera
            camera_config = {
                'device_id': self.config.camera.device_id,
                'resolution': self.config.camera.resolution,
                'capture_interval': getattr(self.config.camera, 'capture_interval', 1.0 / self.config.camera.fps)
            }
            
            camera_id = self.camera_manager.add_camera(
                camera_config, 
                self._process_frame_callback
            )
            
            if not camera_id:
                self.logger.error(" Failed to initialize camera")
                return False
            
            self.logger.info(f"   Camera {camera_id} initialized")
            
            # 6. Start display service
            if self.display_service:
                self.display_service.start()
                self.logger.info("   Display service started")
            
            # 7. Initialize API service
            self.logger.info("   Setting up API service...")
            api_config = {
                'host': self.config.api.host,
                'port': self.config.api.port,
                'debug': getattr(self.config.api, 'debug', False),
                'cors_enabled': getattr(self.config.api, 'cors_enabled', True)
            }
            
            self.api_service = APIService(
                api_config,
                self.db_manager,
                self.notification_service,
                self.camera_manager
            )
            
            self.logger.info(" All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f" Component initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _process_frame_callback(self, frame_data: FrameData):
        """Callback function to process each frame"""
        try:
            start_time = time.time()
            
            # Process frame for fall detection
            result = self.fall_detector.process_image(frame_data.image)
            
            # Update display
            if self.display_service:
                self.display_service.update_frame(frame_data, result)
            
            processing_time = time.time() - start_time
            
            if result is None:
                return
            
            # Update statistics
            self.stats['frames_processed'] += 1
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            # Log every 100 frames
            if self.stats['frames_processed'] % 100 == 0:
                self.logger.info(
                    f" Processed {self.stats['frames_processed']} frames, "
                    f"Avg FPS: {self.performance['avg_fps']:.1f}, "
                    f"Avg processing: {self.performance['avg_processing_time']:.3f}s"
                )
            
            # Check for fall detection
            if result.get('fall_detected', False):
                self._handle_fall_detection(frame_data, result)
            else:
                # Reset consecutive counter nếu không detect fall
                self.consecutive_fall_detections = 0
            
        except Exception as e:
            self.stats['system_errors'] += 1
            self.logger.error(f" Frame processing error: {e}")
    
    def _handle_fall_detection(self, frame_data: FrameData, detection_result: Dict[str, Any]):
        """Handle fall detection event với enhanced cooldown logic và safe type conversion"""
        current_time = time.time()
        
        # Local fall detection cooldown (ngăn spam detection)
        time_since_last_fall = current_time - self.last_fall_detection_time
        
        if time_since_last_fall < self.fall_detection_cooldown:
            self.logger.debug(f" Fall detection suppressed locally (cooldown: {self.fall_detection_cooldown - time_since_last_fall:.1f}s)")
            return
        
        # Track consecutive falls
        self.consecutive_fall_detections += 1
        self.last_fall_detection_time = current_time
        
        try:
            self.stats['falls_detected'] += 1
            
            # Safe extraction với debugging
            self.logger.debug(f" Detection result keys: {list(detection_result.keys())}")
            self.logger.debug(f" Detection result: {detection_result}")
            
            # Safe confidence extraction
            confidence_raw = detection_result.get('confidence', 0)
            if isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
            elif isinstance(confidence_raw, dict):
                # Nếu confidence là dict, lấy value hoặc tính average
                if 'value' in confidence_raw:
                    confidence = float(confidence_raw['value'])
                elif 'avg' in confidence_raw:
                    confidence = float(confidence_raw['avg'])
                else:
                    # Nếu confidence dict có multiple values, lấy trung bình
                    numeric_values = [v for v in confidence_raw.values() if isinstance(v, (int, float))]
                    confidence = float(sum(numeric_values) / len(numeric_values)) if numeric_values else 0.0
            else:
                confidence = 0.0
            
            # Safe angle extraction  
            angle_candidates = [
                detection_result.get('leaning_angle'),
                detection_result.get('body_angle'),
                detection_result.get('angle'),
                detection_result.get('pose_angle')
            ]
            
            angle = 0.0
            for angle_candidate in angle_candidates:
                if angle_candidate is not None:
                    if isinstance(angle_candidate, (int, float)):
                        angle = float(angle_candidate)
                        break
                    elif isinstance(angle_candidate, dict) and 'value' in angle_candidate:
                        angle = float(angle_candidate['value'])
                        break
            
            # Safe keypoint correlation extraction
            keypoint_corr_candidates = [
                detection_result.get('keypoint_correlation'),
                detection_result.get('keypoint_corr'),
                detection_result.get('stability'),
                detection_result.get('pose_stability')
            ]
            
            keypoint_corr = 0.0
            for corr_candidate in keypoint_corr_candidates:
                if corr_candidate is not None:
                    if isinstance(corr_candidate, (int, float)):
                        keypoint_corr = float(corr_candidate)
                        break
                    elif isinstance(corr_candidate, dict) and 'value' in corr_candidate:
                        keypoint_corr = float(corr_candidate['value'])
                        break
            
            self.logger.warning(f" FALL DETECTED #{self.consecutive_fall_detections}")
            self.logger.warning(f"   Confidence: {confidence:.1%}")
            self.logger.warning(f"   Body Angle: {angle:.1f}°")
            self.logger.warning(f"   Keypoint Correlation: {keypoint_corr:.3f}")
            self.logger.warning(f"   Camera: {frame_data.camera_id}")
            self.logger.warning(f"   Time: {datetime.fromtimestamp(frame_data.timestamp)}")
            self.logger.warning(f"   Time since last: {time_since_last_fall:.1f}s")
            
            # Check notification service cooldown status
            if self.notification_service:
                cooldown_status = self.notification_service.get_cooldown_status()
                self.logger.info(f"    Notification cooldown: {cooldown_status['remaining_seconds']:.1f}s remaining")
            
            # Prepare incident data với safe type conversion
            incident_data = {
                'camera_id': str(frame_data.camera_id),
                'timestamp': frame_data.timestamp,
                'confidence': confidence,  # Already converted to float
                'angle': angle,  # Already converted to float  
                'keypoint_corr': keypoint_corr,  # Already converted to float
                'pose_data': detection_result.get('pose_data'),  # Keep as dict
                'location': self._get_camera_location(frame_data.camera_id)
            }
            
            # Debug log incident data
            self.logger.debug(f" Incident data: {incident_data}")
            
            # Save to database
            incident = self.db_manager.save_incident(incident_data)
            self.logger.info(f" Incident saved with ID: {incident.id}")
            
            # Send notifications asynchronously
            notification_thread = threading.Thread(
                target=self._send_notifications_async,
                args=(incident, frame_data),
                daemon=True
            )
            notification_thread.start()
            
            # Force longer cooldown nếu quá nhiều consecutive falls
            if self.consecutive_fall_detections >= self.max_consecutive_falls:
                self.logger.warning(f" Too many consecutive falls ({self.consecutive_fall_detections}). Extending local cooldown.")
                self.fall_detection_cooldown = 30.0  # Extend to 30 seconds
                self.consecutive_fall_detections = 0
            
        except Exception as e:
            self.stats['system_errors'] += 1
            self.logger.error(f" Fall detection handling error: {e}")
            self.logger.error(f"   Detection result: {detection_result}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _send_notifications_async(self, incident, frame_data: FrameData):
        """Send notifications in background thread"""
        try:
            # Capture frame image
            frame_image = self._capture_frame_image(frame_data)
            
            # Prepare notification data
            alert_event = AlertEvent(
                incident_id=incident.id,
                timestamp=incident.timestamp,
                confidence=incident.confidence,
                location=incident.location,
                camera_id=incident.camera_id,
                leaning_angle=incident.leaning_angle,
                image_data=frame_image
            )
            
            # Send notifications (cooldown handled in service)
            results = self.notification_service.send_fall_alert(alert_event)
            
            # Update statistics
            if results:
                successful_notifications = sum(1 for r in results if r.success)
                self.stats['notifications_sent'] += successful_notifications
                
                self.logger.info(f" Notifications: {successful_notifications}/{len(results)} sent successfully")
                
                # Log notification details
                for result in results:
                    status = "" if result.success else "FAILED"
                    self.logger.info(f"   {status} {result.method} to {result.contact_name}: {result.message}")
            else:
                self.stats['notifications_suppressed'] += 1
                self.logger.info(" Notifications suppressed by cooldown")
            
            # Update incident with notification results
            if results:
                notifications_sent = [
                    {
                        'method': result.method,
                        'success': result.success,
                        'message': result.message,
                        'timestamp': result.timestamp.isoformat(),
                        'contact_name': result.contact_name
                    }
                    for result in results
                ]
                
                # Update database
                session = self.db_manager.get_session()
                try:
                    # Refresh incident and update
                    session.refresh(incident)
                    # Add notifications_sent field if your model supports it
                    # incident.notifications_sent = notifications_sent
                    session.commit()
                    self.logger.debug(f" Updated incident {incident.id} with notification results")
                except Exception as db_error:
                    self.logger.warning(f" Failed to update incident with notifications: {db_error}")
                    session.rollback()
                finally:
                    session.close()
            
        except Exception as e:
            self.stats['system_errors'] += 1
            self.logger.error(f" Notification sending error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _get_camera_location(self, camera_id: str) -> str:
        """Get location name for camera"""
        location_map = {
            'camera_0': 'Living Room',
            'camera_1': 'Bedroom', 
            'camera_2': 'Kitchen',
            'camera_3': 'Bathroom',
            'camera_4': 'Hallway'
        }
        return location_map.get(camera_id, f'Camera {camera_id}')
    
    def _capture_frame_image(self, frame_data: FrameData) -> Optional[bytes]:
        """Capture frame as image bytes for notification"""
        try:
            if hasattr(frame_data, 'image') and frame_data.image:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                frame_data.image.save(img_byte_arr, format='JPEG', quality=85)
                return img_byte_arr.getvalue()
            return None
        except Exception as e:
            self.logger.warning(f" Failed to capture frame image: {e}")
            return None
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        # Rolling average for processing time
        alpha = 0.1  # Smoothing factor
        self.performance['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.performance['avg_processing_time']
        )
        
        # Calculate FPS
        if processing_time > 0:
            current_fps = 1.0 / processing_time
            self.performance['avg_fps'] = (
                alpha * current_fps + 
                (1 - alpha) * self.performance['avg_fps']
            )
    
    def _log_system_health(self):
        """Log system health metrics periodically"""
        while self.running:
            try:
                # Get system metrics
                system_metrics = get_system_metrics()
                
                # Get camera stats
                camera_stats = self.camera_manager.get_all_stats()
                healthy_cameras = self.camera_manager.get_healthy_cameras()
                
                # Update performance metrics
                self.performance['memory_usage'] = float(system_metrics.get('memory_usage', 0))
                self.performance['cpu_usage'] = float(system_metrics.get('cpu_usage', 0))
                
                # Log health data
                health_data = {
                    'timestamp': datetime.now(),
                    'cpu_usage': float(system_metrics.get('cpu_usage', 0)),
                    'memory_usage': float(system_metrics.get('memory_usage', 0)),
                    'disk_usage': float(system_metrics.get('disk_usage', 0)),
                    'camera_status': 'online' if healthy_cameras else 'offline',
                    'frame_rate': float(self.performance['avg_fps']),
                    'detection_latency': float(self.performance['avg_processing_time']),
                    'network_status': 'online',
                    'api_response_time': 0.0
                }
                
                self.db_manager.log_system_health(health_data)
                
                # Log to file every 5 minutes
                self.logger.info(
                    f" System Health - CPU: {system_metrics.get('cpu_usage', 0):.1f}%, "
                    f"Memory: {system_metrics.get('memory_usage', 0):.1f}%, "
                    f"Cameras: {len(healthy_cameras)}/{len(camera_stats)}, "
                    f"FPS: {self.performance['avg_fps']:.1f}"
                )
                
                # Update uptime
                if self.start_time:
                    self.stats['uptime_seconds'] = int(time.time() - self.start_time)
                
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f" Health monitoring error: {e}")
                time.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including cooldowns"""
        uptime = int(time.time() - self.start_time) if self.start_time else 0
        
        return {
            'system': {
                'running': self.running,
                'uptime_seconds': uptime,
                'uptime_formatted': f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s"
            },
            'fall_detection': {
                'consecutive_falls': self.consecutive_fall_detections,
                'local_cooldown_seconds': self.fall_detection_cooldown,
                'last_detection': datetime.fromtimestamp(self.last_fall_detection_time) if self.last_fall_detection_time > 0 else None
            },
            'notifications': self.notification_service.get_cooldown_status() if self.notification_service else {},
            'cameras': self.camera_manager.get_all_stats() if self.camera_manager else {},
            'performance': self.performance,
            'statistics': self.stats
        }
    
    def start(self):
        """Start the fall detection system"""
        try:
            self.logger.info(" Starting Fall Detection System...")
            
            # Initialize components
            if not self.initialize_components():
                self.logger.error(" Failed to initialize components")
                return False
            
            self.running = True
            self.start_time = time.time()
            
            # Start health monitoring
            health_thread = threading.Thread(target=self._log_system_health, daemon=True)
            health_thread.start()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info(" Fall Detection System started successfully!")
            self.logger.info(f" Web API: http://{self.config.api.host}:{self.config.api.port}")
            self.logger.info(f" Dashboard: http://{self.config.api.host}:{self.config.api.port}/docs")
            self.logger.info(" Monitoring for falls... Press Ctrl+C to stop")
            
            # Print initial system status
            self._print_startup_info()
            
            # Start API server (this blocks)
            self.api_service.start()
            
        except KeyboardInterrupt:
            self.logger.info(" Received interrupt signal")
        except Exception as e:
            self.logger.error(f" System startup failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self.stop()
        
        return True
    
    def _print_startup_info(self):
        """Print startup information"""
        self.logger.info(" System Configuration:")
        self.logger.info(f"    Camera Device: {self.config.camera.device_id}")
        self.logger.info(f"    Resolution: {self.config.camera.resolution}")
        self.logger.info(f"    Confidence Threshold: {self.config.detection.confidence_threshold}")
        self.logger.info(f"    Cooldown Period: {getattr(self.config.notification, 'cooldown_period', 'N/A')}s")
        self.logger.info(f"    Email Enabled: {getattr(self.config.notification, 'email_enabled', False)}")
        self.logger.info(f"    Telegram Enabled: {getattr(self.config.notification, 'telegram_enabled', False)}")
    
    def stop(self):
        """Stop the fall detection system gracefully"""
        if not self.running:
            return
        
        self.logger.info(" Stopping Fall Detection System...")
        self.running = False
        
        try:
            # Stop components in reverse order
            if self.api_service:
                self.logger.info("    Stopping API service...")
                self.api_service.stop()
            
            if self.camera_manager:
                self.logger.info("    Stopping cameras...")
                self.camera_manager.stop_all()
            
            if self.display_service:
                self.logger.info("    Stopping display...")
                self.display_service.stop()
            
            # Calculate final statistics
            uptime = time.time() - self.start_time if self.start_time else 0
            
            self.logger.info(" Final System Statistics:")
            self.logger.info(f"      Total uptime: {uptime:.1f} seconds ({uptime/3600:.1f} hours)")
            self.logger.info(f"     Frames processed: {self.stats['frames_processed']:,}")
            self.logger.info(f"     Falls detected: {self.stats['falls_detected']}")
            self.logger.info(f"     Notifications sent: {self.stats['notifications_sent']}")
            self.logger.info(f"     Notifications suppressed: {self.stats['notifications_suppressed']}")
            self.logger.info(f"     System errors: {self.stats['system_errors']}")
            self.logger.info(f"     Average FPS: {self.performance['avg_fps']:.1f}")
            self.logger.info(f"    Average processing time: {self.performance['avg_processing_time']:.3f}s")
            
            if self.stats['frames_processed'] > 0:
                error_rate = (self.stats['system_errors'] / self.stats['frames_processed']) * 100
                detection_rate = (self.stats['falls_detected'] / self.stats['frames_processed']) * 100
                self.logger.info(f"     Error rate: {error_rate:.3f}%")
                self.logger.info(f"     Detection rate: {detection_rate:.3f}%")
            
            self.logger.info(" System stopped successfully")
            
        except Exception as e:
            self.logger.error(f" Error during shutdown: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f" Received signal {signum}")
        self.stop()
        sys.exit(0)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print(" Checking prerequisites...")
    
    issues = []
    
    # Check Python packages
    required_packages = [
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('tensorflow', 'tensorflow'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('sqlalchemy', 'sqlalchemy'),
        ('requests', 'requests')
    ]
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {pip_name}")
    
    # Check model file
    model_path = Path("models/posenet_mobilenet_v1.tflite")
    if not model_path.exists():
        issues.append(f"Model file not found: {model_path}")
        issues.append("Run: python download_model.py")
    
    # Check config file
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        issues.append(f"Config file not found: {config_path}")
        issues.append("Copy config/config_template.yaml to config/config.yaml")
    
    # Check directories
    required_dirs = ["logs", "data", "models", "config"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            issues.append(f"Directory not found: {dir_name}")
    
    if issues:
        print(" Prerequisites not met:")
        for issue in issues:
            print(f"   • {issue}")
        print("\ To fix:")
        print("   1. pip install -r requirements.txt")
        print("   2. python download_model.py")
        print("   3. mkdir -p logs data models config")
        return False
    
    print(" Prerequisites check completed")
    return True

def main():
    """Main entry point"""
    print(" Fall Detection System for Elderly Care")
    print("=" * 60)
    print("Version: 1.0.0")
    print("Mode: Local Production")
    print("Author: MSE Fall Detection Team")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print(" Prerequisites not met. Please fix the issues above.")
        return False
    
    # Create and start application
    try:
        app = FallDetectionApp()
        return app.start()
    except Exception as e:
        print(f" Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)