# app/main.py - Modified to use cloud service
"""
Fall Detection System - Cloud-Enhanced Local Runner
"""

import asyncio
import signal
import sys
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import io

# Import our modules
import app.core.config as config_
from app.models.pipeline_fall_detector import ProductionPipelineFallDetector
from app.services.camera_service import MultiCameraManager, FrameData
from app.services.display_services import DisplayService
from app.services.cloud_client import CloudClient
from app.models.h5_fall_detector_realtime import H5FallDetectorRealtime
from app.core.logger import setup_logging
from app.utils.helpers import setup_directories, get_system_metrics

class CloudEnhancedFallDetectionSystem:
    """Fall Detection System with Cloud Backend"""
    
    def __init__(self):
        # Setup logging
        setup_logging(enable=False)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = config_.config
        
        # Create directories
        setup_directories()
        
        # Initialize components
        self.fall_detector = None
        self.camera_manager = None
        self.display_service = None
        self.cloud_client = None
        
        # System state
        self.running = False
        self.start_time = None
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'falls_detected': 0,
            'cloud_incidents_sent': 0,
            'local_processing_time': 0.0,
            'uptime_start': None
        }
        
        self.logger.info(" Cloud-Enhanced Fall Detection System initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info(" Initializing Cloud-Enhanced Fall Detection System...")
            
            # 1. Initialize cloud client first
            self.logger.info("   Setting up cloud client...")
            # print("Cloud server config:", self.config.cloud_server.to_dict())
            self.cloud_client = CloudClient(self.config.cloud_server.to_dict())
            
            # Test cloud connection
            if self.cloud_client.ping():
                self.logger.info("Cloud service connected")
                
                # Configure cloud notifications
                if hasattr(self.config, 'notification'):
                    self.logger.info("   Configuring cloud notifications...")
                    notification_config = self.config.notification.to_dict()
                    
                    self.cloud_client.configure_notifications(notification_config)
            else:
                self.logger.warning(" Cloud service not available, running in offline mode")
            
            # 2. Initialize display service
            self.logger.info("   Setting up display service...")
            self.display_service = DisplayService(enable_display=True)
            
            # 3. Initialize fall detector
            self.logger.info("   Loading fall detection model...")
            if not Path(self.config.detection.model_path).exists():
                self.logger.error(f" Model file not found: {self.config.detection.model_path}")
                return False
            
            # pipeline_config = {
            #     'temporal_window': 5,
            #     'confidence_boost_factor': 1.3,
            #     'false_positive_threshold': 2,
            #     'edgetpu_model_path': None
            # }
            check_path = self.config.detection.model_path.endswith('.tflite')
            self.fall_detector = ProductionPipelineFallDetector(
                model_path=self.config.detection.model_path,
                model_name="mobilenet",
                confidence_threshold=self.config.detection.confidence_threshold,
            
            ) if check_path else H5FallDetectorRealtime(
                    model_path=self.config.detection.model_path,
                    confidence_threshold=self.config.detection.confidence_threshold
            )
            
            # 4. Initialize camera manager
            self.logger.info("   Setting up camera system...")
            self.camera_manager = MultiCameraManager()
            
            camera_config = {
                'device_id': self.config.camera.device_id,
                'resolution': self.config.camera.resolution,
                'capture_interval': getattr(self.config.camera, 'capture_interval', 1.0 / self.config.camera.fps)
            }
            print("Camera config:", camera_config)
            
            camera_id = self.camera_manager.add_camera(
                camera_config, 
                self._process_frame_cloud_enhanced
            )
            
            if not camera_id:
                self.logger.error(" Failed to initialize camera")
                return False
            
            # 5. Start display service
            if self.display_service:
                self.display_service.start()
                self.logger.info("   Display service started")
            
            self.logger.info(" All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f" Component initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _process_frame_cloud_enhanced(self, frame_data: FrameData):
        """Enhanced frame processing with cloud integration"""
        try:
            start_time = time.time()
            
            # Local fall detection
            result = self.fall_detector.process_image(frame_data.image)
   
            processing_time = time.time() - start_time
            self.stats['local_processing_time'] = processing_time
            
            if result is None:
                return
            
            # Update statistics
            self.stats['total_frames_processed'] += 1
            
            # Update display
            if self.display_service:
                self.display_service.update_frame(frame_data, result)
            
            # Handle fall detection
            if result.get('fall_detected', False):
                self._handle_fall_detection_cloud(frame_data, result)
            
        except Exception as e:
            self.logger.error(f" Frame processing error: {e}")
    
    def _handle_fall_detection_cloud(self, frame_data: FrameData, detection_result: Dict[str, Any]):
        """Handle fall detection with cloud integration"""
        try:
            self.stats['falls_detected'] += 1
            
            confidence = detection_result.get('confidence', 0)
            leaning_angle = detection_result.get('leaning_angle', 0)
            
            self.logger.warning("  DETECTED - Cloud Enhanced!")
            self.logger.warning(f"    Confidence: {confidence:.1%}")
            self.logger.warning(f"    Leaning Angle: {leaning_angle:.1f}°")
            self.logger.warning(f"    Camera: {frame_data.camera_id}")
            self.logger.warning(f"    Time: {datetime.fromtimestamp(frame_data.timestamp)}")
            
            # Prepare incident data for cloud
            incident_data = {
                'camera_id': str(frame_data.camera_id),
                'timestamp': frame_data.timestamp,
                'confidence': float(detection_result.get('confidence', 0)),
                'angle': float(detection_result.get('leaning_angle', 0)),
                'keypoint_corr': float(detection_result.get('keypoint_correlation', 0)),
                'pose_data': detection_result.get('pose_data'),
                'location': self._get_location(frame_data.camera_id)
            }
            
            # Capture frame image for cloud
            frame_image = self._capture_frame_image(frame_data)
            
            # Send to cloud service in background
            threading.Thread(
                target=self._send_to_cloud_async,
                args=(incident_data, frame_image),
                daemon=True
            ).start()
            
        except Exception as e:
            self.logger.error(f" Fall detection handling error: {e}")
    
    def _send_to_cloud_async(self, incident_data: Dict[str, Any], image_data: Optional[bytes]):
        """Send incident to cloud service asynchronously"""
        try:
            if self.cloud_client:
                success = self.cloud_client.send_fall_incident(incident_data, image_data)
                
                if success:
                    self.stats['cloud_incidents_sent'] += 1
                    self.logger.info(" Incident sent to cloud service - notifications will be handled automatically")
                else:
                    self.logger.warning(" Failed to send incident to cloud service")
            else:
                self.logger.warning(" Cloud client not available")
                
        except Exception as e:
            self.logger.error(f" Cloud sending error: {e}")
    
    def _get_location(self, camera_id: str) -> str:
        """Get location for camera"""
        locations = {
            'camera_0': 'Living Room',
            'camera_1': 'Bedroom',
            'camera_2': 'Kitchen',
            'camera_3': 'Bathroom'
        }
        return locations.get(camera_id, f'Camera {camera_id}')
    
    def _capture_frame_image(self, frame_data: FrameData) -> Optional[bytes]:
        """Capture frame image for cloud service"""
        try:
            if hasattr(frame_data, 'image') and frame_data.image:
                img_byte_arr = io.BytesIO()
                frame_data.image.save(img_byte_arr, format='JPEG', quality=85)
                return img_byte_arr.getvalue()
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Frame capture failed: {e}")
            return None
    
    def _get_health_data(self) -> Dict[str, Any]:
        """Get system health data for cloud reporting"""
        try:
            # Get system metrics
            system_metrics = get_system_metrics()
            
            # Get camera stats
            camera_stats = self.camera_manager.get_all_stats() if self.camera_manager else {}
            healthy_cameras = self.camera_manager.get_healthy_cameras() if self.camera_manager else []
            
            return {
                'timestamp': datetime.now(),
                'cpu_usage': float(system_metrics.get('cpu_usage', 0)),
                'memory_usage': float(system_metrics.get('memory_usage', 0)),
                'camera_status': 'online' if healthy_cameras else 'offline',
                'frame_rate': float(sum(stats.get('avg_fps', 0) for stats in camera_stats.values())),
                'detection_latency': float(self.stats.get('local_processing_time', 0) * 1000)  # Convert to ms
            }
        except Exception as e:
            self.logger.error(f"Failed to get health data: {e}")
            return {}
    
    def start(self):
        """Start the cloud-enhanced system"""
        try:
            self.logger.info(" Starting Cloud-Enhanced Fall Detection System...")
            
            # Initialize components
            if not self.initialize_components():
                self.logger.error(" Failed to initialize components")
                return False
            
            self.running = True
            self.start_time = time.time()
            self.stats['uptime_start'] = self.start_time
            
            # Start cloud health monitoring
            if self.cloud_client:
                self.cloud_client.start_health_monitoring(self._get_health_data)
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info(" Cloud-Enhanced System started successfully!")
            self.logger.info(" Cloud service: Database and notifications handled remotely")
            self.logger.info(" Local processing: Real-time fall detection")
            self.logger.info(" Monitoring for falls... Press Ctrl+C to stop")
            
            # Keep running
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info(" System interrupted by user")
        except Exception as e:
            self.logger.error(f" System startup failed: {e}")
            return False
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the cloud-enhanced system"""
        if not self.running:
            return
        
        self.logger.info(" Stopping Cloud-Enhanced Fall Detection System...")
        self.running = False
        
        try:
            # Stop cloud monitoring
            if self.cloud_client:
                self.cloud_client.stop_health_monitoring()
            
            # Stop local components
            if self.camera_manager:
                self.logger.info("    Stopping cameras...")
                self.camera_manager.stop_all()
            
            if self.display_service:
                self.logger.info("    Stopping display...")
                self.display_service.stop()
            
            # Final statistics
            uptime = time.time() - self.start_time if self.start_time else 0
            
            self.logger.info(" FINAL STATISTICS")
            self.logger.info(f"    Total Runtime: {uptime/60:.1f} minutes")
            self.logger.info(f"    Frames Processed: {self.stats['total_frames_processed']:,}")
            self.logger.info(f"    Falls Detected: {self.stats['falls_detected']}")
            self.logger.info(f"    Cloud Incidents: {self.stats['cloud_incidents_sent']}")
            
            # Get cloud stats
            if self.cloud_client:
                cloud_stats = self.cloud_client.get_client_stats()
                self.logger.info(f"    Cloud Health Reports: {cloud_stats.get('health_reports_sent', 0)}")
                self.logger.info(f"    Failed Requests: {cloud_stats.get('failed_requests', 0)}")
            
            self.logger.info(" System stopped successfully")
            
        except Exception as e:
            self.logger.error(f" Error during shutdown: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f" Received signal {signum}")
        self.stop()
        sys.exit(0)

def main():
    """Main entry point"""
    print(" Cloud-Enhanced Fall Detection System")
    print("=" * 60)
    print("Version: 1.0.0 Cloud")
    print("Mode: Local Processing + Cloud Backend")
    print("Features:")
    print("   Local real-time fall detection")
    print("   Cloud database and notifications")
    print("   Automatic health monitoring")
    print("   Email and Telegram alerts")
    print("=" * 60)
    
    try:
        # Create and start system
        system = CloudEnhancedFallDetectionSystem()
        return system.start()
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)