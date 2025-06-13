"""
Cloud client for communicating with Replit backend service
"""
import requests
import logging
import base64
import json
from typing import Dict, Any, Optional
from datetime import datetime
import threading
import time
from io import BytesIO

class CloudClient:
    """Client for Fall Detection Cloud Service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cloud service configuration
        self.base_url = config.get('cloud_service_url', 'https://your-replit-url.replit.app')
        self.api_key = config.get('api_key', '')
        self.system_id = config.get('system_id', f'system_{int(time.time())}')
        self.timeout = config.get('timeout', 10)
        
        # Health monitoring
        self.health_interval = config.get('health_interval', 300)  # 5 minutes
        self.health_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'incidents_sent': 0,
            'health_reports_sent': 0,
            'failed_requests': 0,
            'last_successful_ping': None
        }
        
        self.logger.info(f"Cloud client initialized for {self.base_url}")
        self.logger.info(f"System ID: {self.system_id}")
    
    def configure_notifications(self, notification_config: Dict[str, Any]) -> bool:
        """Configure cloud service notifications"""
        try:
            smtp_config = notification_config.get('email', {})
            smtp_server = smtp_config.get('server', 'smtp.gmail.com')
            smtp_port = smtp_config.get('port', 587)
            sender_email = smtp_config.get('sender_email', '')
            sender_password = smtp_config.get('sender_password', '')
            emergency_contacts = notification_config.get('emergency_contacts', [])

            
            # Prepare notification config
            _notification_config = {
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "sender_email": sender_email,
                "sender_password": sender_password,
                "emergency_contacts": emergency_contacts,
            }
            response = requests.post(
                f"{self.base_url}/config/notifications",
                json=_notification_config,
                timeout=self.timeout
            )
            print(f"Response: {response.status_code} {response.text}")
            
            if response.status_code == 200:
                self.logger.info(" Cloud notification config updated")
                return True
            else:
                self.logger.error(f" Failed to update notification config: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f" Failed to configure notifications: {e}")
            self.stats['failed_requests'] += 1
            return False
    
    def send_fall_incident(self, incident_data: Dict[str, Any], image_data: Optional[bytes] = None) -> bool:
        """Send fall incident to cloud service"""
        try:
            # Prepare incident data
            cloud_incident = {
                'camera_id': str(incident_data.get('camera_id', 'camera_0')),
                'timestamp': self._format_timestamp(incident_data.get('timestamp')),
                'confidence': float(incident_data.get('confidence', 0)),
                'leaning_angle': float(incident_data.get('angle', 0)) if incident_data.get('angle') else None,
                'keypoint_correlation': float(incident_data.get('keypoint_corr', 0)) if incident_data.get('keypoint_corr') else None,
                'pose_data': incident_data.get('pose_data'),
                'location': str(incident_data.get('location', 'unknown'))
            }
            
            # Add image data if available
            if image_data:
                try:
                    cloud_incident['image_data'] = base64.b64encode(image_data).decode('utf-8')
                except Exception as e:
                    self.logger.warning(f" Failed to encode image data: {e}")
            
            # Send to cloud
            response = requests.post(
                f"{self.base_url}/incidents",
                json=cloud_incident,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                incident_id = result.get('incident_id')
                
                self.stats['incidents_sent'] += 1
                self.logger.info(f" Fall incident sent to cloud: ID={incident_id}")
                
                return True
            else:
                self.logger.error(f" Failed to send incident: {response.status_code} - {response.text}")
                self.stats['failed_requests'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f" Failed to send fall incident: {e}")
            self.stats['failed_requests'] += 1
            return False
    
    def send_health_data(self, health_data: Dict[str, Any]) -> bool:
        """Send system health data to cloud"""
        try:
            cloud_health = {
                'timestamp': self._format_timestamp(health_data.get('timestamp')),
                'cpu_usage': float(health_data.get('cpu_usage', 0)),
                'memory_usage': float(health_data.get('memory_usage', 0)),
                'camera_status': str(health_data.get('camera_status', 'unknown')),
                'frame_rate': float(health_data.get('frame_rate', 0)),
                'detection_latency': float(health_data.get('detection_latency', 0)),
                'system_id': self.system_id
            }
            
            print(f"Sending health data: {cloud_health} {self.base_url}"    )
            response = requests.post(
                f"{self.base_url}/health",
                json=cloud_health,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.stats['health_reports_sent'] += 1
                self.stats['last_successful_ping'] = datetime.now()
                self.logger.debug(f" Health data sent to cloud")
                return True
            else:
                self.logger.warning(f" Failed to send health data: {response.status_code}")
                self.stats['failed_requests'] += 1
                return False
                
        except Exception as e:
            self.logger.debug(f"Failed to send health data: {e}")
            self.stats['failed_requests'] += 1
            return False
    
    def get_incidents(self, limit: int = 10) -> Optional[Dict]:
        """Get incidents from cloud service"""
        try:
            response = requests.get(
                f"{self.base_url}/incidents",
                params={'limit': limit},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f" Failed to get incidents: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to get incidents: {e}")
            return None
    
    def get_cloud_stats(self) -> Optional[Dict]:
        """Get statistics from cloud service"""
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f" Failed to get cloud stats: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to get cloud stats: {e}")
            return None
    
    def start_health_monitoring(self, get_health_data_callback):
        """Start background health monitoring"""
        self.running = True
        self.get_health_data = get_health_data_callback
        
        self.health_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_thread.start()
        
        self.logger.info(f" Health monitoring started (interval: {self.health_interval}s)")
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.health_thread:
            self.health_thread.join(timeout=5)
        
        self.logger.info(" Health monitoring stopped")
    
    def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.running:
            try:
                # Get health data from callback
                health_data = self.get_health_data()
                if health_data:
                    self.send_health_data(health_data)
                
                # Wait for next interval
                time.sleep(self.health_interval)
                
            except Exception as e:
                self.logger.error(f" Health monitor error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp for cloud service"""
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).isoformat()
        elif isinstance(timestamp, datetime):
            return timestamp.isoformat()
        else:
            return datetime.now().isoformat()
    
    def ping(self) -> bool:
        """Test connection to cloud service"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self.stats,
            'system_id': self.system_id,
            'cloud_url': self.base_url,
            'connection_status': 'connected' if self.ping() else 'disconnected',
            'health_monitoring': self.running
        }