#Not used
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import logging
import numpy as np
from datetime import datetime, timedelta
# SQLAlchemy: ORM (Object-Relational Mapping) để làm việc với database
# Column types: Định nghĩa các kiểu dữ liệu cột trong database
# declarative_base: Base class cho tất cả models
# sessionmaker: Tạo database sessions
# JSON: Lưu trữ dữ liệu phức tạp dưới dạng JSON

Base = declarative_base()
# Tạo base class cho tất cả database models
# Tất cả tables sẽ inherit từ Base
# SQLAlchemy sử dụng để auto-generate SQL schema

class Incident(Base):
    '''Database model for fall detection incidents
    Timing Fields:

        timestamp: Thời điểm phát hiện té ngã
        created_at: Thời điểm tạo record trong DB
        updated_at: Tự động update khi record thay đổi
        Detection Data:

        confidence: 0.0-1.0, độ tin cậy của detection
        leaning_angle: Góc nghiêng cơ thể (0°=đứng, 90°=nằm)
        keypoint_correlation: Mức độ chính xác keypoints'''
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)  # Fix: use datetime.now instead of func.now()
    camera_id = Column(String, default="camera_0")
    
    # Detection results
    confidence = Column(Float, nullable=False)      # Độ tin cậy fall detection
    leaning_angle = Column(Float)                   # Góc nghiêng cơ thể
    keypoint_correlation = Column(Float)            # Correlation score keypoints
    
    # Pose data (stored as JSON)
    pose_data = Column(JSON)
    
    # Status
    status = Column(String, default="detected")  # detected, confirmed, false_alarm
    # "detected": Mới phát hiện, chưa xác nhận
    # "confirmed": Đã xác nhận là té ngã thật
    # "false_alarm": Xác nhận là báo nhầm
    # "resolved": Đã xử lý xong
    resolved_at = Column(DateTime)
    resolved_by = Column(String)
    
    # Location info
    location = Column(String, default="living_room")
    
    # Notification info
    notifications_sent = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.now)  # Fix: use datetime.now
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)  # Fix: use datetime.now

class SystemHealth(Base):
    '''Database model for system health metrics
    System Metrics:

        CPU Usage: Monitor xem AI model có làm CPU quá tải không
        Memory Usage: Theo dõi memory leaks
        Disk Usage: Đảm bảo đủ không gian lưu logs/images
        Camera Metrics:

        Frame Rate: FPS thực tế (có thể < config nếu CPU chậm)
        Detection Latency: Thời gian từ capture frame → kết quả detection
        Network Metrics:

        API Response Time: Monitor web dashboard performance'''
    __tablename__ = "system_health"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)  # Fix: use datetime.now
    
    # System performance metrics
    cpu_usage = Column(Float)      # % CPU usage (0-100)
    memory_usage = Column(Float)   # % RAM usage (0-100)  
    disk_usage = Column(Float)     # % Disk usage (0-100)
    
    # Camera performance
    camera_status = Column(String)        # "online", "offline", "error"
    frame_rate = Column(Float)            # FPS thực tế
    detection_latency = Column(Float)     # Thời gian xử lý 1 frame (ms)
    
    # Network performance
    network_status = Column(String)       # "connected", "disconnected"
    api_response_time = Column(Float)     # Response time API calls (ms)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.logger = logging.getLogger(__name__)
        
        try:
            self.engine = create_engine(database_url, echo=False)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.create_tables()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def save_incident(self, incident_data: Dict[str, Any]) -> Incident:
        """Save fall detection incident với fixed JSON serialization"""
        session = self.get_session()
        try:
            # Convert timestamp to datetime if needed
            timestamp = incident_data.get('timestamp')
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            # Sanitize pose_data để fix JSON serialization
            pose_data = incident_data.get('pose_data')
            if pose_data:
                pose_data = self._sanitize_data_for_json(pose_data)
            
            # Sanitize confidence và các float values
            confidence = incident_data.get('confidence', 0)
            if isinstance(confidence, np.floating):
                confidence = float(confidence)
            
            leaning_angle = incident_data.get('angle') or incident_data.get('leaning_angle')
            if leaning_angle and isinstance(leaning_angle, np.floating):
                leaning_angle = float(leaning_angle)
            
            keypoint_corr = incident_data.get('keypoint_corr') or incident_data.get('keypoint_correlation')
            if keypoint_corr and isinstance(keypoint_corr, np.floating):
                keypoint_corr = float(keypoint_corr)
            
            incident = Incident(
                timestamp=timestamp,
                confidence=confidence,
                leaning_angle=leaning_angle,
                keypoint_correlation=keypoint_corr,
                pose_data=pose_data,  # Đã sanitized
                camera_id=str(incident_data.get('camera_id', 'camera_0')),
                location=str(incident_data.get('location', 'unknown'))
            )
            
            session.add(incident)
            session.commit()
            session.refresh(incident)
            
            self.logger.info(f"Incident saved with ID: {incident.id}")
            return incident
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save incident: {e}")
            self.logger.error(f"Incident data: {incident_data}")  # Log data để debug
            raise e
        finally:
            session.close()
    
    def update_incident_status(self, incident_id: int, status: str, resolved_by: str = None):
        """Update incident status"""
        session = self.get_session()
        try:
            incident = session.query(Incident).filter(Incident.id == incident_id).first()
            if incident:
                incident.status = status
                incident.resolved_at = datetime.now() if status in ['confirmed', 'false_alarm'] else None
                incident.resolved_by = resolved_by
                incident.updated_at = datetime.now()  # Ensure datetime object
                session.commit()
                self.logger.info(f"Incident {incident_id} status updated to {status}")
            else:
                self.logger.warning(f"Incident {incident_id} not found")
                
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to update incident {incident_id}: {e}")
            raise e
        finally:
            session.close()
    
    def get_recent_incidents(self, limit: int = 10) -> List[Incident]:
        """Get recent incidents"""
        session = self.get_session()
        try:
            incidents = session.query(Incident).order_by(Incident.timestamp.desc()).limit(limit).all()
            return incidents
        except Exception as e:
            self.logger.error(f"Failed to get recent incidents: {e}")
            return []
        finally:
            session.close()
    
    def log_system_health(self, health_data: Dict[str, Any]):
        """Log system health metrics"""
        session = self.get_session()
        try:
            # Convert timestamp to datetime if needed
            timestamp = health_data.get('timestamp')
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            # Ensure all numeric values are proper types
            health = SystemHealth(
                timestamp=timestamp,  # Ensure datetime object
                cpu_usage=float(health_data.get('cpu_usage', 0)),
                memory_usage=float(health_data.get('memory_usage', 0)),
                disk_usage=float(health_data.get('disk_usage', 0)),
                camera_status=str(health_data.get('camera_status', 'unknown')),
                frame_rate=float(health_data.get('frame_rate', 0)),
                detection_latency=float(health_data.get('detection_latency', 0)),
                network_status=str(health_data.get('network_status', 'unknown')),
                api_response_time=float(health_data.get('api_response_time', 0))
            )
            
            session.add(health)
            session.commit()
            
            self.logger.debug("System health logged successfully")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to log system health: {e}")
            # Don't raise exception for health logging to avoid breaking main loop
        finally:
            session.close()
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get system health history"""
        session = self.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            health_records = session.query(SystemHealth).filter(
                SystemHealth.timestamp >= cutoff_time
            ).order_by(SystemHealth.timestamp.desc()).all()
            
            return health_records
            
        except Exception as e:
            self.logger.error(f"Failed to get health history: {e}")
            return []
        finally:
            session.close()
    
    def cleanup_old_records(self, days: int = 30):
        """Clean up old records to prevent database bloat"""
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Delete old health records
            deleted_health = session.query(SystemHealth).filter(
                SystemHealth.timestamp < cutoff_date
            ).delete()
            
            # Keep incidents but can delete very old resolved ones
            if days > 90:  # Only delete very old incidents
                deleted_incidents = session.query(Incident).filter(
                    Incident.timestamp < cutoff_date,
                    Incident.status.in_(['resolved', 'false_alarm'])
                ).delete()
            else:
                deleted_incidents = 0
            
            session.commit()
            
            self.logger.info(f"Cleanup completed: {deleted_health} health records, {deleted_incidents} incidents deleted")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database cleanup failed: {e}")
        finally:
            session.close()
    def _sanitize_data_for_json(self, data):
        """Convert numpy types to Python native types for JSON serialization"""
        if data is None:
            return None
            
        if isinstance(data, dict):
            return {key: self._sanitize_data_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data_for_json(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.bool_, bool)):
            return bool(data)
        else:
            return data

# Database này cung cấp:
#  Persistent Storage - Lưu trữ tất cả incidents và system health
#  Status Tracking - Theo dõi lifecycle của mỗi incident
#  Performance Monitoring - Monitor system health realtime
#  Data Maintenance - Auto cleanup để tránh bloat
#  JSON Support - Lưu complex data structures
#  Transaction Safety - ACID compliance với proper error handling