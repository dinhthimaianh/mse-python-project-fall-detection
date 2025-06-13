# app/services/api_service.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
# FastAPI: Framework web API hiện đại cho Python
# HTTPException: Class để ném lỗi HTTP với status code
# Depends: Dependency injection system của FastAPI
# BackgroundTasks: Chạy tasks trong background mà không block response

from fastapi.middleware.cors import CORSMiddleware
# CORS Middleware để cho phép frontend từ domain khác gọi API
from fastapi.responses import JSONResponse
# Custom JSON response với status code tùy chỉnh

from pydantic import BaseModel
# BaseModel để định nghĩa data schemas với validation tự động

from typing import Dict, List, Any, Optional
# Type hints để làm rõ kiểu dữ liệu
import logging          # Ghi logs
import uvicorn         # ASGI server để chạy FastAPI
from datetime import datetime  # Xử lý thời gian
import threading       # Chạy background tasks trong thread riêng
import time           # Xử lý thời gian Unix timestamp
# Pydantic models for API
class IncidentResponse(BaseModel):
    """Schema cho response khi trả về thông tin incident"""
    id: int                                    # ID của incident trong database
    timestamp: datetime                        # Thời điểm xảy ra incident
    camera_id: str                            # ID của camera phát hiện
    confidence: float                         # Độ tin cậy của detection (0.0-1.0)
    leaning_angle: Optional[float]            # Góc nghiêng cơ thể (có thể None)
    keypoint_correlation: Optional[float]     # Correlation score của keypoints
    status: str                               # Trạng thái: "detected", "confirmed", "false_alarm"
    location: str                             # Vị trí xảy ra incident

class SystemHealthResponse(BaseModel):
    """Schema cho response health check"""
    timestamp: datetime                       # Thời điểm check health
    camera_status: Dict[str, str]            # Status của từng camera {camera_id: status}
    system_metrics: Dict[str, float]         # Metrics hệ thống {metric_name: value}
    detection_metrics: Dict[str, float]      # Metrics detection {metric_name: value}

class NotificationTestRequest(BaseModel):
    """Schema cho request test notification"""
    method: str                              # Phương thức gửi: "sms", "email", "call"
    contact_index: Optional[int] = 0         # Index của contact trong list (default: 0)

class APIService:
    def __init__(self, config: Dict[str, Any], 
                 database_manager, 
                 notification_service, 
                 camera_manager):
        """
        Khởi tạo API Service với các dependencies
        
        Args:
            config: Configuration dict từ config.yaml
            database_manager: Instance của DatabaseManager
            notification_service: Instance của NotificationService  
            camera_manager: Instance của CameraManager
        """
        
        # Lưu references đến các services khác
        self.config = config                          # Config từ YAML
        self.db = database_manager                    # Database operations
        self.notification_service = notification_service  # Gửi alerts
        self.camera_manager = camera_manager          # Quản lý cameras
        self.logger = logging.getLogger(__name__)     # Logger cho class này
        
        # Tạo FastAPI application instance
        self.app = FastAPI(
            title="Fall Detection System API",       # Tiêu đề API trong docs
            description="API for Fall Detection System monitoring and control",  # Mô tả
            version="1.0.0"                         # Version hiển thị trong docs
        )
        # Add CORS middleware nếu được enable trong config
        if config.get('cors_enabled', True):  # Default: True nếu không có trong config
            self.app.add_middleware(
                CORSMiddleware,                # Middleware class
                allow_origins=["*"],          # Cho phép tất cả origins (* = wildcard)
                                             # Production nên specify cụ thể: ["http://localhost:3000"]
                allow_credentials=True,       # Cho phép gửi cookies/credentials
                allow_methods=["*"],          # Cho phép tất cả HTTP methods (GET, POST, PUT, DELETE)
                allow_headers=["*"],          # Cho phép tất cả headers
            )
        
        # Setup tất cả API routes
        self._setup_routes()
        
        # Initialize background task variables
        self.background_thread = None    # Thread chạy background monitoring
        self.running = False            # Flag để control background thread
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Fall Detection System API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now()
            }
        
        @self.app.get("/health")
        async def health_check():
            """System health check - kiểm tra tình trạng hệ thống"""
            try:
                # Lấy stats từ tất cả cameras
                camera_stats = self.camera_manager.get_all_stats()
                
                # Lấy danh sách cameras đang hoạt động tốt
                healthy_cameras = self.camera_manager.get_healthy_cameras()
                
                return {
                    # Status tổng thể: "healthy" nếu có camera hoạt động, "warning" nếu không
                    "status": "healthy" if healthy_cameras else "warning",
                    "timestamp": datetime.now(),              # Thời điểm check
                    "cameras": {
                        "total": len(camera_stats),           # Tổng số cameras
                        "healthy": len(healthy_cameras),      # Số cameras hoạt động tốt
                        "stats": camera_stats                 # Chi tiết stats từng camera
                    },
                    "database": "connected",                  # Database status (cần implement thực tế)
                    "notification_service": "active"         # Notification service status
                }
            except Exception as e:
                # Nếu có lỗi trong quá trình check health
                self.logger.error(f"Health check failed: {e}")
                
                # Trả về JSONResponse với status code 500 (Internal Server Error)
                return JSONResponse(
                    status_code=500,                         # HTTP status code
                    content={"status": "error", "message": str(e)}  # Error details
                )
            
        @self.app.get("/incidents", response_model=List[IncidentResponse])
        # GET /incidents với response schema là List của IncidentResponse
        async def get_incidents(limit: int = 10):  # Query parameter với default value
            """Lấy danh sách incidents gần đây"""
            try:
                # Gọi database để lấy incidents gần đây
                incidents = self.db.get_recent_incidents(limit)
                
                # Convert từ database objects sang Pydantic models
                return [
                    IncidentResponse(                        # Tạo IncidentResponse object
                        id=incident.id,                      # Map field id
                        timestamp=incident.timestamp,        # Map field timestamp
                        camera_id=incident.camera_id,        # Map field camera_id
                        confidence=incident.confidence,      # Map field confidence
                        leaning_angle=incident.leaning_angle,  # Map field leaning_angle
                        keypoint_correlation=incident.keypoint_correlation,  # Map field
                        status=incident.status,              # Map field status
                        location=incident.location           # Map field location
                    )
                    for incident in incidents  # List comprehension để convert tất cả incidents
                ]
                
            except Exception as e:
                # Log error
                self.logger.error(f"Failed to get incidents: {e}")
                
                # Ném HTTPException với status 500
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/incidents/{incident_id}")
        async def get_incident(incident_id: int):  # FastAPI tự động convert string → int
            """Lấy chi tiết một incident cụ thể"""
            try:
                # Lấy database session
                session = self.db.get_session()
                
                # Query incident theo ID
                incident = session.query(self.db.Incident).filter(
                    self.db.Incident.id == incident_id  # WHERE id = incident_id
                ).first()  # Lấy record đầu tiên (hoặc None nếu không tìm thấy)
                
                # Đóng session
                session.close()
                
                # Kiểm tra nếu không tìm thấy incident
                if not incident:
                    raise HTTPException(status_code=404, detail="Incident not found")
                
                # Trả về chi tiết đầy đủ (bao gồm cả pose_data và notifications_sent)
                return {
                    "id": incident.id,
                    "timestamp": incident.timestamp,
                    "camera_id": incident.camera_id,
                    "confidence": incident.confidence,
                    "leaning_angle": incident.leaning_angle,
                    "keypoint_correlation": incident.keypoint_correlation,
                    "status": incident.status,
                    "location": incident.location,
                    "pose_data": incident.pose_data,                    # JSON data
                    "notifications_sent": incident.notifications_sent   # JSON data
                }
                
            except HTTPException:
                # Re-raise HTTPException (404) mà không thay đổi
                raise
                
            except Exception as e:
                # Log và ném lỗi 500 cho các lỗi khác
                self.logger.error(f"Failed to get incident {incident_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/incidents/{incident_id}/resolve")
        async def resolve_incident(incident_id: int,         # Path parameter
                              status: str,               # Request body field
                              resolved_by: str = "api"): # Request body field với default
            """Resolve một incident (đánh dấu đã xử lý)"""
            try:
                # Validate status value
                if status not in ['confirmed', 'false_alarm']:
                    raise HTTPException(status_code=400, detail="Invalid status")
                    # 400 = Bad Request
                
                # Update incident status trong database
                self.db.update_incident_status(incident_id, status, resolved_by)
                
                # Trả về success message
                return {"message": f"Incident {incident_id} marked as {status}"}
                
            except HTTPException:
                # Re-raise HTTPException (400) từ validation
                raise
                
            except Exception as e:
                # Log và ném lỗi 500 cho database errors
                self.logger.error(f"Failed to resolve incident {incident_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/record_fall")
        async def record_fall(background_tasks: BackgroundTasks,  # FastAPI dependency injection
                            fall_data: Dict[str, Any]):          # Request body as dict
            """Record fall incident mới (được gọi bởi detection system)"""
            try:
                # Lưu incident vào database ngay lập tức
                incident = self.db.save_incident(fall_data)
                
                # Thêm task gửi notifications vào background queue
                # Task này sẽ chạy sau khi response được trả về client
                background_tasks.add_task(
                    self._send_fall_notifications,  # Function sẽ được gọi
                    incident,                       # Argument 1
                    fall_data                       # Argument 2
                )
                
                # Trả về response ngay lập tức (không chờ notifications)
                return {
                    "message": "Fall recorded successfully",
                    "incident_id": incident.id,          # ID của incident vừa tạo
                    "timestamp": incident.timestamp      # Timestamp của incident
                }
                
            except Exception as e:
                self.logger.error(f"Failed to record fall: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        #
        # Background Tasks giải thích:

        # Response trả về ngay lập tức
        # Notifications được gửi trong background
        # Không làm chậm API response
        
        @self.app.get("/cameras")
        async def get_cameras():
            """Lấy status của tất cả cameras"""
            try:
                # Gọi camera manager để lấy stats
                return self.camera_manager.get_all_stats()
                # Trả về dict: {camera_id: {stats...}, camera_id2: {stats...}}
                
            except Exception as e:
                self.logger.error(f"Failed to get cameras: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/notifications/test")
        async def test_notification(request: NotificationTestRequest):  # Pydantic model validation
            """Gửi test notification để kiểm tra system"""
            try:
                # Gọi notification service để gửi test
                result = self.notification_service.send_test_notification(request.method)
                
                # Trả về kết quả test
                return {
                    "success": result.success,        # Boolean: thành công hay không
                    "message": result.message,        # Message từ notification service
                    "method": result.method,          # Method đã test (sms/email/call)
                    "timestamp": result.timestamp     # Timestamp test
                }
                
            except Exception as e:
                self.logger.error(f"Failed to send test notification: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/notifications/history")
        async def get_notification_history(limit: int = 10):  # Query parameter
            """Lấy lịch sử gửi notifications"""
            try:
                return self.notification_service.get_notification_history(limit)
                # Trả về list các notification đã gửi
                
            except Exception as e:
                self.logger.error(f"Failed to get notification history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_system_stats():
            """Lấy statistics tổng hợp của toàn hệ thống"""
            try:
                # Lấy camera statistics
                camera_stats = self.camera_manager.get_all_stats()
                
                # Lấy notification statistics  
                notification_stats = self.notification_service.get_stats()
                
                # Tính incident statistics từ database
                recent_incidents = self.db.get_recent_incidents(100)  # Lấy 100 incidents gần đây
                
                incident_stats = {
                    'total_incidents': len(recent_incidents),        # Tổng số incidents
                    
                    # Đếm resolved incidents (status != 'detected')
                    'resolved_incidents': len([i for i in recent_incidents if i.status != 'detected']),
                    
                    # Đếm false alarms
                    'false_alarms': len([i for i in recent_incidents if i.status == 'false_alarm']),
                    
                    # Đếm confirmed falls
                    'confirmed_falls': len([i for i in recent_incidents if i.status == 'confirmed'])
                }
                
                # Trả về comprehensive stats
                return {
                    "timestamp": datetime.now(),
                    "cameras": camera_stats,              # Stats từng camera
                    "notifications": notification_stats,  # Notification stats
                    "incidents": incident_stats,          # Incident breakdown
                    "system": {
                        # Tính uptime nếu có start_time
                        "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                        "status": "running"
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get system stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _send_fall_notifications(self, incident, fall_data):
        """
        Background task để gửi fall notifications
        Chạy sau khi API response đã được trả về
        """
        try:
            # Chuẩn bị data cho notification
            notification_data = {
                'id': incident.id,                    # Incident ID
                'timestamp': incident.timestamp,      # Thời điểm xảy ra
                'confidence': incident.confidence,    # Độ tin cậy
                'location': incident.location,        # Vị trí
                'camera_id': incident.camera_id,      # Camera ID
                'angle': incident.leaning_angle       # Góc nghiêng
            }
            
            # Gửi notifications qua notification service
            results = self.notification_service.send_fall_alert(notification_data)
            
            # Convert results sang format để lưu database
            notifications_sent = [
                {
                    'method': result.method,                           # sms/email/call
                    'success': result.success,                         # Boolean
                    'message': result.message,                         # Result message
                    'timestamp': result.timestamp.isoformat()          # ISO format timestamp
                }
                for result in results  # List comprehension cho tất cả results
            ]
            
            # Update incident trong database với notification results
            session = self.db.get_session()
            try:
                incident.notifications_sent = notifications_sent  # Update JSON field
                session.commit()                                  # Persist changes
            finally:
                session.close()                                   # Always close session
                
        except Exception as e:
            self.logger.error(f"Failed to send notifications for incident {incident.id}: {e}")
            # Không raise exception vì đây là background task
    
    def start(self):
        """Khởi động API server"""
        
        # Ghi lại thời điểm start để tính uptime
        self.start_time = time.time()
        
        # Khởi động background health monitoring
        self.running = True  # Flag để control background thread
        
        # Tạo và start background thread
        self.background_thread = threading.Thread(
            target=self._background_tasks,  # Function sẽ chạy trong thread
            daemon=True                     # Thread sẽ terminate khi main program exit
        )
        self.background_thread.start()
        
        # Khởi động API server với uvicorn
        uvicorn.run(
            self.app,                              # FastAPI app instance
            host=self.config.get('host', '0.0.0.0'),  # Listen address (default: all interfaces)
            port=self.config.get('port', 8000),       # Listen port (default: 8000)
            log_level="info"                       # Uvicorn log level
        )
    
    def stop(self):
        """Dừng API server"""
        
        # Signal background thread để dừng
        self.running = False
        
        # Đợi background thread finish (tối đa 5 giây)
        if self.background_thread:
            self.background_thread.join(timeout=5)
            # join() = đợi thread kết thúc
            # timeout=5 = đợi tối đa 5 giây rồi force quit
    
    def _background_tasks(self):
        """Background monitoring tasks chạy trong thread riêng"""
        
        # Loop vô hạn cho đến khi self.running = False
        while self.running:
            try:
                # Collect system health metrics
                camera_stats = self.camera_manager.get_all_stats()
                
                # Tạo health data để log vào database
                health_data = {
                    'timestamp': datetime.now(),
                    
                    # Camera status: 'online' nếu có camera healthy, 'offline' nếu không
                    'camera_status': 'online' if self.camera_manager.get_healthy_cameras() else 'offline',
                    
                    # Tính tổng frame rate của tất cả cameras
                    'frame_rate': sum(stats.get('avg_fps', 0) for stats in camera_stats.values()),
                    
                    # Placeholder cho các metrics khác (cần implement thực tế)
                    'detection_latency': 0,      # Thời gian detection trung bình
                    'cpu_usage': 0,              # % CPU usage
                    'memory_usage': 0,           # % Memory usage  
                    'disk_usage': 0,             # % Disk usage
                    'network_status': 'online',  # Network connectivity
                    'api_response_time': 0       # API response time trung bình
                }
                
                # Log health data vào database
                self.db.log_system_health(health_data)
                
                # Sleep 60 giây trước khi log tiếp
                time.sleep(60)  # Log mỗi phút
                
            except Exception as e:
                # Log error và tiếp tục (không crash background thread)
                self.logger.error(f"Background task error: {e}")
                time.sleep(60)  # Đợi 60 giây rồi retry