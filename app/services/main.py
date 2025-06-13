# main.py (Replit)
"""
Fall Detection Backend Service on Replit
Handles database operations and notifications
"""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import sqlite3
import json
import base64
import threading
import time
from collections import defaultdict

cooldown_lock = threading.Lock()
camera_cooldowns = defaultdict(lambda: {
    'last_alert_time': 0,
    'cooldown_period': 10,  # 10 seconds default
    'alert_count': 0,
    'suppressed_count': 0
})

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fall Detection Backend Service",
    description="Cloud service for fall detection database and notifications",
    version="1.0.0")

# CORS for local fall detection system
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database setup
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('fall_detection.db')
    cursor = conn.cursor()

    # Create incidents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            camera_id TEXT DEFAULT 'camera_0',
            confidence REAL NOT NULL,
            leaning_angle REAL,
            keypoint_correlation REAL,
            pose_data TEXT,
            status TEXT DEFAULT 'detected',
            location TEXT DEFAULT 'unknown',
            notifications_sent TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create system_health table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cpu_usage REAL,
            memory_usage REAL,
            camera_status TEXT,
            frame_rate REAL,
            detection_latency REAL,
            system_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized")


# Initialize database on startup
init_database()


# Pydantic models
class FallIncident(BaseModel):
    camera_id: str
    timestamp: str
    confidence: float
    leaning_angle: Optional[float] = None
    keypoint_correlation: Optional[float] = None
    pose_data: Optional[Dict] = None
    location: str = "unknown"
    image_data: Optional[str] = None  # Base64 encoded image


class SystemHealthData(BaseModel):
    timestamp: str
    cpu_usage: float
    memory_usage: float
    camera_status: str
    frame_rate: float
    detection_latency: float
    system_id: str


class NotificationConfig(BaseModel):
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str
    sender_password: str
    emergency_contacts: List[Dict[str, str]]
    # Cooldown settings
    cooldown_period: int = 10
    max_alerts_per_period: int = 1
    enable_cooldown: bool = True


# Global notification config
notification_config = None
def check_camera_cooldown(camera_id: str, confidence: float) -> Dict[str, Any]:
    """
    Kiểm tra cooldown cho camera cụ thể
    
    Returns:
        Dict với thông tin cooldown status
    """
    
    with cooldown_lock:
        current_time = time.time()
        camera_data = camera_cooldowns[camera_id]
        
        # Lấy cooldown period từ config hoặc default
        if notification_config and notification_config.enable_cooldown:
            cooldown_period = notification_config.cooldown_period
            max_alerts = notification_config.max_alerts_per_period
        else:
            cooldown_period = 10  # Default 10 seconds
            max_alerts = 1
        
        # Cập nhật cooldown period
        camera_data['cooldown_period'] = cooldown_period
        
        # Kiểm tra thời gian từ alert cuối
        time_since_last = current_time - camera_data['last_alert_time']
        
        # Reset counter nếu đã hết cooldown period
        if time_since_last >= cooldown_period:
            camera_data['alert_count'] = 0
        
        # Kiểm tra có nên suppress không
        should_suppress = False
        remaining_cooldown = 0
        
        if time_since_last < cooldown_period and camera_data['alert_count'] >= max_alerts:
            should_suppress = True
            remaining_cooldown = cooldown_period - time_since_last
            camera_data['suppressed_count'] += 1
        
        # Nếu không suppress, cập nhật counters
        if not should_suppress:
            camera_data['last_alert_time'] = current_time
            camera_data['alert_count'] += 1
        
        return {
            'should_suppress': should_suppress,
            'remaining_cooldown': remaining_cooldown,
            'camera_id': camera_id,
            'alert_count': camera_data['alert_count'],
            'suppressed_count': camera_data['suppressed_count'],
            'time_since_last': time_since_last,
            'cooldown_period': cooldown_period,
            'confidence': confidence
        }

@app.post("/config/notifications")
async def update_notification_config(config: NotificationConfig):
    """Update notification configuration"""
    global notification_config
    notification_config = config
    logger.info("Notification config updated")
    return {"status": "success", "message": "Notification config updated"}


@app.post("/incidents")
async def create_incident(incident: FallIncident,
                          background_tasks: BackgroundTasks):
    """Receive fall incident from local system"""
    try:
        # Kiểm tra cooldown cho camera này
        cooldown_status = check_camera_cooldown(incident.camera_id, incident.confidence)
        # Nếu trong cooldown period
        if cooldown_status['should_suppress']:
            # Vẫn save incident nhưng đánh dấu suppressed
            # Save to database
            conn = sqlite3.connect('fall_detection.db')
            cursor = conn.cursor()
            cursor.execute(
            '''
            INSERT INTO incidents 
            (timestamp, camera_id, confidence, leaning_angle, keypoint_correlation, 
             pose_data, location, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (incident.timestamp, incident.camera_id, incident.confidence,
              incident.leaning_angle, incident.keypoint_correlation,
              json.dumps(incident.pose_data) if incident.pose_data else None,
              incident.location, 'suppressed'))
   
            
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Log suppression information
            logger.warning(f" INCIDENT SUPPRESSED - Camera {incident.camera_id}")
            logger.warning(f"    Incident ID: {incident_id}")
            logger.warning(f"    Remaining cooldown: {cooldown_status['remaining_cooldown']:.1f}s")
            logger.warning(f"    Alert count in period: {cooldown_status['alert_count']}")
            logger.warning(f"    Total suppressed: {cooldown_status['suppressed_count']}")
            
            return {
                "status": "suppressed",
                "incident_id": incident_id,
                "message": f"Incident recorded but notifications suppressed (cooldown active)",
                "cooldown_info": {
                    "remaining_seconds": round(cooldown_status['remaining_cooldown'], 1),
                    "suppressed_count": cooldown_status['suppressed_count'],
                    "alert_count": cooldown_status['alert_count']
                },
                "timestamp": datetime.now().isoformat()
            }
        # Save to database
        conn = sqlite3.connect('fall_detection.db')
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO incidents 
            (timestamp, camera_id, confidence, leaning_angle, keypoint_correlation, 
             pose_data, location, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (incident.timestamp, incident.camera_id, incident.confidence,
              incident.leaning_angle, incident.keypoint_correlation,
              json.dumps(incident.pose_data) if incident.pose_data else None,
              incident.location, 'detected'))

        incident_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f" Fall incident saved: ID={incident_id}")

        # Send notifications in background
        if notification_config:
            background_tasks.add_task(send_fall_notifications, incident_id,
                                      incident.dict())

        return {
            "status": "success",
            "incident_id": incident_id,
            "message": "Incident saved and notifications queued"
        }

    except Exception as e:
        logger.error(f"Failed to save incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incidents")
async def get_incidents(limit: int = 10):
    """Get recent incidents"""
    try:
        conn = sqlite3.connect('fall_detection.db')
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT * FROM incidents 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit, ))

        incidents = []
        for row in cursor.fetchall():
            incidents.append({
                'id':
                row[0],
                'timestamp':
                row[1],
                'camera_id':
                row[2],
                'confidence':
                row[3],
                'leaning_angle':
                row[4],
                'keypoint_correlation':
                row[5],
                'pose_data':
                json.loads(row[6]) if row[6] else None,
                'status':
                row[7],
                'location':
                row[8],
                'notifications_sent':
                json.loads(row[9]) if row[9] else None,
                'created_at':
                row[10],
                'updated_at':
                row[11]
            })

        conn.close()
        return {"incidents": incidents}

    except Exception as e:
        logger.error(f"Failed to get incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/health")
async def log_system_health(health_data: SystemHealthData):
    """Log system health from local system"""
    try:
        conn = sqlite3.connect('fall_detection.db')
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO system_health 
            (timestamp, cpu_usage, memory_usage, camera_status, 
             frame_rate, detection_latency, system_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (health_data.timestamp, health_data.cpu_usage,
              health_data.memory_usage, health_data.camera_status,
              health_data.frame_rate, health_data.detection_latency,
              health_data.system_id))

        conn.commit()
        conn.close()

        return {"status": "success", "message": "Health data logged"}

    except Exception as e:
        logger.error(f"Failed to log health data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def send_fall_notifications(incident_id: int, incident_data: Dict):
    """Send email and telegram notifications"""
    try:
        if not notification_config:
            logger.warning("No notification config available")
            return

        notification_results = []

        # Send emails
        for contact in notification_config.emergency_contacts:
            try:
                result = await send_email_notification(contact, incident_data)
                notification_results.append(result)
            except Exception as e:
                logger.error(
                    f"Email notification failed for {contact.get('name', 'unknown')}: {e}"
                )
                notification_results.append({
                    'method':
                    'email',
                    'contact':
                    contact.get('name', 'unknown'),
                    'success':
                    False,
                    'error':
                    str(e),
                    'timestamp':
                    datetime.now().isoformat()
                })

        # Update incident with notification results
        conn = sqlite3.connect('fall_detection.db')
        cursor = conn.cursor()

        cursor.execute(
            '''
            UPDATE incidents 
            SET notifications_sent = ?
            WHERE id = ?
        ''', (json.dumps(notification_results), incident_id))

        conn.commit()
        conn.close()

        successful = sum(1 for r in notification_results
                         if r.get('success', False))
        logger.info(
            f"Notifications sent for incident {incident_id}: {successful}/{len(notification_results)} successful"
        )

    except Exception as e:
        logger.error(
            f"Failed to send notifications for incident {incident_id}: {e}")


async def send_email_notification(contact: Dict, incident_data: Dict):
    """Send email notification"""
    try:
        # Create email content
        subject = f"🚨 EMERGENCY: Fall Detected at {incident_data['location']}"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .alert-header {{ background-color: #dc3545; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; border: 2px solid #dc3545; }}
        .detail-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .detail-table td {{ padding: 10px; border: 1px solid #ddd; }}
        .detail-table .label {{ background-color: #f8f9fa; font-weight: bold; width: 30%; }}
    </style>
</head>
<body>
    <div class="alert-header">
        <h1>🚨 FALL DETECTION EMERGENCY ALERT 🚨</h1>
    </div>

    <div class="content">
        <p><strong>This is an automated emergency alert from the Fall Detection System.</strong></p>
        <p>A fall has been detected and requires immediate attention.</p>

        <table class="detail-table">
            <tr>
                <td class="label">Time:</td>
                <td>{incident_data['timestamp']}</td>
            </tr>
            <tr>
                <td class="label">Location:</td>
                <td>{incident_data['location']}</td>
            </tr>
            <tr>
                <td class="label">Camera ID:</td>
                <td>{incident_data['camera_id']}</td>
            </tr>
            <tr>
                <td class="label">Confidence Level:</td>
                <td>{incident_data['confidence']:.1%}</td>
            </tr>
            {f'<tr><td class="label">Body Angle:</td><td>{incident_data["leaning_angle"]:.1f}°</td></tr>' if incident_data.get('leaning_angle') else ''}
        </table>

        <p style="color: #dc3545; font-weight: bold; font-size: 18px;">
            ⚠️ Please check on the person immediately!
        </p>
    </div>

    <div class="footer">
        <p>Fall Detection System - Cloud Service<br>
        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
        """

        # Setup email
        msg = MIMEMultipart('alternative')
        msg['From'] = notification_config.sender_email
        msg['To'] = contact['email']
        msg['Subject'] = subject

        # Attach HTML content
        msg.attach(MIMEText(html_body, 'html'))

        # Attach image if available
        if incident_data.get('image_data'):
            try:
                image_data = base64.b64decode(incident_data['image_data'])
                image = MIMEImage(image_data)
                image.add_header('Content-Disposition',
                                 f'attachment; filename="fall_detection.jpg"')
                msg.attach(image)
            except Exception as e:
                logger.warning(f"Failed to attach image: {e}")

        # Send email
        with smtplib.SMTP(notification_config.smtp_server,
                          notification_config.smtp_port) as server:
            server.starttls()
            server.login(notification_config.sender_email,
                         notification_config.sender_password)
            server.send_message(msg)

        logger.info(f"📧 Email sent to {contact['name']} ({contact['email']})")

        return {
            'method': 'email',
            'contact': contact['name'],
            'success': True,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Fall Detection Backend Service",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect('fall_detection.db')
        cursor = conn.cursor()

        # Count incidents
        cursor.execute("SELECT COUNT(*) FROM incidents")
        total_incidents = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM incidents WHERE status = 'detected'")
        pending_incidents = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM incidents WHERE status = 'confirmed'")
        confirmed_incidents = cursor.fetchone()[0]

        # Get recent health data
        cursor.execute("""
            SELECT system_id, COUNT(*) as reports, MAX(timestamp) as last_report
            FROM system_health 
            GROUP BY system_id
        """)

        systems = []
        for row in cursor.fetchall():
            systems.append({
                'system_id': row[0],
                'health_reports': row[1],
                'last_report': row[2]
            })

        conn.close()

        return {
            "database_stats": {
                "total_incidents": total_incidents,
                "pending_incidents": pending_incidents,
                "confirmed_incidents": confirmed_incidents
            },
            "connected_systems": systems,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
