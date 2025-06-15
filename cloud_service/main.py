# main.py (Replit)
"""
Fall Detection Backend Service on Replit
Handles database operations and notifications
"""
from fastapi.responses import HTMLResponse
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
camera_cooldowns = defaultdict(
    lambda: {
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

        if time_since_last < cooldown_period and camera_data[
                'alert_count'] >= max_alerts:
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
        cooldown_status = check_camera_cooldown(incident.camera_id,
                                                incident.confidence)
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
            logger.warning(
                f" INCIDENT SUPPRESSED - Camera {incident.camera_id}")
            logger.warning(f"    Incident ID: {incident_id}")
            logger.warning(
                f"    Remaining cooldown: {cooldown_status['remaining_cooldown']:.1f}s"
            )
            logger.warning(
                f"    Alert count in period: {cooldown_status['alert_count']}")
            logger.warning(
                f"    Total suppressed: {cooldown_status['suppressed_count']}")

            return {
                "status": "suppressed",
                "incident_id": incident_id,
                "message":
                f"Incident recorded but notifications suppressed (cooldown active)",
                "cooldown_info": {
                    "remaining_seconds":
                    round(cooldown_status['remaining_cooldown'], 1),
                    "suppressed_count":
                    cooldown_status['suppressed_count'],
                    "alert_count":
                    cooldown_status['alert_count']
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


# =============================================================================
# WEB MONITOR DASHBOARD ROUTES
# =============================================================================


@app.get("/monitor", response_class=HTMLResponse)
async def web_monitor_dashboard():
    """Web Monitor Dashboard - Trang chính để theo dõi hệ thống"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fall Detection System - Cloud Monitor</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .header { 
                background: rgba(255,255,255,0.95); 
                backdrop-filter: blur(10px);
                padding: 20px; 
                text-align: center; 
                box-shadow: 0 2px 20px rgba(0,0,0,0.1);
                position: sticky;
                top: 0;
                z-index: 1000;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                padding: 20px; 
            }
            .system-status {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .status-card { 
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                padding: 25px; 
                border-radius: 15px; 
                text-align: center;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                transition: transform 0.3s ease;
            }
            .status-card:hover {
                transform: translateY(-5px);
            }
            .status-value { 
                font-size: 2.5em; 
                font-weight: bold; 
                margin: 10px 0;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .status-label { 
                color: #666; 
                font-size: 0.9em; 
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .chart-section {
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                padding: 30px;
                border-radius: 15px;
                margin: 20px 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
            }
            .incidents-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .incident-card { 
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                border-left: 5px solid #e74c3c; 
                padding: 20px; 
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }
            .incident-card:hover {
                transform: scale(1.02);
            }
            .incident-card.high { border-color: #e74c3c; }
            .incident-card.medium { border-color: #f39c12; }
            .incident-card.low { border-color: #27ae60; }
            .incident-card.suppressed { border-color: #95a5a6; }
            .severity-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 15px;
                font-size: 0.8em;
                font-weight: bold;
                color: white;
                margin-left: 10px;
            }
            .severity-high { background: #e74c3c; }
            .severity-medium { background: #f39c12; }
            .severity-low { background: #27ae60; }
            .controls { 
                text-align: center; 
                margin: 20px 0; 
            }
            .btn { 
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white; 
                border: none; 
                padding: 12px 25px; 
                border-radius: 25px; 
                cursor: pointer; 
                margin: 5px;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .btn:hover { 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            .loading { 
                text-align: center; 
                color: #666; 
                padding: 40px;
                font-style: italic;
            }
            .online { color: #27ae60; }
            .offline { color: #e74c3c; }
            .warning { color: #f39c12; }
            .real-time-indicator {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(39, 174, 96, 0.9);
                color: white;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            .error-message {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 Fall Detection System - Cloud Monitor</h1>
            <p>Real-time System Monitoring & Analytics Dashboard</p>
        </div>

        <div class="container">
            <div class="controls">
                <button class="btn" onclick="refreshData()">🔄 Refresh Now</button>

            </div>

            <div class="system-status">
                <div class="status-card">
                    <div class="status-value online" id="system-status">🟢 ONLINE</div>
                    <div class="status-label">System Status</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="uptime">0s</div>
                    <div class="status-label">System Uptime</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="total-incidents">-</div>
                    <div class="status-label">Total Incidents</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="active-systems">-</div>
                    <div class="status-label">Connected Systems</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="notifications-sent">-</div>
                    <div class="status-label">Notifications Sent</div>
                </div>
            </div>


            <div class="chart-section">
                <h3>🚨 Incident Analysis</h3>
                <div id="incident-analysis-chart" style="height: 350px;">
                    <div class="loading">Loading incident analysis...</div>
                </div>
            </div>

            <div class="chart-section">
                <h3>📋 Recent Incidents</h3>
                <div id="incidents-grid" class="incidents-grid">
                    <div class="loading">Loading recent incidents...</div>
                </div>
            </div>

            <div class="chart-section">
                <h3>🖥️ Connected Systems Health</h3>
                <div id="systems-health">
                    <div class="loading">Loading system health data...</div>
                </div>
            </div>
        </div>

        <div class="real-time-indicator" id="live-indicator">
            🟢 Live Data
        </div>

        <script>
            let performanceData = [];
            let incidentData = [];

            async function refreshData() {
                try {
                    document.getElementById('live-indicator').style.background = 'rgba(241, 196, 15, 0.9)';
                    document.getElementById('live-indicator').innerHTML = '🟡 Updating...';

                    // Fetch system stats
                    const statsResponse = await fetch('/stats');
                    const stats = await statsResponse.json();

                    updateSystemStats(stats);

                    // Fetch incidents
                    const incidentsResponse = await fetch('/incidents?limit=20');
                    const incidentsData = await incidentsResponse.json();

                    updateIncidents(incidentsData.incidents || []);

                    // Fetch system health
                    const healthResponse = await fetch('/monitor/system-health');
                    const healthData = await healthResponse.json();

                    updateSystemHealth(healthData);

                    // Update charts
                    updatePerformanceChart(stats);
                    updateIncidentChart(incidentsData.incidents || []);

                    document.getElementById('live-indicator').style.background = 'rgba(39, 174, 96, 0.9)';
                    document.getElementById('live-indicator').innerHTML = '🟢 Live Data';

                } catch (error) {
                    console.error('Error refreshing data:', error);
                    document.getElementById('live-indicator').style.background = 'rgba(231, 76, 60, 0.9)';
                    document.getElementById('live-indicator').innerHTML = '🔴 Error';
                    showErrorMessage('Failed to refresh data: ' + error.message);
                }
            }

            function updateSystemStats(stats) {
                // Calculate uptime
                const uptimeSeconds = Math.floor((Date.now() - new Date(stats.timestamp).getTime()) / 1000) + 3600; // Rough estimate
                document.getElementById('uptime').textContent = formatUptime(uptimeSeconds);

                // Update stats
                const dbStats = stats.database_stats || {};
                document.getElementById('total-incidents').textContent = dbStats.total_incidents || 0;
                document.getElementById('active-systems').textContent = (stats.connected_systems || []).length;

                // Mock additional stats
                document.getElementById('notifications-sent').textContent = Math.floor(dbStats.total_incidents * 0.8) || 0;
     
            }

            function updateIncidents(incidents) {
                const grid = document.getElementById('incidents-grid');

                if (!incidents || incidents.length === 0) {
                    grid.innerHTML = '<div class="incident-card"><h4>✅ No Recent Incidents</h4><p>System is operating normally</p></div>';
                    return;
                }

                const html = incidents.slice(0, 6).map(incident => {
                    const severity = calculateSeverity(incident.confidence);
                    const timeAgo = getTimeAgo(incident.timestamp);

                    return `
                        <div class="incident-card ${severity}">
                            <h4>🚨 Incident #${incident.id}</h4>
                            <span class="severity-badge severity-${severity}">${severity.toUpperCase()}</span>
                            <p><strong>Location:</strong> ${incident.location || 'Unknown'}</p>
                            <p><strong>Camera:</strong> ${incident.camera_id}</p>
                            <p><strong>Confidence:</strong> ${(incident.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Status:</strong> ${incident.status}</p>
                            <p><strong>Time:</strong> ${timeAgo}</p>
                            ${incident.leaning_angle ? `<p><strong>Body Angle:</strong> ${incident.leaning_angle.toFixed(1)}°</p>` : ''}
                        </div>
                    `;
                }).join('');

                grid.innerHTML = html;
            }

            function updateSystemHealth(healthData) {
                const systemsDiv = document.getElementById('systems-health');

                if (!healthData || healthData.length === 0) {
                    systemsDiv.innerHTML = '<div class="error-message">⚠️ No system health data available</div>';
                    return;
                }

                const html = healthData.map(system => `
                    <div class="incident-card" style="border-color: #3498db;">
                        <h4>🖥️ ${system.system_id}</h4>
                        <p><strong>CPU Usage:</strong> ${system.cpu_usage?.toFixed(1) || 'N/A'}%</p>
                        <p><strong>Memory Usage:</strong> ${system.memory_usage?.toFixed(1) || 'N/A'}%</p>
                        <p><strong>Frame Rate:</strong> ${system.frame_rate?.toFixed(1) || 'N/A'} FPS</p>
                        <p><strong>Camera Status:</strong> ${system.camera_status || 'Unknown'}</p>
                        <p><strong>Last Report:</strong> ${getTimeAgo(system.timestamp)}</p>
                    </div>
                `).join('');

                systemsDiv.innerHTML = html;
            }

            function updatePerformanceChart(stats) {
                // Mock performance data
                const now = new Date();
                const times = [];
                const incidents = [];
                const requests = [];

                for (let i = 23; i >= 0; i--) {
                    const time = new Date(now.getTime() - i * 60 * 60 * 1000);
                    times.push(time);
                    incidents.push(Math.floor(Math.random() * 5));
                    requests.push(Math.floor(Math.random() * 100 + 50));
                }

                const trace1 = {
                    x: times,
                    y: incidents,
                    name: 'Incidents/Hour',
                    type: 'scatter',
                    line: { color: '#e74c3c', width: 3 }
                };

                const trace2 = {
                    x: times,
                    y: requests,
                    name: 'API Requests/Hour',
                    type: 'scatter',
                    yaxis: 'y2',
                    line: { color: '#3498db', width: 3 }
                };

                const layout = {
                    title: 'System Performance Over 24 Hours',
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Incidents', side: 'left' },
                    yaxis2: { title: 'API Requests', side: 'right', overlaying: 'y' },
                    showlegend: true,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                };

                
            }

            function updateIncidentChart(incidents) {
                if (!incidents || incidents.length === 0) {
                    document.getElementById('incident-analysis-chart').innerHTML = 
                        '<div class="loading">No incident data to analyze</div>';
                    return;
                }

                // Analyze incidents by location
                const locationCounts = {};
                const severityCounts = { high: 0, medium: 0, low: 0 };

                incidents.forEach(incident => {
                    const location = incident.location || 'Unknown';
                    locationCounts[location] = (locationCounts[location] || 0) + 1;

                    const severity = calculateSeverity(incident.confidence);
                    severityCounts[severity]++;
                });

                const trace = {
                    labels: Object.keys(locationCounts),
                    values: Object.values(locationCounts),
                    type: 'pie',
                    marker: {
                        colors: ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
                    }
                };

                const layout = {
                    title: 'Incidents by Location',
                    showlegend: true,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                };

                Plotly.newPlot('incident-analysis-chart', [trace], layout);
            }

            function calculateSeverity(confidence) {
                if (confidence >= 0.8) return 'high';
                if (confidence >= 0.6) return 'medium';
                return 'low';
            }

            function formatUptime(seconds) {
                const days = Math.floor(seconds / 86400);
                const hours = Math.floor((seconds % 86400) / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);

                if (days > 0) return `${days}d ${hours}h`;
                if (hours > 0) return `${hours}h ${minutes}m`;
                return `${minutes}m`;
            }

            function getTimeAgo(timestamp) {
                const now = new Date();
                const time = new Date(timestamp);
                const diffMs = now - time;
                const diffMins = Math.floor(diffMs / 60000);

                if (diffMins < 1) return 'Just now';
                if (diffMins < 60) return `${diffMins}m ago`;
                const diffHours = Math.floor(diffMins / 60);
                if (diffHours < 24) return `${diffHours}h ago`;
                const diffDays = Math.floor(diffHours / 24);
                return `${diffDays}d ago`;
            }

            function showErrorMessage(message) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.innerHTML = `❌ ${message}`;
                document.querySelector('.container').insertBefore(errorDiv, document.querySelector('.controls').nextSibling);

                setTimeout(() => errorDiv.remove(), 5000);
            }

            function exportReport() {
                alert('📊 Export functionality: This would generate a comprehensive PDF/Excel report of system statistics, incidents, and performance metrics.');
            }

            function viewLogs() {
                window.open('/monitor/logs', '_blank');
            }

            // Auto refresh every 30 seconds
            setInterval(refreshData, 30000);

            // Initial load
            refreshData();
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/monitor/system-health")
async def get_system_health():
    """API endpoint để lấy system health data"""
    try:
        conn = sqlite3.connect('fall_detection.db')
        cursor = conn.cursor()

        # Get recent health data grouped by system
        cursor.execute("""
            SELECT system_id, 
                   AVG(cpu_usage) as avg_cpu,
                   AVG(memory_usage) as avg_memory,
                   AVG(frame_rate) as avg_fps,
                   camera_status,
                   MAX(timestamp) as last_report,
                   COUNT(*) as report_count
            FROM system_health 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY system_id
            ORDER BY last_report DESC
        """)

        health_data = []
        for row in cursor.fetchall():
            health_data.append({
                'system_id': row[0],
                'cpu_usage': row[1],
                'memory_usage': row[2],
                'frame_rate': row[3],
                'camera_status': row[4],
                'timestamp': row[5],
                'report_count': row[6]
            })

        conn.close()
        return health_data

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return []





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
