# web_monitor.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="Fall Detection Monitor")

# Mock data for testing
incidents = [
    {
        "id": 1,
        "timestamp": "2024-01-15 10:30:00",
        "confidence": 0.85,
        "status": "detected",
        "location": "living_room"
    }
]

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fall Detection Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            .header { 
                text-align: center; 
                color: #333; 
                border-bottom: 2px solid #007bff; 
                padding-bottom: 10px; 
            }
            .stats { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }
            .stat-card { 
                background: linear-gradient(135deg, #007bff, #0056b3); 
                color: white; 
                padding: 20px; 
                border-radius: 8px; 
                text-align: center; 
            }
            .stat-value { 
                font-size: 2em; 
                font-weight: bold; 
            }
            .incident { 
                background: #fff3cd; 
                border: 1px solid #ffeaa7; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px; 
            }
            .alert { 
                background: #f8d7da; 
                border: 1px solid #f5c6cb; 
                color: #721c24; 
                padding: 15px; 
                border-radius: 5px; 
                margin: 10px 0; 
            }
            .status { 
                display: inline-block; 
                padding: 5px 10px; 
                border-radius: 15px; 
                color: white; 
                font-size: 0.8em; 
            }
            .status.detected { background: #dc3545; }
            .status.resolved { background: #28a745; }
            .refresh-btn { 
                background: #007bff; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer; 
                margin: 10px 0; 
            }
            .refresh-btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Fall Detection System Dashboard</h1>
                <p>Local Testing Interface</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="uptime">0s</div>
                    <div>System Uptime</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="cameras">1</div>
                    <div>Active Cameras</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="incidents">0</div>
                    <div>Total Incidents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="status">🟢</div>
                    <div>System Status</div>
                </div>
            </div>
            
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            
            <h3>📋 Recent Incidents</h3>
            <div id="incidents-list">
                <div class="incident">
                    <strong>🚨 Fall Detected</strong>
                    <span class="status detected">DETECTED</span>
                    <br>
                    <small>Time: 2024-01-15 10:30:00 | Confidence: 85% | Location: Living Room</small>
                </div>
                <div class="alert">
                    <strong>ℹ️ System Note:</strong> This is a local testing interface. 
                    In production, real incidents would appear here with full details.
                </div>
            </div>
            
            <h3>📊 System Logs</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em;">
                <div id="logs">
                    [INFO] System started successfully<br>
                    [INFO] Camera initialized on device 0<br>
                    [INFO] Fall detection engine loaded<br>
                    [INFO] Monitoring started...<br>
                </div>
            </div>
        </div>
        
        <script>
            let startTime = Date.now();
            
            function updateUptime() {
                const uptime = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById('uptime').textContent = uptime + 's';
            }
            
            function refreshData() {
                // Mock refresh
                const incidents = Math.floor(Math.random() * 5);
                document.getElementById('incidents').textContent = incidents;
                
                const logs = document.getElementById('logs');
                const newLog = `[${new Date().toLocaleTimeString()}] Frame processed - Motion detected<br>`;
                logs.innerHTML += newLog;
                
                // Keep only last 10 log entries
                const logLines = logs.innerHTML.split('<br>');
                if (logLines.length > 10) {
                    logs.innerHTML = logLines.slice(-10).join('<br>');
                }
            }
            
            // Update uptime every second
            setInterval(updateUptime, 1000);
            
            // Auto refresh every 5 seconds
            setInterval(refreshData, 5000);
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/stats")
async def get_stats():
    return {
        "uptime": 300,
        "cameras": 1,
        "incidents": len(incidents),
        "status": "running"
    }

@app.get("/api/incidents")
async def get_incidents():
    return incidents

if __name__ == "__main__":
    print("🌐 Starting Web Monitor on http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)