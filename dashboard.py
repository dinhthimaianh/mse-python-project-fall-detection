# dashboard.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
from datetime import datetime
import sqlite3

app = FastAPI(title="Fall Detection Dashboard")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Fall Detection Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .stat-label { color: #666; font-size: 0.9em; }
        .status-running { color: #27ae60; }
        .status-stopped { color: #e74c3c; }
        .alert { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .incidents-list { background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .incident { border-left: 4px solid #e74c3c; padding: 15px; margin: 10px 0; background: #fff5f5; }
        .controls { text-align: center; margin: 20px 0; }
        .btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #229954; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Fall Detection System</h1>
        <p>Real-time Monitoring Dashboard</p>
    </div>
    
    <div class="container">
        <div class="controls">
            <button class="btn" onclick="refreshData()">🔄 Refresh</button>
            <button class="btn btn-success" onclick="testNotification()">📱 Test Notification</button>
            <button class="btn btn-danger" onclick="emergencyStop()">🛑 Emergency Stop</button>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value status-running" id="status">🟢 RUNNING</div>
                <div class="stat-label">System Status</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="uptime">0s</div>
                <div class="stat-label">Uptime</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="frames">0</div>
                <div class="stat-label">Frames Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">0.0</div>
                <div class="stat-label">Current FPS</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="incidents">0</div>
                <div class="stat-label">Total Incidents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="notifications">0</div>
                <div class="stat-label">Notifications Sent</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📊 System Performance</h3>
            <div id="performance-chart" style="height: 400px;"></div>
        </div>
        
        <div class="chart-container">
            <h3>🚨 Detection History</h3>
            <div id="detection-chart" style="height: 300px;"></div>
        </div>
        
        <div class="incidents-list">
            <h3>📋 Recent Incidents</h3>
            <div id="incidents-list">
                <div class="success">✅ System monitoring active. No incidents detected.</div>
            </div>
        </div>
    </div>
    
    <script>
        let startTime = Date.now();
        let performanceData = { time: [], fps: [], cpu: [], memory: [] };
        let detectionData = { time: [], detections: [] };
        
        async function refreshData() {
            try {
                const response = await fetch('https://00c4a11e-b805-4819-9901-d2653b58bea9-00-2qtiolcxbukam.sisko.replit.dev/stats');
                const data = await response.json();
                
                // Update stats
                document.getElementById('uptime').textContent = formatDuration(data.system?.uptime || 0);
                document.getElementById('frames').textContent = data.incidents?.total_incidents || 0;
                document.getElementById('fps').textContent = (Math.random() * 5 + 2).toFixed(1); // Mock FPS
                document.getElementById('incidents').textContent = data.incidents?.total_incidents || 0;
                document.getElementById('notifications').textContent = data.notifications?.total_notifications || 0;
                
                // Update charts
                updatePerformanceChart();
                updateDetectionChart();
                
                // Update incidents
                updateIncidentsList();
                
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }
        
        function formatDuration(seconds) {
            if (seconds < 60) return seconds.toFixed(0) + 's';
            if (seconds < 3600) return (seconds / 60).toFixed(1) + 'm';
            return (seconds / 3600).toFixed(1) + 'h';
        }
        
        function updatePerformanceChart() {
            const now = new Date();
            performanceData.time.push(now);
            performanceData.fps.push(Math.random() * 3 + 2);
            performanceData.cpu.push(Math.random() * 30 + 20);
            performanceData.memory.push(Math.random() * 20 + 40);
            
            // Keep only last 50 points
            if (performanceData.time.length > 50) {
                performanceData.time.shift();
                performanceData.fps.shift();
                performanceData.cpu.shift();
                performanceData.memory.shift();
            }
            
            const trace1 = {
                x: performanceData.time,
                y: performanceData.fps,
                name: 'FPS',
                type: 'scatter',
                line: { color: '#3498db' }
            };
            
            const trace2 = {
                x: performanceData.time,
                y: performanceData.cpu,
                name: 'CPU %',
                type: 'scatter',
                yaxis: 'y2',
                line: { color: '#e74c3c' }
            };
            
            const layout = {
                title: 'Real-time Performance',
                xaxis: { title: 'Time' },
                yaxis: { title: 'FPS', side: 'left' },
                yaxis2: { title: 'CPU %', side: 'right', overlaying: 'y' },
                showlegend: true
            };
            
            Plotly.newPlot('performance-chart', [trace1, trace2], layout);
        }
        
        function updateDetectionChart() {
            const now = new Date();
            detectionData.time.push(now);
            detectionData.detections.push(Math.random() > 0.9 ? 1 : 0); // 10% chance of detection
            
            if (detectionData.time.length > 20) {
                detectionData.time.shift();
                detectionData.detections.shift();
            }
            
            const trace = {
                x: detectionData.time,
                y: detectionData.detections,
                name: 'Fall Detections',
                type: 'bar',
                marker: { color: '#e74c3c' }
            };
            
            const layout = {
                title: 'Fall Detection Events',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Detections' }
            };
            
            Plotly.newPlot('detection-chart', [trace], layout);
        }
        
        function updateIncidentsList() {
            // Mock incident data
            const incidents = [
                { id: 1, time: '10:30:25', confidence: '89%', status: 'Resolved', location: 'Living Room' },
                { id: 2, time: '14:22:10', confidence: '76%', status: 'False Alarm', location: 'Kitchen' }
            ];
            
            let html = '';
            incidents.forEach(incident => {
                html += `
                    <div class="incident">
                        <strong>🚨 Incident #${incident.id}</strong>
                        <span style="float: right;">${incident.time}</span><br>
                        <small>Confidence: ${incident.confidence} | Location: ${incident.location} | Status: ${incident.status}</small>
                    </div>
                `;
            });
            
            if (!html) {
                html = '<div class="success">✅ No recent incidents</div>';
            }
            
            document.getElementById('incidents-list').innerHTML = html;
        }
        
        async function testNotification() {
            try {
                const response = await fetch('/api/notifications/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ method: 'sms' })
                });
                
                if (response.ok) {
                    alert('✅ Test notification sent successfully!');
                } else {
                    alert('❌ Failed to send test notification');
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        }
        
        function emergencyStop() {
            if (confirm('⚠️ Are you sure you want to stop the system?')) {
                alert('🛑 Emergency stop activated (not implemented in demo)');
            }
        }
        
        // Auto refresh every 5 seconds
        setInterval(refreshData, 5000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
    """

@app.get("/stats")
async def get_stats():
    # This will connect to the real API when system is running
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {"uptime": 3600, "status": "running"},
        "incidents": {"total_incidents": 2, "resolved": 1, "false_alarms": 1},
        "notifications": {"total_notifications": 3, "success_rate": 0.95},
        "cameras": {"camera_0": {"status": "online", "fps": 4.2}}
    }

if __name__ == "__main__":
    print("🌐 Starting Enhanced Dashboard on http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)