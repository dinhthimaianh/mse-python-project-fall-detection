

from email.mime.image import MIMEImage
import smtplib
import time
import logging
import requests
import json
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading

@dataclass
class NotificationContact:
    name: str
    email: str
    telegram_chat_id: Optional[str] = None
    priority: int = 3  # 1=highest, 5=lowest

@dataclass
class NotificationResult:
    method: str           # "email" hoặc "telegram"
    success: bool         # True/False
    message: str          # Result message
    timestamp: datetime   # When notification was sent
    contact_name: str     # Name of contact who received notification

@dataclass
class AlertEvent:
    """Sự kiện cảnh báo"""
    incident_id: int
    timestamp: datetime
    confidence: float
    location: str
    camera_id: str
    leaning_angle: Optional[float] = None
    image_data: Optional[bytes] = None
    
class NotificationService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load emergency contacts
        self.contacts = self._load_contacts()
        self.is_in_cooldown = False
        # Cooldown management
        self.last_alert_time = 0.0
        self.alert_count_in_period = 0
        self.cooldown_period = float(config.get('cooldown_period', 30))
        self.max_alerts_per_period = config.get('max_alerts_per_period', 1)  # Max 1 alert per cooldown
        
        # Email configuration
        self.email_config = self.config.get('email', {})
        
        # Telegram configuration
        self.telegram_config = config.get('telegram', {})
        self.bot_token = self.telegram_config.get('bot_token', '')
        
        # Notification history
        self.notification_history = []
        
        # Statistics
        self.stats = {
            'total_sent': 0,
            'email_sent': 0,
            'telegram_sent': 0,
            'failed_notifications': 0,
            'last_notification_time': None,
            'suppressed_alerts': 0,  # New: track suppressed alerts
        }
        
        self.logger.info("Notification Service initialized (Email + Telegram)")
        self.logger.info(f"Loaded {len(self.contacts)} emergency contacts")
    
    def _check_cooldown(self) -> bool:
        """
        Kiểm tra cooldown status
        
        Returns:
            bool: True nếu đang trong cooldown (nên suppress), False nếu có thể gửi
        """
        current_time = time.time()
        
        # Reset cooldown nếu đã hết thời gian
        if self.is_in_cooldown and (current_time - self.cooldown_start_time) >= self.cooldown_period:
            self.is_in_cooldown = False
            self.alert_count_in_period = 0
            self.logger.info("Cooldown period ended, notifications re-enabled")
        
        # Kiểm tra nếu đã đạt max alerts trong period
        time_since_last = current_time - self.last_alert_time
        
        if time_since_last < self.cooldown_period:
            if self.alert_count_in_period >= self.max_alerts_per_period:
                if not self.is_in_cooldown:
                    self.is_in_cooldown = True
                    self.cooldown_start_time = current_time
                    remaining_cooldown = self.cooldown_period - time_since_last
                    self.logger.warning(f"Entering cooldown mode. No more alerts for {remaining_cooldown:.1f} seconds")
                
                return True  # Suppress alert
        
        return False  # Allow alert   
    def _load_contacts(self) -> List[NotificationContact]:
        """Load emergency contacts từ config"""
        contacts = []
        
        contacts_config = self.config.get('emergency_contacts', [])
        for contact_data in contacts_config:
            try:
                contact = NotificationContact(
                    name=contact_data['name'],
                    email=contact_data['email'],
                    telegram_chat_id=contact_data.get('telegram_chat_id'),
                    priority=contact_data.get('priority', 3)
                )
                contacts.append(contact)
            except KeyError as e:
                self.logger.error(f"Invalid contact config: missing {e}")
        
        # Sort by priority (1=highest)
        contacts.sort(key=lambda x: x.priority)
        return contacts
    
    def send_fall_alert(self, alert_event: AlertEvent) -> List[NotificationResult]:
        """
        Gửi fall alert qua Email và Telegram
        
        Args:
            alert_event: Thông tin sự kiện cảnh báo
            
        Returns:
            List các NotificationResult
        """
        try:
            current_time = time.time()
            
                     # Check cooldown
            if self._check_cooldown():
                time_since_last = current_time - self.last_alert_time
                remaining_cooldown = self.cooldown_period - time_since_last
                
                self.stats['suppressed_alerts'] += 1
                self.logger.warning(f" Alert suppressed due to cooldown. Remaining: {remaining_cooldown:.1f}s")
                self.logger.warning(f"   Suppressed alerts in this session: {self.stats['suppressed_alerts']}")
                
                # Return empty result but log the incident
                self._log_suppressed_alert(alert_event)
                return []
            
            
            self.logger.critical(" FALL DETECTED - Sending emergency alerts!")
            self.logger.critical(f"   Incident ID: {alert_event.incident_id}")
            self.logger.critical(f"   Confidence: {alert_event.confidence:.1%}")
            self.logger.critical(f"   Location: {alert_event.location}")
            self.logger.critical(f"   Camera: {alert_event.camera_id}")
            
            results = []
            
            # Gửi alerts tới tất cả contacts
            for contact in self.contacts:
                try:
                    # Send Email
                    if self.email_config.get('enabled', False) and contact.email:
                        email_result = self._send_email_alert(contact, alert_event)
                        results.append(email_result)
                    
                    # Send Telegram
                    if self.telegram_config.get('enabled', False) and contact.telegram_chat_id:
                        telegram_result = self._send_telegram_alert(contact, alert_event)
                        results.append(telegram_result)
                        
                except Exception as e:
                    self.logger.error(f"Failed to send alerts to {contact.name}: {e}")
                    # Fix: Use correct field names
                    results.append(NotificationResult(
                        method="error",
                        success=False,
                        message=str(e),
                        timestamp=datetime.now(),
                        contact_name=contact.name
                    ))
            
            # Update statistics
            self.last_alert_time = current_time
            successful_notifications = sum(1 for r in results if r.success)
            self.stats['total_sent'] += successful_notifications
            self.stats['failed_notifications'] += len(results) - successful_notifications
            self.stats['last_notification_time'] = datetime.now()
            
            # Add to history
            self.notification_history.extend(results)
            if len(self.notification_history) > 100:
                self.notification_history = self.notification_history[-100:]
            
            self.logger.info(f"Sent {successful_notifications}/{len(results)} notifications successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to send fall alerts: {e}")
            return []
    
    def _send_email_alert(self, contact: NotificationContact, alert_event: AlertEvent) -> NotificationResult:
        """Gửi email alert"""
        try:
            # Create email content
            subject = f"🚨 EMERGENCY: Fall Detected at {alert_event.location}"
            
            # HTML email body
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
        .footer {{ margin-top: 20px; font-size: 12px; color: #666; }}
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
                <td>{alert_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
            <tr>
                <td class="label">Location:</td>
                <td>{alert_event.location}</td>
            </tr>
            <tr>
                <td class="label">Camera ID:</td>
                <td>{alert_event.camera_id}</td>
            </tr>
            <tr>
                <td class="label">Confidence Level:</td>
                <td>{alert_event.confidence:.1%}</td>
            </tr>
            <tr>
                <td class="label">Incident ID:</td>
                <td>#{alert_event.incident_id}</td>
            </tr>
            {f'<tr><td class="label">Body Angle:</td><td>{alert_event.leaning_angle:.1f}°</td></tr>' if alert_event.leaning_angle else ''}
        </table>
        
        <p style="color: #dc3545; font-weight: bold; font-size: 18px;">
            ⚠️ Please check on the person immediately!
        </p>
    </div>
    
    <div class="footer">
        <p>Fall Detection System - Automated Alert<br>
        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
            """
            
            # Plain text alternative
            text_body = f"""
EMERGENCY FALL DETECTION ALERT

A fall has been detected at {alert_event.location}.

Details:
- Time: {alert_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Location: {alert_event.location}
- Camera: {alert_event.camera_id}
- Confidence: {alert_event.confidence:.1%}
- Incident ID: #{alert_event.incident_id}
{f'- Body Angle: {alert_event.leaning_angle:.1f}°' if alert_event.leaning_angle else ''}

⚠️ Please check on the person immediately!

This is an automated alert from the Fall Detection System.
            """
            
            # Setup email
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_config['sender_email']
            msg['To'] = contact.email
            msg['Subject'] = subject
            
            # Attach text and HTML versions
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach image if available
            if alert_event.image_data:
                try:
                    image = MIMEImage(alert_event.image_data)
                    image.add_header('Content-Disposition', 
                                   f'attachment; filename="fall_detection_{alert_event.incident_id}.jpg"')
                    msg.attach(image)
                except Exception as e:
                    self.logger.warning(f"Failed to attach image: {e}")
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
            
            self.stats['email_sent'] += 1
            self.logger.info(f"Email sent to {contact.name} ({contact.email})")
            
            # Fix: Use correct field names
            return NotificationResult(
                method="email",
                success=True,
                message=f"Email sent successfully to {contact.email}",
                timestamp=datetime.now(),
                contact_name=contact.name
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send email to {contact.name}: {e}")
            # Fix: Use correct field names
            return NotificationResult(
                method="email",
                success=False,
                message=str(e),
                timestamp=datetime.now(),
                contact_name=contact.name
            )
    
    def _send_telegram_alert(self, contact: NotificationContact, alert_event: AlertEvent) -> NotificationResult:
        """Gửi telegram alert với HTML format (đơn giản hơn MarkdownV2)"""
        try:
            # HTML format message (đơn giản hơn MarkdownV2)
            message_text = f"""🚨 <b>FALL DETECTION EMERGENCY ALERT</b> 🚨

A fall has been detected and requires immediate attention!

📍 <b>Location:</b> {alert_event.location}
📅 <b>Time:</b> {alert_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📹 <b>Camera:</b> {alert_event.camera_id}
📊 <b>Confidence:</b> {alert_event.confidence:.1%}
🆔 <b>Incident ID:</b> #{alert_event.incident_id}"""

            # Add angle if available
            if alert_event.leaning_angle:
                message_text += f"\n📐 <b>Body Angle:</b> {alert_event.leaning_angle:.1f}°"
            
            message_text += f"""

⚠️ <b>Please check on the person immediately!</b>

<i>Fall Detection System - Automated Alert</i>"""
            
            # Telegram API URL
            api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            # Payload với HTML parse mode
            payload = {
                'chat_id': contact.telegram_chat_id,
                'text': message_text,
                'parse_mode': 'HTML'  # HTML đơn giản hơn MarkdownV2
            }
            
            # Send message
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                # Send image if available
                if alert_event.image_data:
                    self._send_telegram_image(contact.telegram_chat_id, alert_event.image_data, alert_event.incident_id)
                
                self.stats['telegram_sent'] += 1
                self.logger.info(f"Telegram message sent to {contact.name} ({contact.telegram_chat_id})")
                
                # Fix: Use correct field names
                return NotificationResult(
                    method="telegram",
                    success=True,
                    message=f"Telegram message sent to {contact.telegram_chat_id}",
                    timestamp=datetime.now(),
                    contact_name=contact.name
                )
            else:
                error_msg = f"Telegram API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                # Fix: Use correct field names
                return NotificationResult(
                    method="telegram",
                    success=False,
                    message=error_msg,
                    timestamp=datetime.now(),
                    contact_name=contact.name
                )
                
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message to {contact.name}: {e}")
            # Fix: Use correct field names
            return NotificationResult(
                method="telegram",
                success=False,
                message=str(e),
                timestamp=datetime.now(),
                contact_name=contact.name
            )
    
    def _send_telegram_image(self, chat_id: str, image_data: bytes, incident_id: int):
        """Gửi ảnh qua Telegram"""
        try:
            api_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            files = {
                'photo': (f'fall_detection_{incident_id}.jpg', image_data, 'image/jpeg')
            }
            
            data = {
                'chat_id': chat_id,
                'caption': f'Fall Detection Image - Incident #{incident_id}'
            }
            
            response = requests.post(api_url, files=files, data=data, timeout=15)
            
            if response.status_code == 200:
                self.logger.info(f"Telegram image sent to {chat_id}")
            else:
                self.logger.warning(f"Failed to send Telegram image: {response.status_code}")
                
        except Exception as e:
            self.logger.warning(f"Failed to send Telegram image: {e}")
    
    def send_test_notification(self, method: str = "both") -> List[NotificationResult]:
        """Gửi test notification"""
        test_event = AlertEvent(
            incident_id=99999,
            timestamp=datetime.now(),
            confidence=0.95,
            location="Test Location",
            camera_id="test_camera",
            leaning_angle=85.5
        )
        
        results = []
        
        for contact in self.contacts[:1]:  # Test với contact đầu tiên
            if method in ["email", "both"] and contact.email:
                result = self._send_email_alert(contact, test_event)
                results.append(result)
            
            if method in ["telegram", "both"] and contact.telegram_chat_id:
                result = self._send_telegram_alert(contact, test_event)
                results.append(result)
        
        return results
    
    def get_notification_history(self, limit: int = 20) -> List[Dict]:
        """Lấy lịch sử notifications"""
        recent_history = self.notification_history[-limit:] if limit else self.notification_history
        
        return [
            {
                'method': result.method,
                'success': result.success,
                'message': result.message,
                'timestamp': result.timestamp.isoformat(),
                'contact_name': result.contact_name  # Now this field exists
            }
            for result in recent_history
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics"""
        return {
            **self.stats,
            'contacts_count': len(self.contacts),
            'email_enabled': self.email_config.get('enabled', False),
            'telegram_enabled': self.telegram_config.get('enabled', False),
            'success_rate': (self.stats['total_sent'] / max(1, self.stats['total_sent'] + self.stats['failed_notifications'])) * 100
        }
        
    def _log_suppressed_alert(self, alert_event: AlertEvent):
        """Log suppressed alert for tracking"""
        suppressed_entry = {
            'action': 'suppressed',
            'incident_id': alert_event.incident_id,
            'timestamp': alert_event.timestamp.isoformat(),
            'confidence': alert_event.confidence,
            'location': alert_event.location,
            'camera_id': alert_event.camera_id,
            'suppressed_at': datetime.now().isoformat(),
            'reason': 'cooldown_active'
        }
        
        # Add to history for tracking
        suppressed_result = NotificationResult(
            method="suppressed",
            success=False,
            message=f"Alert suppressed due to cooldown (incident #{alert_event.incident_id})",
            timestamp=datetime.now(),
            contact_name="system"
        )
        
        self.notification_history.append(suppressed_result)
        
        self.logger.info(f" Logged suppressed alert: Incident #{alert_event.incident_id}")

    def get_cooldown_status(self) -> Dict[str, Any]:
        """Get current cooldown status"""
        current_time = time.time()
        
        if self.is_in_cooldown:
            remaining_cooldown = self.cooldown_period - (current_time - self.cooldown_start_time)
            remaining_cooldown = max(0, remaining_cooldown)
        else:
            time_since_last = current_time - self.last_alert_time
            remaining_cooldown = max(0, self.cooldown_period - time_since_last) if time_since_last < self.cooldown_period else 0
        
        return {
            'is_in_cooldown': self.is_in_cooldown,
            'remaining_seconds': remaining_cooldown,
            'alert_count_in_period': self.alert_count_in_period,
            'max_alerts_per_period': self.max_alerts_per_period,
            'cooldown_period': self.cooldown_period,
            'last_alert_time': datetime.fromtimestamp(self.last_alert_time) if self.last_alert_time > 0 else None,
            'suppressed_alerts_count': self.stats['suppressed_alerts']
        }

    def reset_cooldown(self):
        """Manually reset cooldown (for testing or emergency)"""
        self.is_in_cooldown = False
        self.alert_count_in_period = 0
        self.cooldown_start_time = 0.0
        self.logger.warning(" Cooldown manually reset by admin")