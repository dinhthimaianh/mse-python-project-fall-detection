# app/services/display_service.py
import cv2
import threading
import queue
import time
import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime

class DisplayService:
    """Service to display camera feed with fall detection overlay"""
    
    def __init__(self, enable_display: bool = True):
        self.enable_display = enable_display
        self.logger = logging.getLogger(__name__)
        
        if not self.enable_display:
            return
        
        # Display state
        self.current_frame = None
        self.detection_result = None
        self.camera_id = None
        
        # Display thread
        self.display_thread = None
        self.running = False
        
        # Stats
        self.stats = {
            'frames_displayed': 0,
            'falls_detected': 0,
            'display_fps': 0,
            'last_time': time.time()
        }
        
        # Window name
        self.window_name = "Fall Detection Monitor"
        
        self.logger.info("Display service initialized")
    
    def start(self):
        """Start display service"""
        if not self.enable_display:
            return
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        self.logger.info("Display service started")
    
    def stop(self):
        """Stop display service"""
        if not self.enable_display:
            return
        
        self.running = False
        
        if self.display_thread:
            self.display_thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        self.logger.info("Display service stopped")
    
    def update_frame(self, frame_data, detection_result: Optional[Dict[str, Any]] = None):
        """Update frame for display"""
        if not self.enable_display:
            return
        
        try:
            # Convert PIL to OpenCV if needed
            if hasattr(frame_data, 'image'):
                # FrameData object
                import numpy as np
                self.current_frame = np.array(frame_data.image)
                self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                self.camera_id = frame_data.camera_id
            else:
                # Direct numpy array
                self.current_frame = frame_data
            
            self.detection_result = detection_result
            
            if detection_result and detection_result.get('fall_detected'):
                self.stats['falls_detected'] += 1
            
        except Exception as e:
            self.logger.error(f"Frame update error: {e}")
    
    def _display_loop(self):
        """Main display loop"""
        try:
            while self.running:
                if self.current_frame is not None:
                    # Draw overlay
                    display_frame = self._draw_overlay(self.current_frame.copy())
                    
                    # Show frame
                    cv2.imshow(self.window_name, display_frame)
                    
                    self.stats['frames_displayed'] += 1
                    
                    # Update FPS
                    current_time = time.time()
                    if current_time - self.stats['last_time'] >= 1.0:
                        elapsed = current_time - self.stats['last_time']
                        self.stats['display_fps'] = self.stats['frames_displayed'] / elapsed
                        self.stats['frames_displayed'] = 0
                        self.stats['last_time'] = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Display quit requested")
                    break
                elif key == ord('s'):
                    self._save_screenshot()
                elif key == ord('r'):
                    self.stats['falls_detected'] = 0
                    self.logger.info("Fall counter reset")
                
                time.sleep(0.033)  # ~30 FPS display
                
        except Exception as e:
            self.logger.error(f"Display loop error: {e}")
    
    def _draw_overlay(self, frame):
        """Draw overlay information on frame"""
        if frame is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Colors
        colors = {
            'normal': (0, 255, 0),      # Green
            'warning': (0, 255, 255),   # Yellow  
            'danger': (0, 0, 255),      # Red
            'text': (255, 255, 255),    # White
            'bg': (0, 0, 0)             # Black
        }
        
        # Draw header background
        cv2.rectangle(frame, (0, 0), (width, 80), colors['bg'], -1)
        
        # Title
        cv2.putText(frame, "Fall Detection System", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['text'], 2)
        
        # Camera info
        camera_text = f"Camera: {self.camera_id or 'Unknown'}"
        cv2.putText(frame, camera_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1)
        
        # Time
        time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        # Stats on right side
        fps_text = f"FPS: {self.stats['display_fps']:.1f}"
        cv2.putText(frame, fps_text, (width - 120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1)
        
        falls_text = f"Falls: {self.stats['falls_detected']}"
        cv2.putText(frame, falls_text, (width - 120, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1)
        
        # Detection status
        if self.detection_result:
            fall_detected = self.detection_result.get('fall_detected', False)
            confidence = self.detection_result.get('confidence', 0)
            
            if fall_detected:
                # Flash red border
                cv2.rectangle(frame, (0, 0), (width-1, height-1), colors['danger'], 8)
                
                # Alert text
                alert_text = "🚨 FALL DETECTED! 🚨"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                alert_x = (width - text_size[0]) // 2
                alert_y = height - 60
                
                # Alert background
                cv2.rectangle(frame, (alert_x - 10, alert_y - 35), 
                             (alert_x + text_size[0] + 10, alert_y + 10), 
                             colors['danger'], -1)
                
                # Alert text
                cv2.putText(frame, alert_text, (alert_x, alert_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors['text'], 3)
            
            # Confidence meter
            if confidence > 0:
                meter_x = width - 250
                meter_y = height - 60
                meter_w = 200
                meter_h = 20
                
                # Background
                cv2.rectangle(frame, (meter_x, meter_y), 
                             (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
                
                # Fill
                fill_w = int(meter_w * confidence)
                fill_color = colors['danger'] if confidence > 0.7 else colors['warning']
                cv2.rectangle(frame, (meter_x, meter_y), 
                             (meter_x + fill_w, meter_y + meter_h), fill_color, -1)
                
                # Text
                # conf_text = f"Risk: {confidence:.1%}"
                # cv2.putText(frame, conf_text, (meter_x, meter_y - 5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        # Instructions at bottom
        instructions = [
            "Controls: 'q' quit, 's' screenshot, 'r' reset counter",
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, height - 20 + (i * 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
        
        return frame
    
    def _save_screenshot(self):
        """Save current frame as screenshot"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_detection_screenshot_{timestamp}.jpg"
            
            # Draw overlay before saving
            screenshot = self._draw_overlay(self.current_frame.copy())
            cv2.imwrite(filename, screenshot)
            
            self.logger.info(f"Screenshot saved: {filename}")
        else:
            self.logger.warning("No frame available for screenshot")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get display statistics"""
        return self.stats.copy()