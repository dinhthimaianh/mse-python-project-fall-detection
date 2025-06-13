# app/services/display_service.py
import cv2                    # OpenCV để hiển thị và xử lý ảnh
import threading             # Multi-threading để chạy display loop riêng biệt
import queue                 # Thread-safe queue (không dùng trong code này nhưng import sẵn)
import time                  # Xử lý thời gian và FPS calculation
import numpy as np           # NumPy để xử lý arrays ảnh
from typing import Optional, Dict, Any  # Type hints
import logging              # Ghi logs
from datetime import datetime  # Hiển thị timestamp trên screen

class DisplayService:
    """Service để hiển thị camera feed với overlay thông tin fall detection"""
    
    def __init__(self, enable_display: bool = True):
        """
        Khởi tạo Display Service
        
        Args:
            enable_display: Có hiển thị giao diện hay không (False cho headless mode)
        """
        
        self.enable_display = enable_display    # Flag để enable/disable display
        self.logger = logging.getLogger(__name__)  # Logger cho class này
        
        # Nếu không enable display thì return sớm (headless mode)
        if not self.enable_display:
            return
        
        # Display state - lưu trữ frame và kết quả detection hiện tại
        self.current_frame = None       # Frame hiện tại để hiển thị (numpy array)
        self.detection_result = None    # Kết quả fall detection gần nhất
        self.camera_id = None          # ID của camera đang hiển thị
        
        # Display thread để chạy display loop riêng biệt
        self.display_thread = None     # Thread object
        self.running = False           # Flag để control thread
        
        # Statistics tracking cho display
        self.stats = {
            'frames_displayed': 0,     # Số frames đã hiển thị
            'falls_detected': 0,       # Số lần phát hiện fall
            'display_fps': 0,          # FPS hiển thị thực tế
            'last_time': time.time()   # Timestamp để tính FPS
        }
        
        # Tên cửa sổ OpenCV
        self.window_name = "Fall Detection Monitor"
        
        self.logger.info("Display service initialized")
    
    def start(self):
        """Khởi động display service"""
        
        # Nếu display bị disable thì không làm gì
        if not self.enable_display:
            return
        
        # Set flag để thread chạy
        self.running = True
        
        # Tạo và khởi động display thread
        self.display_thread = threading.Thread(
            target=self._display_loop,  # Function sẽ chạy trong thread
            daemon=True                 # Daemon thread tự terminate khi main exit
        )
        self.display_thread.start()
        
        self.logger.info("Display service started")
    
    def stop(self):
        """Dừng display service"""
        
        # Nếu display bị disable thì không làm gì
        if not self.enable_display:
            return
        
        # Signal thread để dừng
        self.running = False
        
        # Đợi thread kết thúc
        if self.display_thread:
            self.display_thread.join(timeout=2)  # Đợi tối đa 2 giây
        
        # Đóng tất cả cửa sổ OpenCV
        cv2.destroyAllWindows()
        
        self.logger.info("Display service stopped")
    
    def update_frame(self, frame_data, detection_result: Optional[Dict[str, Any]] = None):
        """
        Update frame để hiển thị
        
        Args:
            frame_data: Frame data (có thể là FrameData object hoặc numpy array)
            detection_result: Kết quả fall detection (optional)
        
        Format conversion:

            PIL Image (RGB) → numpy array (RGB) → OpenCV (BGR)
            OpenCV hiển thị BGR, PIL/most libraries dùng RGB

        """
        
        # Nếu display bị disable thì không làm gì
        if not self.enable_display:
            return
        
        try:
            # Convert PIL to OpenCV format nếu cần
            if hasattr(frame_data, 'image'):
                # frame_data là FrameData object (có attribute image)
                import numpy as np
                
                # Convert PIL Image sang numpy array
                self.current_frame = np.array(frame_data.image)
                
                # Convert RGB (PIL format) sang BGR (OpenCV format)
                self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                
                # Lấy camera ID từ FrameData
                self.camera_id = frame_data.camera_id
            else:
                # frame_data đã là numpy array (BGR format)
                self.current_frame = frame_data
            
            # Lưu kết quả detection
            self.detection_result = detection_result
            
            # Nếu có fall detection, tăng counter
            if detection_result and detection_result.get('fall_detected'):
                self.stats['falls_detected'] += 1
            
        except Exception as e:
            self.logger.error(f"Frame update error: {e}")
    
    def _display_loop(self):
        """Main display loop - chạy trong display thread"""
        try:
            # Loop vô hạn cho đến khi self.running = False
            while self.running:
                # Chỉ hiển thị nếu có frame
                if self.current_frame is not None:
                    # Vẽ overlay lên frame
                    display_frame = self._draw_overlay(self.current_frame.copy())
                    # copy() để không modify frame gốc
                    
                    # Hiển thị frame trong cửa sổ OpenCV
                    cv2.imshow(self.window_name, display_frame)
                    
                    # Update statistics
                    self.stats['frames_displayed'] += 1
                    
                    # Tính FPS hiển thị
                    current_time = time.time()
                    # Nếu đã qua 1 giây thì tính FPS
                    if current_time - self.stats['last_time'] >= 1.0:
                        elapsed = current_time - self.stats['last_time']  # Thời gian trôi qua
                        
                        # FPS = số frames / thời gian
                        self.stats['display_fps'] = self.stats['frames_displayed'] / elapsed
                        
                        # Reset counter và timestamp
                        self.stats['frames_displayed'] = 0
                        self.stats['last_time'] = current_time
                
                # Xử lý key presses
                key = cv2.waitKey(1) & 0xFF  # Đợi key press trong 1ms
                # & 0xFF để lấy 8 bit cuối (ASCII value)
                
                if key == ord('q'):  # Phím 'q' - quit
                    self.logger.info("Display quit requested")
                    break  # Thoát khỏi loop
                    
                elif key == ord('s'):  # Phím 's' - screenshot
                    self._save_screenshot()
                    
                elif key == ord('r'):  # Phím 'r' - reset counter
                    self.stats['falls_detected'] = 0
                    self.logger.info("Fall counter reset")
                
                # Sleep để maintain ~30 FPS display
                time.sleep(0.033)  # 1/30 = 0.033 giây
                
        except Exception as e:
            self.logger.error(f"Display loop error: {e}")
    
    def _draw_overlay(self, frame):
        """Vẽ overlay thông tin lên frame"""
        
        # Kiểm tra frame có hợp lệ không
        if frame is None:
            return frame
        
        # Lấy kích thước frame
        height, width = frame.shape[:2]  # shape = (height, width, channels)
        
        # Định nghĩa màu sắc (BGR format)
        colors = {
            'normal': (0, 255, 0),      # Green (B=0, G=255, R=0)
            'warning': (0, 255, 255),   # Yellow (B=0, G=255, R=255)  
            'danger': (0, 0, 255),      # Red (B=0, G=0, R=255)
            'text': (255, 255, 255),    # White
            'bg': (0, 0, 0)             # Black
        }
        # Vẽ header background (màu đen)
        cv2.rectangle(frame, (0, 0), (width, 80), colors['bg'], -1)
        # (0, 0) = top-left corner, (width, 80) = bottom-right corner
        # -1 = filled rectangle
        
        # Title text
        cv2.putText(frame, "Fall Detection System", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['text'], 2)
        # (10, 25) = position, 0.8 = font scale, 2 = thickness
        
        # Camera info
        camera_text = f"Camera: {self.camera_id or 'Unknown'}"
        cv2.putText(frame, camera_text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1)
        
        # Current time
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
        
        # Hiển thị detection status nếu có kết quả
        if self.detection_result:
            # Extract thông tin detection
            fall_detected = self.detection_result.get('fall_detected', False)
            confidence = self.detection_result.get('confidence', 0)
            
            if fall_detected:
                # Vẽ border đỏ flash khi detect fall
                cv2.rectangle(frame, (0, 0), (width-1, height-1), colors['danger'], 8)
                # Border thickness = 8 pixels
                
                # Alert text ở giữa bottom
                alert_text = "FALL DETECTED!"
                
                # Tính kích thước text để center
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                # [0] = (width, height) của text
                
                alert_x = (width - text_size[0]) // 2    # Center horizontally
                alert_y = height - 60                    # 60 pixels từ bottom
                
                # Vẽ background cho alert text
                cv2.rectangle(frame, (alert_x - 10, alert_y - 35), 
                            (alert_x + text_size[0] + 10, alert_y + 10), 
                            colors['danger'], -1)
                # Padding 10 pixels xung quanh text
                
                # Vẽ alert text
                cv2.putText(frame, alert_text, (alert_x, alert_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors['text'], 3)
            # Font scale = 1.2, thickness = 3 (bold)
            
        # Confidence meter (thanh hiển thị độ tin cậy)
        if confidence > 0:
            # Vị trí và kích thước meter
            meter_x = width - 250      # 250 pixels từ bên phải
            meter_y = height - 60      # 60 pixels từ bottom
            meter_w = 200              # Width = 200 pixels
            meter_h = 20               # Height = 20 pixels
            
            # Vẽ background meter (màu xám đậm)
            cv2.rectangle(frame, (meter_x, meter_y), 
                         (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
            
            # Tính width của fill dựa trên confidence
            fill_w = int(meter_w * confidence)  # confidence = 0.8 → fill_w = 160
            
            # Chọn màu dựa trên confidence level
            fill_color = colors['danger'] if confidence > 0.7 else colors['warning']
            # Red nếu > 70%, Yellow nếu <= 70%
            
            # Vẽ fill portion
            cv2.rectangle(frame, (meter_x, meter_y), 
                         (meter_x + fill_w, meter_y + meter_h), fill_color, -1)
            
            # Text hiển thị % confidence
            conf_text = f"Risk: {confidence:.1%}"  # .1% = 1 decimal place percent
            cv2.putText(frame, conf_text, (meter_x, meter_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
            # meter_y - 5 = 5 pixels phía trên meter
        
        return frame
    
    def _save_screenshot(self):
        """Lưu frame hiện tại thành screenshot"""
        
        # Kiểm tra có frame để save không
        if self.current_frame is not None:
            # Tạo filename với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_detection_screenshot_{timestamp}.jpg"
            # Example: fall_detection_screenshot_20241215_143022.jpg
            
            # Vẽ overlay lên frame trước khi save
            screenshot = self._draw_overlay(self.current_frame.copy())
            
            # Lưu ảnh với OpenCV
            cv2.imwrite(filename, screenshot)  # Lưu thành JPG format
            
            self.logger.info(f"Screenshot saved: {filename}")
        else:
            self.logger.warning("No frame available for screenshot")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get display statistics"""
        return self.stats.copy()#Return copy để tránh modification từ bên ngoài