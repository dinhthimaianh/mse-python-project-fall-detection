# app/services/camera_service.py
import cv2                    # OpenCV - thư viện computer vision để xử lý camera
import threading             # Thư viện multi-threading để chạy các tasks đồng thời
import queue                 # Thread-safe queue để truyền data giữa threads
import time                  # Thư viện xử lý thời gian
import logging              # Thư viện ghi logs
from PIL import Image       # Python Imaging Library để xử lý ảnh
from typing import Optional, Callable, Dict, Any  # Type hints
from dataclasses import dataclass  # Decorator để tạo data classes
import numpy as np          # NumPy cho xử lý arrays (dùng với OpenCV)

@dataclass  # Decorator tự động tạo __init__, __repr__, __eq__ methods
class FrameData:
    """Data structure chứa thông tin một frame từ camera
    
    Giải thích FrameData:

    image: Ảnh ở format PIL Image (RGB)
    timestamp: Thời điểm capture để tracking timing
    camera_id: Để phân biệt khi có nhiều cameras
"""
    image: Image.Image      # PIL Image object chứa ảnh
    timestamp: float        # Unix timestamp khi capture frame
    camera_id: str         # ID của camera capture frame này

##CAMERASERVICE CLASS CONSTRUCTOR
class CameraService:
    def __init__(self, camera_config: Dict[str, Any], frame_callback: Callable[[FrameData], None]):
       
        """
        Khởi tạo Camera Service
        
        Args:
            camera_config: Dict chứa cấu hình camera từ config.yaml
            frame_callback: Function được gọi khi có frame mới
        """
        # Lưu config và callback function
        self.config = camera_config                              # Config dict từ YAML
        self.frame_callback = frame_callback                     # Function để xử lý frame
        self.logger = logging.getLogger(__name__)                # Logger cho class này
        
        # Extract camera configuration từ config dict
        self.camera_id = f"camera_{camera_config.get('device_id', 0)}"  # Tạo unique camera ID
        self.device_id = camera_config.get('device_id', 0)              # Camera device number (0, 1, 2...)
        self.resolution = tuple(camera_config.get('resolution', (640, 480)))  # (width, height)
        self.capture_interval = camera_config.get('capture_interval', 0.2)    # Giây giữa các lần capture
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=100)    # Thread-safe queue, tối đa 100 frames
        self.capture_thread = None                     # Thread để capture từ camera
        self.process_thread = None                     # Thread để xử lý frames
        self.running = False                           # Flag để control threads
        
        # Camera object
        self.cap = None                                # OpenCV VideoCapture object
        
        # Statistics tracking
        self.stats = {
            'frames_captured': 0,     # Tổng số frames đã capture
            'frames_processed': 0,    # Tổng số frames đã xử lý
            'frames_dropped': 0,      # Số frames bị drop do queue full
            'last_frame_time': 0,     # Timestamp frame cuối cùng
            'avg_fps': 0,            # FPS trung bình thực tế
            'status': 'stopped'      # Trạng thái hiện tại
        }
        ##Threading model:

        # capture_thread: Continuously capture từ camera → queue
        # process_thread: Lấy frame từ queue → gọi callback
        # queue: Buffer giữa capture và processing
    def _apply_optimizations(self):
        """Apply camera optimizations for better performance"""
        try:
            # Set resolution
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Set codec for better performance
            fourcc = self.config.get('fourcc', 'MJPG')
            if fourcc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            
            # Reduce buffer size để minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Disable auto-exposure for consistent timing
            if not self.config.get('auto_exposure', True):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            
            # Set brightness/contrast if specified
            if 'brightness' in self.config:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config['brightness'] / 100.0)
            
            if 'contrast' in self.config:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.config['contrast'] / 100.0)
            
            # Log actual settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f" Resolution: {actual_width}x{actual_height}")
            self.logger.info(f" FPS: {actual_fps}")
            
        except Exception as e:
            self.logger.warning(f" Optimization failed: {e}")
    #START CAMERA METHOD
    def start(self) -> bool:
        """Khởi động camera capture và processing"""
        try:
  
            # Khởi tạo camera với OpenCV
            self.cap = cv2.VideoCapture(self.device_id)
            # VideoCapture(0) = camera đầu tiên, VideoCapture(1) = camera thứ hai
            
            # Kiểm tra camera có mở thành công không
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.device_id}")
                return False  # Return False để báo lỗi
            # self._apply_optimizations()
            # Cấu hình camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])   # Set width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])  # Set height
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                     # Minimize latency
            # BUFFERSIZE = 1 nghĩa là chỉ giữ 1 frame trong buffer, giảm độ trễ
            
            # Set flags và status
            self.running = True                    # Enable threads
            self.stats['status'] = 'running'       # Update status
            
            # Tạo và khởi động capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop,         # Function sẽ chạy trong thread
                daemon=True                        # Thread tự terminate khi main program exit
            )
            
            # Tạo và khởi động processing thread  
            self.process_thread = threading.Thread(
                target=self._process_loop,         # Function sẽ chạy trong thread
                daemon=True                        # Daemon thread
            )
            
            # Start cả 2 threads
            self.capture_thread.start()
            self.process_thread.start()
            
            self.logger.info(f"Camera {self.camera_id} started successfully")
            return True  # Báo thành công
        
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            return False  # Báo thất bại
    
    def stop(self):
        """Dừng camera capture"""
        
        # Signal threads để dừng
        self.running = False                   # Set flag = False
        self.stats['status'] = 'stopped'       # Update status
        
        # Đợi capture thread kết thúc
        if self.capture_thread:
            self.capture_thread.join(timeout=2)  # Đợi tối đa 2 giây
            # join() = đợi thread kết thúc trước khi tiếp tục
        
        # Đợi process thread kết thúc
        if self.process_thread:
            self.process_thread.join(timeout=2)   # Đợi tối đa 2 giây
        
        # Giải phóng camera resource
        if self.cap:
            self.cap.release()  # Đóng camera, giải phóng device
        
        self.logger.info(f"Camera {self.camera_id} stopped")
    
    def _capture_loop(self):
        """Camera capture loop - chạy trong capture thread
        Rate limiting giải thích:

        capture_interval = 0.2: Capture mỗi 200ms = 5 FPS
        Giúp giảm CPU load, không cần capture 30 FPS nếu chỉ cần 5 FPS để detect

        """
        
        last_capture_time = 0  # Tracking thời gian capture cuối cùng
        
        # Loop vô hạn cho đến khi self.running = False
        while self.running:
            try:
                current_time = time.time()  # Lấy timestamp hiện tại
                
                # Rate limiting - kiểm tra có đủ thời gian giữa các lần capture chưa
                if current_time - last_capture_time < self.capture_interval:
                    time.sleep(0.01)  # Sleep 10ms rồi check lại
                    continue          # Skip phần còn lại của loop
                
                # Capture frame từ camera
                ret, frame = self.cap.read()
                # ret = True/False (success/fail), frame = numpy array (BGR format)
                
                # Kiểm tra capture có thành công không
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)  # Đợi 100ms rồi thử lại
                    continue
                
                # Convert frame từ BGR (OpenCV format) sang RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # OpenCV dùng BGR, nhưng hầu hết thư viện khác dùng RGB
                
                # Tạo PIL Image từ numpy array
                pil_image = Image.fromarray(frame_rgb)
                
                # Tạo FrameData object
                frame_data = FrameData(
                    image=pil_image,        # PIL Image
                    timestamp=current_time, # Thời điểm capture
                    camera_id=self.camera_id  # ID camera
                )
                
                # Thêm frame vào queue để process thread xử lý
                try:
                    self.frame_queue.put_nowait(frame_data)  # Non-blocking put
                    # put_nowait() = thêm vào queue, nếu full thì raise exception
                    
                    self.stats['frames_captured'] += 1  # Tăng counter
                    last_capture_time = current_time     # Update thời gian capture
                    
                except queue.Full:
                    # Queue đầy, drop frame này
                    self.stats['frames_dropped'] += 1
                    self.logger.warning("Frame queue full, dropping frame")
                
            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)  # Đợi 100ms rồi thử lại
    
    def _process_loop(self):
        """Frame processing loop - chạy trong process thread
        # Example với 30 timestamps:
            frame_times = [1000.0, 1000.1, 1000.2, ..., 1002.9]
            time_span = 1002.9 - 1000.0 = 2.9 seconds
            frames = 30 - 1 = 29 frames
            fps = 29 / 2.9 = 10.0 FPS
        """
        
        frame_times = []  # List để track timestamps để tính FPS
        
        # Loop vô hạn cho đến khi self.running = False
        while self.running:
            try:
                # Lấy frame từ queue
                frame_data = self.frame_queue.get(timeout=1.0)
                # get(timeout=1.0) = đợi tối đa 1 giây, nếu không có frame thì raise Empty
                
                # Update statistics
                current_time = time.time()
                frame_times.append(current_time)  # Thêm timestamp vào list
                
                # Giữ chỉ 30 timestamps gần nhất để tính FPS
                if len(frame_times) > 30:
                    frame_times = frame_times[-30:]  # Slice để lấy 30 phần tử cuối
                
                # Tính average FPS từ timestamps
                if len(frame_times) > 1:
                    # Thời gian từ frame đầu đến frame cuối
                    time_span = frame_times[-1] - frame_times[0]
                    
                    # FPS = số frames / thời gian
                    self.stats['avg_fps'] = (len(frame_times) - 1) / time_span
                
                # Update counters
                self.stats['frames_processed'] += 1   # Tăng số frames đã xử lý
                self.stats['last_frame_time'] = current_time  # Update timestamp cuối
                
                # Gọi callback function để xử lý frame
                self.frame_callback(frame_data)
                # Callback sẽ nhận FrameData và xử lý (fall detection, etc.)
                
            except queue.Empty:
                # Không có frame trong queue (timeout 1 giây)
                continue  # Tiếp tục loop
                
            except Exception as e:
                self.logger.error(f"Process loop error: {e}")
                # Log error nhưng không crash thread
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy camera statistics"""
        return self.stats.copy()  # Return copy để tránh modification từ bên ngoài
        # copy() tạo shallow copy của dict
    
    def is_healthy(self) -> bool:
        """Kiểm tra camera có healthy không"""
        
        # Nếu không running thì không healthy
        if not self.running:
            return False
        
        # Kiểm tra có nhận frames gần đây không
        time_since_last_frame = time.time() - self.stats['last_frame_time']
        return time_since_last_frame < 5.0  # Threshold: 5 giây
        # Nếu > 5 giây không có frame mới thì coi là unhealthy

class MultiCameraManager:
    def __init__(self):
        self.cameras = {} # Dict để lưu {camera_id: CameraService}
        self.logger = logging.getLogger(__name__)
    
    def add_camera(self, camera_config: Dict[str, Any], frame_callback: Callable[[FrameData], None]) -> str:
        """Thêm camera mới vào manager"""
        
        # Tạo CameraService instance
        camera_service = CameraService(camera_config, frame_callback)
        camera_id = camera_service.camera_id  # Lấy camera ID
        
        # Thử start camera
        if camera_service.start():
            # Nếu start thành công, thêm vào dict
            self.cameras[camera_id] = camera_service
            self.logger.info(f"Camera {camera_id} added successfully")
            return camera_id  # Return camera ID
        else:
            # Nếu start thất bại
            self.logger.error(f"Failed to add camera {camera_id}")
            return None  # Return None để báo lỗi
    
    def remove_camera(self, camera_id: str):
        """Xóa camera khỏi manager"""
        
        # Kiểm tra camera có tồn tại không
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()  # Dừng camera
            del self.cameras[camera_id]     # Xóa khỏi dict
            self.logger.info(f"Camera {camera_id} removed")
    
    def stop_all(self):
        """Dừng tất cả cameras"""
        
        # Loop qua tất cả cameras và stop
        for camera in self.cameras.values():
            camera.stop()
        
        # Clear dict
        self.cameras.clear()  # Xóa tất cả entries
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Lấy statistics của tất cả cameras"""
        
        # Dictionary comprehension để lấy stats từng camera
        return {
            camera_id: camera.get_stats()  # Key: camera_id, Value: stats dict
            for camera_id, camera in self.cameras.items()
        }
        
        # Kết quả:
        # {
        #   "camera_0": {"frames_captured": 1500, "avg_fps": 9.8, ...},
        #   "camera_1": {"frames_captured": 1480, "avg_fps": 10.1, ...}
        # }
    
    def get_healthy_cameras(self) -> list[str]:
        """Lấy danh sách camera IDs healthy"""
        
        # List comprehension với condition
        return [
            camera_id                              # Element để add vào list
            for camera_id, camera in self.cameras.items()  # Loop qua dict
            if camera.is_healthy()                 # Condition: chỉ lấy healthy cameras
        ]
        
        # Example result: ["camera_0", "camera_2"]  # camera_1 không healthy
    
