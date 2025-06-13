# app/models/pipeline_fall_detector.py
import sys                          # Thư viện hệ thống để thao tác với Python interpreter
import os                           # Thư viện hệ điều hành để làm việc với files/folders
from pathlib import Path            # Thư viện hiện đại để xử lý đường dẫn files
import numpy as np                  # Thư viện tính toán số học, xử lý arrays
from PIL import Image              # Python Imaging Library để xử lý ảnh
import logging                     # Thư viện ghi log, debug và monitor
from typing import Dict, Any, Optional, List  # Type hints để làm rõ kiểu dữ liệu
import time                        # Thư viện xử lý thời gian
import json                        # Thư viện xử lý dữ liệu JSON
from collections import deque      # Circular buffer với kích thước cố định

# Add src to path
# Thêm thư mục src vào Python path để có thể import pipeline
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from src.pipeline.fall_detect import FallDetector as PipelineFallDetector
except ImportError as e:
     # Nếu import thất bại, in thông báo lỗi rõ ràng
    print(f" Cannot import pipeline fall detector: {e}")
    print("Please ensure pipeline files are in src/pipeline/")
    raise # Ném lại exception để dừng chương trình

class ProductionPipelineFallDetector:
    """
    Class wrapper kết hợp pipeline với enhanced features
    """
    def __init__(self, 
                 model_path: str,   # Đường dẫn đến file model .tflite
                 model_name: str = "mobilenet",  # Loại model: "mobilenet" hoặc "movenet"  
                 confidence_threshold: float = 0.6, # Ngưỡng tin cậy
                 config: Dict = None): # Cấu hình bổ sung (optional)
        
        # Khởi tạo logger để ghi log cho class này
        self.logger = logging.getLogger(__name__)
        # Nếu không có config thì dùng dict rỗng
        # config or {} = nếu config là None thì dùng {}
        self.config = config or {}
        
        # Model configuration
        # Chuẩn bị cấu hình model cho pipeline
        # Pipeline yêu cầu dict có 2 keys: 'tflite' và 'edgetpu'
        model_config = {
            'tflite': model_path,  # Model TensorFlow Lite thông thường
            'edgetpu': self.config.get('edgetpu_model_path', None)# Model EdgeTPU (hardware acceleration)
        }
        
        # Create labels file
        # Pipeline yêu cầu file labels (dù không dùng cho pose detection)
        # Tạo file labels giả để thỏa mãn yêu cầu của pipeline
        labels_path = self._create_labels_file()
        
        try:
            # Initialize pipeline fall detector
            # Khởi tạo pipeline fall detector
            self.fall_detector = PipelineFallDetector(
                model=model_config,                    # Cấu hình model
                labels=labels_path,                    # File labels (giả)
                confidence_threshold=confidence_threshold,  # Ngưỡng tin cậy
                model_name=model_name                  # Loại model
            )
            # Ghi log thành công
            self.logger.info(f" Pipeline Fall Detector initialized")
            self.logger.info(f"   Model: {model_name}")
            self.logger.info(f"   Confidence threshold: {confidence_threshold}")
            # Kiểm tra xem có EdgeTPU không
            self.logger.info(f"   EdgeTPU: {'Yes' if model_config['edgetpu'] else 'No'}")
            
        except Exception as e:
            # Nếu khởi tạo thất bại, ghi log lỗi và ném exception
            self.logger.error(f" Failed to initialize pipeline fall detector: {e}")
            raise
        
        ##KHỞI TẠO TRACKING SYSTEM
        # Enhanced tracking
        # Enhanced tracking với circular buffers
        # deque = double-ended queue, efficient cho add/remove ở đầu/cuối
        # maxlen=20 = tự động xóa phần tử cũ nhất khi vượt quá 20 phần tử
        self.detection_history = deque(maxlen=20)     # Lưu 20 kết quả detection gần nhất
        self.performance_history = deque(maxlen=100)  # Lưu 100 thời gian xử lý gần nhấ
        
        # Statistics
        # Dictionary lưu trữ thống kê tổng quan
        self.stats = {
            'total_processed': 0,              # Tổng số frame đã xử lý từ đầu session
            'falls_detected': 0,               # Tổng số lần phát hiện té ngã
            'false_positives_filtered': 0,     # Số lần filter false positive
            'avg_processing_time': 0.0,        # Thời gian xử lý trung bình (giây)
            'avg_confidence': 0.0,             # Confidence trung bình của tất cả detections
            'last_detection_time': None,       # Timestamp lần cuối phát hiện té ngã
            'session_start_time': time.time()  # Thời điểm bắt đầu session (Unix timestamp)
        }
        # Configuration
        # Cấu hình cho temporal analysis (phân tích theo thời gian)
        # .get() = lấy giá trị từ dict, nếu không có thì dùng default value
        self.temporal_window = self.config.get('temporal_window', 3) # Xem 3 detections gần nhất
        self.confidence_boost_factor = self.config.get('confidence_boost_factor', 1.2)# Tăng confidence 20% khi có temporal evidence
        self.false_positive_threshold = self.config.get('false_positive_threshold', 2) #Cần ít nhất 2 falls trong window để confirm
        
    def _create_labels_file(self) -> str:
        """
        Tạo file labels giả vì pipeline yêu cầu labels file
        Dù không thực sự dùng cho pose detection
        """
        labels_path = "temp_labels.txt"  # Tên file labels tạm thời
        
        # Kiểm tra nếu file chưa tồn tại thì tạo mới
        if not os.path.exists(labels_path):
            # Mở file ở mode write ('w') và tự động đóng khi xong
            with open(labels_path, 'w') as f:
                f.write("person\n")
        
        return labels_path
    
    ##HÀM XỬ LÝ CHÍNH
    def process_image(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Hàm xử lý chính - nhận ảnh PIL và trả về kết quả detection
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict chứa kết quả detection hoặc None nếu thất bại
        """
        try:
            # Ghi lại thời điểm bắt đầu xử lý
            start_time = time.time()
            
            # Core pipeline detection
            # Gọi pipeline fall detector (code từ GitHub)
            # fall_detect() trả về (inference_result, thumbnail)
            inference_result, thumbnail = self.fall_detector.fall_detect(image=image)
            
            # Tính thời gian xử lý
            processing_time = time.time() - start_time
            
            # Convert and enhance result
            # Convert kết quả pipeline sang format chuẩn của hệ thống
            result = self._process_detection_result(
                inference_result,    # Kết quả raw từ pipeline
                thumbnail,          # Ảnh thumbnail đã resize
                processing_time,    # Thời gian xử lý
                image.size         # Kích thước ảnh gốc (width, height)
            )
            
            # Update tracking and stats
            # Cập nhật tracking và statistics
            self._update_tracking(result, processing_time)
            
            # Enhanced decision making
            # Áp dụng enhanced decision making với temporal analysis
            result = self._enhance_decision(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f" Image processing failed: {e}")
            return None
    
    ##XỬ LÝ KẾT QUẢ PIPELINE
    def _process_detection_result(self, inference_result, thumbnail, processing_time, image_size):
        """Convert kết quả raw từ pipeline sang format chuẩn"""
        
        # Tạo structure kết quả cơ bản
        base_result = {
            "fall_detected": False,           # Có phát hiện té ngã không
            "confidence": 0.0,               # Độ tin cậy (0.0 - 1.0)
            "processing_time": float(processing_time),  # Thời gian xử lý (giây)
            "timestamp": time.time(),        # Timestamp hiện tại (Unix time)
            "image_size": image_size,        # (width, height) của ảnh gốc
            "pose_data": None,               # Dữ liệu pose sẽ được điền sau
            "thumbnail": thumbnail           # Ảnh thumbnail từ pipeline
        }
        
        # Kiểm tra nếu pipeline không trả về kết quả gì
        if not inference_result:
            # Cập nhật thông tin cho trường hợp không có detection
            base_result.update({
                "reason": "no_fall_detected",       # Lý do không detect
                "pipeline_result": None             # Không có kết quả từ pipeline
            })
            return base_result
        
        # Get primary detection result
        # Parse kết quả từ pipeline
        # Pipeline trả về list, lấy kết quả đầu tiên (confidence cao nhất)
        primary_result = inference_result[0]
        label, confidence, leaning_angle, keypoint_corr = primary_result
        
        # Enhanced fall detection
        # Logic quyết định có phải té ngã không
        # Phải có label "FALL" VÀ confidence > threshold
        fall_detected = (label == "FALL" and confidence > self.fall_detector.confidence_threshold)
        
        # Extract detailed pose data
        # Extract thông tin pose chi tiết từ keypoint correlation
        pose_data = self._extract_detailed_pose_data(keypoint_corr, leaning_angle)
        
        # Enhanced result
        # Tạo kết quả enhanced với thông tin chi tiết
        enhanced_result = base_result.copy()
        enhanced_result.update({
            "fall_detected":  bool(fall_detected),
            "confidence": float(confidence),        # Ensure Python float (not numpy)
            "leaning_angle": float(leaning_angle),  # Góc nghiêng cơ thể (độ)
            "label": str(label),                         # Label từ pipeline ("FALL" hoặc khác)
            "pose_data": pose_data,                 # Structured pose data
            "keypoint_correlation": keypoint_corr,  # Raw keypoint data từ pipeline
            "pipeline_result": {                    # Metadata từ pipeline
                "raw_inference": inference_result,     # Toàn bộ kết quả raw
                "spinal_vector_score": float(confidence),     # Pipeline's spinal vector score
                "body_line_detected": bool(keypoint_corr)  # Có detect được body line không
            }
        })
        
        return enhanced_result
    
    ##EXTRACT POSE DATA CHI TIẾT
    def _extract_detailed_pose_data(self, keypoint_corr, leaning_angle):
        """Extract dữ liệu pose structured từ keypoint correlation của pipeline"""
        # Nếu không có keypoint correlation thì return None
        if not keypoint_corr:
            return None
        
        # Khởi tạo lists để lưu keypoints và body lines
        keypoints = []      # List các keypoint đã structured
        body_lines = []     # List các đường kết nối cơ thể
    
        
        # Mapping từ tên keypoint của pipeline sang format chuẩn
        # Pipeline dùng: 'left shoulder', hệ thống dùng: 'left_shoulder'
        # Tuple: (tên_chuẩn, index_trong_COCO_format)
        keypoint_mapping = {
            'left shoulder': ('left_shoulder', 5),
            'right shoulder': ('right_shoulder', 6),
            'left hip': ('left_hip', 11),
            'right hip': ('right_hip', 12)
        }
        
        # Extract keypoints
        # Loop qua tất cả keypoints trong correlation
        for name, coords in keypoint_corr.items():
            # Kiểm tra nếu có coordinates VÀ tên keypoint nằm trong mapping
            if coords and name in keypoint_mapping:
                # Lấy tên chuẩn và index từ mapping
                kp_name, kp_index = keypoint_mapping[name]
                # Unpack coordinates: (x, y)
                x, y = coords
                # Tạo keypoint object theo format chuẩn
                keypoints.append({
                'name': kp_name,              # Tên keypoint chuẩn
                'index': kp_index,            # Index trong COCO format
                'x': float(x),                # Tọa độ X (pixel)
                'y': float(y),                # Tọa độ Y (pixel)
                'confidence': 1.0             # Pipeline không cung cấp individual confidence
            })
        
        # Extract body lines (đường kết nối shoulder-hip)
        # Kiểm tra nếu có cả left shoulder và left hip
        if ('left shoulder' in keypoint_corr and 'left hip' in keypoint_corr):
            left_line = {
                'type': 'left_body_line',                    # Loại đường kết nối
                'start':[float(keypoint_corr['left shoulder'][0]), 
                     float(keypoint_corr['left shoulder'][1])],     # Điểm bắt đầu (x1, y1)
                'end': [float(keypoint_corr['left hip'][0]), 
                   float(keypoint_corr['left hip'][1])]             # Điểm kết thúc (x2, y2)
            }
            body_lines.append(left_line)
        
        # Tương tự cho right body line
        if ('right shoulder' in keypoint_corr and 'right hip' in keypoint_corr):
            right_line = {
                'type': 'right_body_line', 
                'start': [float(keypoint_corr['right shoulder'][0]), 
                     float(keypoint_corr['right shoulder'][1])],
                'end': [float(keypoint_corr['right hip'][0]), 
                   float(keypoint_corr['right hip'][1])]
            }
            body_lines.append(right_line)
        # Trả về structured pose data
        return {
        'keypoints': keypoints,                    # List keypoints đã format
        'body_lines': body_lines,                  # List đường kết nối cơ thể
        'leaning_angle': float(leaning_angle),            # Góc nghiêng từ pipeline
        'keypoint_count': len(keypoints),          # Số keypoints detected
        'body_line_count': len(body_lines),        # Số body lines detected  
        'timestamp': time.time()                   # Timestamp tạo data
    }
    
    ##CẬP NHẬT TRACKING VÀ STATISTICS
    def _update_tracking(self, result, processing_time):
        """Update detection tracking and statistics"""
        
        # Add to history
        # Tạo entry cho detection history
        history_entry = {
            'timestamp': result['timestamp'],               # Thời điểm detection
            'fall_detected': result['fall_detected'],       # Có té ngã không
            'confidence': result['confidence'],             # Độ tin cậy
            'leaning_angle': result.get('leaning_angle', 0), # Góc nghiêng (default 0 nếu không có)
            'processing_time': processing_time              # Thời gian xử lý frame này
        }
        
        # Thêm vào detection history
        # deque với maxlen=20 sẽ tự động remove phần tử cũ nhất khi add phần tử thứ 21
        self.detection_history.append(history_entry)
        
        # Thêm processing time vào performance history  
        # deque với maxlen=100 sẽ tự động remove thời gian cũ nhất
        self.performance_history.append(processing_time)
        
        # Cập nhật counter statistics
        self.stats['total_processed'] += 1  # Tăng số frame đã xử lý
        
        # Nếu phát hiện té ngã
        if result['fall_detected']:
            self.stats['falls_detected'] += 1              # Tăng counter té ngã
            self.stats['last_detection_time'] = time.time() # Cập nhật thời điểm detect cuối
        
        # Cập nhật moving averages
        if self.stats['total_processed'] == 1:
            # Frame đầu tiên - khởi tạo averages
            self.stats['avg_processing_time'] = processing_time
            self.stats['avg_confidence'] = result['confidence']
        else:
            # Exponential Moving Average (EMA) để smooth averages
            # EMA = α × current_value + (1-α) × previous_average
            # α = 0.1 = weight của giá trị hiện tại (10%), 90% weight cho history
            alpha = 0.1
            
            # Cập nhật average processing time
            self.stats['avg_processing_time'] = (
                alpha * processing_time +                           # 10% current
                (1 - alpha) * self.stats['avg_processing_time']     # 90% history
            )
            
            # Cập nhật average confidence
            self.stats['avg_confidence'] = (
                alpha * result['confidence'] +                      # 10% current  
                (1 - alpha) * self.stats['avg_confidence']          # 90% history
            )
    
    def _enhance_decision(self, result):
        """
        Enhanced decision making với temporal analysis và false positive filtering
        """
        
        # Nếu chưa có đủ history thì return nguyên result
        if len(self.detection_history) < 2:
            return result
        
        # Lấy các detections gần nhất trong temporal window
        # [-self.temporal_window:] = lấy N phần tử cuối cùng
        # list() = convert deque sang list để dễ xử lý
        recent_detections = list(self.detection_history)[-self.temporal_window:]
        
        # Đếm số lần phát hiện fall trong window
        # sum() với generator expression để đếm
        recent_falls = sum(1 for d in recent_detections if d['fall_detected'])
        
        # Lấy tất cả confidence scores của các fall detections
        # List comprehension với condition
        recent_confidences = [d['confidence'] for d in recent_detections if d['fall_detected']]
        
        # Logic temporal confirmation
        # Cần ít nhất 2 falls HOẶC ít nhất một nửa window size
        # max() để đảm bảo ít nhất 2 falls
        required_falls = max(2, self.temporal_window // 2)  # temporal_window=3 → required=2
        temporal_confirmation = recent_falls >= required_falls
        
        # Confidence boosting nếu có temporal evidence
        if temporal_confirmation and recent_confidences:
            # Lấy confidence cao nhất trong các falls gần đây
            max_confidence = max(recent_confidences)
            
            # Boost confidence bằng cách nhân với factor
            boosted_confidence = max_confidence * self.confidence_boost_factor  # 1.2
            
            # Giới hạn confidence tối đa ở 0.95 (95%)
            boosted_confidence = min(0.95, boosted_confidence)
            
            # Cập nhật kết quả
            result['confidence'] = boosted_confidence
            result['confidence_boosted'] = True      # Flag để biết đã boost
        else:
            result['confidence_boosted'] = False     # Không boost
        
        # False positive filtering
        if result['fall_detected'] and not temporal_confirmation:
            # Nếu detect fall NHƯNG không có temporal confirmation
            
            # Kiểm tra nếu số falls < threshold
            if recent_falls < self.false_positive_threshold:  # threshold=2
                # Filter as false positive
                self.stats['false_positives_filtered'] += 1  # Tăng counter FP
                result['fall_detected'] = False              # Set thành không có fall
                result['filtered_as_false_positive'] = True  # Flag để biết đã filter
                self.logger.debug("Filtered potential false positive")  # Log debug
        
        # Thêm metadata về temporal analysis vào result
        result.update({
            'temporal_confirmation': temporal_confirmation,     # Có temporal confirmation không
            'recent_falls_count': recent_falls,                # Số falls trong window
            'temporal_window_size': len(recent_detections),    # Kích thước window thực tế
            'confidence_trend': self._calculate_confidence_trend()  # Trend của confidence
        })
        
        return result  # Trả về kết quả đã enhanced
    
    def _calculate_confidence_trend(self):
        """
        Tính trend của confidence qua các detections gần đây
        """
        
        # Cần ít nhất 3 detections để tính trend
        if len(self.detection_history) < 3:
            return 0.0  # Không đủ data
        
        # Lấy 3 detections gần nhất
        recent = list(self.detection_history)[-3:]
        
        # Extract confidence values
        confidences = [d['confidence'] for d in recent]
        
        # Tính trend đơn giản: confidence_latest - confidence_previous
        if len(confidences) >= 2:
            # confidences[-1] = confidence mới nhất
            # confidences[-2] = confidence trước đó
            trend = confidences[-1] - confidences[-2]
            return trend  # Positive = tăng, Negative = giảm
        
        return 0.0  # Default nếu không tính được
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Lấy statistics chi tiết với performance analysis
        """
        current_time = time.time()  # Thời điểm hiện tại
        
        # Tính thời gian session đã chạy
        session_duration = current_time - self.stats['session_start_time']
        
        # Tính performance statistics từ history
        performance_stats = {}  # Dict rỗng làm default
        
        if self.performance_history:  # Nếu có dữ liệu performance
            # Convert deque sang list để dễ xử lý
            times = list(self.performance_history)
            
            performance_stats = {
                'min_processing_time': min(times),              # Thời gian xử lý nhanh nhất
                'max_processing_time': max(times),              # Thời gian xử lý chậm nhất
                'median_processing_time': np.median(times),     # Median (giá trị giữa)
                'std_processing_time': np.std(times)            # Standard deviation (độ biến thiên)
            }
        
        # Trả về dictionary comprehensive
        return {
            **self.stats,                                       # Unpack tất cả basic stats
            'session_duration': session_duration,              # Tổng thời gian session
            
            # Tính detection rate = falls/total_frames
            'detection_rate': self.stats['falls_detected'] / max(1, self.stats['total_processed']),
            
            # Tính processing FPS = 1/avg_processing_time  
            'processing_fps': 1 / max(0.001, self.stats['avg_processing_time']),
            
            # Tính false positive rate = FP_filtered/total_frames
            'false_positive_rate': self.stats['false_positives_filtered'] / max(1, self.stats['total_processed']),
            
            'performance_stats': performance_stats,             # Chi tiết performance
            'history_size': len(self.detection_history)        # Kích thước history hiện tại
        }
    
    def export_detection_history(self, filepath: str = None):
        """
        Export toàn bộ detection history ra file JSON để phân tích
        """
        
        # Nếu không chỉ định filepath thì tự tạo
        if not filepath:
            timestamp = int(time.time())  # Unix timestamp hiện tại
            filepath = f"detection_history_{timestamp}.json"  # f-string formatting
        
        # Tạo comprehensive data để export
        history_data = {
            'export_timestamp': time.time(),                    # Thời điểm export
            'stats': self.get_detailed_stats(),                 # Tất cả statistics
            'detection_history': list(self.detection_history),  # Convert deque→list
            'config': self.config                               # Configuration đã dùng
        }
        
        # Ghi ra file JSON
        with open(filepath, 'w') as f:
            json.dump(
                history_data,           # Data cần ghi
                f,                      # File handle
                indent=2,               # Pretty print với indent 2 spaces
                default=str             # Convert non-serializable objects sang string
            )
        
        # Log thông báo đã export
        self.logger.info(f" Detection history exported to: {filepath}")
        return filepath  # Trả về đường dẫn file đã tạo
    
    def reset_stats(self):
        """
        Reset tất cả statistics và history để bắt đầu session mới
        """
        
        # Clear tất cả history
        self.detection_history.clear()      # Xóa detection history
        self.performance_history.clear()    # Xóa performance history
        
        # Reset stats về giá trị ban đầu
        self.stats = {
            'total_processed': 0,                   # Reset về 0
            'falls_detected': 0,                    # Reset về 0  
            'false_positives_filtered': 0,          # Reset về 0
            'avg_processing_time': 0.0,             # Reset về 0.0
            'avg_confidence': 0.0,                  # Reset về 0.0
            'last_detection_time': None,            # Reset về None
            'session_start_time': time.time()       # Reset về thời điểm hiện tại
        }
        
        # Log thông báo đã reset
        self.logger.info("📊 Statistics and history reset")

# Alias for easier import
EnhancedPipelineFallDetector = ProductionPipelineFallDetector

'''
# 1. Khởi tạo
detector = ProductionPipelineFallDetector(model_path, model_name, threshold, config)

# 2. Xử lý ảnh
result = detector.process_image(pil_image)
# ↓
# 2a. Gọi pipeline: inference_result, thumbnail = fall_detector.fall_detect()
# 2b. Convert result: _process_detection_result()
# 2c. Update tracking: _update_tracking()  
# 2d. Enhanced decision: _enhance_decision()

# 3. Kết quả enhanced với temporal analysis
{
    "fall_detected": True,
    "confidence": 0.85,
    "leaning_angle": 67.5,
    "temporal_confirmation": True,
    "confidence_boosted": True,
    "recent_falls_count": 3,
    "pose_data": {...}
}
'''