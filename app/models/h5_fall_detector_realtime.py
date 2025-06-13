# app/models/h5_fall_detector_realtime.py
"""
H5 Fall Detector tương tự logic trong fall_detection_realtime.py
Tích hợp với hệ thống run_local.py
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import time
import logging
from typing import Dict, Any, Optional
from PIL import Image

class H5FallDetectorRealtime:
    """Fall detector sử dụng .h5 model với logic tương tự fall_detection_realtime.py"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        """
        Khởi tạo H5 Fall Detector
        
        Args:
            model_path: Đường dẫn đến file model .h5
            confidence_threshold: Ngưỡng confidence để xác định ngã
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Loading H5 fall detection model...")
        self.model = keras.models.load_model(model_path)
        
        self.img_size = 224  # Default
        try:
            model_input_shape = self.model.input_shape
            if model_input_shape[1] == 224:
                self.img_size = 224
            elif model_input_shape[1] == 416:
                self.img_size = 416
            else:
                self.img_size = model_input_shape[1]
                
            self.logger.info(f"Model input shape: {model_input_shape}")
            self.logger.info(f"Using image size: {self.img_size}")
        except Exception as e:
            self.logger.warning(f"Could not determine model input shape: {e}")
            self.img_size = 224  # Default to 224
        
        # Class names
        self.class_names = ['fall', 'not_fallen']
        
        # Khởi tạo MediaPipe cho pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'falls_detected': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
            'last_detection_time': None
        }
        
        self.logger.info(f"H5 Model loaded successfully!")
        self.logger.info(f"   Image size: {self.img_size}")
        self.logger.info(f"   Classes: {self.class_names}")
        self.logger.info(f"   Confidence threshold: {self.confidence_threshold}")
    
    def preprocess_frame(self, frame):
        """
        Tiền xử lý frame để đưa vào model (giống fall_detection_realtime.py)
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            # Resize và normalize giống fall_detection_realtime.py
            img = cv2.resize(frame, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            self.logger.error(f"Failed to preprocess frame: {e}")
            return None
    
    def detect_pose_landmarks(self, frame):
        """
        Phát hiện các điểm landmark trên cơ thể người
        """
        try:
            # Convert to RGB for MediaPipe
            if isinstance(frame, Image.Image):
                rgb_frame = np.array(frame)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.pose.process(rgb_frame)
            return results
        except Exception as e:
            self.logger.error(f"Failed to detect pose landmarks: {e}")
            return None
    
    def calculate_body_angle(self, landmarks):
        """
        Tính góc nghiêng của cơ thể để hỗ trợ phát hiện ngã
        """
        if not landmarks:
            return None
        
        try:
            # Lấy tọa độ vai và hông
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Tính trung điểm vai và hông
            shoulder_center = [(left_shoulder.x + right_shoulder.x) / 2, 
                              (left_shoulder.y + right_shoulder.y) / 2]
            hip_center = [(left_hip.x + right_hip.x) / 2, 
                         (left_hip.y + right_hip.y) / 2]
            
            # Tính góc so với trục dọc
            dx = hip_center[0] - shoulder_center[0]
            dy = hip_center[1] - shoulder_center[1]
            angle = np.degrees(np.arctan2(dx, dy))
            
            return abs(angle)
        except Exception as e:
            self.logger.error(f"Failed to calculate body angle: {e}")
            return None
    
    def predict_fall(self, frame):
        """
        Dự đoán có ngã hay không (giống fall_detection_realtime.py)
        """
        try:
            # Tiền xử lý frame
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return 1, 0.0  # Default to "not_fallen" with 0 confidence
            
            # Dự đoán
            prediction = self.model.predict(processed_frame, verbose=0)
            
            # Process prediction based on model output
            if prediction.shape[-1] == 1:  # Binary output
                confidence = float(prediction[0][0])
                class_id = 0 if confidence > self.confidence_threshold else 1
            elif prediction.shape[-1] == 2:  # Two-class output
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
            else:
                # Multiple classes - assume first is "fall"
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
            
            return class_id, float(confidence)
        except Exception as e:
            self.logger.error(f"Failed to predict fall: {e}")
            return 1, 0.0
    
    def process_image(self, image) -> Optional[Dict[str, Any]]:
        """
        Xử lý một frame và trả về kết quả (interface tương thích với hệ thống)
        """
        try:
            start_time = time.time()
            
            # Convert PIL to numpy array if needed
            if isinstance(image, Image.Image):
                frame = np.array(image)
            else:
                frame = image
            
            # Phát hiện pose
            pose_results = self.detect_pose_landmarks(frame)
            
            # Dự đoán ngã
            class_id, confidence = self.predict_fall(frame)
            
            # Tính góc cơ thể
            body_angle = None
            if pose_results and pose_results.pose_landmarks:
                body_angle = self.calculate_body_angle(pose_results.pose_landmarks)
            
            # Xác định có ngã hay không
            # Logic giống fall_detection_realtime.py
            is_fall = (class_id == 0 and confidence > self.confidence_threshold) or \
                      (body_angle and body_angle > 45)  # Góc nghiêng > 45 độ
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(is_fall, confidence, processing_time)
            
            # Tạo kết quả tương thích với hệ thống
            result = {
                'fall_detected': bool(is_fall),
                'confidence': float(confidence),
                'body_angle': float(body_angle) if body_angle is not None else 0.0,
                'leaning_angle': float(body_angle) if body_angle is not None else 0.0,  # Alias
                'class_id': int(class_id),
                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                'pose_detected': pose_results is not None and pose_results.pose_landmarks is not None,
                'processing_time': float(processing_time),
                'timestamp': time.time(),
                'model_type': 'h5_realtime',
                'raw_predictions': confidence  # Simplified
            }
            
            # Thêm pose data nếu có
            if pose_results and pose_results.pose_landmarks:
                result['pose_data'] = {
                    'landmarks_detected': True,
                    'body_angle': body_angle,
                    'pose_landmarks_count': len(pose_results.pose_landmarks.landmark)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return None
    
    def _update_stats(self, is_fall: bool, confidence: float, processing_time: float):
        """Update detection statistics"""
        self.stats['total_processed'] += 1
        
        if is_fall:
            self.stats['falls_detected'] += 1
            self.stats['last_detection_time'] = time.time()
        
        # Rolling average for processing time
        alpha = 0.1
        self.stats['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.stats['avg_processing_time']
        )
        
        # Rolling average for confidence
        self.stats['avg_confidence'] = (
            alpha * confidence + 
            (1 - alpha) * self.stats['avg_confidence']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_processed': 0,
            'falls_detected': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
            'last_detection_time': None
        }
