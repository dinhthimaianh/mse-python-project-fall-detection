# app/models/h5_fall_detector.py
import tensorflow as tf
import numpy as np
import cv2
from typing import Dict, Any, Optional
import logging
from PIL import Image

class H5FallDetector:
    """Fall detector sử dụng .h5 model"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = None
        self.input_shape = None
        self.output_shape = None
        
        self._load_model()
    
    def _load_model(self):
        """Load .h5 model"""
        try:
            self.logger.info(f"Loading .h5 model from: {self.model_path}")
            
            # Load Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Get model info
            self.input_shape = self.model.input_shape
            self.output_shape = self.model.output_shape
            
            self.logger.info(f" Model loaded successfully")
            self.logger.info(f"   Input shape: {self.input_shape}")
            self.logger.info(f"   Output shape: {self.output_shape}")
            
        except Exception as e:
            self.logger.error(f" Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get target size from model input shape
            target_height = self.input_shape[1]
            target_width = self.input_shape[2]
            
            # Resize image
            image_resized = cv2.resize(image, (target_width, target_height))
            
            # Normalize to [0, 1] or [-1, 1] depending on model
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            return image_batch
            
        except Exception as e:
            self.logger.error(f"Image preprocessing error: {e}")
            return None
    
    def process_image(self, image) -> Optional[Dict[str, Any]]:
        """Process image and detect fall"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None
            
            # Run inference
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Process predictions (adjust based on your model output)
            result = self._process_predictions(predictions)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            return None
    
    def _process_predictions(self, predictions) -> Dict[str, Any]:
        """Process model predictions"""
        try:
            # Adjust this based on your model's output format
            # Example for classification model:
            if isinstance(predictions, np.ndarray):
                if predictions.shape[-1] == 1:  # Binary classification
                    confidence = float(predictions[0][0])
                    fall_detected = confidence > self.confidence_threshold
                elif predictions.shape[-1] == 2:  # Two-class classification
                    fall_confidence = float(predictions[0][1])
                    normal_confidence = float(predictions[0][0])
                    confidence = fall_confidence
                    fall_detected = fall_confidence > self.confidence_threshold
                else:
                    # Multi-output model - adjust as needed
                    confidence = float(np.max(predictions))
                    fall_detected = confidence > self.confidence_threshold
            else:
                # Multiple outputs
                confidence = 0.5
                fall_detected = False
            
            result = {
                'fall_detected': fall_detected,
                'confidence': confidence,
                'raw_predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'model_type': 'h5_keras'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction processing error: {e}")
            return {
                'fall_detected': False,
                'confidence': 0.0,
                'error': str(e)
            }