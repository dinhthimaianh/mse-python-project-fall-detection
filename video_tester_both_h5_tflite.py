# video_tester.py - Enhanced version hỗ trợ .h5 và adaptive model manager
import cv2
import time
import sys
import os
from PIL import Image
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(Path(__file__).parent / "src"))

# Import detectors
DETECTORS_AVAILABLE = {}

try:
    from app.models.h5_fall_detector import H5FallDetector
    DETECTORS_AVAILABLE['h5'] = True
    print(" H5 detector available")
except ImportError as e:
    print(f" H5 detector not available: {e}")
    DETECTORS_AVAILABLE['h5'] = False

try:
    from app.models.pipeline_fall_detector import ProductionPipelineFallDetector
    DETECTORS_AVAILABLE['tflite'] = True
    print(" TFLite detector available")
except ImportError as e:
    print(f" TFLite detector not available: {e}")
    DETECTORS_AVAILABLE['tflite'] = False

try:
    from app.models.adaptive_model_manager import AdaptiveModelManager
    DETECTORS_AVAILABLE['adaptive'] = True
    print(" Adaptive Model Manager available")
except ImportError as e:
    print(f" Adaptive Model Manager not available: {e}")
    DETECTORS_AVAILABLE['adaptive'] = False

class VideoFallTester:
    def __init__(self, video_path, model_path=None, detector_type="auto"):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback FPS
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f" Video: {Path(video_path).name}")
        print(f"    Properties: {self.total_frames} frames, {self.fps:.1f} FPS, {self.duration:.1f}s")
        print(f"    Resolution: {self.width}x{self.height}")
        
        # Initialize fall detector
        self.fall_detector = None
        self.detector_type = "none"
        self.model_path = model_path
        
        self._initialize_detector(detector_type, model_path)
        
        # Test results
        self.results = {
            'video_path': video_path,
            'model_path': model_path,
            'detector_type': self.detector_type,
            'start_time': datetime.now().isoformat(),
            'frames_processed': 0,
            'falls_detected': 0,
            'fall_frames': [],
            'confidence_scores': [],
            'processing_times': [],
            'errors': [],
            'video_properties': {
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration': self.duration,
                'resolution': (self.width, self.height)
            }
        }
    
    def _initialize_detector(self, detector_type, model_path):
        """Initialize appropriate detector"""
        try:
            if detector_type == "auto":
                # Auto-detect based on model file extension
                if model_path:
                    ext = Path(model_path).suffix.lower()
                    if ext == '.h5' and DETECTORS_AVAILABLE['h5']:
                        detector_type = "h5"
                    elif ext == '.tflite' and DETECTORS_AVAILABLE['tflite']:
                        detector_type = "tflite"
                    else:
                        detector_type = "adaptive"
                else:
                    detector_type = "adaptive"
            
            # Initialize specific detector
            if detector_type == "h5" and DETECTORS_AVAILABLE['h5']:
                if not model_path or not Path(model_path).exists():
                    raise ValueError(f"H5 model file not found: {model_path}")
                
                self.fall_detector = H5FallDetector(
                    model_path=model_path,
                    confidence_threshold=0.9
                )
                self.detector_type = "h5"
                print(f" Loaded H5 detector: {Path(model_path).name}")
                
            elif detector_type == "tflite" and DETECTORS_AVAILABLE['tflite']:
                if not model_path or not Path(model_path).exists():
                    raise ValueError(f"TFLite model file not found: {model_path}")
                
                self.fall_detector = ProductionPipelineFallDetector(
                    model_path=model_path,
                    model_name="mobilenet",
                    confidence_threshold=0.5
                )
                self.detector_type = "tflite"
                print(f" Loaded TFLite detector: {Path(model_path).name}")
                
            elif detector_type == "adaptive" and DETECTORS_AVAILABLE['adaptive']:
                self.fall_detector = AdaptiveModelManager()
                self.detector_type = "adaptive"
                
                # Get current model info
                status = self.fall_detector.get_system_status()
                active_model = status['current_model']['type']
                print(f" Loaded Adaptive Manager: {active_model}")
                
            else:
                available_types = [k for k, v in DETECTORS_AVAILABLE.items() if v]
                raise ValueError(f"Detector type '{detector_type}' not available. Available: {available_types}")
                
        except Exception as e:
            print(f" Failed to initialize {detector_type} detector: {e}")
            raise
    
    def process_video(self, save_output=True, display_realtime=True, process_speed=1.0, 
                     save_detections=True, confidence_threshold=None):
        """Process video and detect falls"""
        
        print(f" Starting fall detection analysis...")
        print(f"    Detector: {self.detector_type}")
        print(f"    Speed: {process_speed}x")
        
        # Override confidence threshold if specified
        if confidence_threshold is not None:
            if hasattr(self.fall_detector, 'confidence_threshold'):
                self.fall_detector.confidence_threshold = confidence_threshold
                print(f"   🎚️ Confidence threshold: {confidence_threshold}")
        
        # Setup video writer for output
        output_writer = None
        output_filename = None
        # if save_output:
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     output_filename = f'fall_detection_test_{self.detector_type}_{timestamp}.mp4'
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     output_writer = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.width, self.height))
        #     print(f"📹 Output will be saved as: {output_filename}")
        
        # Setup detection log
        detection_log = []
        
        frame_num = 0
        last_print_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_num += 1
                current_time = frame_num / self.fps
                
                # Process frame for fall detection
                start_time = time.time()
                detection_result = self._process_frame(frame, frame_num)
                processing_time = time.time() - start_time
                
                self.results['frames_processed'] += 1
                self.results['processing_times'].append(processing_time)
                
                # Handle detection result
                if detection_result:
                    self._handle_detection_result(detection_result, frame_num, current_time, detection_log)
                
                # Draw overlay
                display_frame = self._draw_overlay(frame.copy(), detection_result, frame_num, current_time)
                
                # Save frame
                if output_writer:
                    output_writer.write(display_frame)
                
                # Display real-time
                if display_realtime:
                    cv2.imshow(f'Fall Detection Test - {self.detector_type.upper()}', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(" Test interrupted by user")
                        break
                    elif key == ord(' '):  # Space to pause
                        print(" Paused - Press any key to continue")
                        cv2.waitKey(0)
                    elif key == ord('s'):  # Save screenshot
                        screenshot_name = f'screenshot_{frame_num}.jpg'
                        cv2.imwrite(screenshot_name, display_frame)
                        print(f" Screenshot saved: {screenshot_name}")
                
                # Progress update
                current_time_real = time.time()
                if current_time_real - last_print_time > 5:  # Every 5 seconds
                    progress = (frame_num / self.total_frames) * 100
                    avg_processing_time = np.mean(self.results['processing_times'][-30:])  # Last 30 frames
                    processing_fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
                    
                    print(f" Progress: {progress:.1f}% ({frame_num}/{self.total_frames}) | "
                          f"Processing FPS: {processing_fps:.1f} | "
                          f"Falls: {self.results['falls_detected']}")
                    last_print_time = current_time_real
                
                # Simulate playback speed
                if process_speed < 1.0:
                    time.sleep((1/self.fps) * (1/process_speed - 1))
        
        except KeyboardInterrupt:
            print(" Test interrupted by keyboard")
        except Exception as e:
            print(f" Error during processing: {e}")
            self.results['errors'].append(str(e))
        
        finally:
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
            
            # Save detection log
            if save_detections and detection_log:
                log_filename = f'detections_{self.detector_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(log_filename, 'w') as f:
                    json.dump(detection_log, f, indent=2)
                print(f" Detection log saved: {log_filename}")
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['output_video'] = output_filename
        self._generate_report()
        
        return self.results
    
    def _process_frame(self, frame, frame_num):
        """Process single frame for fall detection"""
        if not self.fall_detector:
            return None
        
        try:
            # Convert to PIL Image for H5 detector
            if self.detector_type == "h5":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                return self.fall_detector.process_image(pil_image)
            
            # For TFLite and adaptive managers
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                return self.fall_detector.process_image(pil_image)
            
        except Exception as e:
            error_msg = f"Frame {frame_num} processing error: {e}"
            print(f" {error_msg}")
            self.results['errors'].append(error_msg)
            return None
    
    def _handle_detection_result(self, detection_result, frame_num, current_time, detection_log):
        """Handle detection result and update statistics"""
        try:
            confidence = detection_result.get('confidence', 0)
            fall_detected = detection_result.get('fall_detected', False)
            
            # Always log confidence scores
            self.results['confidence_scores'].append(confidence)
            
            # Log all detections for analysis
            detection_entry = {
                'frame': frame_num,
                'time': current_time,
                'confidence': confidence,
                'fall_detected': fall_detected,
                'detector_type': self.detector_type
            }
            
            # Add detector-specific information
            if self.detector_type == "h5":
                detection_entry.update({
                    'model_type': detection_result.get('model_type', 'h5_keras'),
                    'raw_predictions': detection_result.get('raw_predictions')
                })
            elif self.detector_type == "tflite":
                detection_entry.update({
                    'leaning_angle': detection_result.get('leaning_angle', 0),
                    'keypoint_correlation': detection_result.get('keypoint_correlation', {}),
                    'pose_stability': detection_result.get('pose_stability', 0)
                })
            elif self.detector_type == "adaptive":
                detection_entry.update({
                    'active_model': detection_result.get('model_type', 'unknown'),
                    'inference_time': detection_result.get('inference_time', 0)
                })
            
            detection_log.append(detection_entry)
            
            # Handle fall detection
            if fall_detected:
                self.results['falls_detected'] += 1
                
                fall_info = {
                    'frame': frame_num,
                    'time': current_time,
                    'confidence': confidence,
                    'detector_type': self.detector_type
                }
                
                # Add detector-specific fall info
                if self.detector_type == "h5":
                    fall_info.update({
                        'model_output': detection_result.get('raw_predictions', [])
                    })
                elif self.detector_type == "tflite":
                    fall_info.update({
                        'leaning_angle': detection_result.get('leaning_angle', 0),
                        'temporal_confirmation': detection_result.get('temporal_confirmation', False),
                        'keypoint_correlation': detection_result.get('keypoint_correlation', {})
                    })
                elif self.detector_type == "adaptive":
                    fall_info.update({
                        'active_model': detection_result.get('model_type', 'unknown'),
                        'inference_time': detection_result.get('inference_time', 0)
                    })
                
                self.results['fall_frames'].append(fall_info)
                
                # Enhanced logging
                print(f" FALL DETECTED at {current_time:.1f}s (frame {frame_num})")
                print(f"    Confidence: {confidence:.2%}")
                
                if self.detector_type == "h5":
                    model_type = detection_result.get('model_type', 'h5_keras')
                    print(f"    Model: {model_type}")
                elif self.detector_type == "tflite":
                    angle = detection_result.get('leaning_angle', 0)
                    temporal = detection_result.get('temporal_confirmation', False)
                    print(f"    Leaning angle: {angle:.1f}°")
                    print(f"    Temporal confirmation: {temporal}")
                elif self.detector_type == "adaptive":
                    active_model = detection_result.get('model_type', 'unknown')
                    inference_time = detection_result.get('inference_time', 0)
                    print(f"    Active model: {active_model}")
                    print(f"    Inference time: {inference_time*1000:.1f}ms")
                
        except Exception as e:
            error_msg = f"Error handling detection result: {e}"
            print(f" {error_msg}")
            self.results['errors'].append(error_msg)
    
    def _draw_overlay(self, frame, detection_result, frame_num, current_time):
        """Draw enhanced overlay on frame"""
        height, width = frame.shape[:2]
        
        # Header background
        cv2.rectangle(frame, (0, 0), (width, 140), (0, 0, 0), -1)
        
        # Title with detector type
        title = f"Fall Detection Test - {self.detector_type.upper()}"
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_num}/{self.total_frames}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s / {self.duration:.1f}s", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Progress bar
        progress = frame_num / self.total_frames
        bar_width = width - 300
        cv2.rectangle(frame, (10, 85), (10 + bar_width, 100), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 85), (10 + int(bar_width * progress), 100), (0, 255, 0), -1)
        cv2.putText(frame, f"{progress*100:.1f}%", (10 + bar_width + 10, 98), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detector info
        detector_info = f"Detector: {self.detector_type}"
        if self.detector_type == "adaptive" and detection_result:
            active_model = detection_result.get('model_type', 'unknown')
            detector_info += f" ({active_model})"
        
        cv2.putText(frame, detector_info, (10, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Stats in top right
        cv2.putText(frame, f"Falls: {self.results['falls_detected']}", (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Performance info
        if self.results['processing_times']:
            recent_times = self.results['processing_times'][-10:]  # Last 10 frames
            avg_time = np.mean(recent_times)
            fps_estimate = 1 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps_estimate:.1f}", (width - 200, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Detection result overlay
        if detection_result:
            fall_detected = detection_result.get('fall_detected', False)
            confidence = detection_result.get('confidence', 0)
            
            # Status indicator
            status_text = " FALL DETECTED!" if fall_detected else " Monitoring"
            status_color = (0, 0, 255) if fall_detected else (0, 255, 0)
            
            cv2.putText(frame, status_text, (width - 250, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Confidence visualization
            if confidence > 0:
                # Confidence bar
                bar_width = 200
                bar_height = 20
                bar_x = width - 220
                bar_y = height - 60
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                
                confidence_width = int(bar_width * confidence)
                bar_color = (0, 0, 255) if confidence > 0.7 else (0, 255, 255) if confidence > 0.4 else (0, 255, 0)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), bar_color, -1)
                
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detector-specific overlays
            if self.detector_type == "h5":
                model_type = detection_result.get('model_type', 'h5_keras')
                cv2.putText(frame, f"Model: {model_type}", (width - 250, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            elif self.detector_type == "tflite" and fall_detected:
                angle = detection_result.get('leaning_angle', 0)
                cv2.putText(frame, f"Angle: {angle:.1f}°", (width - 250, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                temporal = detection_result.get('temporal_confirmation', False)
                temp_text = "Temporal: ✓" if temporal else "Temporal: ✗"
                temp_color = (0, 255, 0) if temporal else (0, 255, 255)
                cv2.putText(frame, temp_text, (width - 250, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)
            
            elif self.detector_type == "adaptive":
                active_model = detection_result.get('model_type', 'unknown')
                cv2.putText(frame, f"Active: {active_model}", (width - 250, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                inference_time = detection_result.get('inference_time', 0)
                cv2.putText(frame, f"{inference_time*1000:.1f}ms", (width - 250, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Flash effect for fall detection
            if fall_detected:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 30)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Instructions
        instructions = "Q: Quit | SPACE: Pause | S: Screenshot"
        cv2.putText(frame, instructions, (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print(" FALL DETECTION VIDEO TEST REPORT")
        print("="*80)
        
        # Basic info
        print(f" Video: {Path(self.video_path).name}")
        print(f"    Duration: {self.duration:.1f} seconds ({self.total_frames} frames)")
        print(f"    Resolution: {self.width}x{self.height}")
        print(f"    FPS: {self.fps:.1f}")
        
        print(f" Detector: {self.detector_type}")
        if self.model_path:
            print(f"    Model: {Path(self.model_path).name}")
        
        # Processing stats
        print(f" Processing Statistics:")
        print(f"    Frames processed: {self.results['frames_processed']:,}")
        print(f"    Falls detected: {self.results['falls_detected']}")
        
        if self.results['processing_times']:
            processing_times = np.array(self.results['processing_times'])
            print(f"    Avg processing time: {np.mean(processing_times):.3f}s")
            print(f"    Max processing time: {np.max(processing_times):.3f}s")
            print(f"    Min processing time: {np.min(processing_times):.3f}s")
            print(f"    Processing FPS: {1/np.mean(processing_times):.1f}")
        
        # Confidence stats
        if self.results['confidence_scores']:
            confidence_scores = np.array(self.results['confidence_scores'])
            print(f"    Avg confidence: {np.mean(confidence_scores):.2%}")
            print(f"    Max confidence: {np.max(confidence_scores):.2%}")
            print(f"    Confidence std: {np.std(confidence_scores):.2%}")
        
        # Fall detection timeline
        print(f" Fall Detection Timeline:")
        if self.results['fall_frames']:
            for i, fall in enumerate(self.results['fall_frames'], 1):
                base_info = f"   {i:2d}. Frame {fall['frame']:6d} at {fall['time']:6.1f}s - Confidence: {fall['confidence']:6.2%}"
                
                # Add detector-specific info
                if self.detector_type == "h5":
                    print(f"{base_info} | Model: {fall.get('model_type', 'h5')}")
                elif self.detector_type == "tflite":
                    angle = fall.get('leaning_angle', 0)
                    temporal = fall.get('temporal_confirmation', False)
                    print(f"{base_info} | Angle: {angle:5.1f}° | Temporal: {temporal}")
                elif self.detector_type == "adaptive":
                    model = fall.get('active_model', 'unknown')
                    inference = fall.get('inference_time', 0)
                    print(f"{base_info} | Model: {model} | Time: {inference*1000:.1f}ms")
                else:
                    print(base_info)
        else:
            print("    No falls detected in this video")
        
        # Detector-specific statistics
        if self.detector_type == "adaptive" and hasattr(self.fall_detector, 'get_system_status'):
            status = self.fall_detector.get_system_status()
            print(f" Adaptive Manager Status:")
            print(f"    Active model: {status['current_model']['type']}")
            print(f"    Memory usage: {status['performance']['memory_usage_mb']:.1f} MB")
            print(f"    Available models: {', '.join(status['available_models'])}")
        
        # Error summary
        if self.results['errors']:
            print(f" Errors encountered: {len(self.results['errors'])}")
            for error in self.results['errors'][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(self.results['errors']) > 5:
                print(f"   • ... and {len(self.results['errors']) - 5} more errors")
        
        # Output files
        if self.results.get('output_video'):
            print(f"📹 Output video: {self.results['output_video']}")
        
        print(f" Test completed at: {self.results.get('end_time', 'unknown')}")
        print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Fall Detection with Video - Enhanced Version')
    parser.add_argument('video', help='Path to test video file')
    parser.add_argument('--model', help='Path to fall detection model (.h5 or .tflite)')
    parser.add_argument('--detector', choices=['auto', 'h5', 'tflite', 'adaptive'], 
                       default='auto', help='Detector type to use')
    parser.add_argument('--confidence', type=float, help='Confidence threshold override')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save output video')
    parser.add_argument('--speed', type=float, default=1.0, help='Processing speed multiplier')
    parser.add_argument('--compare', action='store_true', help='Compare all available detectors')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f" Video file not found: {args.video}")
        return 1
    
    if args.model and not os.path.exists(args.model):
        print(f" Model file not found: {args.model}")
        return 1
    
    try:
        if args.compare:
            # Compare all available detectors
            print(" Comparing all available detectors...")
            
            detectors_to_test = []
            if args.model:
                ext = Path(args.model).suffix.lower()
                if ext == '.h5' and DETECTORS_AVAILABLE['h5']:
                    detectors_to_test.append(('h5', args.model))
                elif ext == '.tflite' and DETECTORS_AVAILABLE['tflite']:
                    detectors_to_test.append(('tflite', args.model))
            
            if DETECTORS_AVAILABLE['adaptive']:
                detectors_to_test.append(('adaptive', None))
            
            if not detectors_to_test:
                print(" No detectors available for comparison")
                return 1
            
            results = {}
            for detector_type, model_path in detectors_to_test:
                print(f"\n{'='*50}")
                print(f" TESTING {detector_type.upper()} DETECTOR")
                print(f"{'='*50}")
                
                tester = VideoFallTester(args.video, model_path, detector_type)
                result = tester.process_video(
                    save_output=not args.no_save,
                    display_realtime=not args.no_display,
                    process_speed=args.speed,
                    confidence_threshold=args.confidence
                )
                results[detector_type] = result
            
            # Comparison summary
            print(f"\n{'='*80}")
            print(" DETECTOR COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            for detector_type, result in results.items():
                avg_time = np.mean(result['processing_times']) if result['processing_times'] else 0
                print(f"{detector_type.upper():>12}: "
                      f"Falls={result['falls_detected']:2d} | "
                      f"FPS={1/avg_time:5.1f} | "
                      f"Avg Time={avg_time*1000:6.1f}ms")
            
        else:
            # Single detector test
            tester = VideoFallTester(args.video, args.model, args.detector)
            tester.process_video(
                save_output=not args.no_save,
                display_realtime=not args.no_display,
                process_speed=args.speed,
                confidence_threshold=args.confidence
            )
        
        return 0
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())