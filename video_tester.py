# video_tester.py - Test hệ thống với video (Updated for Pipeline)
import cv2
import time
import sys
import os
from PIL import Image
import numpy as np
from pathlib import Path

# Add src to path for pipeline imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(Path(__file__).parent / "src"))

# Import pipeline fall detector
try:
    from app.models.pipeline_fall_detector import EnhancedPipelineFallDetector
    PIPELINE_AVAILABLE = True
    print(" Pipeline detector available")
except ImportError as e:
    print(f" Pipeline detector not available: {e}")
    try:
        PIPELINE_AVAILABLE = False
        print(" Using fallback detector")
    except ImportError:
        print(" No fall detector available")
        sys.exit(1)

class VideoFallTester:
    def __init__(self, video_path, model_path=None, use_pipeline=True):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f" Video: {video_path}")
        print(f" Properties: {self.total_frames} frames, {self.fps} FPS, {self.duration:.1f}s")
        
        # Initialize fall detector
        self.fall_detector = None
        self.detector_type = "none"
        
        if model_path and os.path.exists(model_path):
            try:
                if use_pipeline and PIPELINE_AVAILABLE:
                    self.fall_detector = EnhancedPipelineFallDetector(
                        model_path=model_path,
                        model_name="mobilenet",
                        confidence_threshold=0.15
                    )
                    self.detector_type = "pipeline"
                    print(" Pipeline fall detector loaded")
                    
            except Exception as e:
                print(f" Fall detector failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test results
        self.results = {
            'frames_processed': 0,
            'falls_detected': 0,
            'fall_frames': [],
            'confidence_scores': [],
            'processing_times': [],
            'detector_type': self.detector_type
        }
    
    def process_video(self, save_output=True, display_realtime=True):
        """Process video and detect falls"""
        
        print(f"\ Starting fall detection analysis with {self.detector_type} detector...")
        
        # Setup video writer for output
        output_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_filename = f'test_output_{self.detector_type}_detection.mp4'
            output_writer = cv2.VideoWriter(
                output_filename, 
                fourcc, self.fps, 
                (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                 int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
            print(f"📹 Output will be saved as: {output_filename}")
        
        frame_num = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_num += 1
                current_time = frame_num / self.fps
                
                # Process frame for fall detection
                start_time = time.time()
                detection_result = self._process_frame(frame)
                processing_time = time.time() - start_time
                
                self.results['frames_processed'] += 1
                self.results['processing_times'].append(processing_time)
                
                if detection_result:
                    confidence = detection_result.get('confidence', 0)
                    fall_detected = detection_result.get('fall_detected', False)
                    
                    self.results['confidence_scores'].append(confidence)
                    
                    if fall_detected:
                        self.results['falls_detected'] += 1
                        fall_info = {
                            'frame': frame_num,
                            'time': current_time,
                            'confidence': confidence
                        }
                        
                        # Add pipeline-specific info
                        if self.detector_type == "pipeline":
                            fall_info.update({
                                'leaning_angle': detection_result.get('leaning_angle', 0),
                                'temporal_confirmation': detection_result.get('temporal_confirmation', False),
                                'keypoint_correlation': detection_result.get('keypoint_correlation', {})
                            })
                        
                        self.results['fall_frames'].append(fall_info)
                        
                        # Enhanced logging for pipeline detector
                        if self.detector_type == "pipeline":
                            print(f" FALL DETECTED at {current_time:.1f}s (frame {frame_num})")
                            print(f"   Confidence: {confidence:.2%}")
                            print(f"   Leaning angle: {detection_result.get('leaning_angle', 0):.1f}°")
                            print(f"   Temporal confirmation: {detection_result.get('temporal_confirmation', False)}")
                        else:
                            print(f" FALL DETECTED at {current_time:.1f}s (frame {frame_num}) - Confidence: {confidence:.2%}")
                
                # Draw overlay
                display_frame = self._draw_overlay(frame.copy(), detection_result, frame_num, current_time)
                
                # Save frame
                if output_writer:
                    output_writer.write(display_frame)
                
                # Display real-time
                if display_realtime:
                    cv2.imshow(f'Fall Detection Test - {self.detector_type.title()}', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Test interrupted by user")
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if frame_num % 30 == 0:
                    progress = (frame_num / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_num}/{self.total_frames})")
                
                # Simulate real-time playback
                time.sleep(1/self.fps * 0.1)  # 10x speed for testing
        
        except KeyboardInterrupt:
            print("Test interrupted")
        
        finally:
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
        
        self._generate_report()
    
    def _process_frame(self, frame):
        """Process single frame for fall detection"""
        if not self.fall_detector:
            return None
        
        try:
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run detection
            return self.fall_detector.process_image(pil_image)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None
    
    def _draw_overlay(self, frame, detection_result, frame_num, current_time):
        """Draw overlay on frame"""
        height, width = frame.shape[:2]
        
        # Header with detector type
        cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Fall Detection Test - {self.detector_type.title()}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_num}/{self.total_frames}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {current_time:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Detector: {self.detector_type}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Stats
        cv2.putText(frame, f"Falls: {self.results['falls_detected']}", (width - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Detection result
        if detection_result:
            fall_detected = detection_result.get('fall_detected', False)
            confidence = detection_result.get('confidence', 0)
            
            # Status
            status_text = "FALL DETECTED!" if fall_detected else "Monitoring"
            status_color = (0, 0, 255) if fall_detected else (0, 255, 0)
            
            cv2.putText(frame, status_text, (width - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Pipeline-specific info
            if self.detector_type == "pipeline" and fall_detected:
                leaning_angle = detection_result.get('leaning_angle', 0)
                temporal_conf = detection_result.get('temporal_confirmation', False)
                
                cv2.putText(frame, f"Angle: {leaning_angle:.1f}°", (width - 200, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                temp_text = "Temporal: ✓" if temporal_conf else "Temporal: ✗"
                temp_color = (0, 255, 0) if temporal_conf else (0, 255, 255)
                cv2.putText(frame, temp_text, (width - 200, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)
            
            # Confidence bar
            if confidence > 0:
                bar_width = int(200 * confidence)
                bar_color = (0, 0, 255) if confidence > 0.7 else (0, 255, 255)
                
                cv2.rectangle(frame, (width - 220, height - 40), (width - 20, height - 20), (50, 50, 50), -1)
                cv2.rectangle(frame, (width - 220, height - 40), (width - 220 + bar_width, height - 20), bar_color, -1)
                
                cv2.putText(frame, f"Risk: {confidence:.1%}", (width - 160, height - 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Flash effect for fall detection
            if fall_detected:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 20)
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        return frame
    
    def _generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print(" FALL DETECTION TEST REPORT")
        print("="*60)
        
        print(f" Video: {self.video_path}")
        print(f"  Duration: {self.duration:.1f} seconds")
        print(f" Detector: {self.detector_type}")
        print(f" Frames processed: {self.results['frames_processed']}")
        print(f" Falls detected: {self.results['falls_detected']}")
        
        if self.results['processing_times']:
            avg_processing_time = np.mean(self.results['processing_times'])
            max_processing_time = np.max(self.results['processing_times'])
            print(f" Avg processing time: {avg_processing_time:.3f}s")
            print(f" Max processing time: {max_processing_time:.3f}s")
            print(f" Processing FPS: {1/avg_processing_time:.1f}")
        
        if self.results['confidence_scores']:
            avg_confidence = np.mean(self.results['confidence_scores'])
            max_confidence = np.max(self.results['confidence_scores'])
            print(f" Avg confidence: {avg_confidence:.2%}")
            print(f" Max confidence: {max_confidence:.2%}")
        
        print(f" Fall Detection Timeline:")
        for fall in self.results['fall_frames']:
            base_info = f"   • Frame {fall['frame']} at {fall['time']:.1f}s - Confidence: {fall['confidence']:.2%}"
            
            if self.detector_type == "pipeline":
                extra_info = f" | Angle: {fall.get('leaning_angle', 0):.1f}° | Temporal: {fall.get('temporal_confirmation', False)}"
                print(base_info + extra_info)
            else:
                print(base_info)
        
        if not self.results['fall_frames']:
            print("   • No falls detected in this video")
        
        # Pipeline-specific stats
        if self.detector_type == "pipeline" and hasattr(self.fall_detector, 'get_stats'):
            detector_stats = self.fall_detector.get_stats()
            print(f" Detector Statistics:")
            print(f"   • Total processed: {detector_stats.get('total_processed', 0)}")
            print(f"   • Falls detected: {detector_stats.get('falls_detected', 0)}")
            print(f"   • Avg processing time: {detector_stats.get('avg_processing_time', 0):.3f}s")
        
        print("Test completed!")
        output_file = f'test_output_{self.detector_type}_detection.mp4'
        if os.path.exists(output_file):
            print(f" Output video saved: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Fall Detection with Video')
    parser.add_argument('video', help='Path to test video file')
    parser.add_argument('--model', default='models/posenet_mobilenet_v1.tflite',
                       help='Path to fall detection model')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save output video')
    parser.add_argument('--no-pipeline', action='store_true', help='Use standard detector instead of pipeline')
    parser.add_argument('--compare', action='store_true', help='Compare both detectors')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f" Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f" Model file not found: {args.model}")
        return
    
    try:
        if args.compare and PIPELINE_AVAILABLE:
            # Test both detectors
            print(" Comparing Pipeline vs Standard Detector...")
            
            # Test pipeline detector
            print("\n" + "="*40)
            print("TESTING PIPELINE DETECTOR")
            print("="*40)
            pipeline_tester = VideoFallTester(args.video, args.model, use_pipeline=True)
            pipeline_tester.process_video(
                save_output=not args.no_save,
                display_realtime=not args.no_display
            )
            
            # Test standard detector
            print("\n" + "="*40)
            print("TESTING STANDARD DETECTOR")
            print("="*40)
            standard_tester = VideoFallTester(args.video, args.model, use_pipeline=False)
            standard_tester.process_video(
                save_output=not args.no_save,
                display_realtime=not args.no_display
            )
            
            # Comparison report
            print("\n" + "="*60)
            print(" DETECTOR COMPARISON")
            print("="*60)
            print(f"Pipeline Detector:")
            print(f"   Falls detected: {pipeline_tester.results['falls_detected']}")
            print(f"   Avg processing time: {np.mean(pipeline_tester.results['processing_times']):.3f}s")
            
            print(f"Standard Detector:")
            print(f"   Falls detected: {standard_tester.results['falls_detected']}")
            print(f"   Avg processing time: {np.mean(standard_tester.results['processing_times']):.3f}s")
            
        else:
            # Single detector test
            use_pipeline = not args.no_pipeline and PIPELINE_AVAILABLE
            tester = VideoFallTester(args.video, args.model, use_pipeline=use_pipeline)
            tester.process_video(
                save_output=not args.no_save,
                display_realtime=not args.no_display
            )
            
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()