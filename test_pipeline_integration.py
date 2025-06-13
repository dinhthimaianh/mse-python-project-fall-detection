# test_pipeline_integration.py
from app.models.pipeline_fall_detector import EnhancedPipelineFallDetector
import cv2
from PIL import Image
import sys

def test_pipeline_detector(video_path, model_path):
    """Test pipeline detector"""
    print(" Testing Pipeline Fall Detector Integration...")
    
    try:
        # Initialize detector
        detector = EnhancedPipelineFallDetector(
            model_path=model_path,
            model_name="mobilenet",
            confidence_threshold=0.15
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process frame
            result = detector.process_image(pil_image)
            
            if result:
                if result.get('fall_detected', False):
                    print(f" FALL DETECTED at frame {frame_count}!")
                    print(f"   Confidence: {result.get('confidence', 0):.2%}")
                    print(f"   Leaning angle: {result.get('leaning_angle', 0):.1f}°")
                    print(f"   Temporal confirmation: {result.get('temporal_confirmation', False)}")
                
                # Log every 100 frames
                if frame_count % 100 == 0:
                    stats = detector.get_stats()
                    print(f"Frame {frame_count}: Processed {stats['total_processed']}, "
                          f"Falls: {stats['falls_detected']}, "
                          f"Avg time: {stats['avg_processing_time']:.3f}s")
        
        cap.release()
        
        # Final stats
        final_stats = detector.get_stats()
        print(f" Final Statistics:")
        print(f"   Total frames: {final_stats['total_processed']}")
        print(f"   Falls detected: {final_stats['falls_detected']}")
        print(f"   Average processing time: {final_stats['avg_processing_time']:.3f}s")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_pipeline_integration.py <video_path> <model_path>")
        sys.exit(1)
    
    test_pipeline_detector(sys.argv[1], sys.argv[2])