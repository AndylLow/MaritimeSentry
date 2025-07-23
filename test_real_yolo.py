#!/usr/bin/env python3
"""
Test script for the real YOLO ship detection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_yolo():
    """Test the real YOLO detection system"""
    try:
        logger.info("Testing Real YOLO Ship Detection System")
        logger.info("=" * 50)
        
        # Test imports first
        logger.info("Testing imports...")
        try:
            import torch
            logger.info(f"✅ PyTorch available: {torch.__version__}")
            logger.info(f"   CUDA available: {torch.cuda.is_available()}")
            
            from ultralytics import YOLO
            logger.info("✅ Ultralytics YOLO available")
            
            from yolo_detector_real import RealYOLOShipDetector
            logger.info("✅ Real YOLO detector module imported")
            
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            return False
        
        # Initialize detector
        logger.info("\nInitializing YOLO detector...")
        try:
            detector = RealYOLOShipDetector()
            logger.info("✅ YOLO detector initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize detector: {e}")
            return False
        
        # Test with sample image
        test_images = [
            "uploads/334f0fe1-ecbd-418e-9b76-b4256cbf6978_test2.jpeg",
            "uploads/b47162ea-ec49-4bc9-87ab-eebb3dfe7a2c_test3.jpeg",
        ]
        
        for test_image in test_images:
            if os.path.exists(test_image):
                logger.info(f"\n🔍 Testing detection on: {test_image}")
                
                try:
                    # Run detection
                    results = detector.detect_ships(test_image, confidence_threshold=0.25)
                    
                    # Print results
                    logger.info(f"📊 Detection Results:")
                    logger.info(f"   Ships detected: {results['ship_count']}")
                    logger.info(f"   Processing time: {results.get('processing_time', 0):.2f}s")
                    logger.info(f"   Model: {results.get('model_version', 'Unknown')}")
                    
                    if results['ship_count'] > 0:
                        logger.info(f"   Vessel types: {results['vessel_types']}")
                        logger.info(f"   Confidences: {[f'{c:.3f}' for c in results['confidence_scores']]}")
                        logger.info(f"   YOLO classes: {results.get('yolo_classes', [])}")
                        
                        # Test annotation
                        output_path = f"test_real_yolo_{os.path.basename(test_image)}"
                        logger.info(f"   Creating annotated image: {output_path}")
                        detector.annotate_image(test_image, results, output_path)
                        
                        # Test summary
                        summary = detector.get_detection_summary(results)
                        logger.info(f"   Quality: {summary.get('detection_quality', 'Unknown')}")
                        logger.info(f"   Avg confidence: {summary.get('confidence_stats', {}).get('average', 0):.3f}")
                    
                    logger.info("✅ Detection test passed")
                    
                except Exception as e:
                    logger.error(f"❌ Detection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                break
        else:
            logger.warning("⚠️  No test images found")
            return False
        
        logger.info("\n🎉 All tests completed successfully!")
        logger.info("Real YOLO detection system is working properly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_yolo()
    if success:
        print("\n✅ SUCCESS: Real YOLO ship detection is working!")
    else:
        print("\n❌ FAILED: Real YOLO ship detection has issues.")
    sys.exit(0 if success else 1)