#!/usr/bin/env python3
"""
Test script for the improved ship detection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_detector import YOLOShipDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_detection():
    """Test the improved detection system"""
    try:
        # Initialize detector
        logger.info("Initializing ship detector...")
        detector = YOLOShipDetector()
        
        # Test with one of the uploaded images
        test_image = "uploads/334f0fe1-ecbd-418e-9b76-b4256cbf6978_test2.jpeg"
        
        if not os.path.exists(test_image):
            logger.error(f"Test image not found: {test_image}")
            return False
        
        logger.info(f"Testing detection on: {test_image}")
        
        # Run detection
        results = detector.detect_ships(test_image, confidence_threshold=0.3)
        
        # Print results
        logger.info(f"Detection Results:")
        logger.info(f"  Ships detected: {results['ship_count']}")
        logger.info(f"  Processing time: {results.get('processing_time', 0):.2f}s")
        
        if results['ship_count'] > 0:
            logger.info(f"  Vessel types: {results['vessel_types']}")
            logger.info(f"  Confidences: {[f'{c:.2f}' for c in results['confidence_scores']]}")
            logger.info(f"  Bounding boxes: {results['bounding_boxes']}")
        
        # Test annotation
        output_path = "test_result.jpg"
        logger.info(f"Creating annotated image: {output_path}")
        detector.annotate_image(test_image, results, output_path)
        
        # Test summary
        summary = detector.get_detection_summary(results)
        logger.info(f"Detection Summary: {summary}")
        
        logger.info("✅ Detection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detection()
    sys.exit(0 if success else 1)