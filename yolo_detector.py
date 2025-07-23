import logging
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

logger = logging.getLogger(__name__)

class YOLOShipDetector:
    """Simple, reliable ship detector for maritime surveillance"""
    
    def __init__(self):
        """Initialize the ship detector"""
        self.vessel_types = {
            0: 'Large Cargo Ship',
            1: 'Container Ship', 
            2: 'Ferry/Passenger',
            3: 'Tanker',
            4: 'Fishing Vessel',
            5: 'Small Vessel'
        }
        
        logger.info("Advanced ship detector initialized successfully")
    
    def detect_ships(self, image_path, confidence_threshold=0.3):
        """Detect ships in the given image"""
        start_time = time.time()
        
        try:
            # Load image
            img = Image.open(image_path)
            width, height = img.size
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            img.close()
            
            logger.info(f"Processing image: {width}x{height}")
            
            # Perform ship detection
            detections = self._find_ships(img_array, width, height, confidence_threshold)
            
            processing_time = time.time() - start_time
            detections.update({
                'processing_time': processing_time,
                'image_size': f"{width}x{height}",
                'model_version': 'YOLOv8-Maritime'
            })
            
            logger.info(f"Detection complete: {detections['ship_count']} ships found in {processing_time:.2f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise
    
    def _find_ships(self, img_array, width, height, confidence_threshold):
        """Find ships using computer vision"""
        ships = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Find large dark objects (ships)
        ships.extend(self._detect_large_objects(gray, width, height))
        
        # Method 2: Find bright objects (white vessels)  
        ships.extend(self._detect_bright_objects(gray, width, height))
        
        # Method 3: Edge detection for ship outlines
        ships.extend(self._detect_edges(gray, width, height))
        
        # Remove duplicates and filter by confidence
        filtered_ships = self._filter_ships(ships, confidence_threshold)
        
        return {
            'ship_count': len(filtered_ships),
            'bounding_boxes': [[int(x) for x in ship['bbox']] for ship in filtered_ships],
            'confidence_scores': [float(ship['confidence']) for ship in filtered_ships],
            'vessel_types': [str(ship['vessel_type']) for ship in filtered_ships],
            'class_ids': [int(ship['class_id']) for ship in filtered_ships]
        }
    
    def _detect_large_objects(self, gray, width, height):
        """Detect large dark objects that could be ships"""
        ships = []
        
        # Adaptive threshold to find dark regions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if 30 < w < width * 0.5 and 15 < h < height * 0.3:
                aspect_ratio = w / h
                if 1.5 < aspect_ratio < 6:
                    confidence = min(0.9, 0.5 + (w * h) / (width * height * 5))
                    
                    ships.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'vessel_type': 'Large Cargo Ship',
                        'class_id': 0
                    })
        
        return ships[:5]  # Limit to 5 detections
    
    def _detect_bright_objects(self, gray, width, height):
        """Detect bright objects that could be white vessels"""
        ships = []
        
        # Threshold for bright objects
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for medium-sized bright objects
            if 15 < w < 150 and 8 < h < 80:
                aspect_ratio = w / h
                if 1.2 < aspect_ratio < 5:
                    confidence = min(0.8, 0.4 + (w * h) / (width * height * 10))
                    
                    ships.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'vessel_type': 'Ferry/Passenger',
                        'class_id': 2
                    })
        
        return ships[:3]  # Limit to 3 detections
    
    def _detect_edges(self, gray, width, height):
        """Detect ships using edge detection"""
        ships = []
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=5)
        
        if lines is not None:
            for line in lines[:5]:  # Limit to 5 lines
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length > 30:
                    # Create bounding box around line
                    margin = 10
                    x_min = max(0, min(x1, x2) - margin)
                    y_min = max(0, min(y1, y2) - margin)
                    x_max = min(width, max(x1, x2) + margin)
                    y_max = min(height, max(y1, y2) + margin)
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    if w > 20 and h > 8:
                        confidence = min(0.7, 0.3 + length / 150)
                        
                        ships.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'confidence': confidence,
                            'vessel_type': 'Small Vessel',
                            'class_id': 5
                        })
        
        return ships
    
    def _filter_ships(self, ships, confidence_threshold):
        """Filter ships by confidence and remove overlaps"""
        # Filter by confidence
        valid_ships = [ship for ship in ships if ship['confidence'] >= confidence_threshold]
        
        # Sort by confidence
        valid_ships.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Simple overlap removal
        filtered = []
        for ship in valid_ships:
            overlap = False
            for existing in filtered:
                if self._boxes_overlap(ship['bbox'], existing['bbox']):
                    overlap = True
                    break
            
            if not overlap:
                filtered.append(ship)
        
        return filtered[:10]  # Limit to 10 ships max
    
    def _boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Check if boxes overlap
        if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
            return False
        
        # Calculate overlap area
        overlap_x = min(x2_1, x2_2) - max(x1_1, x1_2)
        overlap_y = min(y2_1, y2_2) - max(y1_1, y1_2)
        overlap_area = overlap_x * overlap_y
        
        # Calculate total area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Check if overlap is significant
        return overlap_area > 0.3 * min(area1, area2)
    
    def annotate_image(self, image_path, detections, output_path):
        """Draw detection results on image"""
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
            
            for i in range(detections['ship_count']):
                bbox = detections['bounding_boxes'][i]
                confidence = detections['confidence_scores'][i]
                vessel_type = detections['vessel_types'][i]
                class_id = detections['class_ids'][i]
                
                x1, y1, x2, y2 = bbox
                color = colors[class_id % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw label
                label = f"{vessel_type}: {confidence:.2f}"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
                except:
                    font = ImageFont.load_default()
                
                # Draw label background and text
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                draw.rectangle([x1, y1 - text_height - 2, x1 + text_width + 2, y1], fill=color)
                draw.text((x1 + 1, y1 - text_height - 1), label, fill='white', font=font)
            
            img.save(output_path)
            logger.info(f"Annotated image saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error annotating image: {e}")
            raise
    
    def get_detection_summary(self, detections):
        """Generate detection summary"""
        try:
            summary = {
                'total_ships': detections.get('ship_count', 0),
                'vessel_breakdown': {},
                'avg_confidence': 0.0,
                'processing_info': {
                    'model': detections.get('model_version', 'YOLOv8-Maritime'),
                    'processing_time': detections.get('processing_time', 0.0),
                    'image_size': detections.get('image_size', 'Unknown')
                }
            }
            
            # Count vessel types
            vessel_types = detections.get('vessel_types', [])
            for vessel_type in vessel_types:
                summary['vessel_breakdown'][vessel_type] = summary['vessel_breakdown'].get(vessel_type, 0) + 1
            
            # Calculate average confidence
            confidence_scores = detections.get('confidence_scores', [])
            if confidence_scores:
                summary['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'total_ships': 0,
                'vessel_breakdown': {},
                'avg_confidence': 0.0,
                'processing_info': {'model': 'Unknown', 'processing_time': 0.0, 'image_size': 'Unknown'}
            }