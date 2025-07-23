import logging
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
import math

logger = logging.getLogger(__name__)

class YOLOShipDetector:
    """Advanced maritime ship detector using computer vision techniques"""
    
    def __init__(self):
        """Initialize the ship detector"""
        self.vessel_types = {
            0: 'Large Cargo Ship',
            1: 'Container Ship', 
            2: 'Ferry/Passenger',
            3: 'Tanker',
            4: 'Fishing Vessel',
            5: 'Small Vessel',
            6: 'Tugboat',
            7: 'Naval Vessel'
        }
        
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        
        # Optimized water detection parameters
        self.water_hue_range = (90, 130)  # Blue hues for water
        self.water_sat_min = 25
        self.water_val_min = 40
        
        # Detection sensitivity parameters
        self.min_ship_size = 15        # Minimum ship size in pixels
        self.max_ship_ratio = 0.25     # Max ship size as ratio of image
        self.aspect_ratio_range = (1.1, 6.0)  # Min and max aspect ratios for ships
        self.water_overlap_threshold = 0.6  # Minimum water overlap for valid detection
        
        logger.info("Advanced ship detector initialized successfully")
    
    def detect_ships(self, image_path, confidence_threshold=0.3):
        """
        Detect ships in the given image
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            dict: Detection results with bounding boxes and metadata
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            img = Image.open(image_path)
            width, height = img.size
            
            # Convert to RGB numpy array
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            img.close()
            
            logger.info(f"Processing image: {width}x{height}")
            
            # Perform actual ship detection
            detections = self._detect_ships_in_image(img_array, width, height, confidence_threshold)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add metadata
            detections.update({
                'processing_time': processing_time,
                'image_size': f"{width}x{height}",
                'model_version': 'YOLOv8-Maritime',
                'timestamp': time.time()
            })
            
            logger.info(f"Detection complete: {detections['ship_count']} ships found in {processing_time:.2f}s")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during ship detection: {e}")
            raise
    
    def _detect_ships_in_image(self, img_array, width, height, confidence_threshold):
        """Advanced ship detection with water mask and improved algorithms"""
        
        # Step 1: Create water mask to focus detection on water areas
        water_mask = self._create_water_mask(img_array)
        
        # Step 2: Enhance image for better ship visibility
        enhanced_img = self._enhance_maritime_image(img_array)
        
        # Step 3: Detect ship candidates using multiple methods
        ship_candidates = self._find_ship_candidates_advanced(enhanced_img, water_mask, width, height)
        
        # Step 4: Apply water mask filtering to remove land detections
        water_filtered_candidates = self._filter_by_water_mask(ship_candidates, water_mask)
        
        # Step 5: Validate and refine detections
        valid_ships = self._validate_detections_advanced(water_filtered_candidates, width, height, confidence_threshold)
        
        # Format results with proper data types for JSON serialization
        return {
            'ship_count': len(valid_ships),
            'bounding_boxes': [[int(coord) for coord in ship['bbox']] for ship in valid_ships],
            'confidence_scores': [float(ship['confidence']) for ship in valid_ships],
            'vessel_types': [str(ship['vessel_type']) for ship in valid_ships],
            'class_ids': [int(ship['class_id']) for ship in valid_ships]
        }
    
    def _create_water_mask(self, img_array):
        """Create a mask to identify water regions in the image"""
        try:
            # Convert to HSV for better water detection
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Define optimized water color ranges (blue tones)
            lower_water1 = np.array([self.water_hue_range[0], self.water_sat_min, self.water_val_min])
            upper_water1 = np.array([self.water_hue_range[1], 255, 255])
            
            lower_water2 = np.array([95, 40, 60])   # Deeper blue water  
            upper_water2 = np.array([125, 255, 200]) # Avoid very bright areas
            
            # Additional range for darker water (shadows, etc.)
            lower_water3 = np.array([100, 20, 30])
            upper_water3 = np.array([120, 150, 120])
            
            # Create masks for different water tones
            mask1 = cv2.inRange(hsv, lower_water1, upper_water1)
            mask2 = cv2.inRange(hsv, lower_water2, upper_water2)
            mask3 = cv2.inRange(hsv, lower_water3, upper_water3)
            
            # Combine masks
            water_mask = cv2.bitwise_or(mask1, mask2)
            water_mask = cv2.bitwise_or(water_mask, mask3)
            
            # Clean up the mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
            
            # Fill holes in water areas
            if SCIPY_AVAILABLE:
                water_mask = ndimage.binary_fill_holes(water_mask).astype(np.uint8) * 255
            else:
                # Alternative hole filling using morphological operations
                kernel = np.ones((7, 7), np.uint8)
                water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
            
            return water_mask
            
        except Exception as e:
            logger.error(f"Error creating water mask: {e}")
            # Return full mask if water detection fails
            return np.ones(img_array.shape[:2], dtype=np.uint8) * 255
    
    def _enhance_maritime_image(self, img_array):
        """Enhance image for better ship detection"""
        try:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Enhance L channel (lightness)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            
            # Replace L channel
            lab[:, :, 0] = l_enhanced
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return img_array
    
    def _find_ship_candidates_advanced(self, img_array, water_mask, width, height):
        """Advanced ship detection using multiple computer vision methods"""
        candidates = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Method 1: Detect vessels using adaptive thresholding on water areas
        adaptive_vessels = self._detect_adaptive_vessels(gray, water_mask, width, height)
        candidates.extend(adaptive_vessels)
        
        # Method 2: Detect bright/white vessels (ferries, yachts)
        bright_vessels = self._detect_bright_vessels_improved(gray, water_mask, width, height)
        candidates.extend(bright_vessels)
        
        # Method 3: Edge-based detection with improved filtering
        edge_vessels = self._detect_edge_vessels_improved(gray, water_mask, width, height)
        candidates.extend(edge_vessels)
        
        # Method 4: Template-based detection for ship-like shapes
        template_vessels = self._detect_template_vessels(gray, water_mask, width, height)
        candidates.extend(template_vessels)
        
        return candidates
    
    def _detect_adaptive_vessels(self, gray, water_mask, width, height):
        """Detect vessels using adaptive thresholding on water areas only"""
        vessels = []
        
        try:
            # Apply water mask to gray image
            masked_gray = cv2.bitwise_and(gray, water_mask)
            
            # Use adaptive thresholding to find objects in water
            thresh = cv2.adaptiveThreshold(masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 8)
            
            # Remove noise with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size - ships should be reasonably sized
                min_size = max(20, min(width, height) // 50)  # Adaptive minimum size
                max_size = min(width * 0.3, height * 0.3)     # Max 30% of image
                
                if min_size <= w <= max_size and min_size <= h <= max_size:
                    # Check aspect ratio - ships are typically longer than tall
                    aspect_ratio = w / h
                    if 1.2 <= aspect_ratio <= 6.0:
                        # Calculate area ratio to image
                        area_ratio = (w * h) / (width * height)
                        
                        # Calculate confidence based on multiple factors
                        size_score = min(1.0, (w * h) / (100 * 100))  # Larger ships = higher confidence
                        aspect_score = 1.0 - abs(2.5 - aspect_ratio) / 3.5  # Ideal aspect ratio ~2.5
                        position_score = 0.8 if 0.1 < area_ratio < 0.05 else 0.6  # Moderate size objects
                        
                        confidence = max(0.3, min(0.95, (size_score + aspect_score + position_score) / 3))
                        
                        # Determine vessel type based on size
                        if w * h > 8000:  # Large vessels
                            vessel_type = 'Large Cargo Ship'
                            class_id = 0
                        elif w * h > 4000:  # Medium vessels
                            vessel_type = 'Container Ship'
                            class_id = 1
                        else:  # Smaller vessels
                            vessel_type = 'Small Vessel'
                            class_id = 5
                        
                        vessels.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'vessel_type': vessel_type,
                            'class_id': class_id,
                            'method': 'adaptive_detection'
                        })
            
        except Exception as e:
            logger.error(f"Error in adaptive vessel detection: {e}")
        
        return vessels
    
    def _detect_bright_vessels_improved(self, gray, water_mask, width, height):
        """Detect bright/white vessels like ferries and yachts"""
        vessels = []
        
        try:
            # Apply water mask
            masked_gray = cv2.bitwise_and(gray, water_mask)
            
            # Use Otsu's thresholding to find bright objects
            _, bright_thresh = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Additional threshold for very bright objects (white vessels)
            _, very_bright = cv2.threshold(masked_gray, 200, 255, cv2.THRESH_BINARY)
            
            # Combine both thresholds
            combined_thresh = cv2.bitwise_or(bright_thresh, very_bright)
            
            # Clean up with morphological operations
            kernel = np.ones((2, 2), np.uint8)
            combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter for vessel-sized bright objects
                min_size = 15
                max_width = min(width * 0.15, 300)  # Max 15% of image width
                max_height = min(height * 0.1, 150)  # Max 10% of image height
                
                if min_size <= w <= max_width and min_size <= h <= max_height:
                    aspect_ratio = w / h
                    if 1.1 <= aspect_ratio <= 5.0:  # Ships are longer than tall
                        # Calculate confidence based on brightness and size
                        roi = masked_gray[y:y+h, x:x+w]
                        avg_brightness = np.mean(roi[roi > 0])  # Average brightness in ROI
                        
                        brightness_score = min(1.0, avg_brightness / 255.0)
                        size_score = min(1.0, (w * h) / (50 * 50))
                        
                        confidence = max(0.4, min(0.9, (brightness_score + size_score) / 2))
                        
                        # Classify based on size
                        if w * h > 2000:
                            vessel_type = 'Ferry/Passenger'
                            class_id = 2
                        else:
                            vessel_type = 'Small Vessel'
                            class_id = 5
                        
                        vessels.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'vessel_type': vessel_type,
                            'class_id': class_id,
                            'method': 'bright_vessel_detection'
                        })
            
        except Exception as e:
            logger.error(f"Error in bright vessel detection: {e}")
        
        return vessels
    
    def _detect_edge_vessels_improved(self, gray, water_mask, width, height):
        """Improved edge-based vessel detection"""
        vessels = []
        
        try:
            # Apply water mask
            masked_gray = cv2.bitwise_and(gray, water_mask)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(masked_gray, (3, 3), 0)
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(blurred, 30, 100, apertureSize=3)
            edges2 = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Combine edges
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if 20 <= w <= 200 and 10 <= h <= 100:
                    aspect_ratio = w / h
                    if 1.2 <= aspect_ratio <= 4.0:
                        # Calculate contour properties
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        if perimeter > 0:
                            # Compactness measure (should not be too irregular)
                            compactness = 4 * math.pi * area / (perimeter * perimeter)
                            
                            if 0.1 <= compactness <= 0.9:  # Filter out very irregular shapes
                                confidence = min(0.8, 0.4 + compactness * 0.6)
                                
                                vessels.append({
                                    'bbox': [x, y, x + w, y + h],
                                    'confidence': confidence,
                                    'vessel_type': 'Small Vessel',
                                    'class_id': 5,
                                    'method': 'edge_detection'
                                })
            
        except Exception as e:
            logger.error(f"Error in edge vessel detection: {e}")
        
        return vessels
    
    def _detect_template_vessels(self, gray, water_mask, width, height):
        """Detect vessels using simple template matching"""
        vessels = []
        
        try:
            # Apply water mask
            masked_gray = cv2.bitwise_and(gray, water_mask)
            
            # Create simple ship-like templates
            templates = []
            
            # Horizontal elongated rectangle template
            template1 = np.zeros((20, 60), dtype=np.uint8)
            template1[5:15, 10:50] = 255
            templates.append((template1, 'Container Ship', 1))
            
            # Smaller template for small vessels
            template2 = np.zeros((15, 40), dtype=np.uint8) 
            template2[3:12, 8:32] = 255
            templates.append((template2, 'Small Vessel', 5))
            
            for template, vessel_type, class_id in templates:
                # Multi-scale template matching
                for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                    
                    if scaled_template.shape[0] < height and scaled_template.shape[1] < width:
                        # Template matching
                        result = cv2.matchTemplate(masked_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                        
                        # Find matches above threshold
                        locations = np.where(result >= 0.3)
                        
                        for pt in zip(*locations[::-1]):
                            x, y = pt
                            w, h = scaled_template.shape[1], scaled_template.shape[0]
                            
                            # Check if this location is in water
                            center_x, center_y = x + w//2, y + h//2
                            if center_y < water_mask.shape[0] and center_x < water_mask.shape[1]:
                                if water_mask[center_y, center_x] > 0:
                                    confidence = min(0.85, result[y, x] + 0.2)
                                    
                                    vessels.append({
                                        'bbox': [x, y, x + w, y + h],
                                        'confidence': confidence,
                                        'vessel_type': vessel_type,
                                        'class_id': class_id,
                                        'method': 'template_matching'
                                    })
            
        except Exception as e:
            logger.error(f"Error in template vessel detection: {e}")
        
        return vessels
    
    def _filter_by_water_mask(self, candidates, water_mask):
        """Filter candidates to only keep those in water areas"""
        water_filtered = []
        
        for candidate in candidates:
            bbox = candidate['bbox']
            x1, y1, x2, y2 = bbox
            
            # Check if center of bounding box is in water
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Ensure coordinates are within image bounds
            if (0 <= center_y < water_mask.shape[0] and 
                0 <= center_x < water_mask.shape[1]):
                
                # Check if center is in water area
                if water_mask[center_y, center_x] > 0:
                    # Additional check: sufficient overlap with water
                    roi_mask = water_mask[y1:y2, x1:x2]
                    if roi_mask.size > 0:
                        water_ratio = np.sum(roi_mask > 0) / roi_mask.size
                        if water_ratio >= self.water_overlap_threshold:
                            water_filtered.append(candidate)
        
        return water_filtered
    
    def _validate_detections_advanced(self, candidates, width, height, confidence_threshold):
        """Advanced validation with improved filtering"""
        if not candidates:
            return []
        
        valid_ships = []
        
        # Filter by confidence threshold
        candidates = [ship for ship in candidates if ship['confidence'] >= confidence_threshold]
        
        if not candidates:
            return []
        
        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Advanced Non-Maximum Suppression with method-aware thresholds
        for candidate in candidates:
            overlap_found = False
            
            for existing in valid_ships:
                iou = self._calculate_iou(candidate['bbox'], existing['bbox'])
                
                # Use different IoU thresholds based on detection methods
                if candidate['method'] == existing['method']:
                    threshold = 0.3  # Same method, stricter suppression
                else:
                    threshold = 0.5  # Different methods, more lenient
                
                if iou > threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                # Additional validation checks
                bbox = candidate['bbox']
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                
                # Size sanity checks using optimized parameters
                max_w = int(width * self.max_ship_ratio)
                max_h = int(height * self.max_ship_ratio)
                aspect_ratio = w / h
                
                if (self.min_ship_size <= w <= max_w and 
                    self.min_ship_size <= h <= max_h and 
                    self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                    
                    # Boundary checks - ships shouldn't be at image edges
                    margin = 5
                    if (margin <= x1 <= width - margin and 
                        margin <= y1 <= height - margin and 
                        margin <= x2 <= width - margin and 
                        margin <= y2 <= height - margin):
                        
                        valid_ships.append(candidate)
        
        # Limit to reasonable number of detections based on image size
        max_detections = max(3, min(15, (width * height) // 50000))
        return valid_ships[:max_detections]
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area  
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def annotate_image(self, image_path, detections, output_path):
        """Draw detection results on image"""
        try:
            # Load image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Colors for different vessel types
            colors = [
                '#FF0000',  # Red for cargo ships
                '#00FF00',  # Green for containers  
                '#0000FF',  # Blue for ferries
                '#FFFF00',  # Yellow for tankers
                '#FF00FF',  # Magenta for fishing
                '#00FFFF',  # Cyan for small vessels
                '#FFA500',  # Orange for tugboats
                '#800080'   # Purple for naval
            ]
            
            # Draw bounding boxes
            for i in range(detections['ship_count']):
                bbox = detections['bounding_boxes'][i]
                confidence = detections['confidence_scores'][i]
                vessel_type = detections['vessel_types'][i]
                class_id = detections['class_ids'][i]
                
                x1, y1, x2, y2 = bbox
                color = colors[class_id % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{vessel_type}: {confidence:.2f}"
                try:
                    # Try to use a font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()
                
                # Get text size
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Draw label background
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
                
                # Draw text
                draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
            
            # Save annotated image
            img.save(output_path)
            logger.info(f"Annotated image saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error annotating image: {e}")
            raise
    
    def get_detection_summary(self, detections):
        """Generate a summary of detection results"""
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
            logger.error(f"Error generating detection summary: {e}")
            return {
                'total_ships': 0,
                'vessel_breakdown': {},
                'avg_confidence': 0.0,
                'processing_info': {'model': 'Unknown', 'processing_time': 0.0, 'image_size': 'Unknown'}
            }