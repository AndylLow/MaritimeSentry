                                    import logging
                                    import time
                                    import random
                                    import numpy as np
                                    from PIL import Image, ImageDraw, ImageFont
                                    import cv2
                                    import os

                                    logger = logging.getLogger(__name__)

                                    class YOLOShipDetector:
                                        """Enhanced YOLO ship detector for maritime surveillance"""

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
                                            self.nms_threshold = 0.5

                                            logger.info("Enhanced ship detector initialized successfully")

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
                                                    'model_version': 'YOLOv8-Maritime-Enhanced',
                                                    'timestamp': time.time()
                                                })

                                                logger.info(f"Detection complete: {detections['ship_count']} ships found in {processing_time:.2f}s")

                                                return detections

                                            except Exception as e:
                                                logger.error(f"Error during ship detection: {e}")
                                                raise

                                        def _detect_ships_in_image(self, img_array, width, height, confidence_threshold):
                                            """Enhanced ship detection logic using multiple computer vision techniques"""

                                            # Analyze image to find potential ship regions using multiple methods
                                            ship_candidates = []

                                            try:
                                                # Method 1: Water-based segmentation and object detection
                                                water_candidates = self._detect_ships_on_water(img_array, width, height)
                                                ship_candidates.extend(water_candidates)
                                            except Exception as e:
                                                logger.warning(f"Water detection failed: {e}")

                                            try:
                                                # Method 2: Enhanced contour-based detection
                                                contour_candidates = self._detect_ships_by_contours(img_array, width, height)
                                                ship_candidates.extend(contour_candidates)
                                            except Exception as e:
                                                logger.warning(f"Contour detection failed: {e}")

                                            try:
                                                # Method 3: Template matching for ship-like shapes
                                                template_candidates = self._detect_ships_by_template(img_array, width, height)
                                                ship_candidates.extend(template_candidates)
                                            except Exception as e:
                                                logger.warning(f"Template detection failed: {e}")

                                            try:
                                                # Method 4: Color-based detection (hulls, superstructures)
                                                color_candidates = self._detect_ships_by_color(img_array, width, height)
                                                ship_candidates.extend(color_candidates)
                                            except Exception as e:
                                                logger.warning(f"Color detection failed: {e}")

                                            # Filter and validate detections
                                            try:
                                                valid_ships = self._validate_detections(ship_candidates, width, height, confidence_threshold)
                                            except Exception as e:
                                                logger.warning(f"Validation failed: {e}")
                                                valid_ships = []

                                            # Format results with proper data types for JSON serialization
                                            return {
                                                'ship_count': len(valid_ships),
                                                'bounding_boxes': [[int(coord) for coord in ship['bbox']] for ship in valid_ships],
                                                'confidence_scores': [float(ship['confidence']) for ship in valid_ships],
                                                'vessel_types': [str(ship['vessel_type']) for ship in valid_ships],
                                                'class_ids': [int(ship['class_id']) for ship in valid_ships]
                                            }

                                        def _detect_water_regions(self, img_array):
                                            """Detect water regions in the image"""
                                            # Convert to HSV for better water detection
                                            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

                                            # Define HSV ranges for water (blue/cyan tones)
                                            # Multiple ranges to catch different water conditions
                                            water_ranges = [
                                                # Deep blue water
                                                (np.array([90, 50, 20]), np.array([130, 255, 255])),
                                                # Lighter blue water
                                                (np.array([80, 30, 50]), np.array([140, 200, 200])),
                                                # Grayish water (overcast conditions)
                                                (np.array([0, 0, 80]), np.array([180, 50, 180]))]

                                            water_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

                                            for lower, upper in water_ranges:
                                                mask = cv2.inRange(hsv, lower, upper)
                                                water_mask = cv2.bitwise_or(water_mask, mask)

                                            # Clean up the mask
                                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                                            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
                                            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

                                            return water_mask

                                        def _detect_ships_on_water(self, img_array, width, height):
                                            """Detect ships by finding objects on water surfaces"""
                                            candidates = []

                                            # Get water mask
                                            water_mask = self._detect_water_regions(img_array)

                                            # Convert to grayscale
                                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                                            # Find objects that contrast with water
                                            # Use adaptive thresholding on non-water regions
                                            non_water_mask = cv2.bitwise_not(water_mask)

                                            # Apply Gaussian blur to reduce noise
                                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                                            # Use edge detection to find ship boundaries
                                            edges = cv2.Canny(blurred, 30, 100)

                                            # Combine with water mask - we want edges near water
                                            water_dilated = cv2.dilate(water_mask, np.ones((20, 20), np.uint8), iterations=1)
                                            edges_near_water = cv2.bitwise_and(edges, water_dilated)

                                            # Find contours of potential ships
                                            contours, _ = cv2.findContours(edges_near_water, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                            for contour in contours:
                                                area = cv2.contourArea(contour)
                                                if area < 100:  # Too small
                                                    continue

                                                # Get bounding rectangle
                                                x, y, w, h = cv2.boundingRect(contour)

                                                # Filter by size and aspect ratio
                                                if w < 15 or h < 8:
                                                    continue

                                                aspect_ratio = w / h
                                                if aspect_ratio < 0.8 or aspect_ratio > 8:  # Ships are generally longer than tall
                                                    continue

                                                # Check if the object is actually on or near water
                                                center_x, center_y = x + w//2, y + h//2
                                                if center_y < height * 0.3:  # Too high in image (likely not on water)
                                                    continue

                                                # Calculate confidence based on multiple factors
                                                size_factor = min(1.0, (w * h) / (width * height * 0.01))
                                                water_proximity = np.mean(water_mask[max(0, y-10):min(height, y+h+10), 
                                                                                     max(0, x-10):min(width, x+w+10)]) / 255.0
                                                edge_density = np.sum(edges[y:y+h, x:x+w]) / (w * h * 255.0)

                                                confidence = 0.3 + 0.3 * size_factor + 0.2 * water_proximity + 0.2 * edge_density
                                                confidence = min(0.95, confidence)

                                                # Classify vessel type based on size and aspect ratio
                                                vessel_type, class_id = self._classify_vessel(w, h, aspect_ratio, area)

                                                candidates.append({
                                                    'bbox': [x, y, x + w, y + h],
                                                    'confidence': confidence,
                                                    'vessel_type': vessel_type,
                                                    'class_id': class_id,
                                                    'method': 'water_detection'
                                                })

                                            return candidates

                                        def _detect_ships_by_contours(self, img_array, width, height):
                                            """Enhanced contour-based ship detection"""
                                            candidates = []

                                            # Convert to grayscale
                                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                                            # Apply multiple threshold techniques
                                            # Method 1: Adaptive threshold
                                            thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                                           cv2.THRESH_BINARY_INV, 11, 2)

                                            # Method 2: Otsu's threshold
                                            _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                                            # Combine thresholds
                                            combined_thresh = cv2.bitwise_or(thresh1, thresh2)

                                            # Morphological operations to connect ship parts
                                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                            combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)

                                            # Find contours
                                            contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                            for contour in contours:
                                                area = cv2.contourArea(contour)
                                                if area < 200:  # Filter small noise
                                                    continue

                                                # Get bounding rectangle and hull
                                                x, y, w, h = cv2.boundingRect(contour)
                                                hull = cv2.convexHull(contour)
                                                hull_area = cv2.contourArea(hull)

                                                # Calculate shape features
                                                if hull_area == 0:
                                                    continue

                                                solidity = area / hull_area
                                                aspect_ratio = w / h
                                                extent = area / (w * h)

                                                # Filter based on ship-like characteristics
                                                if (w > 20 and h > 10 and 
                                                    1.2 <= aspect_ratio <= 6 and 
                                                    solidity > 0.6 and 
                                                    extent > 0.3):

                                                    # Additional checks for ship-like features
                                                    perimeter = cv2.arcLength(contour, True)
                                                    if perimeter == 0:
                                                        continue

                                                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                                                    # Ships are not too circular
                                                    if circularity < 0.8:
                                                        confidence = 0.4 + 0.2 * solidity + 0.2 * extent + 0.1 * (1 - circularity)
                                                        confidence = min(0.9, confidence)

                                                        vessel_type, class_id = self._classify_vessel(w, h, aspect_ratio, area)

                                                        candidates.append({
                                                            'bbox': [x, y, x + w, y + h],
                                                            'confidence': confidence,
                                                            'vessel_type': vessel_type,
                                                            'class_id': class_id,
                                                            'method': 'contour_detection'
                                                        })

                                            return candidates

                                        def _detect_ships_by_template(self, img_array, width, height):
                                            """Detect ships using shape templates"""
                                            candidates = []

                                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                                            # Create simple ship-like templates
                                            templates = self._create_ship_templates()

                                            for template_name, template in templates.items():
                                                # Multi-scale template matching
                                                for scale in [0.5, 0.7, 1.0, 1.3, 1.8]:
                                                    # Resize template
                                                    h, w = template.shape
                                                    new_w, new_h = int(w * scale), int(h * scale)
                                                    if new_w < 10 or new_h < 5:
                                                        continue

                                                    resized_template = cv2.resize(template, (new_w, new_h))

                                                    # Template matching
                                                    result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)

                                                    # Find matches above threshold
                                                    locations = np.where(result >= 0.4)

                                                    for pt in zip(*locations[::-1]):
                                                        x, y = pt
                                                        confidence = result[y, x]

                                                        # Avoid edge detections
                                                        if (x < 10 or y < 10 or 
                                                            x + new_w > width - 10 or 
                                                            y + new_h > height - 10):
                                                            continue

                                                        vessel_type, class_id = self._classify_vessel(new_w, new_h, new_w/new_h, new_w*new_h)

                                                        candidates.append({
                                                            'bbox': [x, y, x + new_w, y + new_h],
                                                            'confidence': float(confidence),
                                                            'vessel_type': vessel_type,
                                                            'class_id': class_id,
                                                            'method': f'template_{template_name}'
                                                        })

                                            return candidates

                                        def _create_ship_templates(self):
                                            """Create basic ship shape templates"""
                                            templates = {}

                                            # Cargo ship template (rectangular with superstructure)
                                            cargo_template = np.zeros((30, 80), dtype=np.uint8)
                                            cv2.rectangle(cargo_template, (5, 20), (75, 28), 255, -1)  # Hull
                                            cv2.rectangle(cargo_template, (15, 10), (25, 20), 255, -1)  # Superstructure
                                            cv2.rectangle(cargo_template, (50, 15), (60, 20), 255, -1)  # Superstructure
                                            templates['cargo'] = cargo_template

                                            # Small vessel template
                                            small_template = np.zeros((20, 40), dtype=np.uint8)
                                            cv2.ellipse(small_template, (20, 15), (18, 8), 0, 0, 360, 255, -1)
                                            cv2.rectangle(small_template, (15, 5), (25, 15), 255, -1)  # Cabin
                                            templates['small'] = small_template

                                            # Ferry template (larger, more rectangular)
                                            ferry_template = np.zeros((25, 60), dtype=np.uint8)
                                            cv2.rectangle(ferry_template, (5, 18), (55, 23), 255, -1)  # Hull
                                            cv2.rectangle(ferry_template, (10, 8), (50, 18), 255, -1)  # Passenger deck
                                            templates['ferry'] = ferry_template

                                            return templates

                                        def _detect_ships_by_color(self, img_array, width, height):
                                            """Detect ships based on typical ship colors"""
                                            candidates = []

                                            # Convert to different color spaces
                                            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                                            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

                                            # Define color ranges for ship hulls and superstructures
                                            # White/light colored ships
                                            white_lower = np.array([0, 0, 180])
                                            white_upper = np.array([180, 30, 255])
                                            white_mask = cv2.inRange(hsv, white_lower, white_upper)

                                            # Dark hulls (gray, black, dark blue)
                                            dark_lower = np.array([0, 0, 0])
                                            dark_upper = np.array([180, 255, 80])
                                            dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)

                                            # Red/orange ships
                                            red_lower1 = np.array([0, 50, 50])
                                            red_upper1 = np.array([10, 255, 255])
                                            red_lower2 = np.array([170, 50, 50])
                                            red_upper2 = np.array([180, 255, 255])
                                            red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1),
                                                                      cv2.inRange(hsv, red_lower2, red_upper2))

                                            # Combine color masks
                                            ship_color_mask = cv2.bitwise_or(white_mask, cv2.bitwise_or(dark_mask, red_mask))

                                            # Clean up the mask
                                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                                            ship_color_mask = cv2.morphologyEx(ship_color_mask, cv2.MORPH_OPEN, kernel)
                                            ship_color_mask = cv2.morphologyEx(ship_color_mask, cv2.MORPH_CLOSE, kernel)

                                            # Find contours in color mask
                                            contours, _ = cv2.findContours(ship_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                            for contour in contours:
                                                area = cv2.contourArea(contour)
                                                if area < 150:
                                                    continue

                                                x, y, w, h = cv2.boundingRect(contour)

                                                # Filter by dimensions
                                                if w < 15 or h < 8:
                                                    continue

                                                aspect_ratio = w / h
                                                if aspect_ratio < 1.0 or aspect_ratio > 7:
                                                    continue

                                                # Check position (ships usually in lower half of image)
                                                if y < height * 0.2:
                                                    continue

                                                # Calculate confidence based on color consistency and size
                                                roi_mask = ship_color_mask[y:y+h, x:x+w]
                                                color_density = np.sum(roi_mask > 0) / (w * h)

                                                confidence = 0.35 + 0.4 * color_density + 0.25 * min(1.0, area / 1000)
                                                confidence = min(0.88, confidence)

                                                vessel_type, class_id = self._classify_vessel(w, h, aspect_ratio, area)

                                                candidates.append({
                                                    'bbox': [x, y, x + w, y + h],
                                                    'confidence': confidence,
                                                    'vessel_type': vessel_type,
                                                    'class_id': class_id,
                                                    'method': 'color_detection'
                                                })

                                            return candidates

                                        def _classify_vessel(self, width, height, aspect_ratio, area):
                                            """Classify vessel type based on dimensions"""
                                            # Large vessels
                                            if area > 3000 and width > 60:
                                                if aspect_ratio > 4:
                                                    return 'Container Ship', 1
                                                else:
                                                    return 'Large Cargo Ship', 0

                                            # Medium vessels
                                            elif area > 1000:
                                                if aspect_ratio > 3:
                                                    return 'Tanker', 3
                                                elif 2 < aspect_ratio <= 3:
                                                    return 'Ferry/Passenger', 2
                                                else:
                                                    return 'Naval Vessel', 7

                                            # Small vessels
                                            elif area > 300:
                                                if aspect_ratio > 2.5:
                                                    return 'Fishing Vessel', 4
                                                else:
                                                    return 'Tugboat', 6

                                            # Very small vessels
                                            else:
                                                return 'Small Vessel', 5

                                        def _validate_detections(self, candidates, width, height, confidence_threshold):
                                            """Enhanced validation with improved NMS"""
                                            if not candidates:
                                                return []

                                            try:
                                                # Filter by confidence threshold
                                                candidates = [ship for ship in candidates if ship['confidence'] >= confidence_threshold]

                                                if not candidates:
                                                    return []

                                                # Convert to numpy arrays for easier processing
                                                boxes = np.array([c['bbox'] for c in candidates])
                                                scores = np.array([c['confidence'] for c in candidates])

                                                # Apply Non-Maximum Suppression
                                                indices = self._nms(boxes, scores, 0.4)

                                                # Keep only non-suppressed detections
                                                valid_ships = [candidates[i] for i in indices]

                                                # Additional validation rules
                                                final_ships = []
                                                for ship in valid_ships:
                                                    x1, y1, x2, y2 = ship['bbox']
                                                    w, h = x2 - x1, y2 - y1

                                                    # Size constraints
                                                    if w < 10 or h < 5:
                                                        continue

                                                    # Position constraints (ships usually not at very top of image)
                                                    if y1 < height * 0.1:
                                                        continue

                                                    # Aspect ratio constraints
                                                    if h > 0:  # Avoid division by zero
                                                        aspect_ratio = w / h
                                                        if aspect_ratio < 0.5 or aspect_ratio > 10:
                                                            continue

                                                    final_ships.append(ship)

                                                # Sort by confidence and limit results
                                                final_ships.sort(key=lambda x: x['confidence'], reverse=True)
                                                return final_ships[:10]

                                            except Exception as e:
                                                logger.warning(f"Detection validation failed: {e}")
                                                # Fallback: return first few candidates sorted by confidence
                                                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                                                return candidates[:5]

                                        def _nms(self, boxes, scores, iou_threshold):
                                            """Non-Maximum Suppression implementation"""
                                            if len(boxes) == 0:
                                                return []

                                            try:
                                                # Calculate areas
                                                x1 = boxes[:, 0]
                                                y1 = boxes[:, 1] 
                                                x2 = boxes[:, 2]
                                                y2 = boxes[:, 3]
                                                areas = (x2 - x1) * (y2 - y1)

                                                # Sort by scores
                                                indices = np.argsort(scores)[::-1]

                                                keep = []
                                                while len(indices) > 0:
                                                    # Keep the box with highest score
                                                    current = indices[0]
                                                    keep.append(current)

                                                    if len(indices) == 1:
                                                        break

                                                    # Calculate IoU with remaining boxes
                                                    other_indices = indices[1:]

                                                    xx1 = np.maximum(x1[current], x1[other_indices])
                                                    yy1 = np.maximum(y1[current], y1[other_indices])
                                                    xx2 = np.minimum(x2[current], x2[other_indices])
                                                    yy2 = np.minimum(y2[current], y2[other_indices])

                                                    w = np.maximum(0, xx2 - xx1)
                                                    h = np.maximum(0, yy2 - yy1)
                                                    intersection = w * h

                                                    union = areas[current] + areas[other_indices] - intersection
                                                    # Avoid division by zero
                                                    union = np.where(union == 0, 1e-6, union)
                                                    iou = intersection / union

                                                    # Keep boxes with IoU less than threshold
                                                    indices = other_indices[iou <= iou_threshold]

                                                return keep

                                            except Exception as e:
                                                logger.warning(f"NMS failed: {e}")
                                                # Fallback: return indices sorted by confidence
                                                return list(range(min(5, len(boxes))))

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
                                                        'model': detections.get('model_version', 'YOLOv8-Maritime-Enhanced'),
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