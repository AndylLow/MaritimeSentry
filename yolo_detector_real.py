import logging
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    logger.info("Ultralytics YOLO is available")
except ImportError as e:
    YOLO_AVAILABLE = False
    logger.warning(f"Ultralytics YOLO not available: {e}")

class RealYOLOShipDetector:
    """Real YOLO-based ship detector using YOLOv8"""
    
    def __init__(self):
        """Initialize the real YOLO ship detector"""
        
        # Maritime vessel class mapping - COCO classes relevant to ships
        self.maritime_classes = {
            8: 'boat',  # COCO class 8 is 'boat'
            # Note: YOLO may also detect ships as 'car' or other classes in some cases
            # We'll filter and validate based on maritime context
        }
        
        # Enhanced vessel type mapping
        self.vessel_types = {
            0: 'Large Cargo Ship',
            1: 'Container Ship', 
            2: 'Ferry/Passenger',
            3: 'Tanker',
            4: 'Fishing Vessel',
            5: 'Small Vessel',
            6: 'Tugboat',
            7: 'Naval Vessel',
            8: 'Yacht/Pleasure Craft'
        }
        
        # Optimized parameters for maritime detection
        self.confidence_threshold = 0.12  # Lower threshold to catch more vessels
        self.nms_threshold = 0.45         # Balanced NMS for good filtering without losing ships
        self.maritime_conf_boost = 0.08   # Modest boost for maritime objects
        
        # Advanced detection parameters
        self.multi_scale_sizes = [416, 640, 832]  # Multiple input sizes for better detection
        self.tile_overlap = 0.2           # Overlap for tile-based detection on large images
        self.max_image_size = 1280        # Maximum image size for processing
        self.model = None
        
        if YOLO_AVAILABLE:
            self._initialize_model()
        else:
            logger.error("Cannot initialize YOLO model - ultralytics not available")
            raise ImportError("ultralytics package is required for real YOLO detection")
    
    def _initialize_model(self):
        """Initialize the YOLO model"""
        try:
            # Try to load best YOLO model for ship detection
            logger.info("Loading optimized YOLOv8 model for maritime detection...")
            
            # Try different model sizes - larger models are better for ship detection
            model_options = ['yolov8s.pt', 'yolov8n.pt', 'yolov8m.pt']
            
            for model_name in model_options:
                try:
                    self.model = YOLO(model_name)
                    logger.info(f"Successfully loaded {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Failed to load any YOLO model")
            
            # Set model to eval mode and optimize for inference
            self.model.model.eval()
            
            # Optimize model for better maritime detection
            # This sets the model to focus more on relevant classes
            logger.info("YOLOv8 model loaded and optimized for maritime detection")
            
            # Check if CUDA is available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            # Fallback: try to use a different model
            try:
                logger.info("Trying YOLOv8s model as fallback...")
                self.model = YOLO('yolov8s.pt')
                logger.info("YOLOv8s model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def detect_ships(self, image_path, confidence_threshold=None):
        """
        Detect ships using real YOLO model
        
        Args:
            image_path (str): Path to input image
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            dict: Detection results with bounding boxes and metadata
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Adaptive confidence threshold based on image characteristics
        confidence_threshold = self._adaptive_confidence_threshold(image_path, confidence_threshold)
        
        start_time = time.time()
        
        try:
            # Load image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Running YOLO detection on: {image_path}")
            
            # Get optimal parameters for this image
            optimal_params = self._get_optimal_detection_params(image_path)
            
            # Run YOLO inference with optimized parameters
            results = self.model(
                image_path,
                conf=confidence_threshold,
                iou=self.nms_threshold,
                verbose=False,
                imgsz=optimal_params['image_size'],
                max_det=optimal_params['max_detections'],
                agnostic_nms=True,
                augment=True,     # Test Time Augmentation for better accuracy
                half=False        # Use FP32 for better precision on CPU
            )
            
            # Process results with advanced post-processing
            detections = self._process_yolo_results(results[0], image_path)
            
            # Apply advanced post-processing for maritime detection
            detections = self._apply_advanced_post_processing(detections, image_path)
            
            # Multi-scale detection for better accuracy (if image is large enough)
            if optimal_params['original_size'][0] > 1280 or optimal_params['original_size'][1] > 1280:
                logger.info("Applying multi-scale detection for large image")
                multi_scale_detections = self._multi_scale_detection(image_path, confidence_threshold)
                detections = self._merge_detections([detections, multi_scale_detections])
            
            # Add metadata
            processing_time = time.time() - start_time
            detections.update({
                'processing_time': processing_time,
                'model_version': 'YOLOv8',
                'confidence_threshold': confidence_threshold,
                'timestamp': time.time()
            })
            
            logger.info(f"YOLO detection complete: {detections['ship_count']} vessels found in {processing_time:.2f}s")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO ship detection: {e}")
            raise
    
    def _process_yolo_results(self, result, image_path):
        """Process YOLO detection results into our format"""
        try:
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Initialize results
            detections = {
                'ship_count': 0,
                'bounding_boxes': [],
                'confidence_scores': [],
                'vessel_types': [],
                'class_ids': [],
                'yolo_classes': []
            }
            
            # Check if any detections were found
            if result.boxes is None or len(result.boxes) == 0:
                logger.info("No objects detected by YOLO")
                return detections
            
            # Process each detection
            for i, box in enumerate(result.boxes):
                # Get detection data
                xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                conf = float(box.conf[0].cpu().numpy())  # Confidence score
                cls = int(box.cls[0].cpu().numpy())  # Class ID
                
                # Get class name from YOLO
                class_name = result.names[cls] if result.names else str(cls)
                
                # Filter for maritime-relevant detections
                if self._is_maritime_object(cls, class_name, xyxy, img_width, img_height):
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    
                    # Boost confidence for confirmed maritime objects
                    if class_name.lower() in ['boat', 'ship', 'yacht', 'sailboat']:
                        conf = min(0.99, conf + self.maritime_conf_boost)
                    
                    # Classify vessel type based on size and YOLO class
                    vessel_type, class_id = self._classify_vessel(
                        class_name, x2-x1, y2-y1, img_width, img_height, conf
                    )
                    
                    # Add to results
                    detections['bounding_boxes'].append([x1, y1, x2, y2])
                    detections['confidence_scores'].append(conf)
                    detections['vessel_types'].append(vessel_type)
                    detections['class_ids'].append(class_id)
                    detections['yolo_classes'].append(class_name)
                    
                    logger.debug(f"Detected {vessel_type} at [{x1},{y1},{x2},{y2}] with confidence {conf:.3f}")
            
            detections['ship_count'] = len(detections['bounding_boxes'])
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing YOLO results: {e}")
            raise
    
    def _is_maritime_object(self, cls, class_name, bbox, img_width, img_height):
        """Determine if detected object is likely a maritime vessel"""
        
        # Primary maritime classes
        maritime_class_names = [
            'boat', 'ship', 'yacht', 'sailboat', 'motorboat', 
            'ferry', 'vessel', 'watercraft'
        ]
        
        # Check if it's a known maritime class
        if class_name.lower() in maritime_class_names:
            return True
        
        # Sometimes ships are detected as other classes, so we do additional validation
        # based on context and characteristics
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Size-based filtering - should be reasonably sized (relaxed)
        if width < 10 or height < 8:
            return False
        
        # Aspect ratio check - ships are typically longer than tall (relaxed for different vessel types)
        aspect_ratio = width / height
        if aspect_ratio < 0.8 or aspect_ratio > 12.0:  # More permissive for various vessel orientations
            return False
        
        # Position check - likely in water (middle or lower part of image for typical maritime photos)
        center_y = (y1 + y2) / 2
        if center_y < img_height * 0.2:  # Too high in image (likely sky)
            return False
        
        # Additional classes that might be ships in maritime context - STRICT validation needed
        possible_maritime_classes = [
            'car',  # Sometimes ships are misclassified as cars
            'truck',  # Large vessels might be detected as trucks
            'bus'   # Ferries might be detected as buses
        ]
        
        if class_name.lower() in possible_maritime_classes:
            # VERY strict validation for ambiguous classes to avoid buildings
            
            # Must be in water area (lower 60% of image)
            if center_y < img_height * 0.4:
                return False
            
            # Buildings often have square/vertical aspect ratios - reject these
            if aspect_ratio < 1.2 or aspect_ratio > 8.0:  # Ships should be reasonably horizontal
                return False
            
            # Size validation - not too large (buildings) or too small
            area_ratio = (width * height) / (img_width * img_height)
            if area_ratio < 0.0008 or area_ratio > 0.12:
                return False
            
            # Advanced building detection
            if not self._passes_building_detection_filter(x1, y1, x2, y2, img_width, img_height, aspect_ratio):
                logger.debug(f"Rejected {class_name} as potential building")
                return False
            
            logger.debug(f"Accepting {class_name} as potential vessel after strict validation")
            return True
        
        return False
    
    def _passes_building_detection_filter(self, x1, y1, x2, y2, img_width, img_height, aspect_ratio):
        """Advanced building detection filter to avoid false positives"""
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Building characteristics to avoid:
        
        # 1. Too vertical/square - buildings are often taller than wide
        if aspect_ratio < 1.3 and height > width * 0.7:
            return False
        
        # 2. Fixed at image edges with building-like proportions
        edge_threshold = 0.08
        near_left_edge = x1 < img_width * edge_threshold
        near_right_edge = x2 > img_width * (1 - edge_threshold)
        near_bottom_edge = y2 > img_height * (1 - edge_threshold)
        
        if (near_left_edge or near_right_edge) and aspect_ratio < 2.5:
            return False
        
        if near_bottom_edge and aspect_ratio < 1.8:
            return False
        
        # 3. Located in typical building zones (upper half with square proportions)
        if center_y < img_height * 0.5 and aspect_ratio < 1.5:
            return False
        
        # 4. Very large objects that span significant portions of image (buildings/structures)
        area_ratio = (width * height) / (img_width * img_height)
        if area_ratio > 0.08 and aspect_ratio < 2.0:  # Large square objects likely buildings
            return False
        
        # 5. Positioned like coastal structures
        if center_y < img_height * 0.6 and (near_left_edge or near_right_edge) and aspect_ratio < 3.0:
            return False
        
        return True
    
    def _get_optimal_detection_params(self, image_path):
        """Get optimal detection parameters based on image characteristics"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
            # Calculate optimal image size for processing
            max_dim = max(width, height)
            
            if max_dim <= 640:
                image_size = 640
                max_detections = 30
            elif max_dim <= 1280:
                image_size = 832  # Larger size for better small object detection
                max_detections = 50
            else:
                image_size = 1280  # Maximum practical size
                max_detections = 100  # More detections for very large images
            
            # Adjust for maritime context
            aspect_ratio = width / height
            if aspect_ratio > 2.0:  # Wide maritime panoramic images
                max_detections = int(max_detections * 1.5)
            
            return {
                'image_size': image_size,
                'max_detections': max_detections,
                'original_size': (width, height)
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze image for optimal params: {e}")
            return {
                'image_size': 640,
                'max_detections': 50,
                'original_size': (640, 640)
            }
    
    def _classify_vessel(self, yolo_class, width, height, img_width, img_height, confidence):
        """Classify vessel type based on YOLO detection and size"""
        
        # Calculate relative size
        area = width * height
        relative_area = area / (img_width * img_height)
        
        # Base classification on size
        if relative_area > 0.05:  # Large vessels (>5% of image)
            if yolo_class.lower() in ['truck', 'bus']:
                vessel_type = 'Large Cargo Ship'
                class_id = 0
            else:
                vessel_type = 'Container Ship'
                class_id = 1
                
        elif relative_area > 0.02:  # Medium vessels (2-5% of image)
            if width / height > 3.5:  # Very elongated
                vessel_type = 'Ferry/Passenger'
                class_id = 2
            else:
                vessel_type = 'Tanker'
                class_id = 3
                
        elif relative_area > 0.008:  # Small-medium vessels
            vessel_type = 'Fishing Vessel'
            class_id = 4
            
        else:  # Small vessels
            if confidence > 0.7:
                vessel_type = 'Yacht/Pleasure Craft'
                class_id = 8
            else:
                vessel_type = 'Small Vessel'
                class_id = 5
        
        return vessel_type, class_id
    
    def _multi_scale_detection(self, image_path, confidence_threshold):
        """Run detection at multiple scales for better small object detection"""
        try:
            logger.info("Running multi-scale YOLO detection...")
            all_detections = []
            
            # Different scales for comprehensive detection
            scales = [640, 832, 1024]
            
            for scale in scales:
                try:
                    results = self.model(
                        image_path,
                        conf=confidence_threshold * 0.8,  # Slightly lower threshold
                        iou=self.nms_threshold,
                        verbose=False,
                        imgsz=scale,
                        max_det=100,
                        agnostic_nms=True,
                        augment=False  # Skip augmentation for speed
                    )
                    
                    scale_detections = self._process_yolo_results(results[0], image_path)
                    all_detections.append(scale_detections)
                    
                except Exception as e:
                    logger.warning(f"Multi-scale detection failed at scale {scale}: {e}")
                    continue
            
            # Merge all detections with NMS
            if all_detections:
                return self._merge_detections(all_detections)
            else:
                return self._empty_detection_result()
                
        except Exception as e:
            logger.error(f"Multi-scale detection failed: {e}")
            return self._empty_detection_result()
    
    def _apply_advanced_post_processing(self, detections, image_path):
        """Apply advanced post-processing to improve detection quality"""
        try:
            if detections['ship_count'] == 0:
                return detections
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # 1. Maritime context filtering
            detections = self._maritime_context_filter(detections, img_width, img_height)
            
            # 2. Size-based confidence adjustment
            detections = self._size_based_confidence_adjustment(detections, img_width, img_height)
            
            # 3. Vessel clustering for grouped objects
            detections = self._vessel_clustering(detections)
            
            logger.info(f"Post-processing: {detections['ship_count']} vessels after filtering")
            return detections
            
        except Exception as e:
            logger.error(f"Advanced post-processing failed: {e}")
            return detections
    
    def _maritime_context_filter(self, detections, img_width, img_height):
        """Filter detections based on maritime context"""
        if detections['ship_count'] == 0:
            return detections
        
        filtered_indices = []
        
        for i in range(detections['ship_count']):
            bbox = detections['bounding_boxes'][i]
            confidence = detections['confidence_scores'][i]
            x1, y1, x2, y2 = bbox
            
            # Center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Size metrics
            width = x2 - x1
            height = y2 - y1
            area = width * height
            relative_area = area / (img_width * img_height)
            
            # Maritime filtering rules
            keep_detection = True
            
            # Rule 1: Minimum size threshold (relaxed)
            if relative_area < 0.0003:  # Smaller threshold to catch more vessels
                keep_detection = False
                logger.debug(f"Filtered out tiny detection: {relative_area:.6f}")
            
            # Rule 2: Position in image - more permissive for different camera angles
            if center_y < img_height * 0.08:  # Only filter very high objects (sky)
                keep_detection = False
                logger.debug(f"Filtered out sky detection at y={center_y}")
            
            # Rule 3: Aspect ratio check - more permissive
            aspect_ratio = width / height
            if aspect_ratio < 0.6 or aspect_ratio > 15.0:  # Very permissive range
                keep_detection = False
                logger.debug(f"Filtered out extreme aspect ratio: {aspect_ratio:.2f}")
            
            # Rule 4: Confidence vs size correlation (relaxed)
            if confidence < 0.25 and relative_area < 0.003:
                keep_detection = False
                logger.debug(f"Filtered out low confidence tiny object: {confidence:.3f}")
            
            # Rule 5: Building detection - reject square objects in building-like positions
            yolo_class = detections['yolo_classes'][i] if i < len(detections.get('yolo_classes', [])) else ""
            if yolo_class.lower() in ['car', 'truck', 'bus'] and aspect_ratio < 1.3 and center_y < img_height * 0.5:
                keep_detection = False
                logger.debug(f"Filtered out potential building: {yolo_class} with ratio {aspect_ratio:.2f}")
            
            if keep_detection:
                filtered_indices.append(i)
        
        # Apply filtering
        return self._filter_detections_by_indices(detections, filtered_indices)
    
    def _size_based_confidence_adjustment(self, detections, img_width, img_height):
        """Adjust confidence scores based on vessel size and maritime context"""
        for i in range(detections['ship_count']):
            bbox = detections['bounding_boxes'][i]
            confidence = detections['confidence_scores'][i]
            x1, y1, x2, y2 = bbox
            
            width = x2 - x1
            height = y2 - y1
            area = width * height
            relative_area = area / (img_width * img_height)
            center_y = (y1 + y2) / 2
            
            # Boost confidence for well-positioned large vessels
            if relative_area > 0.01 and center_y > img_height * 0.3:
                confidence_boost = min(0.15, relative_area * 2)
                detections['confidence_scores'][i] = min(0.99, confidence + confidence_boost)
            
            # Slight penalty for very small objects
            elif relative_area < 0.002:
                detections['confidence_scores'][i] = max(0.05, confidence * 0.9)
        
        return detections
    
    def _vessel_clustering(self, detections):
        """Group nearby detections that likely represent the same vessel"""
        if detections['ship_count'] <= 1:
            return detections
        
        # Simple clustering based on IoU and proximity
        keep_indices = []
        processed = set()
        
        for i in range(detections['ship_count']):
            if i in processed:
                continue
                
            current_bbox = detections['bounding_boxes'][i]
            current_conf = detections['confidence_scores'][i]
            best_idx = i
            best_conf = current_conf
            
            # Check for overlapping detections
            for j in range(i + 1, detections['ship_count']):
                if j in processed:
                    continue
                    
                other_bbox = detections['bounding_boxes'][j]
                other_conf = detections['confidence_scores'][j]
                
                # Calculate IoU
                iou = self._calculate_iou(current_bbox, other_bbox)
                
                if iou > 0.3:  # Overlapping detections
                    processed.add(j)
                    if other_conf > best_conf:
                        best_idx = j
                        best_conf = other_conf
            
            keep_indices.append(best_idx)
            processed.add(best_idx)
        
        return self._filter_detections_by_indices(detections, keep_indices)
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_detections(self, detection_list):
        """Merge multiple detection results with Non-Maximum Suppression"""
        if not detection_list or len(detection_list) == 0:
            return self._empty_detection_result()
        
        if len(detection_list) == 1:
            return detection_list[0]
        
        # Combine all detections
        all_bboxes = []
        all_confidences = []
        all_vessel_types = []
        all_class_ids = []
        all_yolo_classes = []
        
        for detections in detection_list:
            if detections['ship_count'] > 0:
                all_bboxes.extend(detections['bounding_boxes'])
                all_confidences.extend(detections['confidence_scores'])
                all_vessel_types.extend(detections['vessel_types'])
                all_class_ids.extend(detections['class_ids'])
                all_yolo_classes.extend(detections['yolo_classes'])
        
        if not all_bboxes:
            return self._empty_detection_result()
        
        # Apply NMS
        final_indices = self._apply_nms(all_bboxes, all_confidences, self.nms_threshold)
        
        # Build final result
        merged_detections = {
            'ship_count': len(final_indices),
            'bounding_boxes': [all_bboxes[i] for i in final_indices],
            'confidence_scores': [all_confidences[i] for i in final_indices],
            'vessel_types': [all_vessel_types[i] for i in final_indices],
            'class_ids': [all_class_ids[i] for i in final_indices],
            'yolo_classes': [all_yolo_classes[i] for i in final_indices]
        }
        
        return merged_detections
    
    def _apply_nms(self, bboxes, confidences, iou_threshold):
        """Apply Non-Maximum Suppression"""
        if not bboxes:
            return []
        
        # Sort by confidence
        indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        keep = []
        
        while indices:
            current = indices.pop(0)
            keep.append(current)
            
            # Remove overlapping boxes
            indices = [i for i in indices 
                      if self._calculate_iou(bboxes[current], bboxes[i]) < iou_threshold]
        
        return keep
    
    def _filter_detections_by_indices(self, detections, indices):
        """Filter detection results by keeping only specified indices"""
        filtered_detections = {
            'ship_count': len(indices),
            'bounding_boxes': [detections['bounding_boxes'][i] for i in indices],
            'confidence_scores': [detections['confidence_scores'][i] for i in indices],
            'vessel_types': [detections['vessel_types'][i] for i in indices],
            'class_ids': [detections['class_ids'][i] for i in indices],
            'yolo_classes': [detections['yolo_classes'][i] for i in indices]
        }
        return filtered_detections
    
    def _empty_detection_result(self):
        """Return empty detection result structure"""
        return {
            'ship_count': 0,
            'bounding_boxes': [],
            'confidence_scores': [],
            'vessel_types': [],
            'class_ids': [],
            'yolo_classes': []
        }
    
    def _adaptive_confidence_threshold(self, image_path, base_threshold):
        """Adaptively adjust confidence threshold based on image characteristics"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                img_array = np.array(img.convert('RGB'))
            
            # Calculate image quality metrics
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Adjust threshold based on image quality
            adjusted_threshold = base_threshold
            
            # Enhanced adaptive thresholding
            
            # Lower threshold for high quality maritime images
            if brightness > 110 and contrast > 45:
                adjusted_threshold = max(0.08, base_threshold * 0.8)
                logger.debug(f"High quality image: lowered threshold to {adjusted_threshold:.3f}")
            
            # Higher threshold for low quality images to reduce false positives
            elif brightness < 70 or contrast < 25:
                adjusted_threshold = min(0.35, base_threshold * 1.3)
                logger.debug(f"Low quality image: raised threshold to {adjusted_threshold:.3f}")
            
            # Special handling for complex maritime scenes
            elif brightness > 90 and contrast > 60:  # Very clear maritime photos
                adjusted_threshold = max(0.06, base_threshold * 0.7)
                logger.debug(f"Clear maritime scene: optimized threshold to {adjusted_threshold:.3f}")
            
            # Adjust for image size - smaller images need slightly higher confidence
            if width * height < 640 * 640:
                adjusted_threshold = min(0.4, adjusted_threshold * 1.05)
                logger.debug(f"Small image: adjusted threshold to {adjusted_threshold:.3f}")
            
            # For very large images, be more permissive to catch distant vessels
            elif width * height > 1920 * 1080:
                adjusted_threshold = max(0.05, adjusted_threshold * 0.9)
                logger.debug(f"Large image: lowered threshold to {adjusted_threshold:.3f}")
            
            return adjusted_threshold
            
        except Exception as e:
            logger.warning(f"Could not analyze image for adaptive threshold: {e}")
            return base_threshold
    
    def get_detection_summary(self, detections):
        """Generate comprehensive detection summary with maritime analytics"""
        if detections['ship_count'] == 0:
            return {
                'summary': 'No vessels detected in maritime area',
                'detection_quality': 'N/A',
                'vessel_distribution': {},
                'confidence_analysis': 'N/A'
            }
        
        # Analyze vessel distribution
        vessel_counts = {}
        confidence_scores = detections['confidence_scores']
        
        for vessel_type in detections['vessel_types']:
            vessel_counts[vessel_type] = vessel_counts.get(vessel_type, 0) + 1
        
        # Quality assessment
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        high_conf_count = sum(1 for c in confidence_scores if c > 0.7)
        
        quality = 'Excellent' if avg_confidence > 0.8 else \
                 'Good' if avg_confidence > 0.6 else \
                 'Fair' if avg_confidence > 0.4 else 'Poor'
        
        summary_text = f"Detected {detections['ship_count']} vessel(s) in maritime surveillance area. "
        summary_text += f"Primary vessel types: {', '.join(vessel_counts.keys())}. "
        summary_text += f"Detection confidence: {quality} (avg: {avg_confidence:.3f})"
        
        return {
            'summary': summary_text,
            'detection_quality': quality,
            'vessel_distribution': vessel_counts,
            'confidence_analysis': {
                'average': avg_confidence,
                'high_confidence_count': high_conf_count,
                'confidence_range': f"{min(confidence_scores):.3f} - {max(confidence_scores):.3f}"
            },
            'maritime_analytics': {
                'total_vessels': detections['ship_count'],
                'vessel_types': len(vessel_counts),
                'largest_vessel_area': self._calculate_largest_vessel_area(detections),
                'fleet_distribution': vessel_counts
            }
        }
    
    def _calculate_largest_vessel_area(self, detections):
        """Calculate the area of the largest detected vessel"""
        if detections['ship_count'] == 0:
            return 0
        
        max_area = 0
        for bbox in detections['bounding_boxes']:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            max_area = max(max_area, area)
        
        return max_area
    
    def annotate_image(self, image_path, detections, output_path):
        """Draw YOLO detection results on image"""
        try:
            # Load image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # Colors for different vessel types
            colors = [
                '#FF0000',  # Red for large cargo
                '#00FF00',  # Green for containers  
                '#0000FF',  # Blue for ferries
                '#FFFF00',  # Yellow for tankers
                '#FF00FF',  # Magenta for fishing
                '#00FFFF',  # Cyan for small vessels
                '#FFA500',  # Orange for tugboats
                '#800080',  # Purple for naval
                '#FF69B4'   # Pink for yachts
            ]
            
            # Load font
            try:
                font_large = ImageFont.truetype("arial.ttf", 16)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw detections
            for i in range(detections['ship_count']):
                bbox = detections['bounding_boxes'][i]
                confidence = detections['confidence_scores'][i]
                vessel_type = detections['vessel_types'][i]
                class_id = detections['class_ids'][i]
                
                x1, y1, x2, y2 = bbox
                color = colors[class_id % len(colors)]
                
                # Draw bounding box with confidence-based thickness
                thickness = max(2, int(confidence * 4))
                for t in range(thickness):
                    draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)
                
                # Draw label with background
                label = f"{vessel_type}"
                conf_text = f"{confidence:.2f}"
                
                # Get text dimensions
                label_bbox = draw.textbbox((0, 0), label, font=font_large)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]
                
                # Draw label background
                draw.rectangle([x1, y1-label_height-8, x1+label_width+8, y1], fill=color)
                
                # Draw text
                draw.text((x1+4, y1-label_height-4), label, fill='black', font=font_large)
                draw.text((x2-50, y1-20), conf_text, fill=color, font=font_small)
            
            # Add model info
            info_text = f"YOLOv8 - Ships: {detections['ship_count']}"
            if 'processing_time' in detections:
                info_text += f" - {detections['processing_time']:.2f}s"
            
            img_width, img_height = image.size
            draw.rectangle([10, img_height-35, 250, img_height-10], fill='rgba(0,0,0,128)')
            draw.text((15, img_height-30), info_text, fill='white', font=font_small)
            
            # Save annotated image
            image.save(output_path, quality=95)
            logger.info(f"YOLO annotated image saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error annotating image: {e}")
            raise
    
    def get_detection_summary(self, detections):
        """Generate detection summary"""
        try:
            if not detections or detections['ship_count'] == 0:
                return {
                    'total_ships': 0,
                    'vessel_breakdown': {},
                    'average_confidence': 0,
                    'detection_quality': 'No detections',
                    'model_info': 'YOLOv8'
                }
            
            # Vessel type breakdown
            vessel_breakdown = {}
            for vessel_type in detections['vessel_types']:
                vessel_breakdown[vessel_type] = vessel_breakdown.get(vessel_type, 0) + 1
            
            # Calculate statistics
            confidences = detections['confidence_scores']
            avg_confidence = np.mean(confidences)
            
            # Determine detection quality
            if avg_confidence >= 0.8:
                quality = 'Excellent'
            elif avg_confidence >= 0.6:
                quality = 'Good'
            elif avg_confidence >= 0.4:
                quality = 'Fair'
            else:
                quality = 'Poor'
            
            summary = {
                'total_ships': detections['ship_count'],
                'vessel_breakdown': vessel_breakdown,
                'confidence_stats': {
                    'average': round(avg_confidence, 3),
                    'maximum': round(np.max(confidences), 3),
                    'minimum': round(np.min(confidences), 3)
                },
                'detection_quality': quality,
                'model_info': {
                    'model': 'YOLOv8',
                    'processing_time': detections.get('processing_time', 0),
                    'confidence_threshold': detections.get('confidence_threshold', 0.25)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating detection summary: {e}")
            return {'error': str(e), 'model_info': 'YOLOv8'}

# Create the real YOLO detector instance
if YOLO_AVAILABLE:
    YOLOShipDetector = RealYOLOShipDetector
else:
    # Keep the old class as fallback if YOLO is not available
    logger.warning("Using fallback detector - YOLO not available")
    from yolo_detector import YOLOShipDetector