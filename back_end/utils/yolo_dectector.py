import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional
from config import config

logger = logging.getLogger(__name__)

class BananaDetection:
    """Class to represent a single banana detection"""
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 confidence: float, class_id: int = 0):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        
    def get_bbox_int(self) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates as integers"""
        return tuple(int(coord) for coord in self.bbox)
        
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
        
    def get_area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
        
    def to_dict(self) -> Dict:
        """Convert detection to dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'center': self.get_center(),
            'area': self.get_area()
        }

class YOLODetector:
    """YOLO model wrapper for banana detection"""
    
    def __init__(self, model_path: str):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file (.pt)
        """
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.load_model()
            logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
    
    def load_model(self):
        """Load YOLO model from file"""
        try:
            self.model = YOLO(self.model_path)
            
            # Move model to appropriate device
            if self.device == 'cuda':
                self.model.to('cuda')
                
            logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def detect_bananas(self, image: Image.Image, 
                      confidence_threshold: float = None,
                      iou_threshold: float = None) -> List[BananaDetection]:
        """
        Detect bananas in image
        
        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence score (uses config default if None)
            iou_threshold: IoU threshold for NMS (uses config default if None)
            
        Returns:
            List of BananaDetection objects
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Use config defaults if not provided
            if confidence_threshold is None:
                confidence_threshold = config.YOLO_CONFIDENCE_THRESHOLD
            if iou_threshold is None:
                iou_threshold = config.YOLO_IOU_THRESHOLD
            
            # Run inference
            results = self.model(image, 
                               conf=confidence_threshold,
                               iou=iou_threshold,
                               verbose=False)
            
            detections = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        
                        # Get confidence score
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # Get class ID
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filter for banana class (assuming banana is class 0)
                        if class_id == config.BANANA_CLASS_ID:
                            detection = BananaDetection(
                                bbox=tuple(bbox),
                                confidence=confidence,
                                class_id=class_id
                            )
                            detections.append(detection)
            
            logger.info(f"Detected {len(detections)} banana(s) in image")
            return detections
            
        except Exception as e:
            logger.error(f"Error during banana detection: {str(e)}")
            raise
    
    def get_best_detection(self, detections: List[BananaDetection]) -> Optional[BananaDetection]:
        """
        Get the best detection based on confidence and size
        
        Args:
            detections: List of BananaDetection objects
            
        Returns:
            Best BananaDetection or None if list is empty
        """
        if not detections:
            return None
        
        # Sort by confidence score (descending)
        detections_sorted = sorted(detections, 
                                 key=lambda x: x.confidence, 
                                 reverse=True)
        
        # Return highest confidence detection
        best_detection = detections_sorted[0]
        logger.info(f"Best detection: confidence={best_detection.confidence:.3f}, "
                   f"area={best_detection.get_area():.0f}")
        
        return best_detection
    
    def validate_detection(self, detection: BananaDetection, 
                         image_size: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Validate detection quality
        
        Args:
            detection: BananaDetection object
            image_size: Image size (width, height)
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            # Check confidence threshold
            if detection.confidence < config.YOLO_CONFIDENCE_THRESHOLD:
                return False, f"Độ tin cậy thấp ({detection.confidence:.2f})"
            
            # Check bounding box size (not too small or too large)
            image_width, image_height = image_size
            image_area = image_width * image_height
            
            detection_area = detection.get_area()
            area_ratio = detection_area / image_area
            
            if area_ratio < 0.01:  # Too small (less than 1% of image)
                return False, "Vùng phát hiện quá nhỏ"
            
            if area_ratio > 0.8:   # Too large (more than 80% of image)
                return False, "Vùng phát hiện quá lớn"
            
            # Check if bounding box is reasonable
            x1, y1, x2, y2 = detection.bbox
            
            if x2 <= x1 or y2 <= y1:
                return False, "Bounding box không hợp lệ"
            
            # Check if detection is within image bounds
            if (x1 < 0 or y1 < 0 or 
                x2 > image_width or y2 > image_height):
                return False, "Vùng phát hiện ngoài ranh giới ảnh"
            
            return True, "Detection validation passed"
            
        except Exception as e:
            logger.error(f"Detection validation error: {str(e)}")
            return False, "Lỗi kiểm tra phát hiện"
    
    def analyze_image(self, image: Image.Image) -> Dict:
        """
        Complete analysis of image for banana detection
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with detection results and analysis
        """
        try:
            analysis_result = {
                'success': False,
                'detections': [],
                'best_detection': None,
                'message': '',
                'image_info': {
                    'size': image.size,
                    'mode': image.mode
                }
            }
            
            # Detect bananas
            detections = self.detect_bananas(image)
            
            if not detections:
                analysis_result['message'] = config.MESSAGES['no_banana_detected']
                return analysis_result
            
            # Check for multiple bananas
            if len(detections) > 1:
                # Filter detections by confidence and size
                valid_detections = []
                for detection in detections:
                    is_valid, _ = self.validate_detection(detection, image.size)
                    if is_valid:
                        valid_detections.append(detection)
                
                if len(valid_detections) > 1:
                    analysis_result['message'] = config.MESSAGES['multiple_bananas']
                    analysis_result['detections'] = [d.to_dict() for d in valid_detections]
                    return analysis_result
                
                detections = valid_detections
            
            # Get best detection
            best_detection = self.get_best_detection(detections)
            
            if best_detection is None:
                analysis_result['message'] = config.MESSAGES['no_banana_detected']
                return analysis_result
            
            # Validate best detection
            is_valid, validation_message = self.validate_detection(best_detection, image.size)
            
            if not is_valid:
                analysis_result['message'] = f"Phát hiện không hợp lệ: {validation_message}"
                return analysis_result
            
            # Success case
            analysis_result['success'] = True
            analysis_result['detections'] = [d.to_dict() for d in detections]
            analysis_result['best_detection'] = best_detection.to_dict()
            analysis_result['message'] = f"Phát hiện thành công 1 quả chuối với độ tin cậy {best_detection.confidence:.2f}"
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            return {
                'success': False,
                'detections': [],
                'best_detection': None,
                'message': config.MESSAGES['model_error'],
                'error': str(e)
            }