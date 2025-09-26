import cv2
import numpy as np
from PIL import Image, ImageEnhance
import base64
import io
import logging
from typing import Tuple, Optional, Union
from config import config

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Class for handling image preprocessing and manipulation"""
    
    @staticmethod
    def decode_base64_image(base64_string: str) -> Optional[Image.Image]:
        """
        Decode base64 string to PIL Image
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            # Remove header if present (data:image/jpeg;base64,)
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.info(f"Successfully decoded image: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {str(e)}")
            return None
    
    @staticmethod
    def validate_image(image: Image.Image) -> Tuple[bool, str]:
        """
        Validate image format, size and quality
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            # Check image format
            if image.format and image.format.lower() not in ['jpeg', 'jpg', 'png', 'bmp']:
                return False, "Định dạng ảnh không được hỗ trợ. Vui lòng sử dụng JPG, PNG hoặc BMP."
            
            # Check image size
            width, height = image.size
            min_width, min_height = config.MIN_IMAGE_SIZE
            
            if width < min_width or height < min_height:
                return False, f"Ảnh quá nhỏ. Kích thước tối thiểu: {min_width}x{min_height}"
            
            # Check if image is too blurry (using Laplacian variance)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if variance < 100:  # Threshold for blur detection
                return False, "Ảnh bị mờ. Vui lòng chụp ảnh rõ ràng hơn."
            
            return True, "Image validation passed"
            
        except Exception as e:
            logger.error(f"Image validation error: {str(e)}")
            return False, "Lỗi kiểm tra ảnh. Vui lòng thử lại."
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int] = None) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image object
            target_size: Target size (width, height). If None, uses config.MAX_IMAGE_SIZE
            
        Returns:
            Resized PIL Image
        """
        try:
            if target_size is None:
                target_size = config.MAX_IMAGE_SIZE
            
            # Calculate new size maintaining aspect ratio
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            # Calculate scaling factor
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image in center
            final_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            final_image.paste(resized_image, (paste_x, paste_y))
            
            logger.info(f"Image resized from {image.size} to {final_image.size}")
            return final_image
            
        except Exception as e:
            logger.error(f"Image resize error: {str(e)}")
            return image  # Return original if resize fails
    
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better detection
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1.2)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(image)
            image = sharpness_enhancer.enhance(1.1)
            
            # Enhance color
            color_enhancer = ImageEnhance.Color(image)
            image = color_enhancer.enhance(1.1)
            
            logger.info("Image enhancement completed")
            return image
            
        except Exception as e:
            logger.error(f"Image enhancement error: {str(e)}")
            return image  # Return original if enhancement fails
    
    @staticmethod
    def crop_banana_region(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop banana region from image using bounding box
        
        Args:
            image: PIL Image object
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Cropped PIL Image
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Add some padding around the bounding box
            padding = 20
            width, height = image.size
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            cropped_image = image.crop((x1, y1, x2, y2))
            logger.info(f"Cropped banana region: {cropped_image.size}")
            
            return cropped_image
            
        except Exception as e:
            logger.error(f"Image crop error: {str(e)}")
            return image  # Return original if crop fails
    
    @staticmethod
    def preprocess_for_yolo(image: Image.Image) -> np.ndarray:
        """
        Preprocess image for YOLO model input
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed numpy array
        """
        try:
            # Resize to YOLO input size
            yolo_image = ImageProcessor.resize_image(image, config.MAX_IMAGE_SIZE)
            
            # Convert to numpy array
            image_array = np.array(yolo_image)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Add batch dimension and transpose to CHW format if needed
            # YOLO expects (batch, channels, height, width)
            image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
            image_array = np.expand_dims(image_array, axis=0)    # Add batch dim
            
            return image_array
            
        except Exception as e:
            logger.error(f"YOLO preprocessing error: {str(e)}")
            raise
    
    @staticmethod
    def convert_to_opencv(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format
        
        Args:
            image: PIL Image object
            
        Returns:
            OpenCV image array (BGR format)
        """
        try:
            # Convert PIL to numpy array (RGB)
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return opencv_image
            
        except Exception as e:
            logger.error(f"PIL to OpenCV conversion error: {str(e)}")
            raise