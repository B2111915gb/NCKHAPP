import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import json
from config import config
from .image_processing import ImageProcessor

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from banana images for ripeness analysis"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.scaler = StandardScaler()
        self.feature_names = []
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Initialize feature names for better interpretability"""
        # Color features
        color_features = [
            'mean_hue', 'std_hue', 'mean_saturation', 'std_saturation', 
            'mean_value', 'std_value', 'dominant_color_h', 'dominant_color_s', 
            'dominant_color_v', 'yellow_ratio', 'green_ratio', 'brown_ratio',
            'color_uniformity', 'brightness', 'contrast'
        ]
        
        # Texture features  
        texture_features = [
            'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
            'glcm_energy', 'glcm_correlation', 'lbp_uniformity',
            'edge_density', 'smoothness', 'roughness', 'texture_variance'
        ]
        
        # Shape features
        shape_features = [
            'aspect_ratio', 'solidity', 'extent', 'curvature', 'elongation'
        ]
        
        self.feature_names = color_features + texture_features + shape_features
        logger.info(f"Initialized {len(self.feature_names)} features")
    
    def extract_color_features(self, image: np.ndarray) -> List[float]:
        """
        Extract color-based features from banana image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of color features
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # HSV statistics
            h, s, v = cv2.split(hsv)
            features.extend([
                np.mean(h), np.std(h),      # Hue mean, std
                np.mean(s), np.std(s),      # Saturation mean, std  
                np.mean(v), np.std(v)       # Value mean, std
            ])
            
            # Dominant color in HSV
            hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
            hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
            
            dominant_h = np.argmax(hist_h)
            dominant_s = np.argmax(hist_s)
            dominant_v = np.argmax(hist_v)
            
            features.extend([dominant_h, dominant_s, dominant_v])
            
            # Color ratios (important for banana ripeness)
            total_pixels = image.shape[0] * image.shape[1]
            
            # Yellow ratio (ripe banana indicator)
            yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
            yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
            
            # Green ratio (unripe banana indicator)  
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            green_ratio = np.sum(green_mask > 0) / total_pixels
            
            # Brown ratio (overripe banana indicator)
            brown_mask = cv2.inRange(hsv, (8, 50, 20), (20, 255, 200))
            brown_ratio = np.sum(brown_mask > 0) / total_pixels
            
            features.extend([yellow_ratio, green_ratio, brown_ratio])
            
            # Color uniformity
            color_variance = np.var(hsv, axis=(0, 1))
            color_uniformity = 1.0 / (1.0 + np.mean(color_variance))
            features.append(color_uniformity)
            
            # Brightness and contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            
            return features
            
        except Exception as e:
            logger.error(f"Color feature extraction error: {str(e)}")
            return [0.0] * config.COLOR_FEATURES_COUNT
    
    def extract_texture_features(self, image: np.ndarray) -> List[float]:
        """
        Extract texture-based features from banana image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of texture features
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # GLCM (Gray Level Co-occurrence Matrix) features
            glcm_features = self._calculate_glcm_features(gray)
            features.extend(glcm_features)
            
            # LBP (Local Binary Pattern) features
            lbp_uniformity = self._calculate_lbp_uniformity(gray)
            features.append(lbp_uniformity)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Smoothness and roughness
            smoothness = 1.0 - (1.0 / (1.0 + np.var(gray)))
            roughness = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            features.extend([smoothness, roughness])
            
            # Texture variance
            texture_variance = np.var(gray)
            features.append(texture_variance)
            
            return features
            
        except Exception as e:
            logger.error(f"Texture feature extraction error: {str(e)}")
            return [0.0] * config.TEXTURE_FEATURES_COUNT
    
    def _calculate_glcm_features(self, gray: np.ndarray) -> List[float]:
        """Calculate GLCM texture features"""
        try:
            # Simplified GLCM calculation
            # In production, you might want to use skimage.feature.greycomatrix
            
            # Calculate basic texture measures
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Gradient features
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            contrast = np.var(gx) + np.var(gy)
            dissimilarity = np.mean(np.abs(gx)) + np.mean(np.abs(gy))
            homogeneity = 1.0 / (1.0 + contrast)
            energy = np.sum(gray ** 2) / (gray.shape[0] * gray.shape[1])
            correlation = np.corrcoef(gx.flatten(), gy.flatten())[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            return [contrast, dissimilarity, homogeneity, energy, correlation]
            
        except Exception as e:
            logger.error(f"GLCM calculation error: {str(e)}")
            return [0.0] * 5
    
    def _calculate_lbp_uniformity(self, gray: np.ndarray) -> float:
        """Calculate Local Binary Pattern uniformity"""
        try:
            # Simplified LBP calculation
            rows, cols = gray.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray[i, j]
                    code = 0
                    
                    # 8-neighborhood
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                        gray[i+1, j-1], gray[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            # Calculate uniformity
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            uniformity = np.max(hist) / np.sum(hist)
            
            return float(uniformity)
            
        except Exception as e:
            logger.error(f"LBP calculation error: {str(e)}")
            return 0.0
    
    def extract_shape_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[float]:
        """
        Extract shape-based features from banana region
        
        Args:
            image: OpenCV image (BGR format)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            List of shape features
        """
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            features = []
            
            # Aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            features.append(aspect_ratio)
            
            # Extract banana contour for more detailed shape analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (should be the banana)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Solidity
                contour_area = cv2.contourArea(largest_contour)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
                
                # Extent
                bbox_area = width * height
                extent = contour_area / bbox_area if bbox_area > 0 else 0
                
                # Curvature (simplified)
                perimeter = cv2.arcLength(largest_contour, True)
                curvature = perimeter / contour_area if contour_area > 0 else 0
                
                # Elongation
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    elongation = major_axis / minor_axis if minor_axis > 0 else 0
                else:
                    elongation = aspect_ratio
                
                features.extend([solidity, extent, curvature, elongation])
            else:
                # No contour found, use basic measurements
                features.extend([0.8, 0.8, 0.1, aspect_ratio])  # Default values
            
            return features
            
        except Exception as e:
            logger.error(f"Shape feature extraction error: {str(e)}")
            return [0.0] * config.SHAPE_FEATURES_COUNT
    
    def extract_all_features(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract all features from banana image
        
        Args:
            image: PIL Image object
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Convert PIL to OpenCV
            cv_image = ImageProcessor.convert_to_opencv(image)
            
            # Crop banana region
            x1, y1, x2, y2 = bbox
            cropped_image = cv_image[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                logger.warning("Empty cropped image, using full image")
                cropped_image = cv_image
            
            # Extract different types of features
            color_features = self.extract_color_features(cropped_image)
            texture_features = self.extract_texture_features(cropped_image)
            shape_features = self.extract_shape_features(cropped_image, (0, 0, cropped_image.shape[1], cropped_image.shape[0]))
            
            # Combine all features
            all_features = color_features + texture_features + shape_features
            
            # Convert to numpy array
            feature_vector = np.array(all_features, dtype=np.float32)
            
            # Handle any NaN or infinite values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
            logger.info(f"Extracted {len(feature_vector)} features")
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            # Return zero vector as fallback
            total_features = config.COLOR_FEATURES_COUNT + config.TEXTURE_FEATURES_COUNT + config.SHAPE_FEATURES_COUNT
            return np.zeros(total_features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return self.feature_names.copy()
    
    def save_feature_config(self, filepath: str):
        """Save feature configuration to file"""
        try:
            feature_config = {
                'feature_names': self.feature_names,
                'color_features_count': config.COLOR_FEATURES_COUNT,
                'texture_features_count': config.TEXTURE_FEATURES_COUNT,
                'shape_features_count': config.SHAPE_FEATURES_COUNT,
                'total_features': len(self.feature_names)
            }
            
            with open(filepath, 'w') as f:
                json.dump(feature_config, f, indent=2)
                
            logger.info(f"Feature configuration saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving feature config: {str(e)}")
    
    def load_feature_config(self, filepath: str):
        """Load feature configuration from file"""
        try:
            with open(filepath, 'r') as f:
                feature_config = json.load(f)
            
            self.feature_names = feature_config['feature_names']
            logger.info(f"Feature configuration loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading feature config: {str(e)}")
            # Keep default configuration