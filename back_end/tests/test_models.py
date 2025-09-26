import unittest
import numpy as np
import os
import sys
from PIL import Image
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processing import ImageProcessor
from utils.feature_extraction import FeatureExtractor
from utils.yolo_detector import YOLODetector, BananaDetection
from utils.regression_predictor import RegressionPredictor
from config import config

class TestImageProcessor(unittest.TestCase):
    """Test cases for ImageProcessor utility"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()
        self.test_image = Image.new('RGB', (640, 480), color=(255, 255, 0))
    
    def test_decode_base64_image_valid(self):
        """Test decoding valid base64 image"""
        import base64
        import io
        
        # Create base64 encoded image
        buffer = io.BytesIO()
        self.test_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Test decoding
        decoded_image = ImageProcessor.decode_base64_image(base64_string)
        
        self.assertIsNotNone(decoded_image)
        self.assertIsInstance(decoded_image, Image.Image)
        self.assertEqual(decoded_image.mode, 'RGB')
    
    def test_decode_base64_image_invalid(self):
        """Test decoding invalid base64 string"""
        result = ImageProcessor.decode_base64_image('invalid_base64')
        self.assertIsNone(result)
    
    def test_validate_image_valid(self):
        """Test validating valid image"""
        is_valid, message = ImageProcessor.validate_image(self.test_image)
        self.assertTrue(is_valid)
    
    def test_validate_image_too_small(self):
        """Test validating image that's too small"""
        small_image = Image.new('RGB', (100, 100), color=(255, 255, 0))
        is_valid, message = ImageProcessor.validate_image(small_image)
        self.assertFalse(is_valid)
        self.assertIn('quá nhỏ', message)
    
    def test_resize_image(self):
        """Test image resizing"""
        target_size = (320, 320)
        resized = ImageProcessor.resize_image(self.test_image, target_size)
        
        self.assertEqual(resized.size, target_size)
        self.assertEqual(resized.mode, 'RGB')
    
    def test_enhance_image(self):
        """Test image enhancement"""
        enhanced = ImageProcessor.enhance_image(self.test_image)
        
        self.assertIsInstance(enhanced, Image.Image)
        self.assertEqual(enhanced.size, self.test_image.size)
    
    def test_crop_banana_region(self):
        """Test cropping banana region"""
        bbox = (100, 100, 300, 300)
        cropped = ImageProcessor.crop_banana_region(self.test_image, bbox)
        
        self.assertIsInstance(cropped, Image.Image)
        # Should be approximately the bbox size plus padding
        self.assertGreater(cropped.size[0], 150)
        self.assertGreater(cropped.size[1], 150)
    
    def test_convert_to_opencv(self):
        """Test converting PIL to OpenCV format"""
        cv_image = ImageProcessor.convert_to_opencv(self.test_image)
        
        self.assertIsInstance(cv_image, np.ndarray)
        self.assertEqual(len(cv_image.shape), 3)  # Height, Width, Channels
        self.assertEqual(cv_image.shape[2], 3)    # BGR channels

class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor()
        self.test_image = Image.new('RGB', (640, 480), color=(255, 255, 0))
        self.test_bbox = (100, 100, 300, 300)
    
    def test_extract_color_features(self):
        """Test color feature extraction"""
        cv_image = ImageProcessor.convert_to_opencv(self.test_image)
        features = self.extractor.extract_color_features(cv_image)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), config.COLOR_FEATURES_COUNT)
        
        # Check that all features are numeric
        for feature in features:
            self.assertIsInstance(feature, (int, float))
            self.assertFalse(np.isnan(feature))
    
    def test_extract_texture_features(self):
        """Test texture feature extraction"""
        cv_image = ImageProcessor.convert_to_opencv(self.test_image)
        features = self.extractor.extract_texture_features(cv_image)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), config.TEXTURE_FEATURES_COUNT)
        
        # Check that all features are numeric
        for feature in features:
            self.assertIsInstance(feature, (int, float))
            self.assertFalse(np.isnan(feature))
    
    def test_extract_shape_features(self):
        """Test shape feature extraction"""
        cv_image = ImageProcessor.convert_to_opencv(self.test_image)
        features = self.extractor.extract_shape_features(cv_image, self.test_bbox)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), config.SHAPE_FEATURES_COUNT)
        
        # Check that all features are numeric
        for feature in features:
            self.assertIsInstance(feature, (int, float))
            self.assertFalse(np.isnan(feature))
    
    def test_extract_all_features(self):
        """Test extracting all features"""
        features = self.extractor.extract_all_features(self.test_image, self.test_bbox)
        
        self.assertIsInstance(features, np.ndarray)
        
        expected_length = (config.COLOR_FEATURES_COUNT + 
                         config.TEXTURE_FEATURES_COUNT + 
                         config.SHAPE_FEATURES_COUNT)
        self.assertEqual(len(features), expected_length)
        
        # Check for valid numeric values
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        names = self.extractor.get_feature_names()
        
        self.assertIsInstance(names, list)
        expected_length = (config.COLOR_FEATURES_COUNT + 
                         config.TEXTURE_FEATURES_COUNT + 
                         config.SHAPE_FEATURES_COUNT)
        self.assertEqual(len(names), expected_length)

class TestBananaDetection(unittest.TestCase):
    """Test cases for BananaDetection class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bbox = (100.0, 100.0, 300.0, 300.0)
        self.confidence = 0.85
        self.detection = BananaDetection(self.bbox, self.confidence)
    
    def test_initialization(self):
        """Test BananaDetection initialization"""
        self.assertEqual(self.detection.bbox, self.bbox)
        self.assertEqual(self.detection.confidence, self.confidence)
        self.assertEqual(self.detection.class_id, 0)
    
    def test_get_bbox_int(self):
        """Test getting integer bounding box"""
        bbox_int = self.detection.get_bbox_int()
        expected = (100, 100, 300, 300)
        self.assertEqual(bbox_int, expected)
    
    def test_get_center(self):
        """Test getting center point"""
        center = self.detection.get_center()
        expected = (200.0, 200.0)
        self.assertEqual(center, expected)
    
    def test_get_area(self):
        """Test getting bounding box area"""
        area = self.detection.get_area()
        expected = 200.0 * 200.0  # (300-100) * (300-100)
        self.assertEqual(area, expected)
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        result_dict = self.detection.to_dict()
        
        self.assertIn('bbox', result_dict)
        self.assertIn('confidence', result_dict)
        self.assertIn('class_id', result_dict)
        self.assertIn('center', result_dict)
        self.assertIn('area', result_dict)

class TestYOLODetector(unittest.TestCase):
    """Test cases for YOLODetector (Note: requires actual model file)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = Image.new('RGB', (640, 640), color=(255, 255, 0))
        
        # Only run tests if model file exists
        self.model_exists = os.path.exists(config.YOLO_MODEL_PATH)
        if self.model_exists:
            try:
                self.detector = YOLODetector(config.YOLO_MODEL_PATH)
            except Exception as e:
                self.model_exists = False
                print(f"Could not load YOLO model: {e}")
    
    @unittest.skipUnless(os.path.exists(config.YOLO_MODEL_PATH), "YOLO model file not found")
    def test_model_loading(self):
        """Test YOLO model loading"""
        if self.model_exists:
            self.assertIsNotNone(self.detector.model)
            self.assertIn(self.detector.device, ['cuda', 'cpu'])
    
    @unittest.skipUnless(os.path.exists(config.YOLO_MODEL_PATH), "YOLO model file not found")
    def test_detect_bananas(self):
        """Test banana detection"""
        if self.model_exists:
            detections = self.detector.detect_bananas(self.test_image)
            self.assertIsInstance(detections, list)
            
            # Each detection should be a BananaDetection object
            for detection in detections:
                self.assertIsInstance(detection, BananaDetection)
    
    @unittest.skipUnless(os.path.exists(config.YOLO_MODEL_PATH), "YOLO model file not found")
    def test_analyze_image(self):
        """Test complete image analysis"""
        if self.model_exists:
            result = self.detector.analyze_image(self.test_image)
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('detections', result)
            self.assertIn('message', result)
            self.assertIn('image_info', result)

class TestRegressionPredictor(unittest.TestCase):
    """Test cases for RegressionPredictor (Note: requires actual model file)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_features = np.random.rand(30)  # Random features for testing
        
        # Only run tests if model file exists
        self.model_exists = os.path.exists(config.REGRESSION_MODEL_PATH)
        if self.model_exists:
            try:
                self.predictor = RegressionPredictor(config.REGRESSION_MODEL_PATH)
            except Exception as e:
                self.model_exists = False
                print(f"Could not load regression model: {e}")
    
    @unittest.skipUnless(os.path.exists(config.REGRESSION_MODEL_PATH), "Regression model file not found")
    def test_model_loading(self):
        """Test regression model loading"""
        if self.model_exists:
            self.assertIsNotNone(self.predictor.model)
    
    @unittest.skipUnless(os.path.exists(config.REGRESSION_MODEL_PATH), "Regression model file not found")
    def test_predict_remaining_days(self):
        """Test remaining days prediction"""
        if self.model_exists:
            result = self.predictor.predict_remaining_days(self.test_features)
            
            self.assertIsInstance(result, dict)
            self.assertIn('remaining_days', result)
            self.assertIn('confidence', result)
            self.assertIn('ripeness_category', result)
            self.assertIn('advice', result)
            
            # Check value ranges
            self.assertGreaterEqual(result['remaining_days'], 0.0)
            self.assertLessEqual(result['remaining_days'], 14.0)
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
    
    @unittest.skipUnless(os.path.exists(config.REGRESSION_MODEL_PATH), "Regression model file not found")
    def test_validate_features(self):
        """Test feature validation"""
        if self.model_exists:
            # Test valid features
            is_valid, message = self.predictor.validate_features(self.test_features)
            self.assertTrue(is_valid)
            
            # Test invalid features (NaN)
            invalid_features = np.array([np.nan, 1.0, 2.0])
            is_valid, message = self.predictor.validate_features(invalid_features)
            self.assertFalse(is_valid)
            self.assertIn('NaN', message)
    
    @unittest.skipUnless(os.path.exists(config.REGRESSION_MODEL_PATH), "Regression model file not found")
    def test_get_model_info(self):
        """Test getting model information"""
        if self.model_exists:
            info = self.predictor.get_model_info()
            
            self.assertIsInstance(info, dict)
            self.assertIn('model_type', info)
            self.assertIn('model_path', info)
            self.assertIn('has_scaler', info)

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = Image.new('RGB', (640, 640), color=(255, 255, 0))
        self.extractor = FeatureExtractor()
        self.test_bbox = (100, 100, 300, 300)
    
    def test_complete_feature_pipeline(self):
        """Test complete feature extraction pipeline"""
        # Process image
        processed_image = ImageProcessor.resize_image(self.test_image)
        enhanced_image = ImageProcessor.enhance_image(processed_image)
        
        # Extract features
        features = self.extractor.extract_all_features(enhanced_image, self.test_bbox)
        
        # Validate results
        self.assertIsInstance(features, np.ndarray)
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))
        
        expected_length = (config.COLOR_FEATURES_COUNT + 
                         config.TEXTURE_FEATURES_COUNT + 
                         config.SHAPE_FEATURES_COUNT)
        self.assertEqual(len(features), expected_length)

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestImageProcessor))
    suite.addTest(unittest.makeSuite(TestFeatureExtractor))
    suite.addTest(unittest.makeSuite(TestBananaDetection))
    suite.addTest(unittest.makeSuite(TestYOLODetector))
    suite.addTest(unittest.makeSuite(TestRegressionPredictor))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)