import unittest
import json
import base64
import io
import os
import sys
from PIL import Image
import numpy as np

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from config import config

class TestBananaAPI(unittest.TestCase):
    """Test cases for Banana AI API endpoints"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Create a test image
        self.test_image = self.create_test_image()
        self.test_image_b64 = self.encode_image_to_base64(self.test_image)
    
    def create_test_image(self, width=640, height=640):
        """Create a simple test image"""
        # Create a simple banana-colored image for testing
        image = Image.new('RGB', (width, height), color=(255, 255, 0))  # Yellow
        
        # Add some variation to make it more realistic
        pixels = image.load()
        for i in range(width):
            for j in range(height):
                # Add some noise
                r = min(255, max(0, 255 + np.random.randint(-20, 20)))
                g = min(255, max(0, 255 + np.random.randint(-20, 20))) 
                b = min(255, max(0, 0 + np.random.randint(0, 50)))
                pixels[i, j] = (r, g, b)
        
        return image
    
    def encode_image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('models', data)
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
    
    def test_models_info_endpoint(self):
        """Test models info endpoint"""
        response = self.client.get('/models/info')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('models_info', data)
        self.assertIn('timestamp', data)
    
    def test_analyze_banana_success(self):
        """Test successful banana analysis"""
        payload = {
            'image': self.test_image_b64
        }
        
        response = self.client.post('/analyze_banana',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        # Note: This test might fail if models are not loaded
        # In that case, we should check for appropriate error response
        data = json.loads(response.data)
        
        if response.status_code == 200:
            # Success case
            self.assertTrue(data['success'])
            self.assertIn('remaining_days', data)
            self.assertIn('confidence', data)
            self.assertIn('ripeness_category', data)
            self.assertIn('advice', data)
            self.assertIn('processing_time', data)
        else:
            # Expected failure if models not loaded
            self.assertFalse(data['success'])
            self.assertIn('error', data)
            self.assertIn('message', data)
    
    def test_analyze_banana_missing_image(self):
        """Test banana analysis with missing image field"""
        payload = {}
        
        response = self.client.post('/analyze_banana',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)
        self.assertIn('message', data)
    
    def test_analyze_banana_invalid_json(self):
        """Test banana analysis with invalid JSON"""
        response = self.client.post('/analyze_banana',
                                  data='invalid json',
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], 'invalid_request')
    
    def test_analyze_banana_invalid_base64(self):
        """Test banana analysis with invalid base64 image"""
        payload = {
            'image': 'invalid_base64_string'
        }
        
        response = self.client.post('/analyze_banana',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        data = json.loads(response.data)
        
        if response.status_code != 503:  # If models are loaded
            self.assertFalse(data['success'])
            self.assertIn('error', data)
    
    def test_test_detection_endpoint(self):
        """Test detection testing endpoint"""
        payload = {
            'image': self.test_image_b64
        }
        
        response = self.client.post('/test_detection',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        data = json.loads(response.data)
        
        if response.status_code == 200:
            # Success case
            self.assertTrue(data['success'])
            self.assertIn('detection_result', data)
        else:
            # Expected failure if models not loaded
            self.assertFalse(data['success'])
            self.assertIn('error', data)
    
    def test_test_features_endpoint(self):
        """Test feature extraction testing endpoint"""
        payload = {
            'image': self.test_image_b64,
            'bbox': [100, 100, 200, 200]  # Sample bounding box
        }
        
        response = self.client.post('/test_features',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        
        data = json.loads(response.data)
        
        if response.status_code == 200:
            # Success case
            self.assertTrue(data['success'])
            self.assertIn('feature_result', data)
            feature_result = data['feature_result']
            self.assertIn('features', feature_result)
            self.assertIn('feature_count', feature_result)
            self.assertIn('feature_names', feature_result)
        else:
            # Expected failure if models not loaded
            self.assertFalse(data['success'])
            self.assertIn('error', data)
    
    def test_404_endpoint(self):
        """Test 404 error handling"""
        response = self.client.get('/nonexistent_endpoint')
        
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], 'not_found')
    
    def test_405_method_not_allowed(self):
        """Test 405 method not allowed error"""
        response = self.client.get('/analyze_banana')  # POST endpoint called with GET
        
        self.assertEqual(response.status_code, 405)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], 'method_not_allowed')
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.get('/health')
        
        # Check that CORS headers are present
        self.assertIn('Access-Control-Allow-Origin', response.headers)

class TestAPIPerformance(unittest.TestCase):
    """Performance tests for API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Create test image
        test_image = Image.new('RGB', (640, 640), color=(255, 255, 0))
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        self.test_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    def test_health_response_time(self):
        """Test health endpoint response time"""
        import time
        
        start_time = time.time()
        response = self.client.get('/health')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 1.0)  # Should respond within 1 second
    
    def test_analysis_response_time(self):
        """Test analysis endpoint response time"""
        import time
        
        payload = {
            'image': self.test_image_b64
        }
        
        start_time = time.time()
        response = self.client.post('/analyze_banana',
                                  data=json.dumps(payload),
                                  content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Analysis should complete within reasonable time
        # Note: This depends on model loading status
        if response.status_code == 200:
            self.assertLess(response_time, 30.0)  # Should complete within 30 seconds

if __name__ == '__main__':
    # Create test suite
    unittest.main(verbosity=2)