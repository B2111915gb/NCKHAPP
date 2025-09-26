from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import logging
import os
import time
from datetime import datetime
import traceback
from typing import Dict, Any

# Import custom utilities
from utils.image_processing import ImageProcessor
from utils.yolo_detector import YOLODetector
from utils.feature_extraction import FeatureExtractor
from utils.regression_predictor import RegressionPredictor
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE) if os.path.exists(os.path.dirname(config.LOG_FILE)) else logging.NullHandler(),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])  # Allow all origins for development

# Global model instances
yolo_detector = None
feature_extractor = None
regression_predictor = None

def initialize_models():
    """Initialize all AI models"""
    global yolo_detector, feature_extractor, regression_predictor
    
    try:
        logger.info("Initializing AI models...")
        
        # Check if model files exist
        if not os.path.exists(config.YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found: {config.YOLO_MODEL_PATH}")
        
        if not os.path.exists(config.REGRESSION_MODEL_PATH):
            raise FileNotFoundError(f"Regression model not found: {config.REGRESSION_MODEL_PATH}")
        
        # Initialize models
        yolo_detector = YOLODetector(config.YOLO_MODEL_PATH)
        feature_extractor = FeatureExtractor()
        regression_predictor = RegressionPredictor(config.REGRESSION_MODEL_PATH)
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

def create_error_response(message: str, error_type: str = "processing_error", 
                         status_code: int = 400) -> tuple:
    """Create standardized error response"""
    return jsonify({
        'success': False,
        'error': error_type,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }), status_code

def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized success response"""
    response = {
        'success': True,
        'timestamp': datetime.now().isoformat()
    }
    response.update(data)
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_status = {
            'yolo_detector': yolo_detector is not None,
            'feature_extractor': feature_extractor is not None,
            'regression_predictor': regression_predictor is not None
        }
        
        all_loaded = all(models_status.values())
        
        return jsonify({
            'status': 'healthy' if all_loaded else 'unhealthy',
            'models': models_status,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return create_error_response("Health check failed", "system_error", 500)

@app.route('/models/info', methods=['GET'])
def get_models_info():
    """Get information about loaded models"""
    try:
        info = {
            'yolo_detector': yolo_detector.model.__class__.__name__ if yolo_detector else None,
            'regression_predictor': regression_predictor.get_model_info() if regression_predictor else None,
            'feature_extractor': {
                'total_features': len(feature_extractor.get_feature_names()) if feature_extractor else 0,
                'feature_names': feature_extractor.get_feature_names() if feature_extractor else []
            }
        }
        
        return jsonify(create_success_response({'models_info': info}))
        
    except Exception as e:
        logger.error(f"Models info error: {str(e)}")
        return create_error_response("Failed to get models info", "system_error", 500)

@app.route('/analyze_banana', methods=['POST'])
def analyze_banana():
    """Main endpoint for banana ripeness analysis"""
    start_time = time.time()
    
    try:
        # Check if models are loaded
        if not all([yolo_detector, feature_extractor, regression_predictor]):
            return create_error_response(
                "AI models not properly loaded", 
                "system_error", 
                503
            )
        
        # Validate request
        if not request.is_json:
            return create_error_response(
                "Request must be JSON", 
                "invalid_request"
            )
        
        data = request.get_json()
        
        if 'image' not in data:
            return create_error_response(
                "Missing 'image' field in request", 
                "invalid_request"
            )
        
        # Decode image
        logger.info("Decoding image...")
        image = ImageProcessor.decode_base64_image(data['image'])
        
        if image is None:
            return create_error_response(
                "Failed to decode image. Please check image format and encoding.",
                "image_decode_error"
            )
        
        # Validate image
        logger.info("Validating image...")
        is_valid, validation_message = ImageProcessor.validate_image(image)
        
        if not is_valid:
            return create_error_response(validation_message, "image_validation_error")
        
        # Preprocess image
        logger.info("Preprocessing image...")
        processed_image = ImageProcessor.resize_image(image)
        processed_image = ImageProcessor.enhance_image(processed_image)
        
        # Detect banana using YOLO
        logger.info("Detecting banana...")
        detection_result = yolo_detector.analyze_image(processed_image)
        
        if not detection_result['success']:
            return create_error_response(
                detection_result['message'], 
                "detection_error"
            )
        
        best_detection = detection_result['best_detection']
        bbox = best_detection['bbox']
        detection_confidence = best_detection['confidence']
        
        # Extract features
        logger.info("Extracting features...")
        bbox_int = tuple(int(coord) for coord in bbox)
        features = feature_extractor.extract_all_features(processed_image, bbox_int)
        
        # Validate features
        is_valid, validation_message = regression_predictor.validate_features(features)
        if not is_valid:
            return create_error_response(
                f"Feature validation failed: {validation_message}",
                "feature_error"
            )
        
        # Predict ripeness
        logger.info("Predicting ripeness...")
        prediction_result = regression_predictor.predict_remaining_days(features)
        
        # Combine results
        processing_time = time.time() - start_time
        
        final_result = {
            'remaining_days': prediction_result['remaining_days'],
            'confidence': min(detection_confidence, prediction_result['confidence']),
            'ripeness_category': prediction_result['ripeness_category'],
            'advice': prediction_result['advice'],
            'detection_info': {
                'bbox': bbox,
                'detection_confidence': detection_confidence,
                'banana_area': best_detection['area']
            },
            'processing_time': round(processing_time, 2),
            'message': f"Phân tích thành công! Chuối còn sử dụng được khoảng {prediction_result['remaining_days']:.1f} ngày."
        }
        
        logger.info(f"Analysis completed in {processing_time:.2f}s - {prediction_result['remaining_days']:.1f} days remaining")
        
        return jsonify(create_success_response(final_result))
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_trace = traceback.format_exc()
        
        logger.error(f"Analysis error after {processing_time:.2f}s: {str(e)}")
        logger.debug(f"Full traceback: {error_trace}")
        
        return create_error_response(
            config.MESSAGES.get('processing_error', 'Lỗi xử lý ảnh. Vui lòng thử lại.'),
            "processing_error",
            500
        )

@app.route('/test_detection', methods=['POST'])
def test_detection():
    """Test endpoint for YOLO detection only"""
    try:
        if not yolo_detector:
            return create_error_response("YOLO detector not loaded", "system_error", 503)
        
        data = request.get_json()
        if 'image' not in data:
            return create_error_response("Missing 'image' field", "invalid_request")
        
        # Decode and process image
        image = ImageProcessor.decode_base64_image(data['image'])
        if image is None:
            return create_error_response("Failed to decode image", "image_decode_error")
        
        processed_image = ImageProcessor.resize_image(image)
        
        # Run detection
        detection_result = yolo_detector.analyze_image(processed_image)
        
        return jsonify(create_success_response({'detection_result': detection_result}))
        
    except Exception as e:
        logger.error(f"Test detection error: {str(e)}")
        return create_error_response("Detection test failed", "processing_error", 500)

@app.route('/test_features', methods=['POST'])
def test_features():
    """Test endpoint for feature extraction"""
    try:
        if not feature_extractor:
            return create_error_response("Feature extractor not loaded", "system_error", 503)
        
        data = request.get_json()
        if 'image' not in data or 'bbox' not in data:
            return create_error_response("Missing 'image' or 'bbox' field", "invalid_request")
        
        # Decode image
        image = ImageProcessor.decode_base64_image(data['image'])
        if image is None:
            return create_error_response("Failed to decode image", "image_decode_error")
        
        bbox = tuple(data['bbox'])
        processed_image = ImageProcessor.resize_image(image)
        
        # Extract features
        features = feature_extractor.extract_all_features(processed_image, bbox)
        feature_names = feature_extractor.get_feature_names()
        
        result = {
            'features': features.tolist(),
            'feature_count': len(features),
            'feature_names': feature_names[:len(features)]  # Match length
        }
        
        return jsonify(create_success_response({'feature_result': result}))
        
    except Exception as e:
        logger.error(f"Test features error: {str(e)}")
        return create_error_response("Feature test failed", "processing_error", 500)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return create_error_response("Endpoint not found", "not_found", 404)

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return create_error_response("Method not allowed", "method_not_allowed", 405)

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return create_error_response("Internal server error", "server_error", 500)

@app.before_first_request
def startup():
    """Initialize models before first request"""
    initialize_models()

if __name__ == '__main__':
    try:
        # Create necessary directories
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialize models
        initialize_models()
        
        logger.info(f"Starting Banana AI server on {config.HOST}:{config.PORT}")
        logger.info(f"Debug mode: {config.DEBUG}")
        
        # Start Flask server
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        exit(1)