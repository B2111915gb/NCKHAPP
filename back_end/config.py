import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Model paths
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'yolo_banana.pt')
    REGRESSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'regression_model.pkl')
    MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model_config.json')
    
    # Image processing settings
    MAX_IMAGE_SIZE = (640, 640)  # YOLO input size
    MIN_IMAGE_SIZE = (224, 224)  # Minimum acceptable size
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
    
    # Model settings
    YOLO_CONFIDENCE_THRESHOLD = 0.5
    YOLO_IOU_THRESHOLD = 0.45
    BANANA_CLASS_ID = 0  # Assuming banana is class 0 in your YOLO model
    
    # API settings
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'app.log')
    
    # Feature extraction settings
    COLOR_FEATURES_COUNT = 15  # Number of color features
    TEXTURE_FEATURES_COUNT = 10  # Number of texture features
    SHAPE_FEATURES_COUNT = 5   # Number of shape features
    
    # Regression model settings
    RIPENESS_SCALE = {
        'green': (5, 7),      # Green banana: 5-7 days
        'yellow': (2, 4),     # Yellow banana: 2-4 days  
        'spotted': (1, 2),    # Spotted banana: 1-2 days
        'brown': (0, 1)       # Brown banana: 0-1 days
    }
    
    # Response messages
    MESSAGES = {
        'no_banana_detected': 'Không phát hiện chuối trong ảnh. Vui lòng chụp lại với ảnh rõ ràng hơn.',
        'multiple_bananas': 'Phát hiện nhiều chuối trong ảnh. Vui lòng chụp từng quả một.',
        'low_confidence': 'Độ tin cậy thấp. Vui lòng chụp ảnh rõ ràng hơn.',
        'processing_error': 'Lỗi xử lý ảnh. Vui lòng thử lại.',
        'model_error': 'Lỗi model AI. Vui lòng liên hệ hỗ trợ.'
    }

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'WARNING'

# Choose configuration based on environment
config = DevelopmentConfig() if Config.DEBUG else ProductionConfig()