#!/usr/bin/env python3
"""
Production runner script for Banana AI Backend

This script provides a production-ready way to run the Flask application
with proper error handling, logging, and configuration management.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, initialize_models
    from config import config
except ImportError as e:
    print(f"Failed to import application modules: {e}")
    sys.exit(1)

def setup_logging():
    """Set up production logging"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured successfully. Log file: {config.LOG_FILE}")
        return True
        
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        return False

def check_model_files():
    """Check if required model files exist"""
    missing_files = []
    
    if not os.path.exists(config.YOLO_MODEL_PATH):
        missing_files.append(f"YOLO model: {config.YOLO_MODEL_PATH}")
    
    if not os.path.exists(config.REGRESSION_MODEL_PATH):
        missing_files.append(f"Regression model: {config.REGRESSION_MODEL_PATH}")
    
    if missing_files:
        print("ERROR: Missing required model files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease place your model files in the 'models/' directory:")
        print("  - yolo_banana.pt (YOLO v11 model)")
        print("  - regression_model.pkl (Regression model)")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'torch', 'ultralytics', 'flask', 'opencv-python', 
        'pillow', 'scikit-learn', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def print_startup_banner():
    """Print startup banner with system information"""
    print("="*60)
    print("🍌 BANANA AI BACKEND SERVER")
    print("="*60)
    print(f"Version: 1.0.0")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Host: {config.HOST}")
    print(f"Port: {config.PORT}")
    print(f"Debug mode: {config.DEBUG}")
    print(f"Log level: {config.LOG_LEVEL}")
    print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Banana AI Backend Server')
    parser.add_argument('--host', default=config.HOST, 
                       help='Host to bind to (default: %(default)s)')
    parser.add_argument('--port', type=int, default=config.PORT,
                       help='Port to bind to (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=config.DEBUG,
                       help='Enable debug mode (default: %(default)s)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check system requirements, do not start server')
    parser.add_argument('--skip-model-check', action='store_true',
                       help='Skip model file existence check')
    
    args = parser.parse_args()
    
    # Print startup banner
    print_startup_banner()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    print("✓ All dependencies satisfied")
    
    # Check model files (unless skipped)
    if not args.skip_model_check:
        print("Checking model files...")
        if not check_model_files():
            return 1
        print("✓ All model files found")
    else:
        print("⚠ Model file check skipped")
    
    # Set up logging
    print("Setting up logging...")
    if not setup_logging():
        return 1
    print("✓ Logging configured")
    
    logger = logging.getLogger(__name__)
    
    # If check-only mode, exit here
    if args.check_only:
        print("✅ All checks passed!")
        return 0
    
    try:
        # Initialize models
        print("Initializing AI models...")
        logger.info("Starting model initialization...")
        
        if not args.skip_model_check:
            initialize_models()
            logger.info("✓ All models initialized successfully")
            print("✓ AI models loaded")
        else:
            logger.warning("⚠ Model initialization skipped")
            print("⚠ Model initialization skipped")
        
        # Start Flask server
        logger.info(f"Starting Flask server on {args.host}:{args.port}")
        print(f"🚀 Starting server on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        print("="*60)
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False  # Disable reloader in production
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\n👋 Server stopped by user")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        print(f"\n❌ Failed to start server: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())