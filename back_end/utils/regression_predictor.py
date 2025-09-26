import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import joblib
from config import config

logger = logging.getLogger(__name__)

class RegressionPredictor:
    """Wrapper for regression model to predict banana ripeness/remaining days"""
    
    def __init__(self, model_path: str):
        """
        Initialize regression predictor
        
        Args:
            model_path: Path to pickled regression model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = {}
        
        try:
            self.load_model()
            logger.info("Regression model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load regression model: {str(e)}")
            raise
    
    def load_model(self):
        """Load regression model and associated components"""
        try:
            # Try loading with joblib first (recommended for sklearn models)
            try:
                model_data = joblib.load(self.model_path)
            except:
                # Fallback to pickle
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Handle different model save formats
            if isinstance(model_data, dict):
                # Model saved as dictionary with metadata
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
                self.model_info = model_data.get('info', {})
            else:
                # Model saved directly
                self.model = model_data
                self.scaler = StandardScaler()  # Create default scaler
                self.feature_names = []
                self.model_info = {}
            
            # Validate model
            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded object doesn't have predict method")
            
            logger.info(f"Model type: {type(self.model).__name__}")
            if self.model_info:
                logger.info(f"Model info: {self.model_info}")
                
        except Exception as e:
            logger.error(f"Error loading regression model: {str(e)}")
            raise
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features before prediction
        
        Args:
            features: Raw feature vector
            
        Returns:
            Preprocessed feature vector
        """
        try:
            # Ensure features is 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Handle missing or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Apply scaling if scaler is available and fitted
            if self.scaler is not None:
                try:
                    # Check if scaler is fitted
                    if hasattr(self.scaler, 'mean_'):
                        features = self.scaler.transform(features)
                    else:
                        logger.warning("Scaler not fitted, using raw features")
                except Exception as e:
                    logger.warning(f"Scaling failed: {str(e)}, using raw features")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preprocessing error: {str(e)}")
            # Return original features if preprocessing fails
            return features.reshape(1, -1) if features.ndim == 1 else features
    
    def predict_remaining_days(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict remaining days for banana consumption
        
        Args:
            features: Feature vector extracted from banana image
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess features
            processed_features = self.preprocess_features(features)
            
            # Make prediction
            prediction = self.model.predict(processed_features)
            
            # Handle prediction output (ensure it's a scalar)
            if isinstance(prediction, np.ndarray):
                remaining_days = float(prediction[0])
            else:
                remaining_days = float(prediction)
            
            # Ensure reasonable bounds
            remaining_days = max(0.0, min(remaining_days, 14.0))  # 0-14 days range
            
            # Get prediction confidence/uncertainty if available
            confidence = self._calculate_confidence(processed_features, remaining_days)
            
            # Determine ripeness category
            ripeness_category = self._categorize_ripeness(remaining_days)
            
            # Generate advice message
            advice = self._generate_advice(remaining_days, ripeness_category)
            
            result = {
                'remaining_days': remaining_days,
                'confidence': confidence,
                'ripeness_category': ripeness_category,
                'advice': advice,
                'prediction_details': {
                    'raw_prediction': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                    'feature_count': len(features),
                    'model_type': type(self.model).__name__
                }
            }
            
            logger.info(f"Predicted {remaining_days:.1f} days remaining with {confidence:.2f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'remaining_days': 2.0,  # Default prediction
                'confidence': 0.3,      # Low confidence
                'ripeness_category': 'unknown',
                'advice': 'Không thể phân tích chính xác. Vui lòng kiểm tra trực tiếp.',
                'error': str(e)
            }
    
    def _calculate_confidence(self, features: np.ndarray, prediction: float) -> float:
        """
        Calculate prediction confidence score
        
        Args:
            features: Processed feature vector
            prediction: Model prediction
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence based on model type and prediction
            base_confidence = 0.7
            
            # Adjust based on prediction range (more confident for typical values)
            if 0.5 <= prediction <= 7.0:  # Typical banana lifespan
                range_bonus = 0.2
            elif 0.0 <= prediction <= 10.0:  # Reasonable range
                range_bonus = 0.1
            else:
                range_bonus = -0.2  # Less confident for extreme values
            
            # Adjust based on feature quality (simplified)
            feature_quality = min(1.0, np.mean(np.abs(features)))
            quality_bonus = min(0.1, feature_quality * 0.1)
            
            confidence = base_confidence + range_bonus + quality_bonus
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation error: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def _categorize_ripeness(self, remaining_days: float) -> str:
        """
        Categorize banana ripeness based on remaining days
        
        Args:
            remaining_days: Predicted remaining days
            
        Returns:
            Ripeness category string
        """
        try:
            if remaining_days >= 5:
                return 'green'      # Unripe/green
            elif remaining_days >= 3:
                return 'yellow'     # Ripe/yellow
            elif remaining_days >= 1:
                return 'spotted'    # Very ripe/spotted
            else:
                return 'brown'      # Overripe/brown
                
        except Exception as e:
            logger.warning(f"Ripeness categorization error: {str(e)}")
            return 'unknown'
    
    def _generate_advice(self, remaining_days: float, category: str) -> str:
        """
        Generate consumption advice based on prediction
        
        Args:
            remaining_days: Predicted remaining days
            category: Ripeness category
            
        Returns:
            Advice string in Vietnamese
        """
        try:
            advice_templates = {
                'green': [
                    f"Chuối còn xanh, có thể bảo quản thêm {remaining_days:.0f} ngày.",
                    "Để ở nhiệt độ phòng để chuối chín đều.",
                    "Có thể cho vào túi giấy cùng táo để chín nhanh hơn."
                ],
                'yellow': [
                    f"Chuối đã chín, tốt nhất nên ăn trong {remaining_days:.0f} ngày tới.",
                    "Bảo quản ở nơi khô ráo, thoáng mát.",
                    "Đây là thời điểm hoàn hảo để thưởng thức!"
                ],
                'spotted': [
                    f"Chuối đã rất chín, nên ăn ngay trong {remaining_days:.0f} ngày.",
                    "Có thể dùng làm sinh tố, bánh chuối hoặc kem.",
                    "Không nên bảo quản lâu ở nhiệt độ phòng."
                ],
                'brown': [
                    "Chuối đã quá chín, nên sử dụng ngay hôm nay.",
                    "Rất thích hợp làm bánh chuối nướng hoặc sinh tố.",
                    "Kiểm tra kỹ trước khi ăn, loại bỏ phần hỏng (nếu có)."
                ]
            }
            
            if category in advice_templates:
                advice_list = advice_templates[category]
                return " ".join(advice_list)
            else:
                return "Vui lòng kiểm tra trực tiếp chất lượng chuối trước khi sử dụng."
                
        except Exception as e:
            logger.warning(f"Advice generation error: {str(e)}")
            return "Không thể tạo lời khuyên. Vui lòng kiểm tra trực tiếp."
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict for multiple feature vectors
        
        Args:
            features_batch: Array of feature vectors (n_samples, n_features)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if features_batch.ndim != 2:
                raise ValueError("Features batch must be 2D array")
            
            results = []
            for features in features_batch:
                result = self.predict_remaining_days(features)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': type(self.model).__name__ if self.model else 'Not loaded',
            'model_path': self.model_path,
            'has_scaler': self.scaler is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown',
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        
        # Add model-specific information
        if self.model:
            try:
                if hasattr(self.model, 'n_features_in_'):
                    info['expected_features'] = self.model.n_features_in_
                if hasattr(self.model, 'feature_importances_'):
                    info['has_feature_importance'] = True
                if hasattr(self.model, 'score'):
                    info['supports_scoring'] = True
            except:
                pass
        
        return info
    
    def validate_features(self, features: np.ndarray) -> Tuple[bool, str]:
        """
        Validate input features
        
        Args:
            features: Feature vector to validate
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            if features is None or len(features) == 0:
                return False, "Features vector is empty"
            
            # Check for expected feature count
            expected_features = getattr(self.model, 'n_features_in_', None)
            if expected_features and len(features) != expected_features:
                return False, f"Expected {expected_features} features, got {len(features)}"
            
            # Check for invalid values
            if np.any(np.isnan(features)):
                return False, "Features contain NaN values"
            
            if np.any(np.isinf(features)):
                return False, "Features contain infinite values"
            
            return True, "Features validation passed"
            
        except Exception as e:
            return False, f"Feature validation error: {str(e)}"