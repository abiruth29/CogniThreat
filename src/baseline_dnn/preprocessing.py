"""
DNN Preprocessing Module

Handles data preprocessing for the DNN baseline including:
- Standard scaling
- SMOTE for class imbalance
- Data validation and cleaning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Any, Optional
import logging

class DNNPreprocessor:
    """
    Preprocessor for DNN baseline following Scientific Reports 2025 methodology
    """
    
    def __init__(self, 
                 test_size: float = 0.3,
                 random_state: int = 42,
                 apply_smote: bool = True,
                 smote_strategy: str = 'auto'):
        """
        Initialize DNN Preprocessor
        
        Args:
            test_size: Test set proportion (default: 0.3 for 70:30 split)
            random_state: Random state for reproducibility
            apply_smote: Whether to apply SMOTE for class imbalance
            smote_strategy: SMOTE sampling strategy
        """
        self.test_size = test_size
        self.random_state = random_state
        self.apply_smote = apply_smote
        self.smote_strategy = smote_strategy
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.smote = None
        
        self.is_fitted = False
        self.feature_names_ = None
        self.n_features_in_ = None
        self.classes_ = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> 'DNNPreprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            X: Training features
            y: Training labels
            feature_names: List of feature names
            
        Returns:
            self: Fitted preprocessor
        """
        self.logger.info("Fitting DNN preprocessor...")
        
        # Store metadata
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(self.n_features_in_)]
        
        # Fit standard scaler
        self.scaler.fit(X)
        
        # Fit label encoder
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        
        # Initialize SMOTE if needed
        if self.apply_smote:
            self.smote = SMOTE(
                sampling_strategy=self.smote_strategy,
                random_state=self.random_state
            )
        
        self.is_fitted = True
        self.logger.info(f"Preprocessor fitted with {self.n_features_in_} features")
        self.logger.info(f"Classes found: {self.classes_}")
        
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray = None, apply_smote: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features and labels
        
        Args:
            X: Features to transform
            y: Labels to transform (optional)
            apply_smote: Whether to apply SMOTE (only for training data)
            
        Returns:
            X_transformed: Scaled features
            y_transformed: Encoded labels (if y provided)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
        
        # Apply SMOTE if requested (typically only for training data)
        if apply_smote and self.apply_smote and y_encoded is not None:
            self.logger.info("Applying SMOTE for class balancing...")
            original_shape = X_scaled.shape
            original_class_dist = np.bincount(y_encoded)
            
            X_scaled, y_encoded = self.smote.fit_resample(X_scaled, y_encoded)
            
            new_shape = X_scaled.shape
            new_class_dist = np.bincount(y_encoded)
            
            self.logger.info(f"SMOTE applied: {original_shape} -> {new_shape}")
            self.logger.info(f"Class distribution before: {original_class_dist}")
            self.logger.info(f"Class distribution after: {new_class_dist}")
        
        if y is not None:
            return X_scaled, y_encoded
        else:
            return X_scaled
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list = None, apply_smote: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform data
        
        Args:
            X: Training features
            y: Training labels
            feature_names: List of feature names
            apply_smote: Whether to apply SMOTE
            
        Returns:
            X_transformed: Scaled features
            y_transformed: Encoded labels
        """
        return self.fit(X, y, feature_names).transform(X, y, apply_smote)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> Dict[str, Any]:
        """
        Complete data preparation pipeline
        
        Args:
            X: Features
            y: Labels
            feature_names: Feature names
            
        Returns:
            Dictionary containing train/test splits and metadata
        """
        self.logger.info("Starting complete data preparation pipeline...")
        
        # Split data first (70:30 as per paper)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        self.logger.info(f"Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        
        # Fit on training data
        self.fit(X_train, y_train, feature_names)
        
        # Transform training data (with SMOTE)
        X_train_scaled, y_train_encoded = self.transform(X_train, y_train, apply_smote=True)
        
        # Transform test data (without SMOTE)
        X_test_scaled, y_test_encoded = self.transform(X_test, y_test, apply_smote=False)
        
        # Calculate class distribution information
        train_class_dist = np.bincount(y_train_encoded)
        test_class_dist = np.bincount(y_test_encoded)
        
        # Prepare results
        results = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_encoded,
            'y_test': y_test_encoded,
            'train_shape': X_train_scaled.shape,
            'test_shape': X_test_scaled.shape,
            'n_features': self.n_features_in_,
            'n_classes': len(self.classes_),
            'classes': self.classes_,
            'feature_names': self.feature_names_,
            'train_class_distribution': train_class_dist,
            'test_class_distribution': test_class_dist,
            'preprocessing_info': {
                'scaler': 'StandardScaler',
                'smote_applied': self.apply_smote,
                'test_size': self.test_size,
                'random_state': self.random_state
            }
        }
        
        self.logger.info("Data preparation completed successfully")
        self.logger.info(f"Final training shape: {results['train_shape']}")
        self.logger.info(f"Final test shape: {results['test_shape']}")
        self.logger.info(f"Training class distribution: {train_class_dist}")
        self.logger.info(f"Test class distribution: {test_class_dist}")
        
        return results
    
    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original format
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Original labels
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing configuration
        
        Returns:
            Preprocessing summary dictionary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "n_features": self.n_features_in_,
            "n_classes": len(self.classes_),
            "classes": self.classes_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "test_size": self.test_size,
            "smote_applied": self.apply_smote,
            "random_state": self.random_state
        }
