"""
Extra Trees Feature Selection Module

Implements feature selection using Extra Trees Classifier
to reduce from 43 to top 8 features as per Scientific Reports 2025.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from typing import List, Tuple, Dict, Any
import logging

class ExtraTreesFeatureSelector:
    """
    Feature selector using Extra Trees Classifier
    Reduces features from 43 to top 8 most important ones
    """
    
    def __init__(self, 
                 n_features: int = 8,
                 n_estimators: int = 100,
                 random_state: int = 42,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        """
        Initialize Extra Trees Feature Selector
        
        Args:
            n_features: Number of top features to select (default: 8)
            n_estimators: Number of trees in the forest
            random_state: Random state for reproducibility
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
        """
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
        self.extra_trees = None
        self.feature_importance_ = None
        self.selected_features_ = None
        self.selected_indices_ = None
        self.feature_names_ = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> 'ExtraTreesFeatureSelector':
        """
        Fit Extra Trees and identify top features
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            feature_names: List of feature names
            
        Returns:
            self: Fitted selector
        """
        self.logger.info(f"Fitting Extra Trees with {X.shape[1]} features...")
        
        # Initialize Extra Trees Classifier
        self.extra_trees = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1
        )
        
        # Fit the classifier
        self.extra_trees.fit(X, y)
        
        # Get feature importance
        self.feature_importance_ = self.extra_trees.feature_importances_
        
        # Select top features
        top_indices = np.argsort(self.feature_importance_)[-self.n_features:][::-1]
        self.selected_indices_ = top_indices
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names_ = feature_names
            self.selected_features_ = [feature_names[i] for i in top_indices]
        else:
            self.selected_features_ = [f"feature_{i}" for i in top_indices]
        
        self.logger.info(f"Selected top {self.n_features} features:")
        for i, (idx, name) in enumerate(zip(top_indices, self.selected_features_)):
            importance = self.feature_importance_[idx]
            self.logger.info(f"  {i+1}. {name} (importance: {importance:.4f})")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to selected features only
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            X_selected: Features with only selected columns (n_samples, n_features_selected)
        """
        if self.selected_indices_ is None:
            raise ValueError("Selector must be fitted before transform")
        
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        Fit selector and transform data
        
        Args:
            X: Training features
            y: Training labels  
            feature_names: List of feature names
            
        Returns:
            X_selected: Transformed features
        """
        return self.fit(X, y, feature_names).transform(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance dictionary
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Selector must be fitted first")
        
        if self.feature_names_ is not None:
            return dict(zip(self.feature_names_, self.feature_importance_))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(self.feature_importance_)}
    
    def get_selected_features_info(self) -> pd.DataFrame:
        """
        Get detailed information about selected features
        
        Returns:
            DataFrame with selected features and their importance
        """
        if self.selected_indices_ is None:
            raise ValueError("Selector must be fitted first")
        
        return pd.DataFrame({
            'feature_name': self.selected_features_,
            'feature_index': self.selected_indices_,
            'importance': self.feature_importance_[self.selected_indices_],
            'rank': range(1, len(self.selected_features_) + 1)
        })
    
    def validate_selection(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Validate feature selection using cross-validation
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            Validation results dictionary
        """
        if self.extra_trees is None:
            raise ValueError("Selector must be fitted first")
        
        self.logger.info("Validating feature selection...")
        
        # Cross-validate with all features
        cv_scores_all = cross_val_score(self.extra_trees, X, y, cv=cv_folds, scoring='accuracy')
        
        # Cross-validate with selected features only
        X_selected = self.transform(X)
        et_selected = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            n_jobs=-1
        )
        cv_scores_selected = cross_val_score(et_selected, X_selected, y, cv=cv_folds, scoring='accuracy')
        
        results = {
            'cv_accuracy_all_features': {
                'mean': cv_scores_all.mean(),
                'std': cv_scores_all.std(),
                'scores': cv_scores_all.tolist()
            },
            'cv_accuracy_selected_features': {
                'mean': cv_scores_selected.mean(),
                'std': cv_scores_selected.std(),
                'scores': cv_scores_selected.tolist()
            },
            'feature_reduction_ratio': X_selected.shape[1] / X.shape[1],
            'performance_retention': cv_scores_selected.mean() / cv_scores_all.mean()
        }
        
        self.logger.info(f"CV Accuracy (all features): {results['cv_accuracy_all_features']['mean']:.4f} ± {results['cv_accuracy_all_features']['std']:.4f}")
        self.logger.info(f"CV Accuracy (selected features): {results['cv_accuracy_selected_features']['mean']:.4f} ± {results['cv_accuracy_selected_features']['std']:.4f}")
        self.logger.info(f"Performance retention: {results['performance_retention']:.4f}")
        
        return results
