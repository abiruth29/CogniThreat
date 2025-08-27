"""
Classical machine learning models for cybersecurity intrusion detection.

This module implements baseline ML algorithms including SVM, Random Forest,
Decision Tree, and Logistic Regression for network traffic classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, Any, Tuple, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ClassicalModels:
    """
    Classical machine learning models for intrusion detection.
    """
    
    def __init__(self):
        """Initialize the classical models."""
        self.models = {}
        self.results = {}
        self.fitted_models = {}
        
        # Initialize models with optimized parameters
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all classical ML models."""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                criterion='entropy',
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        logger.info(f"Initialized {len(self.models)} classical models")
    
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models_to_train: List[str] = None
    ) -> Dict[str, Any]:
        """
        Train specified models on the training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names to train (default: all)
            
        Returns:
            Dictionary with training results
        """
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        logger.info(f"Training {len(models_to_train)} models...")
        
        training_results = {}
        
        for model_name in models_to_train:
            if model_name not in self.models:
                logger.warning(f"Model '{model_name}' not found, skipping...")
                continue
            
            logger.info(f"Training {model_name}...")
            
            try:
                # Train the model
                model = self.models[model_name]
                model.fit(X_train, y_train)
                self.fitted_models[model_name] = model
                
                # Get training accuracy
                train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)
                
                training_results[model_name] = {
                    'train_accuracy': train_accuracy,
                    'status': 'success'
                }
                
                logger.info(f"{model_name} training completed - Accuracy: {train_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return training_results
    
    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models_to_evaluate: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            models_to_evaluate: List of model names to evaluate (default: all fitted)
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        if models_to_evaluate is None:
            models_to_evaluate = list(self.fitted_models.keys())
        
        logger.info(f"Evaluating {len(models_to_evaluate)} models...")
        
        evaluation_results = {}
        
        for model_name in models_to_evaluate:
            if model_name not in self.fitted_models:
                logger.warning(f"Model '{model_name}' not trained yet, skipping...")
                continue
            
            try:
                model = self.fitted_models[model_name]
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
                }
                
                # Add ROC-AUC if probabilities are available
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} evaluation completed - Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        self.results = evaluation_results
        return evaluation_results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.fitted_models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model '{model_name}' does not have feature importance")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Plot comparison of model performances.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            logger.warning("No evaluation results available for plotting")
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = []
            valid_models = []
            
            for model in models:
                if metric in self.results[model] and not isinstance(self.results[model][metric], str):
                    values.append(self.results[model][metric])
                    valid_models.append(model)
            
            if values:
                bars = axes[i].bar(valid_models, values, alpha=0.7, color=['blue', 'green', 'red', 'orange'][:len(valid_models)])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        save_path: str = None
    ) -> None:
        """
        Plot confusion matrix for a specific model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        if model_name not in self.fitted_models:
            logger.error(f"Model '{model_name}' not trained yet")
            return
        
        model = self.fitted_models[model_name]
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> str:
        """
        Get detailed classification report for a model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Classification report as string
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.fitted_models[model_name]
        y_pred = model.predict(X_test)
        
        report = classification_report(
            y_test, y_pred,
            target_names=['Normal', 'Attack'],
            digits=4
        )
        
        return report
    
    def predict(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict
            model_name: Name of the model to use
            
        Returns:
            Predictions
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        return self.fitted_models[model_name].predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """
        Get prediction probabilities using a trained model.
        
        Args:
            X: Features to predict
            model_name: Name of the model to use
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.fitted_models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{model_name}' does not support probability predictions")
        
        return model.predict_proba(X)
    
    def save_results(self, file_path: str) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            file_path: Path to save the results
        """
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(file_path, index=True)
        logger.info(f"Results saved to: {file_path}")
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in self.results.items():
            if metric in metrics and not isinstance(metrics[metric], str):
                if metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_model = model_name
        
        if best_model is None:
            raise ValueError(f"No valid results found for metric: {metric}")
        
        return best_model, best_score
