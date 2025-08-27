"""
Uncertainty Quantification for CogniThreat
==========================================

This module provides tools for quantifying and analyzing uncertainty
in machine learning predictions for network intrusion detection.

Classes:
    UncertaintyQuantifier: Main class for uncertainty analysis
    ConfidenceScorer: Scoring system for prediction confidence
    
Functions:
    confidence_analysis: Analyze prediction confidence patterns
    calculate_prediction_confidence: Calculate confidence for individual predictions
    
Author: CogniThreat Team
Date: August 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore', category=UserWarning)


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for machine learning models.
    
    This class provides multiple methods for estimating and analyzing
    uncertainty in predictions, including aleatoric and epistemic uncertainty.
    """
    
    def __init__(self,
                 base_models: Optional[List[BaseEstimator]] = None,
                 n_bootstrap_samples: int = 100,
                 confidence_threshold: float = 0.8):
        """
        Initialize Uncertainty Quantifier.
        
        Args:
            base_models: List of base models for ensemble uncertainty
            n_bootstrap_samples: Number of bootstrap samples for uncertainty estimation
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.base_models = base_models or self._create_default_ensemble()
        self.n_bootstrap_samples = n_bootstrap_samples
        self.confidence_threshold = confidence_threshold
        
        # Storage for trained models and statistics
        self.ensemble_models = []
        self.bootstrap_models = []
        self.training_statistics = {}
        self.is_fitted = False
        
        # Uncertainty components
        self.aleatoric_uncertainty = None
        self.epistemic_uncertainty = None
        self.total_uncertainty = None
    
    def _create_default_ensemble(self) -> List[BaseEstimator]:
        """Create default ensemble of diverse models."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        
        return [
            RandomForestClassifier(n_estimators=50, random_state=42),
            GradientBoostingClassifier(n_estimators=50, random_state=42),
            SVC(probability=True, random_state=42),
            GaussianNB(),
            MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        ]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit uncertainty quantification models.
        
        Args:
            X: Training features
            y: Training labels
        """
        n_samples, n_features = X.shape
        
        # Train ensemble models
        print("Training ensemble models for uncertainty quantification...")
        self.ensemble_models = []
        
        for i, model in enumerate(self.base_models):
            try:
                model.fit(X, y)
                self.ensemble_models.append(model)
                print(f"  ✓ Model {i+1}/{len(self.base_models)} trained")
            except Exception as e:
                print(f"  ✗ Model {i+1} failed: {e}")
        
        # Bootstrap sampling for uncertainty estimation
        print(f"Training {self.n_bootstrap_samples} bootstrap models...")
        self.bootstrap_models = []
        
        for i in range(self.n_bootstrap_samples):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Train model on bootstrap sample
            model = RandomForestClassifier(
                n_estimators=20, 
                random_state=i,
                max_depth=10
            )
            
            try:
                model.fit(X_bootstrap, y_bootstrap)
                self.bootstrap_models.append(model)
                
                if (i + 1) % 20 == 0:
                    print(f"  Bootstrap models: {i+1}/{self.n_bootstrap_samples}")
            except Exception as e:
                continue
        
        # Calculate training statistics
        self._calculate_training_statistics(X, y)
        
        self.is_fitted = True
        print("✅ Uncertainty quantification models trained successfully!")
    
    def _calculate_training_statistics(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate statistics from training data."""
        # Class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts / len(y)))
        
        # Feature statistics
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)
        
        # Model performance statistics
        ensemble_scores = []
        for model in self.ensemble_models:
            try:
                cv_scores = cross_val_score(model, X, y, cv=3)
                ensemble_scores.append(np.mean(cv_scores))
            except:
                continue
        
        self.training_statistics = {
            'class_distribution': class_distribution,
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'ensemble_performance': {
                'mean_accuracy': np.mean(ensemble_scores) if ensemble_scores else 0.0,
                'std_accuracy': np.std(ensemble_scores) if ensemble_scores else 0.0
            }
        }
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions with comprehensive uncertainty quantification.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, uncertainty_dict)
        """
        if not self.is_fitted:
            raise ValueError("UncertaintyQuantifier must be fitted before prediction")
        
        n_samples = X.shape[0]
        
        # Ensemble predictions
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for model in self.ensemble_models:
            try:
                pred_proba = model.predict_proba(X)
                ensemble_probabilities.append(pred_proba)
                predictions = np.argmax(pred_proba, axis=1)
                ensemble_predictions.append(predictions)
            except Exception as e:
                continue
        
        # Bootstrap predictions
        bootstrap_predictions = []
        bootstrap_probabilities = []
        
        for model in self.bootstrap_models:
            try:
                pred_proba = model.predict_proba(X)
                bootstrap_probabilities.append(pred_proba)
                predictions = np.argmax(pred_proba, axis=1)
                bootstrap_predictions.append(predictions)
            except:
                continue
        
        # Calculate uncertainties
        uncertainty_dict = self._calculate_uncertainties(
            ensemble_probabilities, bootstrap_probabilities, X
        )
        
        # Final predictions (ensemble average)
        if ensemble_probabilities:
            final_probabilities = np.mean(ensemble_probabilities, axis=0)
            final_predictions = np.argmax(final_probabilities, axis=1)
        else:
            final_predictions = np.zeros(n_samples, dtype=int)
        
        return final_predictions, uncertainty_dict
    
    def _calculate_uncertainties(self, 
                               ensemble_probs: List[np.ndarray],
                               bootstrap_probs: List[np.ndarray],
                               X: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate different types of uncertainty."""
        n_samples = X.shape[0]
        uncertainty_dict = {}
        
        if ensemble_probs:
            ensemble_probs_array = np.array(ensemble_probs)
            
            # Epistemic uncertainty (model uncertainty)
            mean_probs = np.mean(ensemble_probs_array, axis=0)
            epistemic_uncertainty = np.var(ensemble_probs_array, axis=0)
            epistemic_uncertainty = np.mean(epistemic_uncertainty, axis=1)
            
            # Prediction confidence
            prediction_confidence = np.max(mean_probs, axis=1)
            
            # Entropy-based uncertainty
            entropy_uncertainty = -np.sum(
                mean_probs * np.log(mean_probs + 1e-10), axis=1
            )
            
            uncertainty_dict.update({
                'epistemic_uncertainty': epistemic_uncertainty,
                'prediction_confidence': prediction_confidence,
                'entropy_uncertainty': entropy_uncertainty,
                'mean_probabilities': mean_probs
            })
        
        if bootstrap_probs:
            bootstrap_probs_array = np.array(bootstrap_probs)
            
            # Aleatoric uncertainty (data uncertainty)
            bootstrap_mean = np.mean(bootstrap_probs_array, axis=0)
            aleatoric_uncertainty = np.var(bootstrap_probs_array, axis=0)
            aleatoric_uncertainty = np.mean(aleatoric_uncertainty, axis=1)
            
            uncertainty_dict['aleatoric_uncertainty'] = aleatoric_uncertainty
        
        # Total uncertainty
        if 'epistemic_uncertainty' in uncertainty_dict and 'aleatoric_uncertainty' in uncertainty_dict:
            total_uncertainty = (
                uncertainty_dict['epistemic_uncertainty'] + 
                uncertainty_dict['aleatoric_uncertainty']
            )
            uncertainty_dict['total_uncertainty'] = total_uncertainty
        
        # Distance-based uncertainty
        distance_uncertainty = self._calculate_distance_uncertainty(X)
        uncertainty_dict['distance_uncertainty'] = distance_uncertainty
        
        return uncertainty_dict
    
    def _calculate_distance_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Calculate uncertainty based on distance to training data."""
        if not hasattr(self, 'training_statistics'):
            return np.ones(X.shape[0]) * 0.5
        
        # Standardize features
        X_scaled = (X - self.training_statistics['feature_means']) / (
            self.training_statistics['feature_stds'] + 1e-10
        )
        
        # Calculate distance to nearest training sample
        # For simplicity, use distance to feature means as proxy
        distances = np.linalg.norm(X_scaled, axis=1)
        
        # Convert distances to uncertainty scores (0-1)
        max_distance = np.percentile(distances, 95)
        uncertainty_scores = np.clip(distances / max_distance, 0, 1)
        
        return uncertainty_scores
    
    def analyze_uncertainty_patterns(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze patterns in uncertainty across the dataset.
        
        Args:
            X: Input features
            y: True labels
            feature_names: Names of features
            
        Returns:
            Dictionary of uncertainty analysis results
        """
        predictions, uncertainties = self.predict_with_uncertainty(X)
        
        # Accuracy vs uncertainty correlation
        correct_predictions = (predictions == y)
        
        analysis_results = {
            'accuracy_overall': np.mean(correct_predictions),
            'uncertainty_stats': {}
        }
        
        # Analyze each uncertainty type
        for uncertainty_type, uncertainty_values in uncertainties.items():
            if uncertainty_type.endswith('_uncertainty') or uncertainty_type == 'prediction_confidence':
                
                # Correlation with correctness
                correlation = np.corrcoef(uncertainty_values, correct_predictions.astype(int))[0, 1]
                
                # High vs low uncertainty performance
                high_uncertainty_mask = uncertainty_values > np.percentile(uncertainty_values, 75)
                low_uncertainty_mask = uncertainty_values < np.percentile(uncertainty_values, 25)
                
                high_uncertainty_accuracy = np.mean(correct_predictions[high_uncertainty_mask])
                low_uncertainty_accuracy = np.mean(correct_predictions[low_uncertainty_mask])
                
                analysis_results['uncertainty_stats'][uncertainty_type] = {
                    'correlation_with_correctness': correlation,
                    'high_uncertainty_accuracy': high_uncertainty_accuracy,
                    'low_uncertainty_accuracy': low_uncertainty_accuracy,
                    'mean_value': np.mean(uncertainty_values),
                    'std_value': np.std(uncertainty_values)
                }
        
        # Feature importance for uncertainty
        if feature_names and 'total_uncertainty' in uncertainties:
            feature_uncertainty_correlation = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], uncertainties['total_uncertainty'])[0, 1]
                feature_uncertainty_correlation.append(abs(corr) if not np.isnan(corr) else 0)
            
            # Sort features by uncertainty correlation
            sorted_indices = np.argsort(feature_uncertainty_correlation)[::-1]
            
            analysis_results['feature_uncertainty_ranking'] = [
                {
                    'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                    'uncertainty_correlation': feature_uncertainty_correlation[i]
                }
                for i in sorted_indices[:10]  # Top 10 features
            ]
        
        return analysis_results
    
    def visualize_uncertainty(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            save_path: str = 'uncertainty_analysis.png') -> None:
        """
        Create comprehensive uncertainty visualization.
        
        Args:
            X: Input features
            y: True labels
            save_path: Path to save visualization
        """
        predictions, uncertainties = self.predict_with_uncertainty(X)
        correct_predictions = (predictions == y)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot 1: Prediction confidence distribution
        if 'prediction_confidence' in uncertainties:
            axes[0].hist(uncertainties['prediction_confidence'], bins=30, alpha=0.7, color='blue')
            axes[0].axvline(self.confidence_threshold, color='red', linestyle='--', 
                           label=f'Threshold ({self.confidence_threshold})')
            axes[0].set_xlabel('Prediction Confidence')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Prediction Confidence Distribution')
            axes[0].legend()
        
        # Plot 2: Uncertainty vs Accuracy
        if 'total_uncertainty' in uncertainties:
            # Bin by uncertainty
            n_bins = 10
            uncertainty_bins = np.linspace(0, np.max(uncertainties['total_uncertainty']), n_bins + 1)
            bin_indices = np.digitize(uncertainties['total_uncertainty'], uncertainty_bins)
            
            bin_centers = []
            bin_accuracies = []
            
            for i in range(1, n_bins + 1):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_centers.append(np.mean(uncertainties['total_uncertainty'][mask]))
                    bin_accuracies.append(np.mean(correct_predictions[mask]))
            
            axes[1].plot(bin_centers, bin_accuracies, 'o-', color='green', linewidth=2)
            axes[1].set_xlabel('Total Uncertainty')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Uncertainty vs Accuracy')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Epistemic vs Aleatoric uncertainty
        if 'epistemic_uncertainty' in uncertainties and 'aleatoric_uncertainty' in uncertainties:
            scatter = axes[2].scatter(
                uncertainties['epistemic_uncertainty'],
                uncertainties['aleatoric_uncertainty'],
                c=correct_predictions.astype(int),
                cmap='RdYlGn',
                alpha=0.6
            )
            axes[2].set_xlabel('Epistemic Uncertainty (Model)')
            axes[2].set_ylabel('Aleatoric Uncertainty (Data)')
            axes[2].set_title('Epistemic vs Aleatoric Uncertainty')
            plt.colorbar(scatter, ax=axes[2], label='Correct Prediction')
        
        # Plot 4: Uncertainty by class
        if 'total_uncertainty' in uncertainties:
            unique_classes = np.unique(y)
            for class_label in unique_classes:
                class_mask = y == class_label
                axes[3].hist(
                    uncertainties['total_uncertainty'][class_mask],
                    bins=20, alpha=0.6, label=f'Class {class_label}',
                    density=True
                )
            axes[3].set_xlabel('Total Uncertainty')
            axes[3].set_ylabel('Density')
            axes[3].set_title('Uncertainty Distribution by Class')
            axes[3].legend()
        
        # Plot 5: Feature importance for uncertainty
        if hasattr(self, 'training_statistics') and X.shape[1] <= 20:
            feature_correlations = []
            for i in range(X.shape[1]):
                if 'total_uncertainty' in uncertainties:
                    corr = np.corrcoef(X[:, i], uncertainties['total_uncertainty'])[0, 1]
                    feature_correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            if feature_correlations:
                feature_indices = list(range(len(feature_correlations)))
                axes[4].bar(feature_indices, feature_correlations, alpha=0.7)
                axes[4].set_xlabel('Feature Index')
                axes[4].set_ylabel('|Correlation| with Uncertainty')
                axes[4].set_title('Feature-Uncertainty Correlation')
                axes[4].tick_params(axis='x', rotation=45)
        
        # Plot 6: Model agreement
        if len(self.ensemble_models) > 1:
            model_agreements = []
            for i in range(len(predictions)):
                # Calculate how many models agree on the prediction
                model_predictions = []
                for model in self.ensemble_models:
                    try:
                        pred = model.predict([X[i]])[0]
                        model_predictions.append(pred)
                    except:
                        continue
                
                if model_predictions:
                    agreement = np.mean(np.array(model_predictions) == predictions[i])
                    model_agreements.append(agreement)
                else:
                    model_agreements.append(0.5)
            
            axes[5].scatter(model_agreements, correct_predictions.astype(int), alpha=0.6)
            axes[5].set_xlabel('Model Agreement')
            axes[5].set_ylabel('Correct Prediction')
            axes[5].set_title('Model Agreement vs Correctness')
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_confidence_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get confidence scores for predictions.
        
        Args:
            X: Input features
            
        Returns:
            Array of confidence scores
        """
        predictions, uncertainties = self.predict_with_uncertainty(X)
        
        if 'prediction_confidence' in uncertainties:
            return uncertainties['prediction_confidence']
        else:
            return np.ones(X.shape[0]) * 0.5


def confidence_analysis(model: BaseEstimator,
                       X: np.ndarray,
                       y: np.ndarray,
                       confidence_metric: str = 'max_prob') -> Dict[str, Any]:
    """
    Analyze prediction confidence patterns for a model.
    
    Args:
        model: Trained model with predict_proba method
        X: Input features
        y: True labels
        confidence_metric: Method for calculating confidence
        
    Returns:
        Dictionary of confidence analysis results
    """
    # Get predictions and probabilities
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate confidence scores
    if confidence_metric == 'max_prob':
        confidence_scores = np.max(y_proba, axis=1)
    elif confidence_metric == 'entropy':
        confidence_scores = 1 + np.sum(y_proba * np.log(y_proba + 1e-10), axis=1) / np.log(y_proba.shape[1])
    else:
        confidence_scores = np.max(y_proba, axis=1)
    
    # Analyze confidence patterns
    correct_predictions = (y_pred == y)
    
    # Confidence thresholds analysis
    thresholds = np.linspace(0.5, 0.95, 10)
    threshold_analysis = []
    
    for threshold in thresholds:
        high_confidence_mask = confidence_scores >= threshold
        
        if np.sum(high_confidence_mask) > 0:
            high_conf_accuracy = np.mean(correct_predictions[high_confidence_mask])
            coverage = np.mean(high_confidence_mask)
            
            threshold_analysis.append({
                'threshold': threshold,
                'accuracy': high_conf_accuracy,
                'coverage': coverage,
                'n_samples': np.sum(high_confidence_mask)
            })
    
    # Calibration analysis
    calibration_bins = 10
    bin_boundaries = np.linspace(0, 1, calibration_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_data = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct_predictions[in_bin].mean()
            avg_confidence_in_bin = confidence_scores[in_bin].mean()
            
            calibration_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'proportion': prop_in_bin
            })
    
    return {
        'overall_accuracy': np.mean(correct_predictions),
        'mean_confidence': np.mean(confidence_scores),
        'confidence_accuracy_correlation': np.corrcoef(confidence_scores, correct_predictions.astype(int))[0, 1],
        'threshold_analysis': threshold_analysis,
        'calibration_data': calibration_data,
        'confidence_scores': confidence_scores
    }


def calculate_prediction_confidence(model: BaseEstimator,
                                  X: np.ndarray,
                                  method: str = 'max_prob') -> np.ndarray:
    """
    Calculate confidence scores for individual predictions.
    
    Args:
        model: Trained model with predict_proba method
        X: Input features
        method: Confidence calculation method
        
    Returns:
        Array of confidence scores
    """
    try:
        probabilities = model.predict_proba(X)
        
        if method == 'max_prob':
            return np.max(probabilities, axis=1)
        elif method == 'entropy':
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            # Normalize entropy to [0, 1] range
            max_entropy = np.log(probabilities.shape[1])
            return 1 - (entropy / max_entropy)
        elif method == 'margin':
            sorted_probs = np.sort(probabilities, axis=1)
            return sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            return np.max(probabilities, axis=1)
            
    except AttributeError:
        # Model doesn't have predict_proba, return uniform confidence
        return np.full(X.shape[0], 0.5)


def demo_uncertainty_quantification():
    """Demonstrate uncertainty quantification capabilities."""
    print("🎯 Uncertainty Quantification Demonstration")
    print("=" * 50)
    
    # Generate synthetic data with different noise levels
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create base dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Add noise to create uncertainty
    noise_level = 0.1
    X_noisy = X + np.random.normal(0, noise_level, X.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=0.3, random_state=42
    )
    
    # Initialize uncertainty quantifier
    uq = UncertaintyQuantifier(n_bootstrap_samples=50)
    
    # Fit models
    uq.fit(X_train, y_train)
    
    # Make predictions with uncertainty
    predictions, uncertainties = uq.predict_with_uncertainty(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"📊 Model Performance:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Ensemble Models: {len(uq.ensemble_models)}")
    print(f"Bootstrap Models: {len(uq.bootstrap_models)}")
    
    # Analyze uncertainty patterns
    analysis = uq.analyze_uncertainty_patterns(X_test, y_test)
    
    print(f"\n🔍 Uncertainty Analysis:")
    print(f"Overall Accuracy: {analysis['accuracy_overall']:.4f}")
    
    for uncertainty_type, stats in analysis['uncertainty_stats'].items():
        print(f"\n{uncertainty_type.replace('_', ' ').title()}:")
        print(f"  Correlation with correctness: {stats['correlation_with_correctness']:.4f}")
        print(f"  High uncertainty accuracy: {stats['high_uncertainty_accuracy']:.4f}")
        print(f"  Low uncertainty accuracy: {stats['low_uncertainty_accuracy']:.4f}")
    
    # Visualize uncertainties
    print(f"\n🎨 Generating uncertainty visualizations...")
    uq.visualize_uncertainty(X_test, y_test, 'demo_uncertainty_analysis.png')
    
    # Test confidence-based filtering
    if 'prediction_confidence' in uncertainties:
        high_confidence_mask = uncertainties['prediction_confidence'] > 0.8
        high_conf_accuracy = accuracy_score(
            y_test[high_confidence_mask], 
            predictions[high_confidence_mask]
        ) if np.sum(high_confidence_mask) > 0 else 0
        
        coverage = np.mean(high_confidence_mask)
        
        print(f"\n📈 Confidence-Based Filtering:")
        print(f"High confidence threshold: 0.8")
        print(f"Coverage: {coverage:.2%}")
        print(f"High confidence accuracy: {high_conf_accuracy:.4f}")
    
    print("✅ Uncertainty quantification demonstration completed successfully!")
    
    return uq, uncertainties


if __name__ == "__main__":
    demo_uncertainty_quantification()
