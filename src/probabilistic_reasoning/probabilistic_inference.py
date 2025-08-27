"""
Probabilistic Inference Engine for CogniThreat
==============================================

This module implements probabilistic inference engines for combining
different models and making uncertainty-aware predictions.

Classes:
    BayesianInferenceEngine: Core inference engine for probabilistic reasoning
    ProbabilisticClassifier: Classifier with built-in uncertainty quantification
    AdaptiveThresholdLearner: Dynamic threshold optimization system
    
Functions:
    adaptive_threshold_learning: Learn optimal decision thresholds
    combine_model_predictions: Ensemble prediction combination
    
Author: CogniThreat Team
Date: August 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize_scalar
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)


class BayesianInferenceEngine:
    """
    Bayesian inference engine for combining multiple models and information sources.
    
    This class implements Bayesian reasoning for network intrusion detection,
    combining evidence from multiple models and prior knowledge.
    """
    
    def __init__(self,
                 models: List[BaseEstimator],
                 model_weights: Optional[List[float]] = None,
                 prior_probabilities: Optional[Dict[int, float]] = None):
        """
        Initialize Bayesian Inference Engine.
        
        Args:
            models: List of trained models
            model_weights: Weights for each model (uniform if None)
            prior_probabilities: Prior class probabilities
        """
        self.models = models
        self.model_weights = model_weights or [1.0] * len(models)
        self.prior_probabilities = prior_probabilities
        
        # Normalize weights
        weight_sum = sum(self.model_weights)
        self.model_weights = [w / weight_sum for w in self.model_weights]
        
        # Model reliability scores
        self.model_reliability = {}
        self.calibrated_models = []
        
        # Evidence combination rules
        self.combination_method = 'weighted_average'
        
    def calibrate_models(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Calibrate model probabilities using Platt scaling.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
        """
        self.calibrated_models = []
        
        for i, model in enumerate(self.models):
            try:
                # Use isotonic regression for calibration
                calibrated_model = CalibratedClassifierCV(
                    model, method='isotonic', cv='prefit'
                )
                calibrated_model.fit(X_cal, y_cal)
                self.calibrated_models.append(calibrated_model)
                
                # Calculate reliability score
                cal_proba = calibrated_model.predict_proba(X_cal)
                cal_pred = np.argmax(cal_proba, axis=1)
                reliability = accuracy_score(y_cal, cal_pred)
                self.model_reliability[i] = reliability
                
            except Exception as e:
                print(f"Calibration failed for model {i}: {e}")
                self.calibrated_models.append(model)
                self.model_reliability[i] = 0.5
    
    def _apply_priors(self, posteriors: np.ndarray) -> np.ndarray:
        """
        Apply prior probabilities to posterior distributions.
        
        Args:
            posteriors: Posterior probabilities from models
            
        Returns:
            Prior-adjusted probabilities
        """
        if self.prior_probabilities is None:
            return posteriors
        
        # Convert priors to array
        n_classes = posteriors.shape[1]
        prior_array = np.ones(n_classes) / n_classes  # Uniform default
        
        for class_idx, prior_prob in self.prior_probabilities.items():
            if 0 <= class_idx < n_classes:
                prior_array[class_idx] = prior_prob
        
        # Normalize priors
        prior_array = prior_array / np.sum(prior_array)
        
        # Apply Bayes' rule: P(class|evidence) ∝ P(evidence|class) * P(class)
        adjusted_posteriors = posteriors * prior_array
        
        # Renormalize
        row_sums = np.sum(adjusted_posteriors, axis=1, keepdims=True)
        adjusted_posteriors = adjusted_posteriors / (row_sums + 1e-10)
        
        return adjusted_posteriors
    
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make probabilistic predictions with uncertainty quantification.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (probabilities, inference_info)
        """
        n_samples = X.shape[0]
        model_predictions = []
        model_confidences = []
        
        # Get predictions from each model
        for i, model in enumerate(self.calibrated_models or self.models):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    confidence = np.max(proba, axis=1)
                else:
                    # For models without predict_proba, use predict and convert to probabilities
                    pred = model.predict(X)
                    n_classes = len(np.unique(pred))
                    proba = np.eye(n_classes)[pred]
                    confidence = np.ones(len(pred))
                
                model_predictions.append(proba)
                model_confidences.append(confidence)
                
            except Exception as e:
                print(f"Prediction failed for model {i}: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("No valid model predictions obtained")
        
        # Combine predictions using Bayesian inference
        combined_proba, inference_info = self._combine_predictions(
            model_predictions, model_confidences, X
        )
        
        # Apply priors
        final_proba = self._apply_priors(combined_proba)
        
        # Update inference info
        inference_info.update({
            'final_probabilities': final_proba,
            'prior_applied': self.prior_probabilities is not None,
            'n_models_used': len(model_predictions)
        })
        
        return final_proba, inference_info
    
    def _combine_predictions(self, 
                           model_predictions: List[np.ndarray],
                           model_confidences: List[np.ndarray],
                           X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Combine predictions from multiple models using Bayesian inference.
        
        Args:
            model_predictions: List of prediction arrays
            model_confidences: List of confidence arrays
            X: Input features
            
        Returns:
            Combined predictions and inference information
        """
        n_samples = X.shape[0]
        n_models = len(model_predictions)
        n_classes = model_predictions[0].shape[1]
        
        if self.combination_method == 'weighted_average':
            # Weighted average with reliability weights
            combined_proba = np.zeros((n_samples, n_classes))
            total_weight = 0.0
            
            for i, proba in enumerate(model_predictions):
                # Calculate dynamic weight based on reliability and confidence
                reliability_weight = self.model_reliability.get(i, 1.0)
                static_weight = self.model_weights[i]
                confidence_weight = np.mean(model_confidences[i])
                
                dynamic_weight = static_weight * reliability_weight * confidence_weight
                
                combined_proba += dynamic_weight * proba
                total_weight += dynamic_weight
            
            combined_proba = combined_proba / (total_weight + 1e-10)
            
        elif self.combination_method == 'bayesian_model_averaging':
            # Bayesian model averaging
            combined_proba = np.zeros((n_samples, n_classes))
            
            for i, proba in enumerate(model_predictions):
                model_evidence = self._calculate_model_evidence(proba, model_confidences[i])
                combined_proba += model_evidence.reshape(-1, 1) * proba
            
            # Normalize
            row_sums = np.sum(combined_proba, axis=1, keepdims=True)
            combined_proba = combined_proba / (row_sums + 1e-10)
        
        else:  # Simple average
            combined_proba = np.mean(model_predictions, axis=0)
        
        # Calculate uncertainty metrics
        model_agreement = self._calculate_model_agreement(model_predictions)
        epistemic_uncertainty = self._calculate_epistemic_uncertainty(model_predictions)
        
        inference_info = {
            'model_predictions': model_predictions,
            'model_confidences': model_confidences,
            'model_agreement': model_agreement,
            'epistemic_uncertainty': epistemic_uncertainty,
            'combination_method': self.combination_method
        }
        
        return combined_proba, inference_info
    
    def _calculate_model_evidence(self, 
                                proba: np.ndarray, 
                                confidence: np.ndarray) -> np.ndarray:
        """Calculate evidence for each model prediction."""
        # Use confidence as proxy for evidence
        # Higher confidence predictions get higher evidence
        evidence = confidence / np.sum(confidence)
        return evidence
    
    def _calculate_model_agreement(self, model_predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate agreement between models for each sample."""
        n_samples = model_predictions[0].shape[0]
        n_models = len(model_predictions)
        
        agreements = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get predicted classes for each model
            predicted_classes = [np.argmax(pred[i]) for pred in model_predictions]
            
            # Calculate agreement as fraction of models agreeing with majority
            from collections import Counter
            class_counts = Counter(predicted_classes)
            majority_class = class_counts.most_common(1)[0][0]
            agreement = class_counts[majority_class] / n_models
            
            agreements[i] = agreement
        
        return agreements
    
    def _calculate_epistemic_uncertainty(self, model_predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate epistemic uncertainty from model disagreement."""
        # Stack predictions
        stacked_predictions = np.stack(model_predictions, axis=0)
        
        # Calculate variance across models for each sample and class
        prediction_variance = np.var(stacked_predictions, axis=0)
        
        # Average variance across classes for each sample
        epistemic_uncertainty = np.mean(prediction_variance, axis=1)
        
        return epistemic_uncertainty
    
    def update_model_weights(self, 
                           X_val: np.ndarray, 
                           y_val: np.ndarray,
                           metric: str = 'accuracy') -> None:
        """
        Update model weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Performance metric to optimize
        """
        model_scores = []
        
        for model in self.models:
            try:
                if metric == 'accuracy':
                    pred = model.predict(X_val)
                    score = accuracy_score(y_val, pred)
                elif metric == 'f1':
                    pred = model.predict(X_val)
                    score = f1_score(y_val, pred, average='weighted')
                else:
                    score = 0.5
                
                model_scores.append(score)
            except:
                model_scores.append(0.1)
        
        # Convert to weights (higher score = higher weight)
        total_score = sum(model_scores)
        if total_score > 0:
            self.model_weights = [score / total_score for score in model_scores]
        else:
            self.model_weights = [1.0 / len(self.models)] * len(self.models)


class ProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    """
    Probabilistic classifier with built-in uncertainty quantification.
    
    This classifier combines multiple models and provides calibrated
    probabilities with uncertainty estimates.
    """
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 inference_engine: Optional[BayesianInferenceEngine] = None,
                 uncertainty_threshold: float = 0.3):
        """
        Initialize Probabilistic Classifier.
        
        Args:
            base_models: List of base classification models
            inference_engine: Bayesian inference engine (created if None)
            uncertainty_threshold: Threshold for high uncertainty predictions
        """
        self.base_models = base_models
        self.uncertainty_threshold = uncertainty_threshold
        
        # Initialize inference engine
        if inference_engine is None:
            self.inference_engine = BayesianInferenceEngine(base_models)
        else:
            self.inference_engine = inference_engine
        
        self.is_fitted = False
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ProbabilisticClassifier':
        """
        Fit the probabilistic classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Fitted classifier
        """
        # Store classes
        self.classes_ = np.unique(y)
        
        # Train base models
        fitted_models = []
        for i, model in enumerate(self.base_models):
            try:
                model.fit(X, y)
                fitted_models.append(model)
            except Exception as e:
                print(f"Model {i} training failed: {e}")
        
        self.inference_engine.models = fitted_models
        
        # Calibrate models using a portion of training data
        from sklearn.model_selection import train_test_split
        if len(X) > 100:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.inference_engine.calibrate_models(X_cal, y_cal)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        proba, _ = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict class probabilities with uncertainty.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (probabilities, uncertainty_info)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        proba, inference_info = self.inference_engine.predict_proba(X)
        
        # Calculate additional uncertainty metrics
        uncertainty_info = self._calculate_prediction_uncertainty(proba, inference_info)
        
        return proba, uncertainty_info
    
    def _calculate_prediction_uncertainty(self, 
                                        proba: np.ndarray,
                                        inference_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive uncertainty metrics for predictions."""
        # Prediction confidence
        prediction_confidence = np.max(proba, axis=1)
        
        # Entropy-based uncertainty
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        normalized_entropy = entropy / np.log(proba.shape[1])
        
        # Uncertainty flags
        high_uncertainty_mask = (
            (prediction_confidence < (1 - self.uncertainty_threshold)) |
            (normalized_entropy > self.uncertainty_threshold) |
            (inference_info.get('epistemic_uncertainty', np.zeros(len(proba))) > self.uncertainty_threshold)
        )
        
        uncertainty_info = {
            'prediction_confidence': prediction_confidence,
            'entropy_uncertainty': normalized_entropy,
            'high_uncertainty_mask': high_uncertainty_mask,
            'uncertain_predictions': np.sum(high_uncertainty_mask),
            'uncertainty_rate': np.mean(high_uncertainty_mask)
        }
        
        # Add inference info
        uncertainty_info.update(inference_info)
        
        return uncertainty_info
    
    def predict_with_rejection(self, 
                             X: np.ndarray,
                             confidence_threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with option to reject uncertain samples.
        
        Args:
            X: Input features
            confidence_threshold: Minimum confidence for acceptance
            
        Returns:
            Tuple of (predictions, acceptance_mask)
        """
        proba, uncertainty_info = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        
        # Create acceptance mask
        confidence_scores = uncertainty_info['prediction_confidence']
        acceptance_mask = confidence_scores >= confidence_threshold
        
        return predictions, acceptance_mask


class AdaptiveThresholdLearner:
    """
    Learn adaptive decision thresholds for probabilistic classifiers.
    
    This class optimizes decision thresholds to balance different metrics
    such as precision, recall, and coverage.
    """
    
    def __init__(self,
                 objective: str = 'f1',
                 constraint_metric: Optional[str] = None,
                 constraint_value: Optional[float] = None):
        """
        Initialize Adaptive Threshold Learner.
        
        Args:
            objective: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            constraint_metric: Metric to constrain
            constraint_value: Value for constraint
        """
        self.objective = objective
        self.constraint_metric = constraint_metric
        self.constraint_value = constraint_value
        
        self.optimal_thresholds = {}
        self.threshold_performance = {}
    
    def learn_threshold(self, 
                       y_true: np.ndarray,
                       y_proba: np.ndarray,
                       class_idx: int = 1) -> float:
        """
        Learn optimal threshold for a specific class.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_idx: Class index to optimize for
            
        Returns:
            Optimal threshold
        """
        # Binary classification case
        if y_proba.ndim == 1 or y_proba.shape[1] == 2:
            if y_proba.ndim == 2:
                scores = y_proba[:, 1]
            else:
                scores = y_proba
            
            # Convert to binary labels
            binary_labels = (y_true == class_idx).astype(int)
            
        else:
            # Multi-class case - use one-vs-rest
            scores = y_proba[:, class_idx]
            binary_labels = (y_true == class_idx).astype(int)
        
        # Define objective function
        def objective_function(threshold):
            predictions = (scores >= threshold).astype(int)
            
            if np.sum(predictions) == 0 or np.sum(1 - predictions) == 0:
                return -1.0  # Invalid threshold
            
            # Calculate metrics
            precision = precision_score(binary_labels, predictions, zero_division=0)
            recall = recall_score(binary_labels, predictions, zero_division=0)
            accuracy = accuracy_score(binary_labels, predictions)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Apply constraints
            if self.constraint_metric and self.constraint_value:
                constraint_met = True
                
                if self.constraint_metric == 'precision' and precision < self.constraint_value:
                    constraint_met = False
                elif self.constraint_metric == 'recall' and recall < self.constraint_value:
                    constraint_met = False
                elif self.constraint_metric == 'accuracy' and accuracy < self.constraint_value:
                    constraint_met = False
                
                if not constraint_met:
                    return -1.0
            
            # Return objective value
            if self.objective == 'f1':
                return f1
            elif self.objective == 'precision':
                return precision
            elif self.objective == 'recall':
                return recall
            elif self.objective == 'accuracy':
                return accuracy
            else:
                return f1
        
        # Search for optimal threshold
        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold = 0.5
        best_score = -1.0
        
        for threshold in thresholds:
            score = objective_function(threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store results
        self.optimal_thresholds[class_idx] = best_threshold
        self.threshold_performance[class_idx] = best_score
        
        return best_threshold
    
    def apply_thresholds(self, 
                        y_proba: np.ndarray,
                        thresholds: Optional[Dict[int, float]] = None) -> np.ndarray:
        """
        Apply learned thresholds to make predictions.
        
        Args:
            y_proba: Predicted probabilities
            thresholds: Custom thresholds (uses learned if None)
            
        Returns:
            Threshold-based predictions
        """
        if thresholds is None:
            thresholds = self.optimal_thresholds
        
        if not thresholds:
            # Default to argmax
            return np.argmax(y_proba, axis=1)
        
        predictions = np.zeros(len(y_proba), dtype=int)
        
        # Apply thresholds for each class
        for class_idx, threshold in thresholds.items():
            if class_idx < y_proba.shape[1]:
                class_predictions = y_proba[:, class_idx] >= threshold
                predictions[class_predictions] = class_idx
        
        return predictions


def adaptive_threshold_learning(classifier: BaseEstimator,
                              X_val: np.ndarray,
                              y_val: np.ndarray,
                              objective: str = 'f1') -> Dict[int, float]:
    """
    Learn adaptive thresholds for a classifier.
    
    Args:
        classifier: Trained classifier with predict_proba method
        X_val: Validation features
        y_val: Validation labels
        objective: Objective metric to optimize
        
    Returns:
        Dictionary mapping class indices to optimal thresholds
    """
    # Get predicted probabilities
    y_proba = classifier.predict_proba(X_val)
    
    # Learn thresholds for each class
    learner = AdaptiveThresholdLearner(objective=objective)
    thresholds = {}
    
    unique_classes = np.unique(y_val)
    for class_idx in unique_classes:
        threshold = learner.learn_threshold(y_val, y_proba, class_idx)
        thresholds[class_idx] = threshold
    
    return thresholds


def combine_model_predictions(predictions: List[np.ndarray],
                            weights: Optional[List[float]] = None,
                            method: str = 'weighted_average') -> np.ndarray:
    """
    Combine predictions from multiple models.
    
    Args:
        predictions: List of prediction arrays
        weights: Model weights (uniform if None)
        method: Combination method
        
    Returns:
        Combined predictions
    """
    if not predictions:
        raise ValueError("No predictions provided")
    
    if weights is None:
        weights = [1.0] * len(predictions)
    
    # Normalize weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    if method == 'weighted_average':
        # Weighted average of probabilities
        combined = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            combined += weight * pred
        return combined
    
    elif method == 'majority_vote':
        # Majority voting
        vote_matrix = np.stack([np.argmax(pred, axis=1) for pred in predictions], axis=1)
        combined_classes = stats.mode(vote_matrix, axis=1)[0].flatten()
        
        # Convert back to probabilities
        n_classes = predictions[0].shape[1]
        combined = np.eye(n_classes)[combined_classes]
        return combined
    
    else:
        # Default to simple average
        return np.mean(predictions, axis=0)


def demo_probabilistic_inference():
    """Demonstrate probabilistic inference capabilities."""
    print("🎲 Probabilistic Inference Demonstration")
    print("=" * 50)
    
    # Generate synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    X, y = make_classification(
        n_samples=800,
        n_features=15,
        n_informative=10,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create ensemble of models
    models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        GradientBoostingClassifier(n_estimators=50, random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # Train models
    for model in models:
        model.fit(X_train, y_train)
    
    # Create probabilistic classifier
    prob_classifier = ProbabilisticClassifier(models, uncertainty_threshold=0.25)
    prob_classifier.fit(X_train, y_train)
    
    # Make predictions with uncertainty
    predictions, uncertainty_info = prob_classifier.predict_proba(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate performance
    accuracy = accuracy_score(y_test, pred_classes)
    
    print(f"📊 Probabilistic Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of models: {len(models)}")
    print(f"Uncertain predictions: {uncertainty_info['uncertain_predictions']}")
    print(f"Uncertainty rate: {uncertainty_info['uncertainty_rate']:.2%}")
    
    # Test rejection option
    pred_with_reject, acceptance_mask = prob_classifier.predict_with_rejection(
        X_test, confidence_threshold=0.8
    )
    
    if np.sum(acceptance_mask) > 0:
        accepted_accuracy = accuracy_score(
            y_test[acceptance_mask], 
            pred_with_reject[acceptance_mask]
        )
        coverage = np.mean(acceptance_mask)
        
        print(f"\n🎯 Prediction with Rejection:")
        print(f"Coverage: {coverage:.2%}")
        print(f"Accepted predictions accuracy: {accepted_accuracy:.4f}")
    
    # Learn adaptive thresholds
    print(f"\n🎚️ Learning Adaptive Thresholds...")
    optimal_thresholds = adaptive_threshold_learning(
        prob_classifier, X_test, y_test, objective='f1'
    )
    
    print("Optimal thresholds by class:")
    for class_idx, threshold in optimal_thresholds.items():
        print(f"  Class {class_idx}: {threshold:.3f}")
    
    # Test threshold-based predictions
    learner = AdaptiveThresholdLearner()
    learner.optimal_thresholds = optimal_thresholds
    threshold_predictions = learner.apply_thresholds(predictions)
    threshold_accuracy = accuracy_score(y_test, threshold_predictions)
    
    print(f"Threshold-based accuracy: {threshold_accuracy:.4f}")
    
    print("✅ Probabilistic inference demonstration completed successfully!")
    
    return prob_classifier, uncertainty_info


if __name__ == "__main__":
    demo_probabilistic_inference()
