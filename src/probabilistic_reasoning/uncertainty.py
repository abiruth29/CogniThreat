"""
Uncertainty Quantification Module

Implements Monte Carlo Dropout and predictive uncertainty estimation
for deep learning models.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation in neural networks.
    
    Performs multiple stochastic forward passes with dropout enabled
    to approximate Bayesian inference and quantify epistemic uncertainty.
    
    Reference:
        Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
    """
    
    def __init__(self, model: tf.keras.Model, n_iterations: int = 30):
        """
        Initialize MC Dropout predictor.
        
        Args:
            model: Keras model with Dropout layers
            n_iterations: Number of stochastic forward passes
        """
        self.model = model
        self.n_iterations = n_iterations
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray,
        return_all: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates using MC Dropout.
        
        Args:
            X: Input data (n_samples, ...)
            return_all: If True, return all predictions
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
            If return_all=True: (mean, std, all_predictions)
        """
        predictions = []
        
        # Perform multiple forward passes with dropout enabled
        for _ in range(self.n_iterations):
            # Set training=True to enable dropout during inference
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # (n_iterations, n_samples, n_classes)
        
        # Compute statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Uncertainty as predictive entropy of mean
        uncertainty = predictive_entropy(mean_pred)
        
        if return_all:
            return mean_pred, uncertainty, predictions
        
        return mean_pred, uncertainty
    
    def epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Compute epistemic (model) uncertainty.
        
        Args:
            X: Input data
            
        Returns:
            Epistemic uncertainty (n_samples,)
        """
        predictions = []
        
        for _ in range(self.n_iterations):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Variance across predictions (model uncertainty)
        variance = predictions.var(axis=0).mean(axis=1)
        
        return variance
    
    def confidence_intervals(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for predictions.
        
        Args:
            X: Input data
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, mean, upper_bound)
        """
        predictions = []
        
        for _ in range(self.n_iterations):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        mean = predictions.mean(axis=0)
        lower = np.percentile(predictions, lower_percentile, axis=0)
        upper = np.percentile(predictions, upper_percentile, axis=0)
        
        return lower, mean, upper


def predictive_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute predictive entropy (total uncertainty).
    
    H(y|x) = -sum_i p(y_i|x) log p(y_i|x)
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        epsilon: Small constant to avoid log(0)
        
    Returns:
        Predictive entropy (n_samples,)
    """
    probs_clipped = np.clip(probs, epsilon, 1.0)
    return -np.sum(probs_clipped * np.log(probs_clipped), axis=1)


def mutual_information(predictions: np.ndarray) -> np.ndarray:
    """
    Compute mutual information (epistemic uncertainty).
    
    I(y; θ|x) = H(E[p(y|x,θ)]) - E[H(p(y|x,θ))]
    
    Args:
        predictions: Array of predictions (n_iterations, n_samples, n_classes)
        
    Returns:
        Mutual information (n_samples,)
    """
    # Entropy of mean prediction (total uncertainty)
    mean_probs = predictions.mean(axis=0)
    total_entropy = predictive_entropy(mean_probs)
    
    # Mean of individual entropies (aleatoric uncertainty)
    individual_entropies = np.array([predictive_entropy(p) for p in predictions])
    mean_entropy = individual_entropies.mean(axis=0)
    
    # Mutual information (epistemic uncertainty)
    mi = total_entropy - mean_entropy
    
    return mi


def variation_ratio(predictions: np.ndarray) -> np.ndarray:
    """
    Compute variation ratio (fraction of disagreeing predictions).
    
    VR = 1 - (count of modal class / total predictions)
    
    Args:
        predictions: Array of predictions (n_iterations, n_samples, n_classes)
        
    Returns:
        Variation ratio (n_samples,)
    """
    # Get class with highest probability for each prediction
    predicted_classes = predictions.argmax(axis=2)  # (n_iterations, n_samples)
    
    # Find mode for each sample
    modes = []
    for i in range(predicted_classes.shape[1]):
        mode = np.bincount(predicted_classes[:, i]).max()
        modes.append(mode)
    
    modes = np.array(modes)
    n_iterations = predictions.shape[0]
    
    # Variation ratio
    vr = 1.0 - (modes / n_iterations)
    
    return vr


def expected_pairwise_kl(predictions: np.ndarray) -> np.ndarray:
    """
    Compute expected pairwise KL divergence.
    
    Measures disagreement between pairs of predictions.
    
    Args:
        predictions: Array of predictions (n_iterations, n_samples, n_classes)
        
    Returns:
        Expected KL divergence (n_samples,)
    """
    n_iterations = predictions.shape[0]
    n_samples = predictions.shape[1]
    
    kl_divs = np.zeros(n_samples)
    
    epsilon = 1e-10
    
    for i in range(n_samples):
        total_kl = 0.0
        count = 0
        
        for j in range(n_iterations):
            for k in range(j + 1, n_iterations):
                p = predictions[j, i] + epsilon
                q = predictions[k, i] + epsilon
                
                # KL(p || q)
                kl = np.sum(p * np.log(p / q))
                total_kl += kl
                count += 1
        
        kl_divs[i] = total_kl / count if count > 0 else 0.0
    
    return kl_divs


class EnsembleUncertaintyEstimator:
    """
    Estimate uncertainty from ensemble of models.
    """
    
    def __init__(self, models: List[tf.keras.Model]):
        """
        Initialize ensemble estimator.
        
        Args:
            models: List of trained models
        """
        self.models = models
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (mean_pred, aleatoric, epistemic)
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Mean prediction
        mean_pred = predictions.mean(axis=0)
        
        # Total uncertainty
        total = predictive_entropy(mean_pred)
        
        # Aleatoric (average entropy)
        individual_entropies = np.array([predictive_entropy(p) for p in predictions])
        aleatoric = individual_entropies.mean(axis=0)
        
        # Epistemic (difference)
        epistemic = total - aleatoric
        
        return mean_pred, aleatoric, epistemic
