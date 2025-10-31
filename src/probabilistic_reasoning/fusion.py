"""
Bayesian Fusion and Ensemble Uncertainty Module

Implements probabilistic model fusion using Bayesian principles
and ensemble uncertainty quantification.
"""

import numpy as np
from typing import List, Optional


def bayesian_fusion(
    prob_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = 'weighted_average'
) -> np.ndarray:
    """
    Fuse predictions from multiple models using Bayesian principles.
    
    Combines probabilistic predictions from heterogeneous models
    (classical, neural, quantum) into a unified posterior distribution.
    
    Args:
        prob_list: List of probability arrays (n_samples, n_classes) from each model
        weights: Optional weights for each model (defaults to uniform)
        method: Fusion method ('weighted_average', 'product', 'geometric_mean')
        
    Returns:
        Fused probabilities (n_samples, n_classes)
        
    Examples:
        >>> rf_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        >>> nn_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> fused = bayesian_fusion([rf_probs, nn_probs])
    """
    if not prob_list:
        raise ValueError("prob_list cannot be empty")
    
    n_models = len(prob_list)
    
    # Validate all models have same shape
    shape = prob_list[0].shape
    for probs in prob_list[1:]:
        if probs.shape != shape:
            raise ValueError(f"All probability arrays must have same shape. Got {probs.shape} vs {shape}")
    
    # Default to uniform weights
    if weights is None:
        weights = [1.0 / n_models] * n_models
    
    if len(weights) != n_models:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({n_models})")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    if method == 'weighted_average':
        # Weighted linear combination
        stacked = np.stack(prob_list, axis=0)  # (n_models, n_samples, n_classes)
        fused = np.tensordot(weights, stacked, axes=(0, 0))
        
    elif method == 'product':
        # Product of experts (Bayesian product rule)
        fused = np.ones_like(prob_list[0])
        for w, probs in zip(weights, prob_list):
            fused *= probs ** w
        # Normalize
        fused = fused / fused.sum(axis=1, keepdims=True)
        
    elif method == 'geometric_mean':
        # Geometric mean of probabilities
        fused = np.ones_like(prob_list[0])
        for probs in prob_list:
            fused *= probs ** (1.0 / n_models)
        # Normalize
        fused = fused / fused.sum(axis=1, keepdims=True)
        
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return fused


def ensemble_uncertainty(
    prob_list: List[np.ndarray],
    return_components: bool = False
) -> np.ndarray:
    """
    Compute ensemble uncertainty using mutual information.
    
    Decomposes total uncertainty into:
    - Aleatoric (data) uncertainty: average entropy of individual predictions
    - Epistemic (model) uncertainty: entropy of average prediction
    
    Args:
        prob_list: List of probability arrays from each model
        return_components: If True, return (total, aleatoric, epistemic)
        
    Returns:
        Total uncertainty (n_samples,) or tuple of uncertainties
        
    Reference:
        Depeweg et al. (2018) "Decomposition of Uncertainty in Bayesian Deep Learning"
    """
    if not prob_list:
        raise ValueError("prob_list cannot be empty")
    
    # Stack predictions
    stacked = np.stack(prob_list, axis=0)  # (n_models, n_samples, n_classes)
    
    # Average prediction (consensus)
    mean_probs = stacked.mean(axis=0)  # (n_samples, n_classes)
    
    # Total uncertainty: entropy of average prediction
    total_uncertainty = entropy(mean_probs)
    
    if return_components:
        # Aleatoric: average of individual entropies
        individual_entropies = np.array([entropy(p) for p in stacked])
        aleatoric = individual_entropies.mean(axis=0)
        
        # Epistemic: total - aleatoric
        epistemic = total_uncertainty - aleatoric
        
        return total_uncertainty, aleatoric, epistemic
    
    return total_uncertainty


def entropy(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute Shannon entropy of probability distributions.
    
    Args:
        probs: Probability array (n_samples, n_classes)
        epsilon: Small constant to avoid log(0)
        
    Returns:
        Entropy values (n_samples,)
    """
    # Clip to avoid numerical issues
    probs_clipped = np.clip(probs, epsilon, 1.0)
    
    # H(p) = -sum(p * log(p))
    return -np.sum(probs_clipped * np.log(probs_clipped), axis=1)


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Compute predictive entropy (total uncertainty).
    
    Alias for entropy() with clearer naming for prediction context.
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        
    Returns:
        Predictive entropy (n_samples,)
    """
    return entropy(probs)


def disagreement_score(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute model disagreement as variance of predicted class.
    
    Args:
        prob_list: List of probability arrays from each model
        
    Returns:
        Disagreement scores (n_samples,)
    """
    # Get predicted classes from each model
    predictions = np.array([probs.argmax(axis=1) for probs in prob_list])
    
    # Compute variance in predictions (0 = full agreement, higher = more disagreement)
    disagreement = predictions.var(axis=0)
    
    return disagreement


def confidence_interval(
    prob_list: List[np.ndarray],
    confidence: float = 0.95
) -> tuple:
    """
    Compute confidence intervals for ensemble predictions.
    
    Args:
        prob_list: List of probability arrays from each model
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, mean, upper_bound) arrays
    """
    stacked = np.stack(prob_list, axis=0)
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean = stacked.mean(axis=0)
    lower = np.percentile(stacked, lower_percentile, axis=0)
    upper = np.percentile(stacked, upper_percentile, axis=0)
    
    return lower, mean, upper
