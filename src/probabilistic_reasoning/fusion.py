"""
Weighted Ensemble Fusion and Learned Fusion Module

Implements probabilistic model fusion methods including:
- Hand-set weighted ensemble (heuristic)
- Learned fusion via stacking meta-model (principled)
- Ensemble uncertainty quantification

Note: The function `bayesian_fusion` is renamed to `weighted_ensemble` for 
clarity, as the original implementation is a heuristic ensemble, not true 
Bayesian model averaging. The `bayesian_fusion` name is kept as an alias 
for backward compatibility.

Reference:
    Wolpert (1992) "Stacked Generalization"
    Breiman (1996) "Stacked Regressions"
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def weighted_ensemble(
    prob_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = 'weighted_average'
) -> np.ndarray:
    """
    Fuse predictions from multiple models using weighted ensemble.
    
    NOTE: This is a heuristic ensemble method, not true Bayesian model averaging.
    For principled learned fusion, use the LearnedFusion class.
    
    Combines probabilistic predictions from heterogeneous models
    (classical, neural, quantum) into a unified probability distribution.
    
    Args:
        prob_list: List of probability arrays (n_samples, n_classes) from each model
        weights: Optional weights for each model (defaults to uniform)
        method: Fusion method ('weighted_average', 'product', 'geometric_mean')
        
    Returns:
        Fused probabilities (n_samples, n_classes)
        
    Examples:
        >>> rf_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        >>> nn_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> fused = weighted_ensemble([rf_probs, nn_probs])
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


# Backward compatibility alias
def bayesian_fusion(
    prob_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = 'weighted_average'
) -> np.ndarray:
    """
    Alias for weighted_ensemble() - kept for backward compatibility.
    
    NOTE: This function name is misleading as this is NOT true Bayesian
    model averaging. Consider using weighted_ensemble() or LearnedFusion instead.
    
    See weighted_ensemble() for full documentation.
    """
    return weighted_ensemble(prob_list, weights, method)


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


class LearnedFusion:
    """
    Meta-learner that learns fusion weights from validation predictions.
    
    Uses logistic regression stacking to learn optimal combination of
    base model predictions. This is the PRINCIPLED alternative to hand-set
    accuracy-based weights.
    
    The learned weights are interpretable as model importance scores.
    
    Reference:
        Wolpert (1992) "Stacked Generalization"
        
    Attributes:
        meta_model: Trained logistic regression meta-learner
        weights: Learned importance weights for each base model
        scaler: Standard scaler for input features
        
    Example:
        >>> # Train base models on training data
        >>> model1_probs = model1.predict_proba(X_val)
        >>> model2_probs = model2.predict_proba(X_val)
        >>> 
        >>> # Learn fusion weights on validation data
        >>> fusion = LearnedFusion()
        >>> fusion.fit([model1_probs, model2_probs], y_val)
        >>> print(f"Learned weights: {fusion.get_weights()}")
        >>> 
        >>> # Apply to test data
        >>> test_probs = fusion.transform([model1_test, model2_test])
    """
    
    def __init__(
        self,
        regularization: float = 1.0,
        use_logits: bool = True,
        calibrate_output: bool = False
    ):
        """
        Initialize LearnedFusion.
        
        Args:
            regularization: L2 regularization strength (higher = more regularization)
            use_logits: If True, convert probabilities to logits before stacking
            calibrate_output: If True, calibrate output probabilities
        """
        self.regularization = regularization
        self.use_logits = use_logits
        self.calibrate_output = calibrate_output
        
        self.meta_model = None
        self.scaler = None
        self.n_models = None
        self.n_classes = None
        self.is_fitted = False
        
    def _probs_to_features(self, prob_list: List[np.ndarray]) -> np.ndarray:
        """Convert list of probability arrays to feature matrix for meta-learner."""
        # Concatenate all model predictions
        # Shape: (n_samples, n_models * n_classes)
        features = np.hstack(prob_list)
        
        if self.use_logits:
            # Convert to logits for better linear separability
            epsilon = 1e-10
            features = np.clip(features, epsilon, 1 - epsilon)
            features = np.log(features / (1 - features))
        
        return features
    
    def fit(
        self,
        val_probs: List[np.ndarray],
        val_labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'LearnedFusion':
        """
        Fit meta-learner on validation predictions.
        
        Args:
            val_probs: List of probability arrays from base models
                       Each array shape: (n_samples, n_classes)
            val_labels: True labels for validation data
            sample_weight: Optional sample weights
            
        Returns:
            self: Fitted fusion model
        """
        self.n_models = len(val_probs)
        self.n_classes = val_probs[0].shape[1] if val_probs[0].ndim > 1 else 2
        
        # Create feature matrix
        X = self._probs_to_features(val_probs)
        y = np.asarray(val_labels)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit logistic regression
        self.meta_model = LogisticRegression(
            C=1.0 / self.regularization,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial' if self.n_classes > 2 else 'auto',
            random_state=42
        )
        
        self.meta_model.fit(X_scaled, y, sample_weight=sample_weight)
        
        self.is_fitted = True
        logger.info(f"LearnedFusion fitted on {self.n_models} models, "
                   f"{len(val_labels)} samples")
        
        return self
    
    def transform(self, prob_list: List[np.ndarray]) -> np.ndarray:
        """
        Apply learned fusion to new predictions.
        
        Args:
            prob_list: List of probability arrays from base models
            
        Returns:
            fused_probs: Fused probability predictions (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("LearnedFusion must be fitted before transform")
        
        if len(prob_list) != self.n_models:
            raise ValueError(f"Expected {self.n_models} models, got {len(prob_list)}")
        
        # Create feature matrix
        X = self._probs_to_features(prob_list)
        X_scaled = self.scaler.transform(X)
        
        # Get meta-model predictions
        fused_probs = self.meta_model.predict_proba(X_scaled)
        
        return fused_probs
    
    def fit_transform(
        self,
        val_probs: List[np.ndarray],
        val_labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(val_probs, val_labels)
        return self.transform(val_probs)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get learned model weights/importance scores.
        
        Returns:
            weights: Dictionary with 'per_class' weights [n_classes, n_models]
                     and 'aggregate' weights [n_models]
        """
        if not self.is_fitted:
            raise RuntimeError("LearnedFusion must be fitted before getting weights")
        
        # Extract weights from logistic regression coefficients
        # Coefficients shape: (n_classes, n_features) where n_features = n_models * n_classes
        coefs = self.meta_model.coef_
        
        if coefs.ndim == 1:
            coefs = coefs.reshape(1, -1)
        
        # Reshape to (n_classes, n_models, n_classes_orig)
        # Then aggregate across output classes
        n_features_per_model = self.n_classes
        
        per_model_importance = []
        for m in range(self.n_models):
            start_idx = m * n_features_per_model
            end_idx = start_idx + n_features_per_model
            model_coefs = coefs[:, start_idx:end_idx]
            # Importance = mean absolute coefficient across classes
            importance = np.abs(model_coefs).mean()
            per_model_importance.append(importance)
        
        per_model_importance = np.array(per_model_importance)
        # Normalize to sum to 1
        normalized_weights = per_model_importance / per_model_importance.sum()
        
        return {
            'aggregate': normalized_weights,
            'raw_coefficients': coefs,
            'model_importance': per_model_importance
        }
    
    def get_model_ranking(self) -> List[Tuple[int, float]]:
        """
        Get models ranked by learned importance.
        
        Returns:
            ranking: List of (model_idx, weight) tuples, sorted by importance
        """
        weights = self.get_weights()['aggregate']
        ranking = [(i, w) for i, w in enumerate(weights)]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking


class BayesianFusion:
    """
    Object-oriented wrapper for ensemble fusion methods.
    
    Provides a unified interface for:
    - Heuristic weighted ensemble (hand-set weights)
    - Learned fusion (stacking meta-learner)
    
    This class is provided for convenience and compatibility with existing code.
    For new code, consider using LearnedFusion directly for principled fusion.
    """
    
    def __init__(
        self,
        num_models: int = 3,
        fusion_method: str = 'weighted_average',
        learn_weights: bool = False
    ):
        """
        Initialize Bayesian Fusion.
        
        Args:
            num_models: Number of base models
            fusion_method: 'weighted_average', 'product', 'geometric_mean', or 'learned'
            learn_weights: If True, use LearnedFusion instead of hand-set weights
        """
        self.num_models = num_models
        self.fusion_method = fusion_method
        self.learn_weights = learn_weights or (fusion_method == 'learned')
        
        self.weights = np.ones(num_models) / num_models  # Uniform default
        self.learned_fusion = None
        self.is_fitted = False
    
    def set_weights(self, weights: List[float]) -> None:
        """
        Set fusion weights manually.
        
        Args:
            weights: List of weights for each model
        """
        if len(weights) != self.num_models:
            raise ValueError(f"Expected {self.num_models} weights, got {len(weights)}")
        
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()
        logger.info(f"Set weights: {self.weights}")
    
    def set_weights_from_accuracy(self, accuracies: List[float]) -> None:
        """
        Set weights proportional to model accuracies.
        
        NOTE: This is a heuristic. For principled weight learning,
        use fit() with learn_weights=True.
        
        Args:
            accuracies: Accuracy of each model
        """
        logger.warning("Setting weights from accuracy is heuristic, not principled. "
                      "Consider using fit() with learn_weights=True.")
        self.set_weights(accuracies)
    
    def fit(
        self,
        val_probs: List[np.ndarray],
        val_labels: np.ndarray
    ) -> 'BayesianFusion':
        """
        Fit fusion weights on validation data.
        
        If learn_weights=True, uses LearnedFusion (stacking).
        Otherwise, sets uniform weights.
        
        Args:
            val_probs: Validation predictions from each model
            val_labels: True validation labels
            
        Returns:
            self: Fitted fusion model
        """
        if self.learn_weights:
            self.learned_fusion = LearnedFusion()
            self.learned_fusion.fit(val_probs, val_labels)
            
            # Update weights from learned model
            weights_info = self.learned_fusion.get_weights()
            self.weights = weights_info['aggregate']
            
            logger.info(f"Learned weights: {self.weights}")
        else:
            # Keep uniform weights
            logger.info("Using uniform weights (learn_weights=False)")
        
        self.is_fitted = True
        return self
    
    def weighted_average(
        self,
        prob_list: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute weighted average of predictions.
        
        Args:
            prob_list: Predictions from each model
            weights: Optional weights (uses self.weights if None)
            
        Returns:
            fused: Fused predictions
        """
        if weights is None:
            weights = self.weights.tolist()
        
        if self.learn_weights and self.is_fitted and self.learned_fusion is not None:
            return self.learned_fusion.transform(prob_list)
        
        return weighted_ensemble(prob_list, weights, 'weighted_average')
    
    def fuse(self, prob_list: List[np.ndarray]) -> np.ndarray:
        """
        Apply fusion to predictions using configured method.
        
        Args:
            prob_list: Predictions from each model
            
        Returns:
            fused: Fused predictions
        """
        if self.learn_weights and self.is_fitted and self.learned_fusion is not None:
            return self.learned_fusion.transform(prob_list)
        
        return weighted_ensemble(prob_list, self.weights.tolist(), self.fusion_method)
    
    def get_weights(self) -> np.ndarray:
        """Get current fusion weights."""
        return self.weights.copy()
