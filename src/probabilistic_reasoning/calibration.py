"""
Probability Calibration Module

Implements calibration metrics and post-hoc calibration methods for
ensuring well-calibrated probability outputs from classifiers.

Key Components:
- Expected Calibration Error (ECE): Measures calibration quality
- Brier Score: Proper scoring rule for probabilistic predictions
- Temperature Scaling: Single-parameter post-hoc calibration
- Isotonic Regression: Non-parametric calibration method
- Reliability Diagrams: Visualization of calibration

Reference:
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles"
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> Tuple[float, dict]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy,
    averaged across confidence bins.
    
    ECE = Σ (|B_m| / n) × |acc(B_m) - conf(B_m)|
    
    Args:
        y_true: True labels (n_samples,) - binary or multi-class
        y_prob: Predicted probabilities (n_samples,) for binary or 
                (n_samples, n_classes) for multi-class
        n_bins: Number of bins for confidence grouping
        strategy: 'uniform' for equal-width bins, 'quantile' for equal-size bins
        
    Returns:
        ece: Expected Calibration Error (lower is better)
        details: Dictionary with per-bin accuracy and confidence
        
    Example:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
        >>> ece, details = expected_calibration_error(y_true, y_prob)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Handle multi-class case: use maximum predicted probability
    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        # Binary case: confidence is prob if pred=1, else 1-prob
        confidences = y_prob
        predictions = (y_prob >= 0.5).astype(int)
    
    # Compute accuracy per sample
    accuracies = (predictions == y_true).astype(float)
    
    # Create bins
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_boundaries = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
        bin_boundaries = np.unique(bin_boundaries)
    
    ece = 0.0
    n_samples = len(y_true)
    bin_details = []
    
    for i in range(len(bin_boundaries) - 1):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # Get samples in bin
        if i == len(bin_boundaries) - 2:
            # Last bin includes upper boundary
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            # Weighted contribution to ECE
            ece += (bin_size / n_samples) * np.abs(bin_accuracy - bin_confidence)
            
            bin_details.append({
                'bin_idx': i,
                'lower': lower,
                'upper': upper,
                'count': int(bin_size),
                'accuracy': float(bin_accuracy),
                'confidence': float(bin_confidence),
                'gap': float(np.abs(bin_accuracy - bin_confidence))
            })
    
    details = {
        'ece': float(ece),
        'n_bins': n_bins,
        'n_samples': n_samples,
        'bins': bin_details
    }
    
    return float(ece), details


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum gap between accuracy and confidence across bins.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        mce: Maximum Calibration Error
    """
    _, details = expected_calibration_error(y_true, y_prob, n_bins)
    
    if not details['bins']:
        return 0.0
    
    gaps = [b['gap'] for b in details['bins']]
    return max(gaps)


def brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute Brier Score for probabilistic predictions.
    
    Brier Score = (1/n) × Σ (p_i - y_i)²
    
    Lower is better. Perfect predictions = 0, random = 0.25 for binary.
    
    Args:
        y_true: True labels (binary: 0/1)
        y_prob: Predicted probabilities for positive class
        sample_weight: Optional sample weights
        
    Returns:
        brier: Brier score
        
    Example:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
        >>> bs = brier_score(y_true, y_prob)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Handle multi-class: compute score for each class and average
    if y_prob.ndim == 2:
        n_classes = y_prob.shape[1]
        scores = []
        
        for c in range(n_classes):
            y_true_c = (y_true == c).astype(float)
            y_prob_c = y_prob[:, c]
            scores.append(np.mean((y_prob_c - y_true_c) ** 2))
        
        return float(np.mean(scores))
    
    # Binary case
    squared_errors = (y_prob - y_true) ** 2
    
    if sample_weight is not None:
        return float(np.average(squared_errors, weights=sample_weight))
    
    return float(np.mean(squared_errors))


def brier_skill_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    reference_prob: Optional[np.ndarray] = None
) -> float:
    """
    Compute Brier Skill Score (BSS).
    
    BSS = 1 - (BS / BS_ref)
    
    BSS = 1 indicates perfect predictions, 0 indicates no skill vs reference.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        reference_prob: Reference probabilities (default: climatological)
        
    Returns:
        bss: Brier Skill Score
    """
    bs = brier_score(y_true, y_prob)
    
    if reference_prob is None:
        # Use climatological reference (base rate)
        base_rate = np.mean(y_true)
        reference_prob = np.full_like(y_prob, base_rate)
    
    bs_ref = brier_score(y_true, reference_prob)
    
    if bs_ref == 0:
        return 1.0 if bs == 0 else 0.0
    
    return float(1 - (bs / bs_ref))


class TemperatureScaling:
    """
    Temperature Scaling for post-hoc probability calibration.
    
    Learns a single temperature parameter T on validation data that scales
    the logits before softmax: p = softmax(z / T)
    
    T > 1: Softer probabilities (less confident)
    T < 1: Sharper probabilities (more confident)
    T = 1: No change
    
    Reference:
        Guo et al. (2017) "On Calibration of Modern Neural Networks"
        
    Attributes:
        temperature (float): Learned temperature parameter
        
    Example:
        >>> ts = TemperatureScaling()
        >>> ts.fit(val_probs, val_labels)
        >>> calibrated = ts.transform(test_probs)
    """
    
    def __init__(self, init_temperature: float = 1.0):
        """
        Initialize Temperature Scaling calibrator.
        
        Args:
            init_temperature: Initial temperature value
        """
        self.temperature = init_temperature
        self.is_fitted = False
        
    def _softmax_with_temperature(
        self,
        probs: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        # Convert probabilities to logits
        epsilon = 1e-10
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        
        if probs.ndim == 1:
            # Binary: convert to 2-class format
            probs_clipped = np.column_stack([1 - probs_clipped, probs_clipped])
        
        logits = np.log(probs_clipped)
        
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        if probs.ndim == 1:
            # Return to binary format
            return calibrated[:, 1]
        
        return calibrated
    
    def _nll_loss(self, temperature: float, probs: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood loss for temperature optimization."""
        calibrated = self._softmax_with_temperature(probs, temperature)
        
        epsilon = 1e-10
        calibrated = np.clip(calibrated, epsilon, 1 - epsilon)
        
        if calibrated.ndim == 1:
            # Binary case
            nll = -np.mean(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
        else:
            # Multi-class
            n_samples = len(labels)
            nll = -np.mean(np.log(calibrated[np.arange(n_samples), labels.astype(int)]))
        
        return nll
    
    def fit(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        method: str = 'nll'
    ) -> 'TemperatureScaling':
        """
        Fit temperature parameter on validation data.
        
        Args:
            val_probs: Validation probabilities (n_samples,) or (n_samples, n_classes)
            val_labels: Validation labels
            method: 'nll' for negative log-likelihood, 'ece' for ECE minimization
            
        Returns:
            self: Fitted calibrator
        """
        val_probs = np.asarray(val_probs)
        val_labels = np.asarray(val_labels)
        
        if method == 'nll':
            loss_fn = lambda t: self._nll_loss(t, val_probs, val_labels)
        else:  # ece
            loss_fn = lambda t: expected_calibration_error(
                val_labels, 
                self._softmax_with_temperature(val_probs, t)
            )[0]
        
        # Optimize temperature
        result = minimize_scalar(
            loss_fn,
            bounds=(0.1, 10.0),
            method='bounded',
            options={'maxiter': 100}
        )
        
        self.temperature = result.x
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted: T = {self.temperature:.4f}")
        
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Args:
            probs: Input probabilities
            
        Returns:
            calibrated: Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("TemperatureScaling not fitted, returning original probabilities")
            return probs
        
        return self._softmax_with_temperature(probs, self.temperature)
    
    def fit_transform(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray
    ) -> np.ndarray:
        """Fit on validation data and transform."""
        self.fit(val_probs, val_labels)
        return self.transform(val_probs)
    
    def get_temperature(self) -> float:
        """Get the learned temperature parameter."""
        return self.temperature


class IsotonicCalibration:
    """
    Isotonic Regression for non-parametric probability calibration.
    
    Fits a monotonic function that maps predicted probabilities to
    calibrated probabilities.
    
    More flexible than temperature scaling but may overfit with small
    validation sets.
    
    Reference:
        Zadrozny & Elkan (2002) "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
    """
    
    def __init__(self):
        """Initialize Isotonic Calibration."""
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray
    ) -> 'IsotonicCalibration':
        """
        Fit isotonic regression on validation data.
        
        Args:
            val_probs: Validation probabilities
            val_labels: Validation labels
            
        Returns:
            self: Fitted calibrator
        """
        val_probs = np.asarray(val_probs)
        val_labels = np.asarray(val_labels)
        
        if val_probs.ndim == 1:
            # Binary case
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(val_probs, val_labels)
            self.calibrators[1] = ir
        else:
            # Multi-class: fit one calibrator per class
            n_classes = val_probs.shape[1]
            for c in range(n_classes):
                y_c = (val_labels == c).astype(float)
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(val_probs[:, c], y_c)
                self.calibrators[c] = ir
        
        self.is_fitted = True
        logger.info(f"Isotonic calibration fitted for {len(self.calibrators)} class(es)")
        
        return self
    
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration to probabilities.
        
        Args:
            probs: Input probabilities
            
        Returns:
            calibrated: Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("IsotonicCalibration not fitted, returning original probabilities")
            return probs
        
        probs = np.asarray(probs)
        
        if probs.ndim == 1:
            return self.calibrators[1].transform(probs)
        
        # Multi-class
        calibrated = np.zeros_like(probs)
        for c, ir in self.calibrators.items():
            calibrated[:, c] = ir.transform(probs[:, c])
        
        # Renormalize
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def fit_transform(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(val_probs, val_labels)
        return self.transform(val_probs)


def get_reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Get data for plotting a reliability diagram.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        data: Dictionary with plotting data
    """
    _, details = expected_calibration_error(y_true, y_prob, n_bins)
    
    bin_means = []
    bin_accuracies = []
    bin_counts = []
    
    for b in details['bins']:
        bin_means.append((b['lower'] + b['upper']) / 2)
        bin_accuracies.append(b['accuracy'])
        bin_counts.append(b['count'])
    
    return {
        'bin_means': bin_means,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'ece': details['ece'],
        'n_samples': details['n_samples']
    }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> dict:
    """
    Generate comprehensive calibration report.
    
    Args:
        y_true: True labels
        y_prob: Original predicted probabilities
        y_prob_calibrated: Optional calibrated probabilities
        n_bins: Number of bins for ECE
        
    Returns:
        report: Dictionary with calibration metrics
    """
    # Original metrics
    ece_orig, _ = expected_calibration_error(y_true, y_prob, n_bins)
    mce_orig = maximum_calibration_error(y_true, y_prob, n_bins)
    brier_orig = brier_score(y_true, y_prob)
    
    report = {
        'original': {
            'ece': ece_orig,
            'mce': mce_orig,
            'brier_score': brier_orig
        }
    }
    
    if y_prob_calibrated is not None:
        ece_cal, _ = expected_calibration_error(y_true, y_prob_calibrated, n_bins)
        mce_cal = maximum_calibration_error(y_true, y_prob_calibrated, n_bins)
        brier_cal = brier_score(y_true, y_prob_calibrated)
        
        report['calibrated'] = {
            'ece': ece_cal,
            'mce': mce_cal,
            'brier_score': brier_cal
        }
        
        report['improvement'] = {
            'ece_reduction': (ece_orig - ece_cal) / ece_orig * 100 if ece_orig > 0 else 0,
            'mce_reduction': (mce_orig - mce_cal) / mce_orig * 100 if mce_orig > 0 else 0,
            'brier_reduction': (brier_orig - brier_cal) / brier_orig * 100 if brier_orig > 0 else 0
        }
    
    return report
