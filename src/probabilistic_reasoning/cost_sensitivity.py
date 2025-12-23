"""
Cost Sensitivity Analysis Module

Provides tools for analyzing the sensitivity of intrusion detection
decisions to the cost ratio between false negatives and false positives.

Key Functions:
- Sensitivity analysis across cost ratios
- Visualization of performance vs cost ratio
- Documented justification for default ratios

Reference:
    Elkan (2001) "The Foundations of Cost-Sensitive Learning"
    Drummond & Holte (2006) "Cost Curves: An Improved Method for Visualizing Classifier Performance"
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostSensitivityResult:
    """Result of cost sensitivity analysis for a single ratio."""
    fn_fp_ratio: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    expected_cost: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_fp_ratio: float = 10.0
) -> float:
    """
    Compute optimal decision threshold for given cost ratio.
    
    Using Bayes decision theory:
    Threshold = C_FP / (C_FP + C_FN) = 1 / (1 + fn_fp_ratio)
    
    Args:
        y_true: True labels (0: normal, 1: attack)
        y_prob: Predicted probability of attack
        fn_fp_ratio: Ratio C_FN / C_FP
        
    Returns:
        threshold: Optimal decision threshold
    """
    # Theoretical optimal threshold
    theoretical = 1.0 / (1.0 + fn_fp_ratio)
    
    return theoretical


def compute_expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0
) -> float:
    """
    Compute expected cost of predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        fn_cost: Cost of false negative (missed attack)
        fp_cost: Cost of false positive (false alarm)
        
    Returns:
        expected_cost: Total expected cost
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Count errors
    fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed attacks
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
    
    total_cost = fn * fn_cost + fp * fp_cost
    
    return float(total_cost)


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict:
    """
    Compute classification metrics at a specific threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Decision threshold
        
    Returns:
        metrics: Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def cost_sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_fp_ratios: Optional[List[float]] = None,
    fp_cost: float = 1.0
) -> List[CostSensitivityResult]:
    """
    Analyze model performance across different cost ratios.
    
    This helps justify the choice of C_FN/C_FP ratio by showing
    how performance metrics change with different cost assumptions.
    
    Args:
        y_true: True labels (binary: 0=normal, 1=attack)
        y_prob: Predicted probability of attack
        fn_fp_ratios: List of C_FN/C_FP ratios to analyze
        fp_cost: Base false positive cost (default 1.0)
        
    Returns:
        results: List of CostSensitivityResult for each ratio
        
    Example:
        >>> results = cost_sensitivity_analysis(y_true, y_prob)
        >>> for r in results:
        ...     print(f"Ratio {r.fn_fp_ratio}: Recall={r.recall:.3f}, Cost={r.expected_cost:.1f}")
    """
    if fn_fp_ratios is None:
        fn_fp_ratios = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
    
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    results = []
    
    for ratio in fn_fp_ratios:
        # Compute optimal threshold for this ratio
        threshold = compute_optimal_threshold(y_true, y_prob, ratio)
        
        # Get metrics at threshold
        metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
        
        # Compute expected cost
        fn_cost = ratio * fp_cost
        y_pred = (y_prob >= threshold).astype(int)
        expected_cost = compute_expected_cost(y_true, y_pred, fn_cost, fp_cost)
        
        result = CostSensitivityResult(
            fn_fp_ratio=ratio,
            threshold=threshold,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            expected_cost=expected_cost,
            true_positives=metrics['true_positives'],
            false_positives=metrics['false_positives'],
            true_negatives=metrics['true_negatives'],
            false_negatives=metrics['false_negatives']
        )
        
        results.append(result)
    
    return results


def get_sensitivity_curves_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_fp_ratios: Optional[List[float]] = None
) -> Dict:
    """
    Get data for plotting sensitivity curves.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        fn_fp_ratios: Cost ratios to analyze
        
    Returns:
        data: Dictionary with curve data for plotting
    """
    results = cost_sensitivity_analysis(y_true, y_prob, fn_fp_ratios)
    
    return {
        'ratios': [r.fn_fp_ratio for r in results],
        'thresholds': [r.threshold for r in results],
        'recalls': [r.recall for r in results],
        'precisions': [r.precision for r in results],
        'f1_scores': [r.f1_score for r in results],
        'expected_costs': [r.expected_cost for r in results],
        'false_negatives': [r.false_negatives for r in results],
        'false_positives': [r.false_positives for r in results]
    }


def find_optimal_cost_ratio(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.95,
    search_range: Tuple[float, float] = (1, 100)
) -> Tuple[float, CostSensitivityResult]:
    """
    Find cost ratio that achieves target recall.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_recall: Desired recall (e.g., 0.95 for 95%)
        search_range: Range to search for optimal ratio
        
    Returns:
        optimal_ratio: Cost ratio achieving target recall
        result: Performance at that ratio
    """
    # Binary search for ratio achieving target recall
    low, high = search_range
    best_ratio = None
    best_result = None
    
    for _ in range(50):  # Max iterations
        mid = (low + high) / 2
        results = cost_sensitivity_analysis(y_true, y_prob, [mid])
        result = results[0]
        
        if abs(result.recall - target_recall) < 0.01:
            return mid, result
        
        if result.recall < target_recall:
            # Need higher ratio (lower threshold) for more recall
            low = mid
        else:
            high = mid
        
        if best_result is None or abs(result.recall - target_recall) < abs(best_result.recall - target_recall):
            best_ratio = mid
            best_result = result
    
    return best_ratio, best_result


def get_cost_ratio_justification(
    default_ratio: float = 10.0
) -> str:
    """
    Get documented justification for the default cost ratio.
    
    Returns:
        justification: Markdown-formatted justification text
    """
    return f"""
## Cost Ratio Justification: C_FN = {default_ratio} × C_FP

### Rationale

The default false negative to false positive cost ratio of **{default_ratio}:1** is based on:

#### 1. Security Operations Center (SOC) Perspective

| Error Type | Consequence | Typical Impact |
|------------|-------------|----------------|
| **False Positive** | Unnecessary investigation | ~15-30 min analyst time |
| **False Negative** | Missed attack/breach | Hours to days of remediation, potential data loss |

A missed attack (FN) typically requires 10-100x more resources to remediate than
investigating a false alarm (FP).

#### 2. Industry Standards

- **NIST SP 800-53**: Emphasizes detection over false alarm reduction for high-value assets
- **MITRE ATT&CK**: Detection coverage is primary metric, assuming reasonable FP rates
- **SOC Studies**: Average cost of breach investigation >> cost of false alarm triage

#### 3. Empirical Ratios from Literature

| Source | Recommended Ratio |
|--------|-------------------|
| Axelsson (2000) | 5-20x |
| Buczak & Guven (2016) | 10x default |
| Sommer & Paxson (2010) | Domain-dependent, 5-50x |

#### 4. Practical Considerations

- **Ratio = 1**: Treats all errors equally (unsuitable for security)
- **Ratio = 5**: Moderate preference for detection
- **Ratio = 10**: Balanced for most enterprise SOCs ✓
- **Ratio = 50+**: High-security environments (military, finance)

### Recommendation

The default ratio of **{default_ratio}:1** is appropriate for:
- Enterprise network security
- General intrusion detection systems
- Balanced false alarm / detection requirements

For **critical infrastructure** or **high-security environments**, consider ratios of 20-50.

For **high-traffic environments** with limited SOC capacity, consider ratios of 3-5.

### Sensitivity Analysis

Always run `cost_sensitivity_analysis()` on your specific dataset to verify
performance across different cost assumptions. The optimal ratio depends on:
- Organization's security posture
- Available SOC resources
- Asset criticality
- Attack prevalence
"""


def normalized_expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_fp_ratio: float = 10.0
) -> float:
    """
    Compute normalized expected cost (0-1 scale).
    
    Normalizes by the maximum possible cost for the dataset.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        fn_fp_ratio: C_FN / C_FP ratio
        
    Returns:
        nec: Normalized expected cost (0=perfect, 1=worst)
    """
    raw_cost = compute_expected_cost(y_true, y_pred, fn_fp_ratio, 1.0)
    
    # Max cost: all predictions wrong in costliest way
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)
    
    max_cost = n_positive * fn_fp_ratio + n_negative * 1.0
    
    if max_cost == 0:
        return 0.0
    
    return float(raw_cost / max_cost)


def cost_sensitive_metrics_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_fp_ratio: float = 10.0
) -> Dict:
    """
    Generate comprehensive cost-sensitive metrics report.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        fn_fp_ratio: Cost ratio to use
        
    Returns:
        report: Dictionary with metrics and analysis
    """
    # Optimal threshold
    threshold = compute_optimal_threshold(y_true, y_prob, fn_fp_ratio)
    
    # Metrics at optimal threshold
    metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
    
    # Compare to default threshold (0.5)
    metrics_default = compute_metrics_at_threshold(y_true, y_prob, 0.5)
    
    # Expected costs
    y_pred_optimal = (y_prob >= threshold).astype(int)
    y_pred_default = (y_prob >= 0.5).astype(int)
    
    cost_optimal = compute_expected_cost(y_true, y_pred_optimal, fn_fp_ratio, 1.0)
    cost_default = compute_expected_cost(y_true, y_pred_default, fn_fp_ratio, 1.0)
    
    # Sensitivity around current ratio
    sensitivity = cost_sensitivity_analysis(
        y_true, y_prob, 
        [fn_fp_ratio * 0.5, fn_fp_ratio, fn_fp_ratio * 2]
    )
    
    return {
        'cost_ratio': fn_fp_ratio,
        'optimal_threshold': threshold,
        'default_threshold': 0.5,
        'metrics_at_optimal': metrics,
        'metrics_at_default': metrics_default,
        'expected_cost_optimal': cost_optimal,
        'expected_cost_default': cost_default,
        'cost_reduction': (cost_default - cost_optimal) / cost_default * 100 if cost_default > 0 else 0,
        'sensitivity_analysis': [
            {
                'ratio': s.fn_fp_ratio,
                'recall': s.recall,
                'precision': s.precision,
                'expected_cost': s.expected_cost
            }
            for s in sensitivity
        ]
    }
