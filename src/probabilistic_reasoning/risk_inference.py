"""
Risk-Based Decision Making Module

Implements expected cost minimization and risk scoring for
network intrusion detection alert prioritization.
"""

import numpy as np
from typing import Dict, Optional


def risk_score(
    calibrated_proba: np.ndarray,
    cost_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute risk scores using expected cost minimization.
    
    Risk score = minimum expected cost across all possible decisions.
    Higher risk scores indicate samples requiring urgent attention.
    
    Args:
        calibrated_proba: Calibrated probabilities (n_samples, n_classes)
        cost_matrix: Cost matrix (n_classes_true, n_classes_pred)
                     cost_matrix[i, j] = cost of predicting j when true class is i
        
    Returns:
        Risk scores (n_samples,)
        
    Example:
        >>> # Cost matrix for (Normal, Attack)
        >>> cost = np.array([[0.0, 1.0],   # FP cost
        ...                  [10.0, 0.0]])  # FN cost (high!)
        >>> risk = risk_score(probs, cost)
    """
    # Expected cost for each possible decision
    # expected_costs[i, j] = sum_k P(k|x_i) * cost[k, j]
    expected_costs = calibrated_proba @ cost_matrix
    
    # Risk score = minimum expected cost (optimal decision's cost)
    risk_scores = expected_costs.min(axis=1)
    
    return risk_scores


def compute_decision_threshold(
    calibrated_proba: np.ndarray,
    cost_matrix: np.ndarray,
    false_positive_weight: float = 1.0,
    false_negative_weight: float = 10.0
) -> float:
    """
    Compute optimal decision threshold for binary classification.
    
    Uses Bayes decision rule to minimize expected cost.
    
    Args:
        calibrated_proba: Calibrated probabilities (n_samples, 2)
        cost_matrix: Cost matrix (2, 2)
        false_positive_weight: Relative cost of false positives
        false_negative_weight: Relative cost of false negatives
        
    Returns:
        Optimal threshold between 0 and 1
    """
    # For binary classification: threshold = C_FP / (C_FP + C_FN)
    # where C_FP = cost of false positive, C_FN = cost of false negative
    
    c_fp = cost_matrix[0, 1] * false_positive_weight
    c_fn = cost_matrix[1, 0] * false_negative_weight
    
    optimal_threshold = c_fp / (c_fp + c_fn)
    
    return optimal_threshold


def expected_cost_decision(
    calibrated_proba: np.ndarray,
    cost_matrix: np.ndarray
) -> np.ndarray:
    """
    Make decisions by minimizing expected cost.
    
    Args:
        calibrated_proba: Calibrated probabilities (n_samples, n_classes)
        cost_matrix: Cost matrix (n_classes_true, n_classes_pred)
        
    Returns:
        Predicted class labels (n_samples,)
    """
    # Expected cost for each decision
    expected_costs = calibrated_proba @ cost_matrix
    
    # Choose decision with minimum expected cost
    optimal_decisions = expected_costs.argmin(axis=1)
    
    return optimal_decisions


def alert_prioritization(
    risk_scores: np.ndarray,
    labels: np.ndarray,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None
) -> Dict:
    """
    Prioritize alerts based on risk scores.
    
    Args:
        risk_scores: Risk scores for each sample
        labels: Predicted labels
        top_k: Return top-k highest risk samples (optional)
        threshold: Return samples above risk threshold (optional)
        
    Returns:
        Dictionary with prioritized indices and statistics
    """
    # Sort by risk (descending)
    sorted_indices = np.argsort(-risk_scores)
    
    if top_k is not None:
        prioritized_indices = sorted_indices[:top_k]
    elif threshold is not None:
        prioritized_indices = sorted_indices[risk_scores[sorted_indices] >= threshold]
    else:
        prioritized_indices = sorted_indices
    
    return {
        'indices': prioritized_indices,
        'risk_scores': risk_scores[prioritized_indices],
        'labels': labels[prioritized_indices],
        'mean_risk': risk_scores[prioritized_indices].mean(),
        'max_risk': risk_scores[prioritized_indices].max(),
        'num_alerts': len(prioritized_indices)
    }


def cost_sensitive_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: np.ndarray
) -> float:
    """
    Compute cost-sensitive accuracy (inverse of normalized cost).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cost_matrix: Cost matrix
        
    Returns:
        Cost-sensitive accuracy between 0 and 1
    """
    # Total cost incurred
    total_cost = sum(cost_matrix[true_label, pred_label] 
                     for true_label, pred_label in zip(y_true, y_pred))
    
    # Worst case cost (always predicting worst class)
    worst_cost = len(y_true) * cost_matrix.max()
    
    # Normalize to [0, 1] where 1 is best
    if worst_cost == 0:
        return 1.0
    
    cost_sensitive_acc = 1.0 - (total_cost / worst_cost)
    
    return max(0.0, cost_sensitive_acc)


def create_cost_matrix(
    n_classes: int,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    correct_cost: float = 0.0
) -> np.ndarray:
    """
    Create a cost matrix for classification.
    
    Args:
        n_classes: Number of classes
        false_positive_cost: Cost of predicting attack when normal
        false_negative_cost: Cost of predicting normal when attack
        correct_cost: Cost of correct prediction (usually 0)
        
    Returns:
        Cost matrix (n_classes, n_classes)
    """
    cost_matrix = np.ones((n_classes, n_classes)) * false_positive_cost
    
    # Diagonal elements (correct predictions)
    np.fill_diagonal(cost_matrix, correct_cost)
    
    # Special handling for binary case
    if n_classes == 2:
        # Assume class 0 = Normal, class 1 = Attack
        cost_matrix[1, 0] = false_negative_cost  # FN: predict normal when attack
        cost_matrix[0, 1] = false_positive_cost  # FP: predict attack when normal
    
    return cost_matrix


def bayesian_risk(
    posterior_probs: np.ndarray,
    loss_function: callable,
    decisions: np.ndarray
) -> np.ndarray:
    """
    Compute Bayesian risk (expected loss under posterior).
    
    Args:
        posterior_probs: Posterior probabilities P(class|x)
        loss_function: Loss function L(true_class, decision)
        decisions: Possible decisions
        
    Returns:
        Expected losses for each decision
    """
    n_samples, n_classes = posterior_probs.shape
    n_decisions = len(decisions)
    
    expected_losses = np.zeros((n_samples, n_decisions))
    
    for i, decision in enumerate(decisions):
        for true_class in range(n_classes):
            loss = loss_function(true_class, decision)
            expected_losses[:, i] += posterior_probs[:, true_class] * loss
    
    return expected_losses
