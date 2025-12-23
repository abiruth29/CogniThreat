"""
Probabilistic Reasoning Module for CogniThreat NIDS

This module implements Bayesian inference, uncertainty quantification,
and risk-based decision making for network intrusion detection.

Components:
- Calibration: ECE, Brier score, Temperature Scaling, Isotonic Regression
- Fusion: Weighted ensemble, Learned fusion via stacking
- Uncertainty: MC Dropout, ensemble uncertainty
- Risk Inference: Cost-sensitive decision making
- Cost Sensitivity: Analysis of cost ratio choices
"""

from .fusion import (
    bayesian_fusion, 
    weighted_ensemble, 
    ensemble_uncertainty,
    LearnedFusion,
    BayesianFusion
)
from .risk_inference import risk_score, compute_decision_threshold, create_cost_matrix
from .uncertainty import MCDropoutPredictor, predictive_entropy
from .pipeline import ProbabilisticPipeline, quick_probabilistic_analysis
from .calibration import (
    expected_calibration_error,
    brier_score,
    TemperatureScaling,
    IsotonicCalibration,
    calibration_report
)
from .cost_sensitivity import (
    cost_sensitivity_analysis,
    get_cost_ratio_justification,
    cost_sensitive_metrics_report
)

__all__ = [
    # Fusion
    'bayesian_fusion',
    'weighted_ensemble',
    'ensemble_uncertainty',
    'LearnedFusion',
    'BayesianFusion',
    # Risk
    'risk_score',
    'compute_decision_threshold',
    'create_cost_matrix',
    # Uncertainty
    'MCDropoutPredictor',
    'predictive_entropy',
    # Pipeline
    'ProbabilisticPipeline',
    'quick_probabilistic_analysis',
    # Calibration
    'expected_calibration_error',
    'brier_score',
    'TemperatureScaling',
    'IsotonicCalibration',
    'calibration_report',
    # Cost Sensitivity
    'cost_sensitivity_analysis',
    'get_cost_ratio_justification',
    'cost_sensitive_metrics_report'
]

__version__ = '2.0.0'
