"""
Probabilistic Reasoning Module for CogniThreat NIDS

This module implements Bayesian inference, uncertainty quantification,
and risk-based decision making for network intrusion detection.
"""

from .fusion import bayesian_fusion, ensemble_uncertainty
from .risk_inference import risk_score, compute_decision_threshold, create_cost_matrix
from .uncertainty import MCDropoutPredictor, predictive_entropy
from .pipeline import ProbabilisticPipeline, quick_probabilistic_analysis

__all__ = [
    'bayesian_fusion',
    'ensemble_uncertainty',
    'risk_score',
    'compute_decision_threshold',
    'create_cost_matrix',
    'MCDropoutPredictor',
    'predictive_entropy',
    'ProbabilisticPipeline',
    'quick_probabilistic_analysis'
]

__version__ = '1.0.0'
