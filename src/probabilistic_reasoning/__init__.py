"""
Probabilistic Reasoning Module for CogniThreat
=============================================

This module implements Bayesian networks and probabilistic reasoning
for uncertainty quantification in network intrusion detection.

Modules:
    bayesian_network: Core Bayesian network implementation
    uncertainty_quantification: Uncertainty analysis and confidence scoring
    probabilistic_inference: Inference engines for probability calculations
    
Author: CogniThreat Team
Date: August 2025
"""

from .bayesian_network import (
    CogniThreatBayesianNetwork,
    NetworkSecurityBN,
    create_intrusion_detection_bn
)

from .uncertainty_quantification import (
    UncertaintyQuantifier,
    confidence_analysis,
    calculate_prediction_confidence
)

from .probabilistic_inference import (
    BayesianInferenceEngine,
    ProbabilisticClassifier,
    adaptive_threshold_learning
)

__all__ = [
    # Bayesian Networks
    'CogniThreatBayesianNetwork',
    'NetworkSecurityBN', 
    'create_intrusion_detection_bn',
    
    # Uncertainty Quantification
    'UncertaintyQuantifier',
    'confidence_analysis',
    'calculate_prediction_confidence',
    
    # Probabilistic Inference
    'BayesianInferenceEngine',
    'ProbabilisticClassifier',
    'adaptive_threshold_learning'
]

__version__ = "1.0.0"
