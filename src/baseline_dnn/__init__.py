"""
CogniThreat DNN Baseline Module

This module implements the robust Deep Neural Network baseline
following the methodology from "Network-based intrusion detection 
using deep learning technique" (Scientific Reports, 2025).

Key components:
- Extra Trees feature selection (43 → 8 features)
- Sequential DNN with 3 hidden layers (800, 800, 400)
- Standard scaling and SMOTE for class imbalance
- Comprehensive evaluation metrics
"""

from .feature_selection import ExtraTreesFeatureSelector
from .preprocessing import DNNPreprocessor
from .train_dnn import DNNBaseline
from .evaluate import DNNEvaluator

__all__ = [
    'ExtraTreesFeatureSelector',
    'DNNPreprocessor', 
    'DNNBaseline',
    'DNNEvaluator'
]
