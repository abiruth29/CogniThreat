"""
Temporal Reasoning Module for CogniThreat

This module provides Hidden Markov Model (HMM) and Markov Chain implementations
for modeling sequential network security events and predicting future attack stages.

Components:
- markov_chain: Basic Markov Chain for observable state sequences
- hmm_model: Hidden Markov Model for latent attack stage modeling
- event_encoder: Converts network events to discrete observations
- temporal_predictor: High-level API for temporal predictions
"""

from .markov_chain import MarkovChain
from .hmm_model import HMMChain
from .event_encoder import EventEncoder
from .temporal_predictor import TemporalPredictor

__all__ = [
    'MarkovChain',
    'HMMChain',
    'EventEncoder',
    'TemporalPredictor'
]

__version__ = '1.0.0'
