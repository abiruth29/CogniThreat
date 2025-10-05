"""
Test configuration and fixtures for CogniThreat tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set random seed for reproducible tests
np.random.seed(42)

@pytest.fixture
def sample_network_data():
    """Fixture providing sample network data for tests."""
    n_samples = 50
    n_features = 8
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (30% attacks)
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    return X, y

@pytest.fixture
def sample_sequence_data():
    """Fixture providing sample sequence data for quantum models."""
    n_samples = 20
    sequence_length = 10
    n_features = 4
    
    # Generate sequence data
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    return X, y
