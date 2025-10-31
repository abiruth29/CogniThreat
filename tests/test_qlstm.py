"""
Unit tests for Quantum LSTM implementation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_models.qlstm import QuantumLSTM


class TestQuantumLSTM:
    """Test cases for Quantum LSTM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_qubits = 4
        self.n_layers = 2
        self.qlstm = QuantumLSTM(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers
        )
    
    def test_initialization(self):
        """Test QLSTM initialization."""
        assert self.qlstm.n_qubits == self.n_qubits
        assert self.qlstm.n_layers == self.n_layers
        assert self.qlstm.dev is not None
        assert self.qlstm.quantum_circuit is not None
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        X, y = self.qlstm.generate_synthetic_data(n_samples=100, sequence_length=10)
        
        assert X.shape == (100, 10, self.n_qubits)
        assert y.shape == (100, 1)
        assert np.all((y == 0) | (y == 1))  # Binary labels
    
    def test_model_structure(self):
        """Test model structure."""
        assert self.qlstm.model is not None
        
        # Test with sample input
        sample_input = np.random.randn(1, 10, self.n_qubits)
        prediction = self.qlstm.model.predict(sample_input)
        
        assert prediction.shape[0] == 1
        assert 0 <= prediction[0, 0] <= 1  # Sigmoid output
    
    def test_training_pipeline(self):
        """Test training pipeline with minimal data."""
        # Generate small dataset for quick test
        X, y = self.qlstm.generate_synthetic_data(n_samples=50, sequence_length=5)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train for 1 epoch
        history = self.qlstm.train(X_train, y_train, epochs=1, batch_size=16)
        
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 1
    
    def test_evaluation(self):
        """Test model evaluation."""
        # Generate test data
        X_test, y_test = self.qlstm.generate_synthetic_data(n_samples=20, sequence_length=5)
        
        # Quick training to have a trained model
        X_train, y_train = self.qlstm.generate_synthetic_data(n_samples=40, sequence_length=5)
        self.qlstm.train(X_train, y_train, epochs=1, batch_size=16)
        
        # Evaluate
        metrics = self.qlstm.evaluate(X_test, y_test)
        
        expected_metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
