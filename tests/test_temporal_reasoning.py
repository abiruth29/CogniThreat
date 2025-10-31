"""
Unit Tests for Temporal Reasoning Module

Tests for MarkovChain, HMMChain, EventEncoder, and TemporalPredictor.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.temporal_reasoning import MarkovChain, HMMChain, EventEncoder, TemporalPredictor


class TestMarkovChain:
    """Tests for MarkovChain class."""
    
    def test_initialization(self):
        """Test Markov Chain initialization."""
        mc = MarkovChain(num_states=3, state_names=["A", "B", "C"])
        
        assert mc.num_states == 3
        assert len(mc.state_names) == 3
        assert mc.transition_matrix.shape == (3, 3)
        assert np.allclose(mc.transition_matrix.sum(axis=1), 1.0)
    
    def test_fit(self):
        """Test training Markov Chain."""
        sequences = [
            [0, 1, 2, 1, 0],
            [0, 1, 1, 2],
            [1, 2, 0, 1]
        ]
        
        mc = MarkovChain(num_states=3)
        mc.fit(sequences)
        
        assert mc.transition_matrix.shape == (3, 3)
        assert np.allclose(mc.transition_matrix.sum(axis=1), 1.0)
    
    def test_prediction(self):
        """Test next state prediction."""
        mc = MarkovChain(num_states=3)
        mc.transition_matrix = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7]
        ])
        
        next_state, confidence = mc.predict_next_state(0)
        assert 0 <= next_state < 3
        assert 0 <= confidence <= 1
    
    def test_sequence_generation(self):
        """Test sequence prediction."""
        mc = MarkovChain(num_states=3)
        sequences = [[0, 1, 2, 1, 0], [0, 1, 1, 2]]
        mc.fit(sequences)
        
        seq, confs = mc.predict_sequence(initial_state=0, length=5)
        assert len(seq) == 5
        assert len(confs) == 5
        assert all(0 <= s < 3 for s in seq)
    
    def test_sequence_probability(self):
        """Test sequence probability computation."""
        mc = MarkovChain(num_states=3)
        sequences = [[0, 1, 2], [1, 2, 0]]
        mc.fit(sequences)
        
        log_prob = mc.compute_sequence_probability([0, 1, 2])
        assert isinstance(log_prob, float)
        assert log_prob <= 0  # Log probability should be negative
    
    def test_online_update(self):
        """Test online learning."""
        mc = MarkovChain(num_states=3)
        mc.fit([[0, 1, 2]])
        
        old_matrix = mc.transition_matrix.copy()
        mc.update_online([2, 1, 0], learning_rate=0.5)
        
        assert not np.allclose(old_matrix, mc.transition_matrix)
    
    def test_save_load(self, tmp_path):
        """Test model persistence."""
        mc = MarkovChain(num_states=3, state_names=["A", "B", "C"])
        mc.fit([[0, 1, 2], [1, 2, 0]])
        
        filepath = tmp_path / "markov_test.pkl"
        mc.save(str(filepath))
        
        loaded_mc = MarkovChain.load(str(filepath))
        
        assert loaded_mc.num_states == mc.num_states
        assert np.allclose(loaded_mc.transition_matrix, mc.transition_matrix)


class TestHMMChain:
    """Tests for HMMChain class."""
    
    def test_initialization(self):
        """Test HMM initialization."""
        hmm = HMMChain(num_hidden_states=3, num_observations=5)
        
        assert hmm.num_hidden_states == 3
        assert hmm.num_observations == 5
        assert hmm.transition_matrix.shape == (3, 3)
        assert hmm.emission_matrix.shape == (3, 5)
    
    def test_supervised_fit(self):
        """Test supervised training."""
        obs_sequences = [[0, 1, 2, 1], [1, 2, 0, 1]]
        state_sequences = [[0, 1, 2, 1], [1, 2, 0, 1]]
        
        hmm = HMMChain(num_hidden_states=3, num_observations=3)
        hmm.fit(obs_sequences, state_sequences=state_sequences)
        
        assert np.allclose(hmm.transition_matrix.sum(axis=1), 1.0)
        assert np.allclose(hmm.emission_matrix.sum(axis=1), 1.0)
    
    def test_unsupervised_fit(self):
        """Test unsupervised training (Baum-Welch)."""
        obs_sequences = [[0, 1, 2], [1, 2, 0], [0, 1, 1]]
        
        hmm = HMMChain(num_hidden_states=2, num_observations=3)
        hmm.fit(obs_sequences, max_iterations=10, verbose=False)
        
        assert np.allclose(hmm.transition_matrix.sum(axis=1), 1.0)
        assert np.allclose(hmm.emission_matrix.sum(axis=1), 1.0)
    
    def test_viterbi(self):
        """Test Viterbi decoding."""
        hmm = HMMChain(num_hidden_states=2, num_observations=3)
        hmm.fit([[0, 1, 2]], state_sequences=[[0, 1, 1]])
        
        states, log_prob = hmm.viterbi([0, 1, 2])
        
        assert len(states) == 3
        assert all(0 <= s < 2 for s in states)
        assert isinstance(log_prob, float)
    
    def test_forward_backward(self):
        """Test forward-backward algorithm."""
        hmm = HMMChain(num_hidden_states=2, num_observations=3)
        hmm.fit([[0, 1, 2]], state_sequences=[[0, 1, 1]])
        
        state_probs = hmm.compute_state_probabilities([0, 1, 2])
        
        assert state_probs.shape == (2, 3)
        assert np.allclose(state_probs.sum(axis=0), 1.0)
    
    def test_prediction(self):
        """Test next state prediction."""
        hmm = HMMChain(num_hidden_states=2, num_observations=3)
        hmm.fit([[0, 1, 2], [1, 2, 0]], state_sequences=[[0, 1, 1], [1, 0, 0]])
        
        next_state, confidence = hmm.predict_next_state([0, 1])
        
        assert 0 <= next_state < 2
        assert 0 <= confidence <= 1
    
    def test_save_load(self, tmp_path):
        """Test model persistence."""
        hmm = HMMChain(num_hidden_states=2, num_observations=3)
        hmm.fit([[0, 1, 2]], state_sequences=[[0, 1, 1]])
        
        filepath = tmp_path / "hmm_test.pkl"
        hmm.save(str(filepath))
        
        loaded_hmm = HMMChain.load(str(filepath))
        
        assert loaded_hmm.num_hidden_states == hmm.num_hidden_states
        assert np.allclose(loaded_hmm.transition_matrix, hmm.transition_matrix)


class TestEventEncoder:
    """Tests for EventEncoder class."""
    
    def test_label_encoding(self):
        """Test label-based encoding."""
        encoder = EventEncoder(encoding_type='label', num_symbols=5)
        labels = [0, 1, 2, 1, 0, 2]
        
        encoder.fit(labels)
        encoded = encoder.transform(labels)
        
        assert len(encoded) == len(labels)
        assert all(0 <= x < 5 for x in encoded)
    
    def test_quantization_encoding(self):
        """Test quantization-based encoding."""
        encoder = EventEncoder(encoding_type='quantization', num_symbols=5)
        data = np.random.randn(100, 1)
        
        encoder.fit(data)
        encoded = encoder.transform(data)
        
        assert len(encoded) == len(data)
        assert all(0 <= x < 5 for x in encoded)
    
    def test_clustering_encoding(self):
        """Test clustering-based encoding."""
        encoder = EventEncoder(encoding_type='clustering', num_symbols=3)
        data = np.random.randn(50, 2)
        
        encoder.fit(data)
        encoded = encoder.transform(data)
        
        assert len(encoded) == len(data)
        assert all(0 <= x < 3 for x in encoded)
    
    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        encoder = EventEncoder(encoding_type='clustering', num_symbols=4)
        data = np.random.randn(30, 1)
        
        encoded = encoder.fit_transform(data)
        
        assert encoder.is_fitted
        assert len(encoded) == len(data)
    
    def test_sequence_transformation(self):
        """Test transforming multiple sequences."""
        encoder = EventEncoder(encoding_type='label', num_symbols=3)
        sequences = [[0, 1, 2], [1, 2, 0], [0, 0, 1]]
        
        encoder.fit([item for seq in sequences for item in seq])
        encoded_seqs = encoder.transform_sequences(sequences)
        
        assert len(encoded_seqs) == len(sequences)
        assert all(len(enc) == len(orig) for enc, orig in zip(encoded_seqs, sequences))
    
    def test_save_load(self, tmp_path):
        """Test encoder persistence."""
        encoder = EventEncoder(encoding_type='clustering', num_symbols=3)
        data = np.random.randn(30, 2)
        encoder.fit(data)
        
        filepath = tmp_path / "encoder_test.pkl"
        encoder.save(str(filepath))
        
        loaded_encoder = EventEncoder.load(str(filepath))
        
        assert loaded_encoder.encoding_type == encoder.encoding_type
        assert loaded_encoder.num_symbols == encoder.num_symbols
        assert loaded_encoder.is_fitted


class TestTemporalPredictor:
    """Tests for TemporalPredictor class."""
    
    def test_initialization_markov(self):
        """Test Markov-based predictor initialization."""
        predictor = TemporalPredictor(
            model_type='markov',
            num_states=3,
            encoding_type='label'
        )
        
        assert predictor.model_type == 'markov'
        assert isinstance(predictor.model, MarkovChain)
    
    def test_initialization_hmm(self):
        """Test HMM-based predictor initialization."""
        predictor = TemporalPredictor(
            model_type='hmm',
            num_states=3,
            num_observations=6,
            encoding_type='clustering'
        )
        
        assert predictor.model_type == 'hmm'
        assert isinstance(predictor.model, HMMChain)
    
    def test_fit_predict_markov(self):
        """Test training and prediction with Markov model."""
        sequences = [
            [0, 1, 2, 1, 0],
            [0, 1, 1, 2],
            [1, 2, 0, 1]
        ]
        
        predictor = TemporalPredictor(model_type='markov', num_states=3)
        predictor.fit(sequences)
        
        assert predictor.is_fitted
        
        # Predict
        state_idx, state_name, conf = predictor.predict_next_state([0, 1])
        
        assert 0 <= state_idx < 3
        assert isinstance(state_name, str)
        assert 0 <= conf <= 1
    
    def test_fit_predict_hmm(self):
        """Test training and prediction with HMM model."""
        obs_sequences = [
            [0, 1, 2, 1, 0],
            [1, 2, 0, 1],
            [0, 1, 1, 2]
        ]
        state_sequences = [
            [0, 1, 2, 1, 0],
            [1, 2, 0, 1],
            [0, 1, 1, 2]
        ]
        
        predictor = TemporalPredictor(
            model_type='hmm',
            num_states=3,
            num_observations=3
        )
        predictor.fit(obs_sequences, state_sequences=state_sequences)
        
        assert predictor.is_fitted
        
        # Predict
        state_idx, state_name, conf = predictor.predict_next_state([0, 1, 2])
        
        assert 0 <= state_idx < 3
        assert isinstance(state_name, str)
        assert 0 <= conf <= 1
    
    def test_predict_attack_sequence(self):
        """Test attack sequence prediction."""
        sequences = [[0, 1, 2], [1, 2, 0], [0, 1, 1]]
        
        predictor = TemporalPredictor(model_type='markov', num_states=3)
        predictor.fit(sequences)
        
        state_indices, state_names, confidences = predictor.predict_attack_sequence(
            initial_events=[0, 1],
            horizon=5
        )
        
        assert len(state_indices) == 5
        assert len(state_names) == 5
        assert len(confidences) == 5
    
    def test_uncertainty_computation(self):
        """Test uncertainty (entropy) computation."""
        sequences = [[0, 1, 2], [1, 2, 0]]
        
        predictor = TemporalPredictor(model_type='markov', num_states=3)
        predictor.fit(sequences)
        
        uncertainty = predictor.get_uncertainty([0, 1])
        
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0
    
    def test_analyze_sequence(self):
        """Test comprehensive sequence analysis."""
        obs_sequences = [[0, 1, 2], [1, 2, 0]]
        state_sequences = [[0, 1, 2], [1, 2, 0]]
        
        predictor = TemporalPredictor(
            model_type='hmm',
            num_states=3,
            num_observations=3
        )
        predictor.fit(obs_sequences, state_sequences=state_sequences)
        
        analysis = predictor.analyze_sequence([0, 1, 2])
        
        assert 'predicted_state_name' in analysis
        assert 'confidence' in analysis
        assert 'uncertainty' in analysis
        assert 'state_distribution' in analysis
        assert 'most_likely_state_sequence' in analysis
    
    def test_temporal_context_interface(self):
        """Test interface for probabilistic reasoning engine."""
        sequences = [[0, 1, 2], [1, 2, 0]]
        
        predictor = TemporalPredictor(model_type='markov', num_states=3)
        predictor.fit(sequences)
        
        context = predictor.receive_temporal_context([0, 1])
        
        assert 'temporal_prediction' in context
        assert 'attack_stage_probabilities' in context
        assert 'risk_level' in context
        assert context['risk_level'] in ['low', 'medium', 'high', 'critical']
    
    def test_save_load(self, tmp_path):
        """Test predictor persistence."""
        sequences = [[0, 1, 2], [1, 2, 0]]
        
        predictor = TemporalPredictor(model_type='markov', num_states=3)
        predictor.fit(sequences)
        
        filepath = tmp_path / "predictor_test.pkl"
        predictor.save(str(filepath))
        
        loaded_predictor = TemporalPredictor.load(str(filepath))
        
        assert loaded_predictor.model_type == predictor.model_type
        assert loaded_predictor.is_fitted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
