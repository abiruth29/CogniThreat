"""
Markov Chain Implementation for Observable State Sequences

This module provides a discrete-time Markov Chain for modeling transitions
between observable network security states.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkovChain:
    """
    Discrete-time Markov Chain for modeling observable state sequences.
    
    Suitable for scenarios where network security states are directly observable
    (e.g., traffic patterns, alert levels, connection states).
    
    Attributes:
        num_states (int): Number of discrete states
        state_names (List[str]): Human-readable state labels
        transition_matrix (np.ndarray): State transition probability matrix [S x S]
        initial_probs (np.ndarray): Initial state distribution [S]
        state_counts (np.ndarray): Observed transition counts for updates
    """
    
    def __init__(
        self,
        num_states: int,
        state_names: Optional[List[str]] = None,
        transition_matrix: Optional[np.ndarray] = None,
        initial_probs: Optional[np.ndarray] = None
    ):
        """
        Initialize Markov Chain.
        
        Args:
            num_states: Number of discrete states
            state_names: Optional list of state labels (default: ["State_0", ..., "State_N"])
            transition_matrix: Optional pre-computed transition matrix [S x S]
            initial_probs: Optional initial state distribution [S]
        """
        self.num_states = num_states
        
        # Initialize state names
        if state_names is None:
            self.state_names = [f"State_{i}" for i in range(num_states)]
        else:
            assert len(state_names) == num_states, "State names must match num_states"
            self.state_names = state_names
        
        # Initialize transition matrix
        if transition_matrix is None:
            # Uniform initialization
            self.transition_matrix = np.ones((num_states, num_states)) / num_states
        else:
            assert transition_matrix.shape == (num_states, num_states), \
                f"Transition matrix must be {num_states}x{num_states}"
            self.transition_matrix = transition_matrix
        
        # Initialize initial state distribution
        if initial_probs is None:
            self.initial_probs = np.ones(num_states) / num_states
        else:
            assert len(initial_probs) == num_states, "Initial probs must match num_states"
            self.initial_probs = initial_probs
        
        # Count matrix for online learning
        self.state_counts = np.zeros((num_states, num_states))
        self.initial_counts = np.zeros(num_states)
        
        logger.info(f"Initialized Markov Chain with {num_states} states")
    
    def fit(
        self,
        sequences: List[List[int]],
        smoothing: float = 1e-6
    ) -> 'MarkovChain':
        """
        Train Markov Chain from observed state sequences.
        
        Args:
            sequences: List of state sequences (each sequence is list of state indices)
            smoothing: Laplace smoothing parameter to avoid zero probabilities
        
        Returns:
            self: Trained model
        """
        logger.info(f"Training on {len(sequences)} sequences...")
        
        # Reset counts
        self.state_counts = np.ones((self.num_states, self.num_states)) * smoothing
        self.initial_counts = np.ones(self.num_states) * smoothing
        
        # Count transitions
        for seq in sequences:
            if len(seq) == 0:
                continue
            
            # Initial state
            self.initial_counts[seq[0]] += 1
            
            # Transitions
            for i in range(len(seq) - 1):
                self.state_counts[seq[i], seq[i + 1]] += 1
        
        # Normalize to probabilities
        self.initial_probs = self.initial_counts / self.initial_counts.sum()
        
        # Normalize each row (from-state) to sum to 1
        row_sums = self.state_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = self.state_counts / row_sums
        
        logger.info("Training complete")
        return self
    
    def predict_next_state(
        self,
        current_state: int,
        return_distribution: bool = False
    ) -> Union[int, Tuple[int, float], np.ndarray]:
        """
        Predict the next state given current state.
        
        Args:
            current_state: Current state index
            return_distribution: If True, return full probability distribution
        
        Returns:
            If return_distribution=False: (next_state, confidence)
            If return_distribution=True: probability distribution over next states
        """
        assert 0 <= current_state < self.num_states, \
            f"Invalid state {current_state}, must be in [0, {self.num_states})"
        
        next_probs = self.transition_matrix[current_state]
        
        if return_distribution:
            return next_probs
        
        next_state = np.argmax(next_probs)
        confidence = next_probs[next_state]
        
        return next_state, confidence
    
    def predict_sequence(
        self,
        initial_state: int,
        length: int
    ) -> Tuple[List[int], List[float]]:
        """
        Generate a predicted sequence starting from initial state.
        
        Args:
            initial_state: Starting state index
            length: Length of sequence to generate
        
        Returns:
            sequence: List of predicted state indices
            confidences: List of confidence scores for each prediction
        """
        sequence = [initial_state]
        confidences = [1.0]  # Initial state has confidence 1.0
        
        current = initial_state
        for _ in range(length - 1):
            next_state, conf = self.predict_next_state(current)
            sequence.append(next_state)
            confidences.append(conf)
            current = next_state
        
        return sequence, confidences
    
    def compute_sequence_probability(
        self,
        sequence: List[int]
    ) -> float:
        """
        Compute the probability of observing a given state sequence.
        
        Args:
            sequence: List of state indices
        
        Returns:
            log_prob: Log probability of the sequence
        """
        if len(sequence) == 0:
            return 0.0
        
        log_prob = np.log(self.initial_probs[sequence[0]] + 1e-10)
        
        for i in range(len(sequence) - 1):
            log_prob += np.log(self.transition_matrix[sequence[i], sequence[i + 1]] + 1e-10)
        
        return log_prob
    
    def get_stationary_distribution(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8
    ) -> np.ndarray:
        """
        Compute the stationary distribution of the Markov Chain.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        
        Returns:
            stationary: Stationary distribution [S]
        """
        # Power iteration method
        current = self.initial_probs.copy()
        
        for _ in range(max_iterations):
            next_dist = current @ self.transition_matrix
            
            if np.allclose(current, next_dist, atol=tolerance):
                return next_dist
            
            current = next_dist
        
        logger.warning("Stationary distribution did not converge")
        return current
    
    def update_online(
        self,
        state_sequence: List[int],
        learning_rate: float = 0.1
    ) -> 'MarkovChain':
        """
        Update transition probabilities with new observations (online learning).
        
        Args:
            state_sequence: New observed state sequence
            learning_rate: Weight for new observations (0-1)
        
        Returns:
            self: Updated model
        """
        if len(state_sequence) < 2:
            return self
        
        # Create temporary count matrix for this sequence
        temp_counts = np.zeros((self.num_states, self.num_states))
        
        for i in range(len(state_sequence) - 1):
            temp_counts[state_sequence[i], state_sequence[i + 1]] += 1
        
        # Normalize
        row_sums = temp_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        temp_transitions = temp_counts / row_sums
        
        # Blend with existing transitions
        for i in range(self.num_states):
            if temp_counts[i].sum() > 0:  # Only update states that appear
                self.transition_matrix[i] = (
                    (1 - learning_rate) * self.transition_matrix[i] +
                    learning_rate * temp_transitions[i]
                )
        
        return self
    
    def get_transition_entropy(
        self,
        state: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Compute entropy of transition distribution(s).
        
        Higher entropy indicates more uncertainty about next state.
        
        Args:
            state: If provided, compute entropy for this state only
        
        Returns:
            entropy: Entropy value(s)
        """
        def compute_entropy(probs):
            # Add small epsilon to avoid log(0)
            probs = probs + 1e-10
            return -np.sum(probs * np.log2(probs))
        
        if state is not None:
            return compute_entropy(self.transition_matrix[state])
        else:
            return np.array([compute_entropy(self.transition_matrix[i]) 
                           for i in range(self.num_states)])
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'num_states': self.num_states,
            'state_names': self.state_names,
            'transition_matrix': self.transition_matrix,
            'initial_probs': self.initial_probs,
            'state_counts': self.state_counts,
            'initial_counts': self.initial_counts
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MarkovChain':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            num_states=model_data['num_states'],
            state_names=model_data['state_names'],
            transition_matrix=model_data['transition_matrix'],
            initial_probs=model_data['initial_probs']
        )
        
        model.state_counts = model_data['state_counts']
        model.initial_counts = model_data['initial_counts']
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def __repr__(self) -> str:
        return (f"MarkovChain(num_states={self.num_states}, "
                f"states={self.state_names})")
