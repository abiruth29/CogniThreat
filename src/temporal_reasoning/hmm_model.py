"""
Hidden Markov Model (HMM) Implementation for Latent State Sequences

This module provides a comprehensive HMM implementation for modeling hidden
attack stages/phases in network security events.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import pickle
import logging
from scipy.special import logsumexp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HMMChain:
    """
    Hidden Markov Model for latent attack stage modeling.
    
    Models scenarios where true attack phases are hidden, but we observe
    network events/alerts that are probabilistically related to hidden states.
    
    Attributes:
        num_hidden_states (int): Number of hidden states (attack stages)
        num_observations (int): Number of observable event types
        state_names (List[str]): Hidden state labels
        observation_names (List[str]): Observable event labels
        transition_matrix (np.ndarray): State transition probabilities [S x S]
        emission_matrix (np.ndarray): Observation emission probabilities [S x O]
        initial_probs (np.ndarray): Initial state distribution [S]
    """
    
    def __init__(
        self,
        num_hidden_states: int,
        num_observations: int,
        state_names: Optional[List[str]] = None,
        observation_names: Optional[List[str]] = None,
        transition_matrix: Optional[np.ndarray] = None,
        emission_matrix: Optional[np.ndarray] = None,
        initial_probs: Optional[np.ndarray] = None
    ):
        """
        Initialize Hidden Markov Model.
        
        Args:
            num_hidden_states: Number of hidden states (e.g., attack stages)
            num_observations: Number of observable events/symbols
            state_names: Optional hidden state labels
            observation_names: Optional observation labels
            transition_matrix: Optional pre-computed transitions [S x S]
            emission_matrix: Optional pre-computed emissions [S x O]
            initial_probs: Optional initial state distribution [S]
        """
        self.num_hidden_states = num_hidden_states
        self.num_observations = num_observations
        
        # Initialize state names
        if state_names is None:
            self.state_names = [
                "Normal", "Reconnaissance", "Exploitation",
                "Lateral_Movement", "Exfiltration"
            ][:num_hidden_states]
            # Pad if needed
            while len(self.state_names) < num_hidden_states:
                self.state_names.append(f"State_{len(self.state_names)}")
        else:
            assert len(state_names) == num_hidden_states
            self.state_names = state_names
        
        # Initialize observation names
        if observation_names is None:
            self.observation_names = [f"Obs_{i}" for i in range(num_observations)]
        else:
            assert len(observation_names) == num_observations
            self.observation_names = observation_names
        
        # Initialize transition matrix
        if transition_matrix is None:
            self.transition_matrix = self._initialize_matrix(
                (num_hidden_states, num_hidden_states)
            )
        else:
            assert transition_matrix.shape == (num_hidden_states, num_hidden_states)
            self.transition_matrix = transition_matrix
        
        # Initialize emission matrix
        if emission_matrix is None:
            self.emission_matrix = self._initialize_matrix(
                (num_hidden_states, num_observations)
            )
        else:
            assert emission_matrix.shape == (num_hidden_states, num_observations)
            self.emission_matrix = emission_matrix
        
        # Initialize initial state distribution
        if initial_probs is None:
            self.initial_probs = np.ones(num_hidden_states) / num_hidden_states
        else:
            assert len(initial_probs) == num_hidden_states
            self.initial_probs = initial_probs
        
        logger.info(f"Initialized HMM with {num_hidden_states} states, "
                   f"{num_observations} observations")
    
    def _initialize_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        """Initialize probability matrix with random values."""
        matrix = np.random.dirichlet(np.ones(shape[1]), size=shape[0])
        return matrix
    
    def fit(
        self,
        sequences: List[List[int]],
        state_sequences: Optional[List[List[int]]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        verbose: bool = True
    ) -> 'HMMChain':
        """
        Train HMM using Baum-Welch algorithm (EM) or supervised learning.
        
        Args:
            sequences: List of observation sequences
            state_sequences: Optional true state sequences (for supervised learning)
            max_iterations: Maximum EM iterations (for unsupervised)
            tolerance: Convergence tolerance
            verbose: Print training progress
        
        Returns:
            self: Trained model
        """
        if state_sequences is not None:
            # Supervised learning
            return self._fit_supervised(sequences, state_sequences)
        else:
            # Unsupervised learning (Baum-Welch)
            return self._fit_baum_welch(sequences, max_iterations, tolerance, verbose)
    
    def _fit_supervised(
        self,
        sequences: List[List[int]],
        state_sequences: List[List[int]]
    ) -> 'HMMChain':
        """Train with known state sequences (supervised)."""
        logger.info(f"Supervised training on {len(sequences)} sequences...")
        
        # Count matrices with Laplace smoothing
        trans_counts = np.ones((self.num_hidden_states, self.num_hidden_states)) * 1e-6
        emiss_counts = np.ones((self.num_hidden_states, self.num_observations)) * 1e-6
        init_counts = np.ones(self.num_hidden_states) * 1e-6
        
        for obs_seq, state_seq in zip(sequences, state_sequences):
            assert len(obs_seq) == len(state_seq)
            
            # Initial state
            init_counts[state_seq[0]] += 1
            
            # Transitions and emissions
            for t in range(len(state_seq)):
                state = state_seq[t]
                obs = obs_seq[t]
                
                emiss_counts[state, obs] += 1
                
                if t < len(state_seq) - 1:
                    next_state = state_seq[t + 1]
                    trans_counts[state, next_state] += 1
        
        # Normalize
        self.initial_probs = init_counts / init_counts.sum()
        self.transition_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        self.emission_matrix = emiss_counts / emiss_counts.sum(axis=1, keepdims=True)
        
        logger.info("Supervised training complete")
        return self
    
    def _fit_baum_welch(
        self,
        sequences: List[List[int]],
        max_iterations: int,
        tolerance: float,
        verbose: bool
    ) -> 'HMMChain':
        """Train using Baum-Welch algorithm (unsupervised)."""
        logger.info(f"Baum-Welch training on {len(sequences)} sequences...")
        
        prev_log_likelihood = float('-inf')
        
        for iteration in range(max_iterations):
            # E-step: compute forward-backward probabilities
            gamma_sum = np.zeros((self.num_hidden_states, self.num_observations))
            xi_sum = np.zeros((self.num_hidden_states, self.num_hidden_states))
            init_sum = np.zeros(self.num_hidden_states)
            
            total_log_likelihood = 0.0
            
            for seq in sequences:
                if len(seq) == 0:
                    continue
                
                # Forward-backward
                alpha = self._forward(seq)
                beta = self._backward(seq)
                
                # Compute gamma (state occupation probabilities)
                gamma = alpha + beta
                gamma -= logsumexp(gamma, axis=0)
                gamma = np.exp(gamma)
                
                # Compute xi (transition probabilities)
                xi = self._compute_xi(seq, alpha, beta)
                
                # Accumulate statistics
                init_sum += gamma[:, 0]
                
                for t in range(len(seq)):
                    gamma_sum[:, seq[t]] += gamma[:, t]
                
                xi_sum += xi.sum(axis=2)
                
                # Compute log likelihood
                total_log_likelihood += logsumexp(alpha[:, -1])
            
            # M-step: update parameters
            self.initial_probs = init_sum / len(sequences)
            self.initial_probs /= self.initial_probs.sum()
            
            self.transition_matrix = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            self.emission_matrix = gamma_sum / gamma_sum.sum(axis=1, keepdims=True)
            
            # Check convergence
            avg_log_likelihood = total_log_likelihood / len(sequences)
            
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                logger.info(f"Iteration {iteration + 1}: "
                          f"Avg Log Likelihood = {avg_log_likelihood:.4f}")
            
            if abs(avg_log_likelihood - prev_log_likelihood) < tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = avg_log_likelihood
        
        return self
    
    def _forward(self, observations: List[int]) -> np.ndarray:
        """
        Forward algorithm for computing forward probabilities.
        
        Returns:
            alpha: Log forward probabilities [S x T]
        """
        T = len(observations)
        alpha = np.zeros((self.num_hidden_states, T))
        
        # Initialize
        alpha[:, 0] = (np.log(self.initial_probs + 1e-10) +
                      np.log(self.emission_matrix[:, observations[0]] + 1e-10))
        
        # Recursion
        for t in range(1, T):
            for j in range(self.num_hidden_states):
                alpha[j, t] = (
                    logsumexp(alpha[:, t-1] + np.log(self.transition_matrix[:, j] + 1e-10)) +
                    np.log(self.emission_matrix[j, observations[t]] + 1e-10)
                )
        
        return alpha
    
    def _backward(self, observations: List[int]) -> np.ndarray:
        """
        Backward algorithm for computing backward probabilities.
        
        Returns:
            beta: Log backward probabilities [S x T]
        """
        T = len(observations)
        beta = np.zeros((self.num_hidden_states, T))
        
        # Initialize (log(1) = 0)
        beta[:, T-1] = 0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.num_hidden_states):
                beta[i, t] = logsumexp(
                    np.log(self.transition_matrix[i, :] + 1e-10) +
                    np.log(self.emission_matrix[:, observations[t+1]] + 1e-10) +
                    beta[:, t+1]
                )
        
        return beta
    
    def _compute_xi(
        self,
        observations: List[int],
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """
        Compute xi (transition probabilities).
        
        Returns:
            xi: Transition probabilities [S x S x T-1]
        """
        T = len(observations)
        xi = np.zeros((self.num_hidden_states, self.num_hidden_states, T-1))
        
        for t in range(T-1):
            for i in range(self.num_hidden_states):
                for j in range(self.num_hidden_states):
                    xi[i, j, t] = (
                        alpha[i, t] +
                        np.log(self.transition_matrix[i, j] + 1e-10) +
                        np.log(self.emission_matrix[j, observations[t+1]] + 1e-10) +
                        beta[j, t+1]
                    )
            
            # Normalize
            xi[:, :, t] -= logsumexp(xi[:, :, t])
            xi[:, :, t] = np.exp(xi[:, :, t])
        
        return xi
    
    def viterbi(self, observations: List[int]) -> Tuple[List[int], float]:
        """
        Viterbi algorithm for most likely state sequence.
        
        Args:
            observations: Observation sequence
        
        Returns:
            states: Most likely state sequence
            log_prob: Log probability of the sequence
        """
        T = len(observations)
        
        # Viterbi trellis (log probabilities)
        viterbi = np.zeros((self.num_hidden_states, T))
        backpointer = np.zeros((self.num_hidden_states, T), dtype=int)
        
        # Initialize
        viterbi[:, 0] = (np.log(self.initial_probs + 1e-10) +
                        np.log(self.emission_matrix[:, observations[0]] + 1e-10))
        
        # Recursion
        for t in range(1, T):
            for j in range(self.num_hidden_states):
                trans_probs = viterbi[:, t-1] + np.log(self.transition_matrix[:, j] + 1e-10)
                backpointer[j, t] = np.argmax(trans_probs)
                viterbi[j, t] = (
                    trans_probs[backpointer[j, t]] +
                    np.log(self.emission_matrix[j, observations[t]] + 1e-10)
                )
        
        # Backtrack
        states = [0] * T
        states[-1] = np.argmax(viterbi[:, -1])
        log_prob = viterbi[states[-1], -1]
        
        for t in range(T-2, -1, -1):
            states[t] = backpointer[states[t+1], t+1]
        
        return states, log_prob
    
    def predict_next_state(
        self,
        observation_sequence: List[int],
        return_distribution: bool = False
    ) -> Union[Tuple[int, float], np.ndarray]:
        """
        Predict next hidden state given observation sequence.
        
        Args:
            observation_sequence: Observed event sequence
            return_distribution: Return full probability distribution
        
        Returns:
            If return_distribution=False: (next_state, confidence)
            If return_distribution=True: probability distribution [S]
        """
        # Use forward algorithm to get current state distribution
        alpha = self._forward(observation_sequence)
        
        # Current state posterior
        current_probs = alpha[:, -1]
        current_probs = np.exp(current_probs - logsumexp(current_probs))
        
        # Predict next state
        next_probs = current_probs @ self.transition_matrix
        
        if return_distribution:
            return next_probs
        
        next_state = np.argmax(next_probs)
        confidence = next_probs[next_state]
        
        return next_state, confidence
    
    def compute_state_probabilities(
        self,
        observation_sequence: List[int]
    ) -> np.ndarray:
        """
        Compute posterior state probabilities for each time step.
        
        Args:
            observation_sequence: Observed event sequence
        
        Returns:
            state_probs: Posterior probabilities [S x T]
        """
        alpha = self._forward(observation_sequence)
        beta = self._backward(observation_sequence)
        
        gamma = alpha + beta
        gamma -= logsumexp(gamma, axis=0)
        gamma = np.exp(gamma)
        
        return gamma
    
    def compute_sequence_probability(
        self,
        observation_sequence: List[int]
    ) -> float:
        """
        Compute probability of observation sequence.
        
        Args:
            observation_sequence: Observed event sequence
        
        Returns:
            log_prob: Log probability
        """
        alpha = self._forward(observation_sequence)
        return logsumexp(alpha[:, -1])
    
    def get_state_entropy(
        self,
        observation_sequence: List[int]
    ) -> np.ndarray:
        """
        Compute state uncertainty (entropy) at each time step.
        
        Args:
            observation_sequence: Observed event sequence
        
        Returns:
            entropy: Entropy values [T]
        """
        state_probs = self.compute_state_probabilities(observation_sequence)
        
        entropy = np.zeros(state_probs.shape[1])
        for t in range(state_probs.shape[1]):
            probs = state_probs[:, t]
            entropy[t] = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            'num_hidden_states': self.num_hidden_states,
            'num_observations': self.num_observations,
            'state_names': self.state_names,
            'observation_names': self.observation_names,
            'transition_matrix': self.transition_matrix,
            'emission_matrix': self.emission_matrix,
            'initial_probs': self.initial_probs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"HMM model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HMMChain':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            num_hidden_states=model_data['num_hidden_states'],
            num_observations=model_data['num_observations'],
            state_names=model_data['state_names'],
            observation_names=model_data['observation_names'],
            transition_matrix=model_data['transition_matrix'],
            emission_matrix=model_data['emission_matrix'],
            initial_probs=model_data['initial_probs']
        )
        
        logger.info(f"HMM model loaded from {filepath}")
        return model
    
    def __repr__(self) -> str:
        return (f"HMMChain(states={self.num_hidden_states}, "
                f"observations={self.num_observations})")
