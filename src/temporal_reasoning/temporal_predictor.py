"""
Temporal Predictor - High-Level API for Temporal Reasoning

This module provides a unified interface for temporal event prediction
combining Markov Chain and HMM models with the CogniThreat system.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Any
import logging
from .markov_chain import MarkovChain
from .hmm_model import HMMChain
from .event_encoder import EventEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalPredictor:
    """
    High-level API for temporal event sequence prediction and attack forecasting.
    
    Integrates event encoding, Markov/HMM models, and provides interface
    for the CogniThreat probabilistic reasoning engine.
    
    Attributes:
        model_type (str): Type of model ('markov' or 'hmm')
        encoder (EventEncoder): Event encoder for discretization
        model (Union[MarkovChain, HMMChain]): Trained temporal model
        attack_stages (List[str]): Attack stage labels
    """
    
    def __init__(
        self,
        model_type: str = 'hmm',
        num_states: int = 5,
        num_observations: Optional[int] = None,
        encoding_type: str = 'clustering',
        state_names: Optional[List[str]] = None,
        observation_names: Optional[List[str]] = None
    ):
        """
        Initialize Temporal Predictor.
        
        Args:
            model_type: 'markov' for observable states or 'hmm' for hidden states
            num_states: Number of hidden/observable states
            num_observations: Number of observation symbols (for HMM)
            encoding_type: Event encoding strategy
            state_names: Optional state labels
            observation_names: Optional observation labels
        """
        assert model_type in ['markov', 'hmm'], \
            f"Invalid model_type: {model_type}, must be 'markov' or 'hmm'"
        
        self.model_type = model_type
        
        # Default attack stage names
        if state_names is None:
            self.state_names = [
                "Normal",
                "Reconnaissance",
                "Exploitation",
                "Lateral_Movement",
                "Exfiltration"
            ][:num_states]
        else:
            self.state_names = state_names
        
        # Initialize encoder
        if num_observations is None:
            num_observations = num_states * 2  # Default: 2x observations per state
        
        self.encoder = EventEncoder(
            encoding_type=encoding_type,
            num_symbols=num_observations
        )
        
        # Initialize model
        if model_type == 'markov':
            self.model = MarkovChain(
                num_states=num_states,
                state_names=self.state_names
            )
        else:  # hmm
            self.model = HMMChain(
                num_hidden_states=num_states,
                num_observations=num_observations,
                state_names=self.state_names,
                observation_names=observation_names
            )
        
        self.is_fitted = False
        
        logger.info(f"Initialized TemporalPredictor: {model_type}, "
                   f"{num_states} states, {num_observations} observations")
    
    def fit(
        self,
        event_sequences: List[List[Any]],
        state_sequences: Optional[List[List[int]]] = None,
        encoding_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'TemporalPredictor':
        """
        Train temporal predictor on event sequences.
        
        Args:
            event_sequences: List of event/feature sequences
            state_sequences: Optional true state sequences (for supervised HMM)
            encoding_data: Optional data for fitting encoder (if different from sequences)
            **kwargs: Additional arguments for model training
        
        Returns:
            self: Trained predictor
        """
        logger.info(f"Training TemporalPredictor on {len(event_sequences)} sequences...")
        
        # Fit encoder
        if encoding_data is not None:
            logger.info("Fitting encoder on provided data...")
            self.encoder.fit(encoding_data)
        else:
            # Concatenate all sequences for encoder training
            logger.info("Fitting encoder on event sequences...")
            all_events = []
            for seq in event_sequences:
                all_events.extend(seq)
            self.encoder.fit(all_events)
        
        # Encode sequences
        logger.info("Encoding sequences...")
        encoded_sequences = self.encoder.transform_sequences(event_sequences)
        
        # Train model
        if self.model_type == 'markov':
            # For Markov Chain, encoded sequences are the states
            self.model.fit(encoded_sequences, **kwargs)
        else:  # hmm
            # For HMM, encoded sequences are observations
            self.model.fit(
                sequences=encoded_sequences,
                state_sequences=state_sequences,
                **kwargs
            )
        
        self.is_fitted = True
        logger.info("Training complete")
        
        return self
    
    def predict_next_state(
        self,
        event_sequence: List[Any],
        return_distribution: bool = False,
        return_confidence: bool = True
    ) -> Union[Tuple[int, str, float], Tuple[int, str, np.ndarray], int, str]:
        """
        Predict the next attack state given current event sequence.
        
        Args:
            event_sequence: Current sequence of events/features
            return_distribution: Return full probability distribution
            return_confidence: Return confidence score
        
        Returns:
            Various formats based on flags:
            - (state_idx, state_name, confidence)
            - (state_idx, state_name, distribution)
            - state_idx
            - state_name
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        
        # Encode sequence
        encoded_seq = self.encoder.transform(event_sequence).tolist()
        
        # Predict
        if self.model_type == 'markov':
            current_state = encoded_seq[-1]  # Last state
            result = self.model.predict_next_state(
                current_state,
                return_distribution=return_distribution
            )
        else:  # hmm
            result = self.model.predict_next_state(
                encoded_seq,
                return_distribution=return_distribution
            )
        
        # Format output
        if return_distribution:
            if isinstance(result, tuple):
                state_idx, dist = result
            else:
                dist = result
                state_idx = np.argmax(dist)
            
            state_name = self.state_names[state_idx]
            return state_idx, state_name, dist
        else:
            state_idx, confidence = result
            state_name = self.state_names[state_idx]
            
            if return_confidence:
                return state_idx, state_name, confidence
            else:
                return state_idx, state_name
    
    def predict_attack_sequence(
        self,
        initial_events: List[Any],
        horizon: int = 5
    ) -> Tuple[List[int], List[str], List[float]]:
        """
        Predict likely attack progression over time horizon.
        
        Args:
            initial_events: Starting event sequence
            horizon: Number of steps to predict ahead
        
        Returns:
            state_indices: Predicted state indices
            state_names: Predicted state names
            confidences: Confidence scores
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        
        if self.model_type == 'markov':
            # Encode and get current state
            encoded = self.encoder.transform(initial_events).tolist()
            current_state = encoded[-1]
            
            # Predict sequence
            state_seq, confidences = self.model.predict_sequence(
                current_state,
                horizon
            )
            
            state_names = [self.state_names[s] for s in state_seq]
            return state_seq, state_names, confidences
        
        else:  # hmm
            # For HMM, encode the initial sequence once
            encoded_seq = self.encoder.transform(initial_events).tolist()
            
            state_indices = []
            state_names = []
            confidences = []
            
            for _ in range(horizon):
                # Get next state prediction from encoded sequence
                next_state, conf = self.model.predict_next_state(encoded_seq)
                
                # Map to state name if available
                state_name = self.state_names[next_state] if self.state_names else f"State_{next_state}"
                
                state_indices.append(next_state)
                state_names.append(state_name)
                confidences.append(conf)
                
                # Append most likely observation for next iteration
                # In practice, this would be the actual observed event
                most_likely_obs = next_state % self.model.num_observations
                encoded_seq.append(most_likely_obs)
            
            return state_indices, state_names, confidences
    
    def compute_attack_probability(
        self,
        event_sequence: List[Any],
        target_state: Union[int, str]
    ) -> float:
        """
        Compute probability of reaching target attack state.
        
        Args:
            event_sequence: Current event sequence
            target_state: Target state (index or name)
        
        Returns:
            probability: Probability of target state
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        
        # Convert state name to index if needed
        if isinstance(target_state, str):
            target_state = self.state_names.index(target_state)
        
        # Get state distribution
        _, _, dist = self.predict_next_state(
            event_sequence,
            return_distribution=True
        )
        
        return dist[target_state]
    
    def get_uncertainty(
        self,
        event_sequence: List[Any]
    ) -> float:
        """
        Compute prediction uncertainty (entropy) for current sequence.
        
        Args:
            event_sequence: Current event sequence
        
        Returns:
            entropy: Uncertainty in bits
        """
        assert self.is_fitted, "Model must be fitted before prediction"
        
        # Get state distribution
        _, _, dist = self.predict_next_state(
            event_sequence,
            return_distribution=True
        )
        
        # Compute entropy
        entropy = -np.sum(dist * np.log2(dist + 1e-10))
        return entropy
    
    def analyze_sequence(
        self,
        event_sequence: List[Any]
    ) -> dict:
        """
        Comprehensive analysis of event sequence.
        
        Args:
            event_sequence: Event sequence to analyze
        
        Returns:
            analysis: Dictionary with predictions, uncertainties, etc.
        """
        assert self.is_fitted, "Model must be fitted before analysis"
        
        # Encode
        encoded_seq = self.encoder.transform(event_sequence).tolist()
        
        # Basic prediction
        state_idx, state_name, confidence = self.predict_next_state(
            event_sequence,
            return_confidence=True
        )
        
        # Uncertainty
        uncertainty = self.get_uncertainty(event_sequence)
        
        # State distribution
        _, _, distribution = self.predict_next_state(
            event_sequence,
            return_distribution=True
        )
        
        analysis = {
            'predicted_state_index': state_idx,
            'predicted_state_name': state_name,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'state_distribution': distribution.tolist(),
            'state_names': self.state_names,
            'encoded_sequence': encoded_seq
        }
        
        # HMM-specific analysis
        if self.model_type == 'hmm':
            # Viterbi decoding
            most_likely_states, log_prob = self.model.viterbi(encoded_seq)
            analysis['most_likely_state_sequence'] = most_likely_states
            analysis['sequence_log_probability'] = float(log_prob)
            
            # State probabilities over time
            state_probs = self.model.compute_state_probabilities(encoded_seq)
            analysis['state_probabilities_over_time'] = state_probs.tolist()
            
            # Uncertainty over time
            entropy = self.model.get_state_entropy(encoded_seq)
            analysis['uncertainty_over_time'] = entropy.tolist()
        
        return analysis
    
    def receive_temporal_context(
        self,
        event_sequence: List[Any]
    ) -> dict:
        """
        Interface for probabilistic reasoning engine to receive temporal context.
        
        This method provides a clean API for the main CogniThreat system
        to incorporate temporal predictions into Bayesian inference.
        
        Args:
            event_sequence: Current event sequence
        
        Returns:
            context: Dictionary with temporal predictions and uncertainties
        """
        analysis = self.analyze_sequence(event_sequence)
        
        # Format for integration with probabilistic reasoning
        context = {
            'temporal_prediction': {
                'next_state': analysis['predicted_state_name'],
                'confidence': analysis['confidence'],
                'uncertainty': analysis['uncertainty']
            },
            'attack_stage_probabilities': {
                name: prob for name, prob in 
                zip(analysis['state_names'], analysis['state_distribution'])
            },
            'risk_level': self._compute_risk_level(analysis),
            'metadata': {
                'model_type': self.model_type,
                'sequence_length': len(event_sequence)
            }
        }
        
        return context
    
    def _compute_risk_level(self, analysis: dict) -> str:
        """
        Compute risk level based on predicted state and confidence.
        
        Args:
            analysis: Analysis dictionary
        
        Returns:
            risk_level: 'low', 'medium', 'high', or 'critical'
        """
        state_name = analysis['predicted_state_name']
        confidence = analysis['confidence']
        
        # Risk based on attack stage
        high_risk_states = ['Exploitation', 'Lateral_Movement', 'Exfiltration']
        medium_risk_states = ['Reconnaissance']
        
        if state_name in high_risk_states and confidence > 0.7:
            return 'critical'
        elif state_name in high_risk_states:
            return 'high'
        elif state_name in medium_risk_states and confidence > 0.6:
            return 'high'
        elif state_name in medium_risk_states:
            return 'medium'
        else:
            return 'low'
    
    def update_online(
        self,
        event_sequence: List[Any],
        true_state: Optional[int] = None,
        learning_rate: float = 0.1
    ) -> 'TemporalPredictor':
        """
        Update model with new observations (online learning).
        
        Args:
            event_sequence: New event sequence
            true_state: Optional true state for supervised update
            learning_rate: Learning rate for updates
        
        Returns:
            self: Updated predictor
        """
        assert self.is_fitted, "Model must be fitted before online updates"
        
        # Encode
        encoded_seq = self.encoder.transform(event_sequence).tolist()
        
        # Update model
        if self.model_type == 'markov':
            self.model.update_online(encoded_seq, learning_rate)
        else:
            # For HMM, online learning is more complex
            # Here we do a single EM iteration
            logger.warning("Online learning for HMM not fully implemented")
        
        return self
    
    def save(self, filepath: str) -> None:
        """Save predictor to file."""
        import pickle
        
        predictor_data = {
            'model_type': self.model_type,
            'state_names': self.state_names,
            'encoder': self.encoder,
            'model': self.model,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(predictor_data, f)
        
        logger.info(f"TemporalPredictor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TemporalPredictor':
        """Load predictor from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            predictor_data = pickle.load(f)
        
        # Create instance
        predictor = cls.__new__(cls)
        predictor.model_type = predictor_data['model_type']
        predictor.state_names = predictor_data['state_names']
        predictor.encoder = predictor_data['encoder']
        predictor.model = predictor_data['model']
        predictor.is_fitted = predictor_data['is_fitted']
        
        logger.info(f"TemporalPredictor loaded from {filepath}")
        return predictor
    
    def __repr__(self) -> str:
        return (f"TemporalPredictor(model={self.model_type}, "
                f"states={len(self.state_names)}, fitted={self.is_fitted})")
