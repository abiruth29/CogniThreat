"""
Temporal Reasoning Integration with CogniThreat System

This script demonstrates how the temporal reasoning module integrates
with the existing quantum models and probabilistic reasoning pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.temporal_reasoning import TemporalPredictor
from src.probabilistic_reasoning.fusion import BayesianFusion
from src.probabilistic_reasoning.uncertainty import UncertaintyQuantifier
from src.probabilistic_reasoning.risk_inference import RiskInference

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CogniThreatTemporalIntegration:
    """
    Integration layer connecting temporal reasoning with CogniThreat's
    quantum models and Bayesian probabilistic reasoning.
    """
    
    def __init__(
        self,
        data_path: str = None,
        use_hmm: bool = True,
        num_attack_stages: int = 5
    ):
        """
        Initialize integrated system.
        
        Args:
            data_path: Path to network traffic data
            use_hmm: Use HMM (True) or Markov Chain (False)
            num_attack_stages: Number of attack stages to model
        """
        self.data_path = data_path
        self.use_hmm = use_hmm
        self.num_attack_stages = num_attack_stages
        
        # Initialize components
        self.temporal_predictor = None
        self.bayesian_fusion = None
        self.uncertainty_quantifier = None
        self.risk_inference = None
        
        # Attack stage definitions
        self.attack_stages = [
            "Normal",
            "Reconnaissance",
            "Exploitation",
            "Lateral_Movement",
            "Exfiltration"
        ][:num_attack_stages]
        
        logger.info("CogniThreat Temporal Integration initialized")
    
    def load_and_prepare_data(self, sample_size: int = 10000):
        """
        Load network traffic data and prepare temporal sequences.
        
        Args:
            sample_size: Number of samples to load
        """
        logger.info("Loading and preparing data...")
        
        if self.data_path and Path(self.data_path).exists():
            # Load real data
            df = pd.read_csv(self.data_path, nrows=sample_size)
            logger.info(f"Loaded {len(df)} samples from {self.data_path}")
        else:
            # Generate synthetic data
            logger.info("Generating synthetic network traffic data...")
            df = self._generate_synthetic_data(sample_size)
        
        # Extract features and labels
        if ' Label' in df.columns:
            labels = df[' Label'].values
            features = df.drop(columns=[' Label']).values
        else:
            # Assume last column is label
            labels = df.iloc[:, -1].values
            features = df.iloc[:, :-1].values
        
        # Map labels to attack stages
        self.label_to_stage = self._create_label_mapping(labels)
        stage_labels = np.array([self.label_to_stage.get(str(l), 0) for l in labels])
        
        # Create temporal sequences (sliding windows)
        self.event_sequences, self.state_sequences = self._create_sequences(
            features, stage_labels, window_size=10
        )
        
        logger.info(f"Created {len(self.event_sequences)} temporal sequences")
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic network traffic data."""
        np.random.seed(42)
        
        # Generate features
        n_features = 10
        features = np.random.randn(n_samples, n_features)
        
        # Generate labels (attack types)
        labels = np.random.choice(
            ['BENIGN', 'DoS', 'DDoS', 'PortScan', 'Brute Force'],
            size=n_samples,
            p=[0.6, 0.15, 0.1, 0.1, 0.05]
        )
        
        # Create DataFrame
        columns = [f'Feature_{i}' for i in range(n_features)] + [' Label']
        data = np.column_stack([features, labels])
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def _create_label_mapping(self, labels: np.ndarray) -> dict:
        """Map traffic labels to attack stages."""
        mapping = {
            'BENIGN': 0,  # Normal
            'PortScan': 1,  # Reconnaissance
            'DoS': 2,  # Exploitation
            'DDoS': 2,  # Exploitation
            'Brute Force': 3,  # Lateral Movement
            'Web Attack': 3,  # Lateral Movement
            'Infiltration': 4,  # Exfiltration
            'Botnet': 4  # Exfiltration
        }
        
        # Add any unknown labels as Normal
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if str(label) not in mapping:
                mapping[str(label)] = 0
        
        return mapping
    
    def _create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int = 10
    ) -> tuple:
        """Create temporal sequences using sliding window."""
        event_sequences = []
        state_sequences = []
        
        for i in range(len(features) - window_size):
            event_seq = features[i:i+window_size].tolist()
            state_seq = labels[i:i+window_size].tolist()
            
            event_sequences.append(event_seq)
            state_sequences.append(state_seq)
        
        return event_sequences, state_sequences
    
    def train_temporal_model(self, supervised: bool = True):
        """
        Train temporal predictor.
        
        Args:
            supervised: Use supervised learning (requires state labels)
        """
        logger.info("Training temporal predictor...")
        
        # Initialize temporal predictor
        model_type = 'hmm' if self.use_hmm else 'markov'
        
        self.temporal_predictor = TemporalPredictor(
            model_type=model_type,
            num_states=self.num_attack_stages,
            num_observations=self.num_attack_stages * 2,
            encoding_type='clustering',
            state_names=self.attack_stages
        )
        
        # Train
        if supervised and self.state_sequences:
            self.temporal_predictor.fit(
                self.event_sequences,
                state_sequences=self.state_sequences,
                max_iterations=50
            )
        else:
            self.temporal_predictor.fit(
                self.event_sequences,
                max_iterations=50
            )
        
        logger.info("Temporal predictor trained successfully")
    
    def initialize_probabilistic_reasoning(self, num_models: int = 3):
        """Initialize Bayesian fusion and uncertainty quantification."""
        logger.info("Initializing probabilistic reasoning components...")
        
        # Bayesian Fusion
        self.bayesian_fusion = BayesianFusion(num_models=num_models)
        
        # Uncertainty Quantifier
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Risk Inference
        self.risk_inference = RiskInference()
        
        logger.info("Probabilistic reasoning components initialized")
    
    def predict_with_temporal_context(
        self,
        event_sequence: list,
        quantum_predictions: np.ndarray = None,
        classical_predictions: np.ndarray = None
    ) -> dict:
        """
        Make prediction incorporating temporal context.
        
        Args:
            event_sequence: Current event sequence
            quantum_predictions: Predictions from quantum model
            classical_predictions: Predictions from classical model
        
        Returns:
            integrated_result: Combined predictions with temporal context
        """
        # Get temporal predictions
        temporal_context = self.temporal_predictor.receive_temporal_context(
            event_sequence
        )
        
        # Analyze sequence
        temporal_analysis = self.temporal_predictor.analyze_sequence(event_sequence)
        
        # If we have model predictions, fuse them
        if quantum_predictions is not None and classical_predictions is not None:
            # Create model predictions list
            model_predictions = [quantum_predictions, classical_predictions]
            
            # Add temporal prediction as a model
            temporal_pred = np.array(temporal_analysis['state_distribution'])
            model_predictions.append(temporal_pred)
            
            # Fuse predictions
            fused_probs = self.bayesian_fusion.weighted_average(
                np.array(model_predictions)
            )
            
            # Compute uncertainty
            uncertainty = self.uncertainty_quantifier.compute_total_uncertainty(
                np.array(model_predictions)
            )
        else:
            # Use only temporal predictions
            fused_probs = np.array(temporal_analysis['state_distribution'])
            uncertainty = temporal_analysis['uncertainty']
        
        # Compute risk
        predicted_class = np.argmax(fused_probs)
        risk_score = self.risk_inference.compute_risk_score(
            fused_probs,
            predicted_class
        )
        
        # Compile result
        result = {
            'predicted_stage': self.attack_stages[predicted_class],
            'predicted_stage_index': int(predicted_class),
            'confidence': float(fused_probs[predicted_class]),
            'uncertainty': float(uncertainty),
            'risk_score': float(risk_score),
            'risk_level': temporal_context['risk_level'],
            'stage_probabilities': {
                stage: float(prob) 
                for stage, prob in zip(self.attack_stages, fused_probs)
            },
            'temporal_context': temporal_context,
            'temporal_analysis': {
                'encoded_sequence': temporal_analysis.get('encoded_sequence', []),
                'next_stage_confidence': temporal_analysis['confidence']
            }
        }
        
        # Add HMM-specific information
        if self.use_hmm and 'most_likely_state_sequence' in temporal_analysis:
            result['temporal_analysis']['most_likely_states'] = [
                self.attack_stages[s] 
                for s in temporal_analysis['most_likely_state_sequence']
            ]
            result['temporal_analysis']['sequence_log_probability'] = \
                temporal_analysis['sequence_log_probability']
        
        return result
    
    def forecast_attack_progression(
        self,
        current_sequence: list,
        horizon: int = 5
    ) -> dict:
        """
        Forecast likely attack progression.
        
        Args:
            current_sequence: Current event sequence
            horizon: Number of steps to forecast
        
        Returns:
            forecast: Predicted attack stages and confidences
        """
        state_indices, state_names, confidences = \
            self.temporal_predictor.predict_attack_sequence(
                current_sequence, horizon
            )
        
        forecast = {
            'forecast_horizon': horizon,
            'predicted_stages': state_names,
            'confidences': [float(c) for c in confidences],
            'trajectory': [
                {
                    'step': i + 1,
                    'stage': state_names[i],
                    'stage_index': int(state_indices[i]),
                    'confidence': float(confidences[i])
                }
                for i in range(horizon)
            ]
        }
        
        # Identify critical transitions
        critical_stages = ['Exploitation', 'Lateral_Movement', 'Exfiltration']
        critical_transitions = [
            step for step in forecast['trajectory']
            if step['stage'] in critical_stages and step['confidence'] > 0.6
        ]
        
        forecast['critical_transitions'] = critical_transitions
        forecast['has_critical_risk'] = len(critical_transitions) > 0
        
        return forecast
    
    def run_demonstration(self):
        """Run complete demonstration of temporal integration."""
        print("\n" + "="*70)
        print(" CogniThreat Temporal Reasoning Integration Demonstration")
        print("="*70 + "\n")
        
        # Step 1: Load data
        print("Step 1: Loading and preparing data...")
        self.load_and_prepare_data(sample_size=5000)
        print(f"  ✓ Created {len(self.event_sequences)} temporal sequences\n")
        
        # Step 2: Train temporal model
        print("Step 2: Training temporal predictor...")
        self.train_temporal_model(supervised=True)
        print(f"  ✓ {self.temporal_predictor}\n")
        
        # Step 3: Initialize probabilistic reasoning
        print("Step 3: Initializing probabilistic reasoning...")
        self.initialize_probabilistic_reasoning()
        print("  ✓ Bayesian fusion, uncertainty quantification, and risk inference ready\n")
        
        # Step 4: Make predictions
        print("Step 4: Making predictions with temporal context...")
        
        # Select test sequence
        test_sequence = self.event_sequences[100]
        
        # Simulate quantum and classical predictions
        quantum_pred = np.array([0.1, 0.2, 0.5, 0.15, 0.05])
        classical_pred = np.array([0.15, 0.15, 0.45, 0.2, 0.05])
        
        result = self.predict_with_temporal_context(
            test_sequence,
            quantum_pred,
            classical_pred
        )
        
        print(f"  Predicted Stage: {result['predicted_stage']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Uncertainty: {result['uncertainty']:.4f}")
        print(f"  Risk Score: {result['risk_score']:.4f}")
        print(f"  Risk Level: {result['risk_level']}\n")
        
        print("  Stage Probabilities:")
        for stage, prob in result['stage_probabilities'].items():
            print(f"    {stage:20s}: {prob:.4f}")
        print()
        
        # Step 5: Forecast attack progression
        print("Step 5: Forecasting attack progression...")
        forecast = self.forecast_attack_progression(test_sequence, horizon=5)
        
        print(f"  Forecast Horizon: {forecast['forecast_horizon']} steps")
        print("  Predicted Trajectory:")
        for step in forecast['trajectory']:
            print(f"    Step {step['step']}: {step['stage']:20s} "
                  f"(confidence: {step['confidence']:.4f})")
        
        if forecast['has_critical_risk']:
            print(f"\n  ⚠ WARNING: {len(forecast['critical_transitions'])} "
                  f"critical transition(s) detected!")
            for trans in forecast['critical_transitions']:
                print(f"    → Step {trans['step']}: {trans['stage']} "
                      f"(confidence: {trans['confidence']:.4f})")
        print()
        
        # Step 6: Demonstrate online learning
        print("Step 6: Online learning demonstration...")
        new_sequence = self.event_sequences[200]
        self.temporal_predictor.update_online(new_sequence, learning_rate=0.1)
        print("  ✓ Model updated with new observations\n")
        
        print("="*70)
        print(" Demonstration Complete!")
        print("="*70 + "\n")
        
        return result, forecast


def main():
    """Main demonstration function."""
    # Create integration instance
    integration = CogniThreatTemporalIntegration(
        use_hmm=True,
        num_attack_stages=5
    )
    
    # Run demonstration
    result, forecast = integration.run_demonstration()
    
    print("\n📊 Integration Benefits:")
    print("  • Temporal context improves attack stage prediction")
    print("  • Early warning for attack progressions")
    print("  • Uncertainty quantification guides analyst decisions")
    print("  • Online learning adapts to evolving threats")
    print("  • Seamless integration with quantum/classical models\n")


if __name__ == '__main__':
    main()
