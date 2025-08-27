"""
Quantum Circuit Layers for CogniThreat
=====================================

This module provides fundamental quantum circuit building blocks for
quantum deep learning models in network intrusion detection.

Classes:
    QuantumLayer: Base class for quantum layers
    QuantumCircuitBuilder: Utility for building parameterized quantum circuits
    
Author: CogniThreat Team
Date: August 2025
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
import tensorflow as tf
from typing import List, Tuple, Optional, Callable
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)


class QuantumCircuitBuilder:
    """
    Utility class for building parameterized quantum circuits for deep learning.
    
    This class provides methods to construct common quantum circuit patterns
    used in quantum machine learning, specifically tailored for intrusion detection.
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 1):
        """
        Initialize quantum circuit builder.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of variational layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = self._calculate_params()
    
    def _calculate_params(self) -> int:
        """Calculate total number of parameters needed."""
        # Each layer has rotation gates (3 params per qubit) + entangling gates
        params_per_layer = 3 * self.n_qubits
        return params_per_layer * self.n_layers
    
    def data_encoding_layer(self, inputs: np.ndarray) -> None:
        """
        Encode classical data into quantum states using angle encoding.
        
        Args:
            inputs: Classical input features (normalized to [0, π])
        """
        # Normalize inputs to [0, π] range for angle encoding
        normalized_inputs = np.pi * inputs / (np.max(np.abs(inputs)) + 1e-8)
        
        for i in range(min(len(normalized_inputs), self.n_qubits)):
            qml.RY(normalized_inputs[i], wires=i)
    
    def variational_layer(self, params: np.ndarray, layer_idx: int = 0) -> None:
        """
        Apply a variational layer with parameterized rotations and entanglement.
        
        Args:
            params: Parameters for the variational layer
            layer_idx: Index of the current layer
        """
        start_idx = layer_idx * 3 * self.n_qubits
        
        # Parameterized rotation gates
        for i in range(self.n_qubits):
            param_idx = start_idx + i * 3
            qml.RX(params[param_idx], wires=i)
            qml.RY(params[param_idx + 1], wires=i)
            qml.RZ(params[param_idx + 2], wires=i)
        
        # Entangling layer (circular connectivity)
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def measurement_layer(self, observables: Optional[List] = None) -> List:
        """
        Define measurement observables for the circuit output.
        
        Args:
            observables: Custom observables, if None uses PauliZ on all qubits
            
        Returns:
            List of expectation values
        """
        if observables is None:
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        else:
            return [qml.expval(obs) for obs in observables]


class QuantumLayer(tf.keras.layers.Layer):
    """
    TensorFlow Keras layer for quantum circuits.
    
    This layer integrates quantum circuits into classical neural networks,
    enabling hybrid quantum-classical deep learning for intrusion detection.
    """
    
    def __init__(self, 
                 n_qubits: int, 
                 n_layers: int = 1,
                 output_dim: int = None,
                 name: str = "quantum_layer",
                 **kwargs):
        """
        Initialize quantum layer.
        
        Args:
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of variational layers
            output_dim: Dimension of classical output (default: n_qubits)
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim or n_qubits
        self.circuit_builder = QuantumCircuitBuilder(n_qubits, n_layers)
        
        # Initialize quantum device (simulator)
        self.device = qml.device('default.qubit', wires=n_qubits)
        
        # Build quantum circuit
        self.qnode = self._build_qnode()
    
    def build(self, input_shape):
        """Build the layer parameters."""
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=(self.circuit_builder.n_params,),
            initializer='random_uniform',
            trainable=True
        )
        
        # Optional classical post-processing weights
        self.post_processing = tf.keras.layers.Dense(
            self.output_dim,
            activation='tanh',
            name=f"{self.name}_post_processing"
        )
        
        super().build(input_shape)
    
    def _build_qnode(self) -> Callable:
        """Build the quantum node (QNode) for PennyLane."""
        
        @qml.qnode(self.device, interface='tf')
        def quantum_circuit(inputs, weights):
            # Data encoding
            self.circuit_builder.data_encoding_layer(inputs)
            
            # Variational layers
            for layer in range(self.n_layers):
                self.circuit_builder.variational_layer(weights, layer)
            
            # Measurements
            return self.circuit_builder.measurement_layer()
        
        return quantum_circuit
    
    def call(self, inputs):
        """Forward pass through quantum layer."""
        batch_size = tf.shape(inputs)[0]
        
        # Process each sample in the batch
        def process_sample(sample):
            # Ensure input is properly normalized
            normalized_sample = tf.nn.l2_normalize(sample, axis=0)
            
            # Execute quantum circuit
            quantum_output = self.qnode(normalized_sample, self.quantum_weights)
            return quantum_output
        
        # Map over batch
        quantum_outputs = tf.map_fn(
            process_sample,
            inputs,
            dtype=tf.float32,
            parallel_iterations=1  # Sequential for stability
        )
        
        # Classical post-processing
        classical_output = self.post_processing(quantum_outputs)
        
        return classical_output
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'output_dim': self.output_dim
        })
        return config


class QuantumPooling(tf.keras.layers.Layer):
    """
    Quantum pooling layer for quantum CNNs.
    
    Implements quantum pooling operations that preserve quantum information
    while reducing dimensionality.
    """
    
    def __init__(self, pool_size: int = 2, name: str = "quantum_pooling", **kwargs):
        """
        Initialize quantum pooling layer.
        
        Args:
            pool_size: Size of pooling window
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.pool_size = pool_size
    
    def call(self, inputs):
        """Apply quantum pooling operation."""
        # Simple implementation: average pooling with quantum-inspired weights
        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[1]
        
        # Reshape for pooling
        pooled_dim = input_dim // self.pool_size
        reshaped = tf.reshape(inputs, [batch_size, pooled_dim, self.pool_size])
        
        # Apply quantum-weighted average
        # Use entanglement-inspired weights
        weights = tf.constant([0.6, 0.4] if self.pool_size == 2 else 
                             [1.0 / self.pool_size] * self.pool_size, dtype=tf.float32)
        pooled = tf.reduce_sum(reshaped * weights, axis=2)
        
        return pooled
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({'pool_size': self.pool_size})
        return config


def create_quantum_device(n_qubits: int, device_type: str = 'default.qubit') -> qml.Device:
    """
    Create a quantum device for circuit execution.
    
    Args:
        n_qubits: Number of qubits
        device_type: Type of quantum device
        
    Returns:
        PennyLane quantum device
    """
    return qml.device(device_type, wires=n_qubits)


def quantum_feature_map(features: np.ndarray, n_qubits: int) -> Callable:
    """
    Create a quantum feature map for encoding classical data.
    
    Args:
        features: Classical features to encode
        n_qubits: Number of qubits available
        
    Returns:
        Quantum feature map function
    """
    def feature_map(x):
        """Apply feature map encoding."""
        # Normalize features to [0, π]
        normalized = np.pi * x / (np.max(np.abs(x)) + 1e-8)
        
        # Apply rotation gates
        for i in range(min(len(normalized), n_qubits)):
            qml.RY(normalized[i], wires=i)
        
        # Add entangling gates for feature correlation
        for i in range(min(len(normalized) - 1, n_qubits - 1)):
            qml.CNOT(wires=[i, i + 1])
    
    return feature_map


# Export quantum metrics for model evaluation
def quantum_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate quantum fidelity between two states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity value between 0 and 1
    """
    return np.abs(np.vdot(state1, state2)) ** 2


def quantum_entanglement_entropy(state: np.ndarray, subsystem_size: int) -> float:
    """
    Calculate entanglement entropy of a quantum state.
    
    Args:
        state: Quantum state vector
        subsystem_size: Size of subsystem for partial trace
        
    Returns:
        Entanglement entropy value
    """
    # Simplified implementation for demonstration
    # In practice, would use proper density matrix calculations
    n_qubits = int(np.log2(len(state)))
    density_matrix = np.outer(state, np.conj(state))
    
    # Calculate reduced density matrix (simplified)
    # This is a placeholder - proper implementation would use tensor operations
    eigenvals = np.linalg.eigvals(density_matrix)
    eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
    
    entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
    return entropy
