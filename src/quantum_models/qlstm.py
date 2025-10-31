"""
Quantum LSTM Implementation for CogniThreat
==========================================

This module implements a Quantum Long Short-Term Memory (QLSTM) network
for temporal pattern recognition in network intrusion detection.

The QLSTM combines classical LSTM architecture with quantum processing
to capture complex temporal dependencies in network traffic data.

Classes:
    QuantumLSTMCell: Core quantum LSTM cell
    QuantumLSTM: Full QLSTM layer for integration with TensorFlow
    
Author: CogniThreat Team
Date: August 2025
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
import tensorflow as tf
from typing import Tuple, Optional, List
import warnings
from .quantum_layers import QuantumLayer, QuantumCircuitBuilder

warnings.filterwarnings('ignore', category=UserWarning)


class QuantumLSTMCell:
    """
    Quantum LSTM Cell implementing quantum gates for memory operations.
    
    This cell replaces classical LSTM gates with quantum circuits that can
    process superposition states and capture quantum correlations in data.
    """
    
    def __init__(self, 
                 n_qubits: int = 4, 
                 n_layers: int = 2,
                 memory_dim: int = 8):
        """
        Initialize Quantum LSTM Cell.
        
        Args:
            n_qubits: Number of qubits in quantum circuits
            n_layers: Number of variational layers in quantum gates
            memory_dim: Dimension of classical memory state
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.memory_dim = memory_dim
        
        # Create quantum devices for different gates
        self.device = qml.device('default.qubit', wires=n_qubits)
        
        # Initialize circuit builders for each gate
        self.forget_gate_builder = QuantumCircuitBuilder(n_qubits, n_layers)
        self.input_gate_builder = QuantumCircuitBuilder(n_qubits, n_layers)
        self.output_gate_builder = QuantumCircuitBuilder(n_qubits, n_layers)
        self.candidate_builder = QuantumCircuitBuilder(n_qubits, n_layers)
        
        # Build quantum nodes for LSTM gates
        self.forget_gate_qnode = self._build_forget_gate()
        self.input_gate_qnode = self._build_input_gate()
        self.output_gate_qnode = self._build_output_gate()
        self.candidate_qnode = self._build_candidate_gate()
    
    def _build_forget_gate(self):
        """Build quantum circuit for forget gate."""
        @qml.qnode(self.device, interface='tf')
        def forget_gate_circuit(inputs, hidden_state, weights):
            # Encode inputs and hidden state
            combined_input = tf.concat([inputs, hidden_state], axis=0)
            normalized_input = combined_input[:self.n_qubits]
            
            # Data encoding
            self.forget_gate_builder.data_encoding_layer(normalized_input)
            
            # Variational layers
            for layer in range(self.n_layers):
                self.forget_gate_builder.variational_layer(weights, layer)
            
            # Measure in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return forget_gate_circuit
    
    def _build_input_gate(self):
        """Build quantum circuit for input gate."""
        @qml.qnode(self.device, interface='tf')
        def input_gate_circuit(inputs, hidden_state, weights):
            # Encode inputs and hidden state
            combined_input = tf.concat([inputs, hidden_state], axis=0)
            normalized_input = combined_input[:self.n_qubits]
            
            # Data encoding with phase rotation
            for i, val in enumerate(normalized_input):
                if i < self.n_qubits:
                    qml.RY(val, wires=i)
                    qml.RZ(val * 0.5, wires=i)  # Additional phase information
            
            # Variational layers
            for layer in range(self.n_layers):
                self.input_gate_builder.variational_layer(weights, layer)
            
            return [qml.expval(qml.PauliY(i)) for i in range(self.n_qubits)]
        
        return input_gate_circuit
    
    def _build_output_gate(self):
        """Build quantum circuit for output gate."""
        @qml.qnode(self.device, interface='tf')
        def output_gate_circuit(inputs, hidden_state, cell_state, weights):
            # Encode all three states
            combined_input = tf.concat([inputs, hidden_state, cell_state], axis=0)
            normalized_input = combined_input[:self.n_qubits]
            
            # Complex encoding with multiple rotation axes
            for i, val in enumerate(normalized_input):
                if i < self.n_qubits:
                    qml.RX(val, wires=i)
                    qml.RY(val * 0.7, wires=i)
                    qml.RZ(val * 0.3, wires=i)
            
            # Entangling gates for information mixing
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Variational layers
            for layer in range(self.n_layers):
                self.output_gate_builder.variational_layer(weights, layer)
            
            return [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
        
        return output_gate_circuit
    
    def _build_candidate_gate(self):
        """Build quantum circuit for candidate values."""
        @qml.qnode(self.device, interface='tf')
        def candidate_circuit(inputs, hidden_state, weights):
            # Encode inputs and hidden state
            combined_input = tf.concat([inputs, hidden_state], axis=0)
            normalized_input = combined_input[:self.n_qubits]
            
            # Advanced encoding with Hadamard gates for superposition
            for i, val in enumerate(normalized_input):
                if i < self.n_qubits:
                    qml.Hadamard(wires=i)  # Create superposition
                    qml.RY(val, wires=i)   # Rotate based on input
            
            # Create entanglement patterns
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            
            # Variational layers
            for layer in range(self.n_layers):
                self.candidate_builder.variational_layer(weights, layer)
            
            return [qml.expval(qml.PauliZ(i) @ qml.PauliX((i + 1) % self.n_qubits)) 
                   for i in range(self.n_qubits)]
        
        return candidate_circuit


class QuantumLSTM(tf.keras.layers.Layer):
    """
    Quantum LSTM Layer for TensorFlow integration.
    
    This layer implements a full Quantum LSTM that can be used as a drop-in
    replacement for classical LSTM in neural network architectures.
    """
    
    def __init__(self, 
                 units: int,
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 name: str = "quantum_lstm",
                 **kwargs):
        """
        Initialize Quantum LSTM layer.
        
        Args:
            units: Number of LSTM units (output dimension)
            n_qubits: Number of qubits in quantum circuits
            n_layers: Number of variational layers
            return_sequences: Whether to return full sequence or last output
            return_state: Whether to return final states
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.units = units
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Initialize quantum LSTM cell
        self.quantum_cell = QuantumLSTMCell(n_qubits, n_layers, units)
        
        # Classical post-processing layers
        self.classical_projection = tf.keras.layers.Dense(units, activation='tanh')
        self.output_projection = tf.keras.layers.Dense(units)
    
    def build(self, input_shape):
        """Build layer weights."""
        # Quantum gate weights
        n_params = self.quantum_cell.forget_gate_builder.n_params
        
        self.forget_weights = self.add_weight(
            name='forget_weights',
            shape=(n_params,),
            initializer='random_uniform',
            trainable=True
        )
        
        self.input_weights = self.add_weight(
            name='input_weights', 
            shape=(n_params,),
            initializer='random_uniform',
            trainable=True
        )
        
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(n_params,),
            initializer='random_uniform', 
            trainable=True
        )
        
        self.candidate_weights = self.add_weight(
            name='candidate_weights',
            shape=(n_params,),
            initializer='random_uniform',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs, initial_state=None, training=None):
        """Forward pass through Quantum LSTM."""
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]
        input_dim = tf.shape(inputs)[2]
        
        # Initialize states
        if initial_state is None:
            hidden_state = tf.zeros([batch_size, self.units])
            cell_state = tf.zeros([batch_size, self.units])
        else:
            hidden_state, cell_state = initial_state
        
        # Process sequence
        outputs = []
        
        for t in range(sequence_length):
            current_input = inputs[:, t, :]
            
            # Apply quantum LSTM cell
            hidden_state, cell_state = self._quantum_lstm_step(
                current_input, hidden_state, cell_state
            )
            
            if self.return_sequences:
                outputs.append(hidden_state)
        
        # Prepare outputs
        if self.return_sequences:
            final_output = tf.stack(outputs, axis=1)
        else:
            final_output = hidden_state
        
        if self.return_state:
            return final_output, hidden_state, cell_state
        else:
            return final_output
    
    def _quantum_lstm_step(self, inputs, hidden_state, cell_state):
        """Single step of quantum LSTM computation."""
        batch_size = tf.shape(inputs)[0]
        
        def process_sample(sample_data):
            """Process a single sample through quantum gates."""
            sample_input, sample_hidden, sample_cell = sample_data
            
            # Normalize inputs for quantum processing
            normalized_input = tf.nn.l2_normalize(sample_input)
            normalized_hidden = tf.nn.l2_normalize(sample_hidden)
            normalized_cell = tf.nn.l2_normalize(sample_cell)
            
            # Quantum forget gate
            forget_output = self.quantum_cell.forget_gate_qnode(
                normalized_input, normalized_hidden, self.forget_weights
            )
            forget_gate = tf.nn.sigmoid(
                self.classical_projection(tf.stack(forget_output))
            )
            
            # Quantum input gate
            input_output = self.quantum_cell.input_gate_qnode(
                normalized_input, normalized_hidden, self.input_weights
            )
            input_gate = tf.nn.sigmoid(
                self.classical_projection(tf.stack(input_output))
            )
            
            # Quantum candidate values
            candidate_output = self.quantum_cell.candidate_qnode(
                normalized_input, normalized_hidden, self.candidate_weights
            )
            candidate_values = tf.nn.tanh(
                self.classical_projection(tf.stack(candidate_output))
            )
            
            # Update cell state
            new_cell_state = forget_gate * sample_cell + input_gate * candidate_values
            
            # Quantum output gate
            output_output = self.quantum_cell.output_gate_qnode(
                normalized_input, normalized_hidden, normalized_cell, self.output_weights
            )
            output_gate = tf.nn.sigmoid(
                self.classical_projection(tf.stack(output_output))
            )
            
            # Update hidden state
            new_hidden_state = output_gate * tf.nn.tanh(new_cell_state)
            
            return new_hidden_state, new_cell_state
        
        # Process batch
        sample_data = (inputs, hidden_state, cell_state)
        new_states = tf.map_fn(
            process_sample,
            sample_data,
            dtype=(tf.float32, tf.float32),
            parallel_iterations=1
        )
        
        return new_states
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config


def create_quantum_lstm_model(input_shape: Tuple[int, int],
                             n_classes: int,
                             lstm_units: int = 64,
                             n_qubits: int = 4,
                             n_layers: int = 2) -> tf.keras.Model:
    """
    Create a complete Quantum LSTM model for intrusion detection.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        n_classes: Number of output classes
        lstm_units: Number of LSTM units
        n_qubits: Number of qubits in quantum circuits
        n_layers: Number of variational layers
        
    Returns:
        Compiled TensorFlow model
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name='sequence_input')
    
    # Quantum LSTM layer
    qlstm_output = QuantumLSTM(
        units=lstm_units,
        n_qubits=n_qubits,
        n_layers=n_layers,
        return_sequences=False,
        name='quantum_lstm'
    )(inputs)
    
    # Classical post-processing
    dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')(qlstm_output)
    dropout1 = tf.keras.layers.Dropout(0.3, name='dropout1')(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu', name='dense2')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2, name='dropout2')(dense2)
    
    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(dropout2)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='QuantumLSTM_Model')
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def prepare_sequence_data(X: np.ndarray, 
                         y: np.ndarray, 
                         sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for sequence modeling with Quantum LSTM.
    
    Args:
        X: Feature matrix
        y: Labels
        sequence_length: Length of sequences to create
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    
    # Create sequences
    X_sequences = np.zeros((n_sequences, sequence_length, n_features))
    y_sequences = y[sequence_length-1:]
    
    for i in range(n_sequences):
        X_sequences[i] = X[i:i+sequence_length]
    
    return X_sequences, y_sequences


def quantum_lstm_metrics(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        model: tf.keras.Model) -> dict:
    """
    Calculate comprehensive metrics for Quantum LSTM model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model: Trained model
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'model_parameters': model.count_params(),
        'quantum_parameters': sum([
            layer.count_params() for layer in model.layers 
            if isinstance(layer, QuantumLSTM)
        ])
    }
    
    return metrics
