"""
Quantum CNN Implementation for CogniThreat
==========================================

This module implements a Quantum Convolutional Neural Network (QCNN)
for spatial pattern recognition in network intrusion detection.

The QCNN uses quantum circuits to implement convolution and pooling operations
that can capture quantum correlations and spatial dependencies in network data.

Classes:
    QuantumConvLayer: Quantum convolution layer
    QuantumPoolingLayer: Quantum pooling layer  
    QuantumCNN: Full QCNN model for TensorFlow integration
    
Author: CogniThreat Team
Date: August 2025
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
import tensorflow as tf
from typing import Tuple, Optional, List, Dict
import warnings
from .quantum_layers import QuantumLayer, QuantumCircuitBuilder, QuantumPooling

warnings.filterwarnings('ignore', category=UserWarning)


class QuantumConvLayer(tf.keras.layers.Layer):
    """
    Quantum Convolution Layer implementing quantum filters.
    
    This layer applies quantum convolution operations using parameterized
    quantum circuits that act as learnable quantum filters.
    """
    
    def __init__(self,
                 n_filters: int,
                 filter_size: int = 2,
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 stride: int = 1,
                 name: str = "quantum_conv",
                 **kwargs):
        """
        Initialize Quantum Convolution Layer.
        
        Args:
            n_filters: Number of quantum filters
            filter_size: Size of each filter
            n_qubits: Number of qubits per filter
            n_layers: Number of variational layers
            stride: Convolution stride
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.stride = stride
        
        # Create quantum devices for filters
        self.devices = [qml.device('default.qubit', wires=n_qubits) 
                       for _ in range(n_filters)]
        
        # Build quantum filter circuits
        self.filter_circuits = [self._build_filter_circuit(i) 
                               for i in range(n_filters)]
    
    def _build_filter_circuit(self, filter_idx: int):
        """Build quantum circuit for a single filter."""
        
        @qml.qnode(self.devices[filter_idx], interface='tf')
        def filter_circuit(patch, weights):
            """
            Quantum filter circuit for convolution operation.
            
            Args:
                patch: Input patch from feature map
                weights: Filter parameters
                
            Returns:
                Quantum filter output
            """
            # Encode input patch
            self._encode_patch(patch)
            
            # Apply variational layers
            for layer in range(self.n_layers):
                self._variational_filter_layer(weights, layer)
            
            # Measure filter response
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return filter_circuit
    
    def _encode_patch(self, patch):
        """Encode input patch into quantum state."""
        # Normalize patch for quantum encoding
        normalized_patch = tf.nn.l2_normalize(patch)
        
        # Angle encoding with rotation gates
        for i in range(min(len(normalized_patch), self.n_qubits)):
            qml.RY(normalized_patch[i], wires=i)
        
        # Add entanglement for spatial correlation
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    def _variational_filter_layer(self, weights, layer_idx):
        """Apply variational layer for quantum filter."""
        start_idx = layer_idx * self.n_qubits * 3
        
        # Parameterized rotations
        for i in range(self.n_qubits):
            param_idx = start_idx + i * 3
            qml.RX(weights[param_idx], wires=i)
            qml.RY(weights[param_idx + 1], wires=i)
            qml.RZ(weights[param_idx + 2], wires=i)
        
        # Controlled entangling gates
        for i in range(0, self.n_qubits - 1, 2):
            qml.CRY(weights[start_idx + i], wires=[i, i + 1])
    
    def build(self, input_shape):
        """Build layer parameters."""
        # Calculate number of parameters per filter
        n_params_per_filter = self.n_layers * self.n_qubits * 3
        
        # Create filter weights
        self.filter_weights = []
        for i in range(self.n_filters):
            filter_weight = self.add_weight(
                name=f'filter_{i}_weights',
                shape=(n_params_per_filter,),
                initializer='random_uniform',
                trainable=True
            )
            self.filter_weights.append(filter_weight)
        
        super().build(input_shape)
    
    def call(self, inputs):
        """Apply quantum convolution to inputs."""
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        
        # Calculate output dimensions
        out_height = (height - self.filter_size) // self.stride + 1
        out_width = (width - self.filter_size) // self.stride + 1
        
        # Initialize output tensor
        outputs = []
        
        # Apply each quantum filter
        for filter_idx in range(self.n_filters):
            filter_output = self._apply_quantum_filter(
                inputs, filter_idx, out_height, out_width
            )
            outputs.append(filter_output)
        
        # Stack filter outputs
        final_output = tf.stack(outputs, axis=-1)
        
        return final_output
    
    def _apply_quantum_filter(self, inputs, filter_idx, out_height, out_width):
        """Apply a single quantum filter to inputs."""
        batch_size = tf.shape(inputs)[0]
        filter_circuit = self.filter_circuits[filter_idx]
        filter_weights = self.filter_weights[filter_idx]
        
        # Extract patches and apply quantum filter
        def process_batch_sample(sample):
            """Process a single sample through quantum filter."""
            sample_outputs = []
            
            for i in range(out_height):
                for j in range(out_width):
                    # Extract patch
                    start_i = i * self.stride
                    start_j = j * self.stride
                    patch = sample[start_i:start_i + self.filter_size,
                                 start_j:start_j + self.filter_size, :]
                    
                    # Flatten patch for quantum processing
                    patch_flat = tf.reshape(patch, [-1])
                    
                    # Apply quantum filter
                    quantum_response = filter_circuit(patch_flat, filter_weights)
                    
                    # Aggregate response (mean of qubit measurements)
                    aggregated_response = tf.reduce_mean(quantum_response)
                    sample_outputs.append(aggregated_response)
            
            # Reshape to output feature map
            output_map = tf.reshape(sample_outputs, [out_height, out_width])
            return output_map
        
        # Process all samples in batch
        batch_outputs = tf.map_fn(
            process_batch_sample,
            inputs,
            dtype=tf.float32,
            parallel_iterations=1
        )
        
        return batch_outputs
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'filter_size': self.filter_size,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'stride': self.stride
        })
        return config


class QuantumCNN(tf.keras.layers.Layer):
    """
    Complete Quantum CNN architecture.
    
    This layer combines multiple quantum convolution and pooling layers
    to create a full quantum CNN for feature extraction.
    """
    
    def __init__(self,
                 conv_layers: List[Dict],
                 n_qubits: int = 4,
                 name: str = "quantum_cnn",
                 **kwargs):
        """
        Initialize Quantum CNN.
        
        Args:
            conv_layers: List of convolution layer configurations
            n_qubits: Number of qubits per quantum circuit
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.conv_layers_config = conv_layers
        self.n_qubits = n_qubits
        
        # Build quantum CNN layers
        self.conv_layers = []
        self.pool_layers = []
        
        for i, config in enumerate(conv_layers):
            # Quantum convolution layer
            conv_layer = QuantumConvLayer(
                n_filters=config.get('filters', 8),
                filter_size=config.get('filter_size', 2),
                n_qubits=n_qubits,
                n_layers=config.get('n_layers', 2),
                stride=config.get('stride', 1),
                name=f'qconv_{i}'
            )
            self.conv_layers.append(conv_layer)
            
            # Quantum pooling layer
            pool_layer = QuantumPooling(
                pool_size=config.get('pool_size', 2),
                name=f'qpool_{i}'
            )
            self.pool_layers.append(pool_layer)
    
    def call(self, inputs):
        """Forward pass through Quantum CNN."""
        x = inputs
        
        # Apply quantum conv-pool blocks
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = conv_layer(x)
            x = tf.nn.relu(x)  # Activation
            
            # Flatten for pooling, then reshape back
            batch_size = tf.shape(x)[0]
            height = tf.shape(x)[1]
            width = tf.shape(x)[2]
            channels = tf.shape(x)[3]
            
            # Reshape for pooling
            x_flat = tf.reshape(x, [batch_size, height * width, channels])
            x_pooled = pool_layer(x_flat)
            
            # Calculate new dimensions after pooling
            new_dim = tf.shape(x_pooled)[1]
            new_height = tf.cast(tf.sqrt(tf.cast(new_dim, tf.float32)), tf.int32)
            new_width = new_height
            
            # Reshape back to spatial format
            x = tf.reshape(x_pooled, [batch_size, new_height, new_width, channels])
        
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'conv_layers': self.conv_layers_config,
            'n_qubits': self.n_qubits
        })
        return config


def create_quantum_cnn_model(input_shape: Tuple[int, int, int],
                            n_classes: int,
                            conv_config: Optional[List[Dict]] = None,
                            n_qubits: int = 4) -> tf.keras.Model:
    """
    Create a complete Quantum CNN model for intrusion detection.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        n_classes: Number of output classes
        conv_config: Configuration for convolution layers
        n_qubits: Number of qubits per quantum circuit
        
    Returns:
        Compiled TensorFlow model
    """
    if conv_config is None:
        conv_config = [
            {'filters': 8, 'filter_size': 2, 'n_layers': 2, 'pool_size': 2},
            {'filters': 16, 'filter_size': 2, 'n_layers': 2, 'pool_size': 2}
        ]
    
    inputs = tf.keras.layers.Input(shape=input_shape, name='spatial_input')
    
    # Quantum CNN layers
    qcnn_output = QuantumCNN(
        conv_layers=conv_config,
        n_qubits=n_qubits,
        name='quantum_cnn'
    )(inputs)
    
    # Global average pooling
    gap = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(qcnn_output)
    
    # Classical post-processing layers
    dense1 = tf.keras.layers.Dense(128, activation='relu', name='dense1')(gap)
    dropout1 = tf.keras.layers.Dropout(0.3, name='dropout1')(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu', name='dense2')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2, name='dropout2')(dense2)
    
    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(dropout2)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='QuantumCNN_Model')
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def prepare_spatial_data(X: np.ndarray, 
                        target_shape: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Prepare data for spatial modeling with Quantum CNN.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        target_shape: Target spatial shape (height, width)
        
    Returns:
        Reshaped data for CNN (n_samples, height, width, channels)
    """
    n_samples, n_features = X.shape
    target_height, target_width = target_shape
    target_size = target_height * target_width
    
    if n_features <= target_size:
        # Pad with zeros if needed
        padding_needed = target_size - n_features
        X_padded = np.pad(X, ((0, 0), (0, padding_needed)), mode='constant')
        
        # Reshape to spatial format
        X_spatial = X_padded.reshape(n_samples, target_height, target_width, 1)
    else:
        # Truncate if too large
        X_truncated = X[:, :target_size]
        X_spatial = X_truncated.reshape(n_samples, target_height, target_width, 1)
    
    return X_spatial


def quantum_cnn_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       model: tf.keras.Model) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for Quantum CNN model.
    
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
            if isinstance(layer, (QuantumConvLayer, QuantumCNN))
        ])
    }
    
    return metrics


def visualize_quantum_filters(model: tf.keras.Model, 
                             layer_name: str = 'quantum_cnn') -> None:
    """
    Visualize learned quantum filter parameters.
    
    Args:
        model: Trained Quantum CNN model
        layer_name: Name of quantum CNN layer to visualize
    """
    import matplotlib.pyplot as plt
    
    # Get quantum CNN layer
    qcnn_layer = None
    for layer in model.layers:
        if layer.name == layer_name and isinstance(layer, QuantumCNN):
            qcnn_layer = layer
            break
    
    if qcnn_layer is None:
        print(f"No Quantum CNN layer found with name '{layer_name}'")
        return
    
    # Visualize filter weights for each convolution layer
    for i, conv_layer in enumerate(qcnn_layer.conv_layers):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Plot first 8 filters
        for j in range(min(8, conv_layer.n_filters)):
            if j < len(conv_layer.filter_weights):
                weights = conv_layer.filter_weights[j].numpy()
                
                axes[j].plot(weights)
                axes[j].set_title(f'Conv Layer {i+1}, Filter {j+1}')
                axes[j].set_xlabel('Parameter Index')
                axes[j].set_ylabel('Parameter Value')
                axes[j].grid(True, alpha=0.3)
        
        plt.suptitle(f'Quantum Filter Parameters - Convolution Layer {i+1}')
        plt.tight_layout()
        plt.show()


def create_hybrid_qcnn_model(input_shape: Tuple[int, int, int],
                            n_classes: int,
                            n_qubits: int = 4) -> tf.keras.Model:
    """
    Create a hybrid classical-quantum CNN model.
    
    Args:
        input_shape: Shape of input data
        n_classes: Number of output classes
        n_qubits: Number of qubits
        
    Returns:
        Hybrid CNN model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Classical convolution layers
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Quantum processing layer
    x = QuantumLayer(
        n_qubits=n_qubits,
        n_layers=2,
        output_dim=32,
        name='quantum_processing'
    )(tf.keras.layers.Flatten()(x))
    
    # Classical output layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='HybridQCNN_Model')
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def demo_quantum_cnn():
    """Demonstrate Quantum CNN capabilities."""
    print("🚀 Quantum CNN Demonstration")
    print("=" * 50)
    
    # Generate synthetic spatial network data
    n_samples = 300
    spatial_shape = (8, 8)
    n_features = spatial_shape[0] * spatial_shape[1]
    
    # Create synthetic network patterns
    X = np.random.randn(n_samples, n_features)
    
    # Add structured attack patterns
    attack_types = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])
    
    for i, attack_type in enumerate(attack_types):
        if attack_type == 1:  # DoS pattern - high activity regions
            high_activity_indices = np.random.choice(n_features, 16, replace=False)
            X[i, high_activity_indices] += np.random.normal(2.5, 0.8, 16)
        elif attack_type == 2:  # Probe pattern - scanning sequence
            scan_pattern = np.linspace(1, 3, n_features)
            X[i] += scan_pattern * np.random.uniform(0.8, 1.2)
    
    # Prepare spatial data
    X_spatial = prepare_spatial_data(X, spatial_shape)
    
    # One-hot encode labels
    from tensorflow.keras.utils import to_categorical
    y_categorical = to_categorical(attack_types, num_classes=3)
    
    # Split data
    split_idx = int(0.8 * len(X_spatial))
    X_train, X_test = X_spatial[:split_idx], X_spatial[split_idx:]
    y_train, y_test = y_categorical[:split_idx], y_categorical[split_idx:]
    
    # Create Quantum CNN model
    model = create_quantum_cnn_model(
        input_shape=X_spatial.shape[1:],
        n_classes=3,
        n_qubits=4
    )
    
    print(f"📊 Model Architecture Summary:")
    model.summary()
    
    # Train model
    print("\n🏋️ Training Quantum CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,  # Short demo
        batch_size=16,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n📈 Quantum CNN Performance:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    
    # Calculate additional metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    metrics = quantum_cnn_metrics(y_true_classes, y_pred_classes, model)
    
    print(f"\n🔬 Detailed Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n✅ Quantum CNN demonstration completed successfully!")
    
    return model, history


if __name__ == "__main__":
    demo_quantum_cnn()
