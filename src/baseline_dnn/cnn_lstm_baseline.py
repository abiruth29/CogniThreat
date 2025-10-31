"""
CNN-LSTM Baseline Model for CogniThreat
=======================================

This module implements a CNN-LSTM baseline model for network intrusion detection
using the CIC-IDS-2017 dataset. The model combines convolutional neural networks
for spatial feature extraction with LSTM for temporal pattern recognition.

Classes:
    CNNLSTMBaseline: Main CNN-LSTM model implementation
    CNNLSTMTrainer: Training and evaluation wrapper
    
Author: CogniThreat Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Flatten, Input, Reshape,
    TimeDistributed, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Optional, Tuple, List
import logging
import json
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

class CNNLSTMBaseline:
    """
    CNN-LSTM Baseline Model for Network Intrusion Detection
    
    Architecture:
    - 1D Convolutional layers for spatial feature extraction
    - LSTM layers for temporal pattern recognition
    - Dense layers for classification
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 n_classes: int,
                 config: Optional[Dict] = None):
        """
        Initialize CNN-LSTM Baseline model.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            n_classes: Number of output classes
            config: Configuration dictionary
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.config = config or self._default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        """Basic configuration for CNN-LSTM baseline (simpler than quantum for comparison)."""
        return {
            'cnn_filters': [32, 64],       # Fewer filters than quantum
            'cnn_kernel_sizes': [3, 3],
            'pool_size': 2,
            'lstm_units': [64, 32],        # Smaller LSTM units
            'dense_units': [128, 64],      # Simpler dense layers
            'dropout_rate': 0.25,          # Less dropout
            'use_bidirectional': False,    # No bidirectional (quantum has this advantage)
            'use_batch_norm': True,
            'activation': 'relu',          # Basic activation vs quantum's swish
            'recurrent_activation': 'sigmoid',
            'optimizer': 'adam',
            'learning_rate': 0.002,        # Higher learning rate (less stable)
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        }
    
    def build_model(self) -> Model:
        """
        Build CNN-LSTM architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape, name='input')
        x = inputs
        
        # CNN layers for spatial feature extraction
        for i, (filters, kernel_size) in enumerate(zip(
            self.config['cnn_filters'], 
            self.config['cnn_kernel_sizes']
        )):
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=self.config['activation'],
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            
            if self.config['use_batch_norm']:
                x = BatchNormalization(name=f'batch_norm_conv_{i+1}')(x)
            
            x = MaxPooling1D(
                pool_size=self.config['pool_size'],
                name=f'max_pool_{i+1}'
            )(x)
            
            x = Dropout(
                rate=self.config['dropout_rate'],
                name=f'dropout_conv_{i+1}'
            )(x)
        
        # LSTM layers for temporal pattern recognition
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = i < len(self.config['lstm_units']) - 1
            
            if self.config['use_bidirectional']:
                x = Bidirectional(
                    LSTM(
                        units=units,
                        return_sequences=return_sequences,
                        recurrent_activation=self.config['recurrent_activation'],
                        name=f'lstm_{i+1}'
                    ),
                    name=f'bidirectional_lstm_{i+1}'
                )(x)
            else:
                x = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    recurrent_activation=self.config['recurrent_activation'],
                    name=f'lstm_{i+1}'
                )(x)
            
            if self.config['use_batch_norm']:
                x = BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
            
            x = Dropout(
                rate=self.config['dropout_rate'],
                name=f'dropout_lstm_{i+1}'
            )(x)
        
        # Dense layers for classification
        for i, units in enumerate(self.config['dense_units']):
            x = Dense(
                units=units,
                activation=self.config['activation'],
                name=f'dense_{i+1}'
            )(x)
            
            if self.config['use_batch_norm']:
                x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            
            x = Dropout(
                rate=self.config['dropout_rate'],
                name=f'dropout_dense_{i+1}'
            )(x)
        
        # Output layer
        outputs = Dense(
            units=self.n_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Baseline')
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """
        Convert feature matrix to sequences for LSTM input.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            sequence_length: Length of sequences to create
            
        Returns:
            Sequence array (n_sequences, sequence_length, n_features)
        """
        n_samples, n_features = X.shape
        n_sequences = n_samples - sequence_length + 1
        
        sequences = np.zeros((n_sequences, sequence_length, n_features))
        
        for i in range(n_sequences):
            sequences[i] = X[i:i + sequence_length]
        
        return sequences
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            sequence_length: int = 10,
            epochs: int = 100,
            batch_size: int = 256,
            verbose: int = 1) -> Dict:
        """
        Train the CNN-LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sequence_length: Length of input sequences
            epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        self.logger.info("Preparing sequence data...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_train_seq = self.prepare_sequences(X_train_scaled, sequence_length)
        y_train_seq = y_train[sequence_length-1:]  # Align labels with sequences
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train_seq)
        y_train_categorical = to_categorical(y_train_encoded, num_classes=self.n_classes)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq = self.prepare_sequences(X_val_scaled, sequence_length)
            y_val_seq = y_val[sequence_length-1:]
            y_val_encoded = self.label_encoder.transform(y_val_seq)
            y_val_categorical = to_categorical(y_val_encoded, num_classes=self.n_classes)
            validation_data = (X_val_seq, y_val_categorical)
        
        # Build model if not already built
        if self.model is None:
            self.input_shape = (sequence_length, X_train_scaled.shape[1])
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.logger.info(f"Training CNN-LSTM model for {epochs} epochs...")
        self.history = self.model.fit(
            X_train_seq,
            y_train_categorical,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale and create sequences
        X_scaled = self.scaler.transform(X)
        X_seq = self.prepare_sequences(X_scaled, sequence_length)
        
        # Predict
        probabilities = self.model.predict(X_seq, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
              sequence_length: int = 10, epochs: int = 50, batch_size: int = 256) -> Dict:
        """
        Simple training interface for main.py compatibility.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (used for validation)
            y_test: Test labels (used for validation)
            sequence_length: Length of sequences
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        self.logger.info("Training CNN-LSTM baseline model...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Train the model
        history = self.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, sequence_length: int = 10) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            sequence_length: Length of input sequences
            
        Returns:
            Evaluation metrics dictionary
        """
        # Make predictions
        y_pred, y_prob = self.predict(X_test, sequence_length)
        
        # Align labels with predictions (account for sequence offset)
        y_test_aligned = y_test[sequence_length-1:]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_aligned, y_pred)
        report = classification_report(y_test_aligned, y_pred, output_dict=True)
        
        # Model evaluation on sequences
        X_test_scaled = self.scaler.transform(X_test)
        X_test_seq = self.prepare_sequences(X_test_scaled, sequence_length)
        y_test_encoded = self.label_encoder.transform(y_test_aligned)
        y_test_categorical = to_categorical(y_test_encoded, num_classes=self.n_classes)
        
        loss, model_accuracy = self.model.evaluate(X_test_seq, y_test_categorical, verbose=0)
        
        # Extract metrics from classification report
        weighted_avg = report.get('weighted avg', {})
        precision = weighted_avg.get('precision', 0.0)
        recall = weighted_avg.get('recall', 0.0)
        f1_score = weighted_avg.get('f1-score', 0.0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'model_accuracy': model_accuracy,
            'loss': loss,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test_aligned, y_pred).tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist()
        }
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessing objects."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save preprocessing objects
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{filepath}_label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save config
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessing objects."""
        # Load model
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        
        # Load preprocessing objects
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{filepath}_label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load config
        with open(f"{filepath}_config.json", 'r') as f:
            self.config = json.load(f)
        
        self.logger.info(f"Model loaded from {filepath}")


class CNNLSTMTrainer:
    """
    Training wrapper for CNN-LSTM baseline model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'sequence_length': 10,
                'batch_size': 256,
                'epochs': 100,
                'validation_split': 0.2,
                'random_state': 42
            }
    
    def train_and_evaluate(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          n_classes: int,
                          save_path: Optional[str] = None) -> Dict:
        """
        Complete training and evaluation pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            n_classes: Number of classes
            save_path: Path to save trained model
            
        Returns:
            Complete evaluation results
        """
        self.logger.info("Starting CNN-LSTM training and evaluation...")
        
        # Initialize model
        sequence_length = self.config['sequence_length']
        input_shape = (sequence_length, X_train.shape[1])
        
        self.model = CNNLSTMBaseline(
            input_shape=input_shape,
            n_classes=n_classes
        )
        
        # Split training data for validation
        val_split = self.config.get('validation_split', 0.2)
        val_size = int(len(X_train) * val_split)
        
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train_split = X_train[val_size:]
        y_train_split = y_train[val_size:]
        
        # Train model
        history = self.model.fit(
            X_train_split, y_train_split,
            X_val, y_val,
            sequence_length=sequence_length,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        # Evaluate model
        results = self.model.evaluate(X_test, y_test, sequence_length)
        
        # Add training history
        results['training_history'] = history
        
        # Save model if path provided
        if save_path:
            self.model.save_model(save_path)
        
        self.logger.info(f"CNN-LSTM training completed. Test accuracy: {results['accuracy']:.4f}")
        
        return results
