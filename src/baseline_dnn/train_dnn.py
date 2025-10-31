"""
DNN Training Module

Implements the Sequential DNN architecture as specified in
"Network-based intrusion detection using deep learning technique" 
(Scientific Reports, 2025):

- 3 hidden layers: 800, 800, 400 neurons
- ReLU activation functions
- Adam optimizer
- Categorical crossentropy loss
- 100 epochs, batch size 50
- Softmax output layer
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple
import logging
import time

class DNNBaseline:
    """
    Deep Neural Network baseline implementation
    Following Scientific Reports 2025 methodology
    """
    
    def __init__(self,
                 hidden_layers: list = [800, 800, 400],
                 activation: str = 'relu',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 loss: str = 'categorical_crossentropy',
                 epochs: int = 100,
                 batch_size: int = 50,
                 dropout_rate: float = 0.0,
                 early_stopping: bool = True,
                 patience: int = 10,
                 random_state: int = 42):
        """
        Initialize DNN Baseline
        
        Args:
            hidden_layers: List of hidden layer sizes [800, 800, 400]
            activation: Activation function ('relu')
            optimizer: Optimizer ('adam')
            learning_rate: Learning rate for optimizer
            loss: Loss function ('categorical_crossentropy')
            epochs: Number of training epochs (100)
            batch_size: Batch size (50)
            dropout_rate: Dropout rate (0.0 = no dropout)
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            random_state: Random seed
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        
        self.model = None
        self.history = None
        self.training_time = None
        self.is_trained = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def build_model(self, input_dim: int, n_classes: int) -> Sequential:
        """
        Build the Sequential DNN model
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        self.logger.info("Building DNN model...")
        self.logger.info(f"Architecture: Input({input_dim}) -> {' -> '.join(map(str, self.hidden_layers))} -> Output({n_classes})")
        
        model = Sequential()
        
        # Input layer + first hidden layer
        model.add(Dense(
            self.hidden_layers[0],
            input_dim=input_dim,
            activation=self.activation,
            name=f'hidden_1_{self.hidden_layers[0]}'
        ))
        
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate, name='dropout_1'))
        
        # Additional hidden layers
        for i, units in enumerate(self.hidden_layers[1:], 2):
            model.add(Dense(
                units,
                activation=self.activation,
                name=f'hidden_{i}_{units}'
            ))
            
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Output layer with softmax
        model.add(Dense(
            n_classes,
            activation='softmax',
            name='output'
        ))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )
        
        # Print model summary
        self.logger.info("Model architecture:")
        model.summary(print_fn=self.logger.info)
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the DNN model
        
        Args:
            X_train: Training features
            y_train: Training labels (encoded)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Starting DNN training...")
        
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        # Build model
        self.model = self.build_model(input_dim, n_classes)
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=n_classes)
        y_val_cat = None
        if y_val is not None:
            y_val_cat = to_categorical(y_val, num_classes=n_classes)
        
        # Setup callbacks
        callbacks = []
        
        if self.early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss' if y_val is not None else 'loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Add learning rate reduction
        lr_reduce = ReduceLROnPlateau(
            monitor='val_loss' if y_val is not None else 'loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reduce)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val_cat)
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        # Training summary
        training_results = {
            'training_time': self.training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1],
            'final_train_accuracy': self.history.history['accuracy'][-1],
            'model_parameters': self.model.count_params(),
            'architecture': {
                'input_dim': input_dim,
                'hidden_layers': self.hidden_layers,
                'n_classes': n_classes,
                'activation': self.activation,
                'optimizer': self.optimizer,
                'learning_rate': self.learning_rate
            }
        }
        
        if validation_data is not None:
            training_results.update({
                'final_val_loss': self.history.history['val_loss'][-1],
                'final_val_accuracy': self.history.history['val_accuracy'][-1]
            })
        
        self.logger.info(f"Training completed in {self.training_time:.2f} seconds")
        self.logger.info(f"Epochs trained: {training_results['epochs_trained']}")
        self.logger.info(f"Final training accuracy: {training_results['final_train_accuracy']:.4f}")
        
        if validation_data is not None:
            self.logger.info(f"Final validation accuracy: {training_results['final_val_accuracy']:.4f}")
        
        return training_results
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions (classes or probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        
        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels (encoded)
            
        Returns:
            Evaluation results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating model performance...")
        
        # Get predictions
        start_time = time.time()
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict(X_test, return_probabilities=True)
        prediction_time = time.time() - start_time
        
        # Convert to categorical for loss calculation
        n_classes = len(np.unique(y_test))
        y_test_cat = to_categorical(y_test, num_classes=n_classes)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'prediction_time': prediction_time,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'model_info': {
                'parameters': self.model.count_params(),
                'architecture': self.hidden_layers,
                'training_time': self.training_time
            }
        }
        
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Test loss: {test_loss:.4f}")
        self.logger.info(f"Prediction time: {prediction_time:.4f} seconds")
        
        return results
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get training history
        
        Returns:
            Training history dictionary
        """
        if self.history is None:
            return {}
        
        return self.history.history
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model
        
        Args:
            filepath: Path to load model from
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Model summary dictionary
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "architecture": {
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "total_parameters": self.model.count_params()
            },
            "training": {
                "epochs_trained": len(self.history.history['loss']) if self.history else 0,
                "training_time": self.training_time,
                "batch_size": self.batch_size,
                "early_stopping": self.early_stopping
            },
            "performance": {
                "final_train_accuracy": self.history.history['accuracy'][-1] if self.history else None,
                "final_train_loss": self.history.history['loss'][-1] if self.history else None
            }
        }
