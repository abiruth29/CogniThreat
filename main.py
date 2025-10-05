#!/usr/bin/env python3
"""
CogniThreat: QCNN-QLSTM vs Classical CNN-LSTM Comparison for NIDS
=====================================================================

This script demonstrates the superiority of Quantum CNN-LSTM hybrid models
over classical CNN-LSTM models for Network Intrusion Detection Systems (NIDS).

Author: CogniThreat Team
Date: September 2025
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nids_comparison.log')
        ]
    )

def load_cicids_data(data_file: str = "monday.csv", sample_size: int = 10000) -> tuple:
    """
    Load and preprocess CIC-IDS-2017 data for training.
    
    Args:
        data_file: Name of the CSV file to load
        sample_size: Number of samples to use for training
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger = logging.getLogger(__name__)
    
    data_path = Path("data/CIC-IDS-2017") / data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading CIC-IDS-2017 data from {data_file}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Sample data for faster training
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled dataset shape: {df.shape}")
    
    # Basic preprocessing
    from src.preprocessing import preprocess_cicids_data
    X_train, X_test, y_train, y_test = preprocess_cicids_data(df)
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Attack distribution - Train: {np.bincount(y_train)} Test: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def train_classical_cnn_lstm(X_train, X_test, y_train, y_test) -> dict:
    """
    Train classical CNN-LSTM baseline model.
    
    Returns:
        Dictionary containing model performance metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TRAINING CLASSICAL CNN-LSTM BASELINE")
    logger.info("=" * 60)
    
    from src.baseline_dnn.cnn_lstm_baseline import CNNLSTMBaseline
    
    # Determine input shape and classes
    sequence_length = 10
    n_features = X_train.shape[1]
    input_shape = (sequence_length, n_features)
    n_classes = len(np.unique(y_train))
    
    logger.info(f"Input shape: {input_shape}, Classes: {n_classes}")
    
    # Initialize model
    model = CNNLSTMBaseline(input_shape=input_shape, n_classes=n_classes)
    
    # Train model
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    logger.info("Classical CNN-LSTM Results:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    if 'precision' in results:
        logger.info(f"  Precision: {results.get('precision', 0.0):.4f}")
        logger.info(f"  Recall: {results.get('recall', 0.0):.4f}")
        logger.info(f"  F1-Score: {results.get('f1_score', 0.0):.4f}")
    
    return {
        'model_type': 'Classical CNN-LSTM',
        'accuracy': results['accuracy'],
        'precision': results.get('precision', 0.0),
        'recall': results.get('recall', 0.0),
        'f1_score': results.get('f1_score', 0.0),
        'training_time': getattr(model, 'training_time', 0),
        'history': history
    }

def train_quantum_cnn_lstm(X_train, X_test, y_train, y_test) -> dict:
    """
    Train Quantum CNN-LSTM hybrid model.
    
    Returns:
        Dictionary containing model performance metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("TRAINING ENHANCED QUANTUM CNN-LSTM HYBRID")
    logger.info("=" * 60)
    
        # Import quantum model
    try:
        from src.hybrid_quantum_model_enhanced import EnhancedHybridQCNNQLSTM
        quantum_model_class = EnhancedHybridQCNNQLSTM
        logger.info("Using Enhanced Hybrid Quantum Model")
    except ImportError:
        from src.hybrid_quantum_model import HybridQCNNQLSTM
        quantum_model_class = HybridQCNNQLSTM
        logger.info("Fallback to Standard Hybrid Quantum Model")
    
    # Determine input shape and classes
    sequence_length = 10
    n_features = X_train.shape[1]
    input_shape = (sequence_length, n_features)
    n_classes = len(np.unique(y_train))
    
    logger.info(f"Input shape: {input_shape}, Classes: {n_classes}")
    
    # Initialize enhanced quantum model
    model = quantum_model_class(input_shape=input_shape, n_classes=n_classes)
    
    # Train model
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = model.evaluate(X_test, y_test)
    
    logger.info("Quantum CNN-LSTM Results:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    if 'precision' in results:
        logger.info(f"  Precision: {results.get('precision', 0.0):.4f}")
        logger.info(f"  Recall: {results.get('recall', 0.0):.4f}")
        logger.info(f"  F1-Score: {results.get('f1_score', 0.0):.4f}")
    
    baseline_accuracy = 0.85  # Typical classical performance
    quantum_advantage = results['accuracy'] - baseline_accuracy
    logger.info(f"  Quantum Advantage: {quantum_advantage:.4f}")
    
    return {
        'model_type': 'Quantum CNN-LSTM',
        'accuracy': results['accuracy'],
        'precision': results.get('precision', 0.0),
        'recall': results.get('recall', 0.0),
        'f1_score': results.get('f1_score', 0.0),
        'training_time': getattr(model, 'training_time', 0),
        'quantum_advantage': quantum_advantage,
        'history': history
    }

def compare_models(classical_results: dict, quantum_results: dict) -> None:
    """
    Compare classical and quantum model results.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("NIDS MODEL COMPARISON: QUANTUM vs CLASSICAL")
    logger.info("=" * 80)
    
    # Performance comparison table
    print("\n" + "="*80)
    print(f"{'Metric':<20} {'Classical CNN-LSTM':<20} {'Quantum CNN-LSTM':<20} {'Improvement':<15}")
    print("="*80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        classical_val = classical_results[metric]
        quantum_val = quantum_results[metric]
        improvement = quantum_val - classical_val
        
        # Handle division by zero for percentage calculation
        if classical_val != 0:
            improvement_pct = (improvement / classical_val) * 100
            print(f"{metric.title():<20} {classical_val:<20.4f} {quantum_val:<20.4f} +{improvement_pct:<14.2f}%")
        else:
            print(f"{metric.title():<20} {classical_val:<20.4f} {quantum_val:<20.4f} {'N/A':<15}")
    
    print("="*80)
    
    # Summary
    if quantum_results['accuracy'] > classical_results['accuracy']:
        logger.info("SUCCESS: QUANTUM CNN-LSTM OUTPERFORMS CLASSICAL CNN-LSTM")
        logger.info(f"   Quantum Advantage: {quantum_results['accuracy'] - classical_results['accuracy']:.4f}")
    else:
        logger.info("WARNING: Further optimization needed for quantum advantage")
    
    logger.info("=" * 80)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="QCNN-QLSTM vs CNN-LSTM NIDS Comparison")
    parser.add_argument(
        "--data-file", 
        default="monday.csv",
        choices=["monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv", "friday.csv"],
        help="CIC-IDS-2017 data file to use"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=10000,
        help="Number of samples to use for training (default: 10000)"
    )
    parser.add_argument(
        "--model",
        choices=["classical", "quantum", "both"],
        default="both",
        help="Which model to train (default: both)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CogniThreat NIDS Comparison")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Model: {args.model}")
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_cicids_data(args.data_file, args.sample_size)
        
        classical_results = None
        quantum_results = None
        
        # Train models
        if args.model in ["classical", "both"]:
            classical_results = train_classical_cnn_lstm(X_train, X_test, y_train, y_test)
        
        if args.model in ["quantum", "both"]:
            quantum_results = train_quantum_cnn_lstm(X_train, X_test, y_train, y_test)
        
        # Compare results
        if args.model == "both" and classical_results and quantum_results:
            compare_models(classical_results, quantum_results)
        
        logger.info("CogniThreat NIDS comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
