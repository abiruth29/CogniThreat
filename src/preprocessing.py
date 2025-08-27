"""
Data preprocessing utilities for CIC-IDS2017 cybersecurity datasets.

This module provides functions for loading, cleaning, and preprocessing
network traffic data for intrusion detection analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_cic_ids_data(
    data_path: str,
    chunk_size: int = 10000,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Load CIC-IDS2017 dataset with memory-efficient processing.
    
    Args:
        data_path: Path to the CSV file
        chunk_size: Size of chunks for reading large files
        sample_size: Optional sample size for testing
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading dataset from: {data_path}")
    
    try:
        chunks = []
        
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            if sample_size and len(pd.concat(chunks)) >= sample_size:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess features in the dataset.
    
    Args:
        df: Raw dataset DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting feature preprocessing...")
    
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Handle missing values
    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_columns] = processed_df[numeric_columns].fillna(0)
    
    # Remove infinite values
    processed_df = processed_df.replace([np.inf, -np.inf], 0)
    
    # Remove duplicates
    initial_shape = processed_df.shape[0]
    processed_df = processed_df.drop_duplicates()
    removed_duplicates = initial_shape - processed_df.shape[0]
    
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle categorical columns if any
    categorical_columns = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'Label':  # Don't encode the target label yet
            processed_df[col] = processed_df[col].astype('category').cat.codes
    
    logger.info(f"Feature preprocessing completed: {processed_df.shape}")
    return processed_df


def encode_labels(df: pd.DataFrame, label_column: str = 'Label') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode target labels for classification.
    
    Args:
        df: Dataset with labels
        label_column: Name of the label column
        
    Returns:
        Tuple of (DataFrame with encoded labels, encoding info)
    """
    logger.info(f"Encoding labels from column: {label_column}")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    # Get unique labels
    unique_labels = df[label_column].unique()
    logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Create binary classification (BENIGN vs ATTACK)
    df_encoded = df.copy()
    df_encoded['Label_Binary'] = (df_encoded[label_column] != 'BENIGN').astype(int)
    
    # Create multi-class encoding
    label_encoder = LabelEncoder()
    df_encoded['Label_Multiclass'] = label_encoder.fit_transform(df_encoded[label_column])
    
    encoding_info = {
        'unique_labels': unique_labels,
        'label_encoder': label_encoder,
        'label_mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
        'binary_mapping': {'BENIGN': 0, 'ATTACK': 1}
    }
    
    logger.info("Label encoding completed")
    return df_encoded, encoding_info


def split_features_labels(
    df: pd.DataFrame,
    target_column: str = 'Label_Binary',
    exclude_columns: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and labels.
    
    Args:
        df: Preprocessed dataset
        target_column: Name of target column
        exclude_columns: Additional columns to exclude from features
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    exclude_cols = ['Label', 'Label_Binary', 'Label_Multiclass']
    if exclude_columns:
        exclude_cols.extend(exclude_columns)
    
    # Remove columns that shouldn't be features
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_columns]
    y = df[target_column]
    
    logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        
    Returns:
        Tuple of (scaled training features, scaled test features, fitted scaler)
    """
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling completed")
    return X_train_scaled, X_test_scaled, scaler


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create stratified train-test split.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Creating train-test split (test_size={test_size})")
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Train label distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test label distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def get_feature_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about dataset features.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        Dictionary with feature information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    return info


def memory_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    logger.info("Optimizing memory usage...")
    
    initial_memory = df.memory_usage(deep=True).sum()
    
    # Downcast integers
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:  # Unsigned
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:  # Signed
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    # Downcast floats
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    final_memory = df.memory_usage(deep=True).sum()
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    logger.info(f"Memory optimization completed: {memory_reduction:.1f}% reduction")
    logger.info(f"Memory usage: {initial_memory / 1e6:.1f}MB -> {final_memory / 1e6:.1f}MB")
    
    return df
