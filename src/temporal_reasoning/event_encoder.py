"""
Event Encoder for Converting Network Events to Discrete Observations

This module provides utilities to map continuous network features or
complex events into discrete observation symbols suitable for Markov models.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventEncoder:
    """
    Encode network events/features into discrete observation symbols.
    
    Supports multiple encoding strategies:
    - Label-based: Direct mapping of categorical labels
    - Quantization: Binning continuous features into discrete ranges
    - Clustering: KMeans clustering of feature vectors
    
    Attributes:
        encoding_type (str): Encoding strategy used
        num_symbols (int): Number of discrete observation symbols
        encoder (object): Trained encoder (scaler, kmeans, etc.)
    """
    
    def __init__(
        self,
        encoding_type: str = 'clustering',
        num_symbols: int = 10,
        **kwargs
    ):
        """
        Initialize Event Encoder.
        
        Args:
            encoding_type: Encoding strategy ('label', 'quantization', 'clustering')
            num_symbols: Number of discrete symbols to generate
            **kwargs: Additional parameters for specific encoders
        """
        assert encoding_type in ['label', 'quantization', 'clustering'], \
            f"Invalid encoding_type: {encoding_type}"
        
        self.encoding_type = encoding_type
        self.num_symbols = num_symbols
        self.kwargs = kwargs
        
        # Encoder components
        self.scaler = None
        self.kmeans = None
        self.label_map = None
        self.quantization_bins = None
        
        self.is_fitted = False
        
        logger.info(f"Initialized EventEncoder: {encoding_type}, {num_symbols} symbols")
    
    def fit(
        self,
        data: Union[List[List[int]], np.ndarray],
        labels: Optional[List[int]] = None
    ) -> 'EventEncoder':
        """
        Fit encoder on training data.
        
        Args:
            data: Training data (labels or feature vectors)
            labels: Optional labels (for label-based encoding)
        
        Returns:
            self: Fitted encoder
        """
        if self.encoding_type == 'label':
            return self._fit_label(labels if labels is not None else data)
        elif self.encoding_type == 'quantization':
            return self._fit_quantization(np.array(data))
        elif self.encoding_type == 'clustering':
            return self._fit_clustering(np.array(data))
    
    def _fit_label(self, labels: List[int]) -> 'EventEncoder':
        """Fit label-based encoder (direct mapping)."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) > self.num_symbols:
            logger.warning(f"Found {len(unique_labels)} unique labels, "
                          f"but num_symbols={self.num_symbols}. "
                          f"Consider increasing num_symbols.")
        
        # Create label mapping
        self.label_map = {label: i % self.num_symbols 
                         for i, label in enumerate(unique_labels)}
        
        self.num_symbols = len(self.label_map)
        self.is_fitted = True
        
        logger.info(f"Label encoder fitted with {len(self.label_map)} unique labels")
        return self
    
    def _fit_quantization(self, data: np.ndarray) -> 'EventEncoder':
        """Fit quantization-based encoder (binning)."""
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Normalize
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Create bins using quantiles
        self.quantization_bins = []
        
        for dim in range(data_scaled.shape[1]):
            percentiles = np.linspace(0, 100, self.num_symbols + 1)
            bins = np.percentile(data_scaled[:, dim], percentiles)
            self.quantization_bins.append(bins)
        
        self.is_fitted = True
        
        logger.info(f"Quantization encoder fitted with {self.num_symbols} bins")
        return self
    
    def _fit_clustering(self, data: np.ndarray) -> 'EventEncoder':
        """Fit clustering-based encoder (KMeans)."""
        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Normalize
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.num_symbols,
            random_state=self.kwargs.get('random_state', 42),
            n_init=10
        )
        self.kmeans.fit(data_scaled)
        
        self.is_fitted = True
        
        logger.info(f"Clustering encoder fitted with {self.num_symbols} clusters")
        return self
    
    def transform(
        self,
        data: Union[List, np.ndarray]
    ) -> np.ndarray:
        """
        Transform data to discrete observation symbols.
        
        Args:
            data: Input data (labels or features)
        
        Returns:
            symbols: Discrete observation indices [N]
        """
        assert self.is_fitted, "Encoder must be fitted before transform"
        
        if self.encoding_type == 'label':
            return self._transform_label(data)
        elif self.encoding_type == 'quantization':
            return self._transform_quantization(np.array(data))
        elif self.encoding_type == 'clustering':
            return self._transform_clustering(np.array(data))
    
    def _transform_label(self, labels: List[int]) -> np.ndarray:
        """Transform using label mapping."""
        # Map known labels, use default for unknown
        default_symbol = 0
        symbols = np.array([
            self.label_map.get(label, default_symbol) for label in labels
        ])
        return symbols
    
    def _transform_quantization(self, data: np.ndarray) -> np.ndarray:
        """Transform using quantization."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Normalize
        data_scaled = self.scaler.transform(data)
        
        # Assign to bins (use first dimension if multi-dimensional)
        symbols = np.digitize(
            data_scaled[:, 0],
            self.quantization_bins[0]
        ) - 1  # Make 0-indexed
        
        # Clip to valid range
        symbols = np.clip(symbols, 0, self.num_symbols - 1)
        
        return symbols
    
    def _transform_clustering(self, data: np.ndarray) -> np.ndarray:
        """Transform using clustering."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Normalize
        data_scaled = self.scaler.transform(data)
        
        # Predict cluster assignments
        symbols = self.kmeans.predict(data_scaled)
        
        return symbols
    
    def fit_transform(
        self,
        data: Union[List, np.ndarray],
        labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Fit encoder and transform data in one step.
        
        Args:
            data: Training data
            labels: Optional labels
        
        Returns:
            symbols: Discrete observation indices
        """
        self.fit(data, labels)
        return self.transform(data)
    
    def transform_sequences(
        self,
        sequences: List[Union[List, np.ndarray]]
    ) -> List[List[int]]:
        """
        Transform multiple sequences.
        
        Args:
            sequences: List of data sequences
        
        Returns:
            symbol_sequences: List of discrete symbol sequences
        """
        assert self.is_fitted, "Encoder must be fitted before transform"
        
        symbol_sequences = []
        for seq in sequences:
            symbols = self.transform(seq)
            symbol_sequences.append(symbols.tolist())
        
        return symbol_sequences
    
    def get_symbol_names(self) -> List[str]:
        """Get human-readable names for observation symbols."""
        if self.encoding_type == 'label' and self.label_map:
            # Reverse mapping
            reverse_map = {v: k for k, v in self.label_map.items()}
            return [f"Label_{reverse_map.get(i, i)}" for i in range(self.num_symbols)]
        else:
            return [f"Symbol_{i}" for i in range(self.num_symbols)]
    
    def save(self, filepath: str) -> None:
        """Save encoder to file."""
        encoder_data = {
            'encoding_type': self.encoding_type,
            'num_symbols': self.num_symbols,
            'kwargs': self.kwargs,
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'label_map': self.label_map,
            'quantization_bins': self.quantization_bins,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
        
        logger.info(f"Encoder saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EventEncoder':
        """Load encoder from file."""
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
        
        encoder = cls(
            encoding_type=encoder_data['encoding_type'],
            num_symbols=encoder_data['num_symbols'],
            **encoder_data['kwargs']
        )
        
        encoder.scaler = encoder_data['scaler']
        encoder.kmeans = encoder_data['kmeans']
        encoder.label_map = encoder_data['label_map']
        encoder.quantization_bins = encoder_data['quantization_bins']
        encoder.is_fitted = encoder_data['is_fitted']
        
        logger.info(f"Encoder loaded from {filepath}")
        return encoder
    
    def __repr__(self) -> str:
        return (f"EventEncoder(type={self.encoding_type}, "
                f"symbols={self.num_symbols}, fitted={self.is_fitted})")
