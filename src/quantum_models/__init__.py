"""
CogniThreat Quantum Models Module
===============================

This module implements quantum deep learning architectures for network intrusion detection:
- QLSTM: Quantum Long Short-Term Memory
- QCNN: Quantum Convolutional Neural Network
- Hybrid quantum-classical models

Author: CogniThreat Team
Date: August 2025
"""

from .quantum_layers import QuantumLayer, QuantumCircuitBuilder
from .qlstm import QuantumLSTM
from .qcnn import QuantumCNN

__all__ = [
    'QuantumLayer',
    'QuantumCircuitBuilder', 
    'QuantumLSTM',
    'QuantumCNN'
]
