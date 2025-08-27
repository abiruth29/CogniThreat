#!/usr/bin/env python3
"""
CogniThreat: AI-driven Intrusion Detection System
Main entry point for Partner B's modules - Quantum Deep Learning and Explainable AI

Author: Partner B
Date: August 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.quantum_models.qlstm import QuantumLSTM
from src.quantum_models.qcnn import QuantumCNN
from src.xai.explainer import ExplainableAI
from src.xai.dashboard import launch_dashboard


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cognithreat.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_quantum_models(model_type: str = "both") -> None:
    """Run quantum model demonstrations."""
    logger = logging.getLogger(__name__)
    
    if model_type in ["qlstm", "both"]:
        logger.info("Running Quantum LSTM demonstration...")
        qlstm = QuantumLSTM(n_qubits=4, n_layers=2)
        qlstm.demonstrate()
    
    if model_type in ["qcnn", "both"]:
        logger.info("Running Quantum CNN demonstration...")
        qcnn = QuantumCNN(n_qubits=4, n_layers=2)
        qcnn.demonstrate()


def run_xai_demo() -> None:
    """Run explainable AI demonstration."""
    logger = logging.getLogger(__name__)
    logger.info("Running Explainable AI demonstration...")
    
    xai = ExplainableAI()
    xai.demonstrate()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CogniThreat: AI-driven Intrusion Detection System"
    )
    parser.add_argument(
        "--mode",
        choices=["quantum", "xai", "dashboard", "all"],
        default="all",
        help="Mode to run (default: all)"
    )
    parser.add_argument(
        "--quantum-model",
        choices=["qlstm", "qcnn", "both"],
        default="both",
        help="Quantum model to run (default: both)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard (default: 8501)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting CogniThreat system...")
    
    try:
        if args.mode in ["quantum", "all"]:
            run_quantum_models(args.quantum_model)
        
        if args.mode in ["xai", "all"]:
            run_xai_demo()
        
        if args.mode in ["dashboard", "all"]:
            logger.info(f"Launching dashboard on port {args.port}...")
            launch_dashboard(port=args.port)
        
        logger.info("CogniThreat execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
