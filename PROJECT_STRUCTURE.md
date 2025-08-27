# 🛡️ CogniThreat Project Structure

## 📁 Repository Organization

```
CogniThreat/
├── 📄 README.md                              # Main project documentation
├── 📄 requirements.txt                       # Python dependencies
├── 📄 main.py                               # Main entry point
├── 📄 .gitignore                            # Git ignore patterns
│
├── 📂 config/                               # Configuration files
│   └── cnn_lstm_config.yaml                # CNN-LSTM model configuration
│
├── 📂 data/                                 # Dataset files (CIC-IDS2017)
│   ├── README.md                           # Data description
│   ├── 02-14-2018.csv                     # Network traffic data
│   ├── 02-15-2018.csv                     # (10 CSV files total)
│   └── ...                                # Additional data files
│
├── 📂 notebooks/                           # Jupyter notebooks
│   ├── CogniThreat_Comprehensive_Analysis.ipynb   # 🎯 MAIN ANALYSIS NOTEBOOK
│   ├── CogniThreat_Demo.ipynb             # System demonstration
│   └── CogniThreat_DNN_Baseline.ipynb     # Baseline model training
│
├── 📂 src/                                 # Source code modules
│   ├── __init__.py
│   ├── classical_models.py                # Classical ML implementations
│   ├── preprocessing.py                   # Data preprocessing utilities
│   │
│   ├── 📂 baseline_dnn/                   # CNN-LSTM baseline model
│   │   ├── __init__.py
│   │   ├── train_dnn.py                   # Training script
│   │   ├── evaluate.py                    # Evaluation utilities
│   │   ├── preprocessing.py               # Model-specific preprocessing
│   │   ├── feature_selection.py           # Feature selection
│   │   ├── cnn_lstm_baseline.ipynb        # Training notebook
│   │   ├── config/cnn_lstm_config.yaml    # Model configuration
│   │   └── artifacts/                     # Trained model artifacts (ignored)
│   │
│   ├── 📂 quantum_models/                 # Quantum ML implementations
│   │   ├── __init__.py
│   │   ├── quantum_layers.py              # Quantum circuit layers
│   │   ├── qcnn.py                        # Quantum CNN implementation
│   │   └── qlstm.py                       # Quantum LSTM implementation
│   │
│   ├── 📂 probabilistic_reasoning/        # Bayesian Network system
│   │   ├── __init__.py
│   │   ├── bayesian_network.py            # Core Bayesian network
│   │   ├── probabilistic_inference.py     # Inference engine
│   │   └── uncertainty_quantification.py  # Uncertainty analysis
│   │
│   └── 📂 xai/                           # Explainable AI components
│       ├── __init__.py
│       ├── explainer.py                   # Model explanation tools
│       └── dashboard.py                   # Visualization dashboard
│
├── 📂 tests/                              # Unit tests
│   ├── __init__.py
│   ├── conftest.py                        # Test configuration
│   └── test_qlstm.py                      # Quantum LSTM tests
│
└── 📂 docs/                              # Additional documentation
    ├── CogniThreat_Model_Analysis_Report.md    # Technical analysis
    ├── model_implementations_report.md         # Implementation details
    └── GITHUB_SETUP.md                         # Setup instructions
```

## 🎯 Key Components

### **Primary Analysis Notebook**
- **`notebooks/CogniThreat_Comprehensive_Analysis.ipynb`** - The main academic presentation notebook with complete system analysis

### **Core Modules**
- **`src/quantum_models/`** - Quantum CNN and LSTM implementations using PennyLane
- **`src/probabilistic_reasoning/`** - Bayesian synthesis engine for uncertainty-aware assessment
- **`src/baseline_dnn/`** - Classical CNN-LSTM baseline for comparison

### **Data & Configuration**
- **`data/`** - CIC-IDS2017 intrusion detection dataset (10 CSV files)
- **`config/`** - Model configuration files

## 🚀 Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run main analysis**: Open `notebooks/CogniThreat_Comprehensive_Analysis.ipynb`
3. **Execute all cells** to see complete system demonstration

## 📊 Academic Contribution

This repository demonstrates:
- **Quantum-enhanced cybersecurity** using QCNN and QLSTM
- **Uncertainty-aware threat assessment** via Bayesian reasoning
- **Alert fatigue reduction** through intelligent prioritization
- **Novel threat detection** via high-uncertainty flagging

---

**Ready for academic review, conference submission, or production deployment! 🎓**
