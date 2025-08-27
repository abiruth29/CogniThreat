# 🛡️ CogniThreat: AI-Driven Intrusion Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-green.svg)](https://pennylane.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange.svg)](https://tensorflow.org/)

CogniThreat is an advanced AI-driven intrusion detection system that leverages **quantum deep learning**, **probabilistic reasoning**, and **explainable AI** to provide next-generation cybersecurity capabilities.

## 🌟 Key Features

### 🔬 Quantum Deep Learning Architecture
- **Quantum LSTM (QLSTM)**: Hybrid classical-quantum LSTM for sequential pattern recognition
- **Quantum CNN (QCNN)**: Quantum convolutional networks for feature extraction
- **Variational Quantum Circuits**: Parameterized quantum circuits for enhanced learning
- **Amplitude Encoding**: Efficient quantum data encoding techniques
- **IBM Quantum Backend Support**: Integration with real quantum hardware

### 🧠 Explainable AI Integration
- **SHAP Integration**: Global and local feature importance analysis
- **LIME Support**: Instance-level model explanations
- **Trust Metrics**: Quantified confidence and reliability scores
- **Interactive Dashboard**: Real-time visualization of threats and explanations
- **Decision Boundary Visualization**: Model behavior understanding

### 📊 Real-Time Monitoring
- **Streamlit Dashboard**: Interactive web interface for threat monitoring
- **Live Attack Alerts**: Real-time intrusion detection and classification
- **Performance Metrics**: Comprehensive model performance tracking
- **Quantum Metrics**: Quantum-specific performance indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/CogniThreat.git
cd CogniThreat
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv cognithreat_env
cognithreat_env\Scripts\activate

# macOS/Linux
python3 -m venv cognithreat_env
source cognithreat_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the System

#### 🎯 Quick Demo - All Components
```bash
python main.py --mode all
```

#### ⚛️ Quantum Models Only
```bash
# Run both QLSTM and QCNN
python main.py --mode quantum --quantum-model both

# Run specific model
python main.py --mode quantum --quantum-model qlstm
python main.py --mode quantum --quantum-model qcnn
```

#### 🔍 Explainable AI Only
```bash
python main.py --mode xai
```

#### 📈 Interactive Dashboard
```bash
python main.py --mode dashboard --port 8501
```

### Example Usage

#### Quantum LSTM Example
```python
from src.quantum_models.qlstm import QuantumLSTM

# Initialize model
qlstm = QuantumLSTM(n_qubits=4, n_layers=2)

# Generate synthetic data
X, y = qlstm.generate_synthetic_data(n_samples=1000)

# Train model
history = qlstm.train(X_train, y_train, epochs=10)

# Evaluate performance
metrics = qlstm.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

#### Explainable AI Example
```python
from src.xai.explainer import ExplainableAI

# Initialize XAI system
xai = ExplainableAI()

# Generate and prepare data
X, y = xai.generate_synthetic_network_data(n_samples=1000)
xai.prepare_model(X, y)

# Get explanations
shap_values = xai.get_shap_explanations(X_test[:10])
lime_exp = xai.get_lime_explanation(X_test[0])

# Calculate trust metrics
trust_metrics = xai.calculate_trust_metrics(shap_values)
```

## 📁 Project Structure

```
CogniThreat/
├── 📁 src/                          # Source code
│   ├── 📁 quantum_models/           # Quantum deep learning models
│   │   ├── 🐍 qlstm.py             # Quantum LSTM implementation
│   │   ├── 🐍 qcnn.py              # Quantum CNN implementation
│   │   └── 🐍 __init__.py
│   └── 📁 xai/                      # Explainable AI module
│       ├── 🐍 explainer.py         # SHAP & LIME integration
│       ├── 🐍 dashboard.py         # Streamlit dashboard
│       └── 🐍 __init__.py
├── 📁 tests/                        # Unit tests
│   ├── 🧪 test_qlstm.py            # QLSTM tests
│   ├── 🧪 test_qcnn.py             # QCNN tests
│   ├── 🧪 test_explainer.py        # XAI tests
│   └── 🧪 conftest.py              # Test configuration
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── 📓 quantum_models_demo.md    # Quantum models demonstration
│   └── 📓 explainable_ai_demo.md    # XAI demonstration
├── 📁 data/                         # Dataset directory
│   └── 📄 README.md                # Data documentation
├── 🐍 main.py                      # Main entry point
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
└── 📄 README.md                    # This file
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_qlstm.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Supported Datasets

### NSL-KDD Dataset
- **Description**: Network Security Laboratory dataset for IDS evaluation
- **Features**: 41 network connection features
- **Classes**: Normal, DoS, Probe, U2R, R2L attacks

### CIC-IDS2017 Dataset
- **Description**: Modern network traffic dataset
- **Features**: 78 flow-based network features
- **Classes**: Benign and various attack types

### Synthetic Data Generation
The system includes sophisticated synthetic data generators for testing without real datasets:

```python
# Network traffic simulation with realistic attack patterns
X, y = generate_synthetic_network_data(
    n_samples=1000,
    attack_types=['dos', 'probe', 'u2r', 'r2l']
)
```

## 🔧 Configuration

### Quantum Backend Configuration

```python
# Use quantum simulator (default)
qlstm = QuantumLSTM(device="default.qubit")

# Use IBM Quantum backend (requires API key)
qlstm = QuantumLSTM(device="qiskit.ibmq")
```

### Model Hyperparameters

```python
# Quantum LSTM configuration
qlstm = QuantumLSTM(
    n_qubits=4,           # Number of qubits
    n_layers=2,           # Quantum circuit depth
    classical_units=64,   # Classical LSTM units
    learning_rate=0.001   # Optimizer learning rate
)

# Quantum CNN configuration
qcnn = QuantumCNN(
    n_qubits=4,           # Number of qubits
    n_layers=2,           # Quantum circuit depth
    classical_filters=32, # Classical CNN filters
    learning_rate=0.001   # Optimizer learning rate
)
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Model Performance
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to attacks
- **F1-Score**: Harmonic mean of precision and recall

### Quantum Metrics
- **Quantum Advantage**: Performance improvement over classical models
- **Circuit Fidelity**: Quantum state preparation accuracy
- **Gate Count**: Quantum circuit complexity

### Trust Metrics
- **Consistency Score**: Feature importance stability
- **Confidence Score**: Prediction confidence levels
- **Stability Score**: Explanation robustness

## 🎯 Use Cases

### Cybersecurity Applications
- **Network Intrusion Detection**: Real-time threat identification
- **Anomaly Detection**: Unusual pattern recognition
- **Threat Intelligence**: Automated threat analysis
- **Security Operations Center (SOC)**: AI-assisted incident response

### Research Applications
- **Quantum Machine Learning**: Novel QML algorithm development
- **Explainable AI**: Interpretable security AI research
- **Hybrid Computing**: Classical-quantum integration studies

## 🤝 Contributing

We welcome contributions to CogniThreat! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📋 Roadmap

### Phase 1: Core Implementation ✅
- [x] Quantum LSTM implementation
- [x] Quantum CNN implementation
- [x] SHAP/LIME integration
- [x] Basic dashboard
- [x] Synthetic data generation

### Phase 2: Enhanced Features 🚧
- [ ] Real dataset integration
- [ ] Advanced quantum circuits
- [ ] Enhanced dashboard features
- [ ] Model optimization
- [ ] Performance benchmarking

### Phase 3: Production Ready 📋
- [ ] Scalability improvements
- [ ] Real-time processing
- [ ] API development
- [ ] Deployment tools
- [ ] Documentation expansion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PennyLane Team**: For the excellent quantum ML framework
- **Qiskit Team**: For quantum computing tools and IBM Quantum access
- **SHAP/LIME Authors**: For explainable AI methodologies
- **Streamlit Team**: For the interactive dashboard framework
- **Cybersecurity Community**: For datasets and domain expertise

## 📞 Contact

- **Project Lead**: Partner B
- **Email**: [your-email@domain.com]
- **GitHub**: [your-github-username]

## 🔗 Related Projects

- [PennyLane](https://pennylane.ai/): Quantum machine learning library
- [Qiskit](https://qiskit.org/): Quantum computing framework
- [SHAP](https://github.com/slundberg/shap): SHapley Additive exPlanations
- [LIME](https://github.com/marcotcr/lime): Local Interpretable Model-agnostic Explanations

---

**⚡ Powered by Quantum Computing & Explainable AI**

*CogniThreat represents the cutting edge of AI-driven cybersecurity, combining quantum computational advantages with transparent, explainable AI for next-generation threat detection.*
