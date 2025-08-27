# 🛡️ CogniThreat Dataset Repository

This directory contains cybersecurity datasets for training and evaluating the CogniThreat intrusion detection system.

## 📊 Supported Datasets

### ✅ Currently Integrated Datasets

#### 1. CIC-IDS2017 Dataset
- **Status**: ✅ **ACTIVE** - Successfully integrated and preprocessed
- **Description**: Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset
- **Usage**: Modern network traffic analysis and intrusion detection
- **Format**: CSV files with flow-based features (78+ features)
- **Current File**: `02-14-2018.csv` (FTP-BruteForce attacks)
- **Size**: 50,000 → 10,345 samples after preprocessing
- **Performance**: Perfect classification (100% accuracy) with classical models
- **Download**: https://www.unb.ca/cic/datasets/ids-2017.html

### 🎯 Recommended Additional Datasets

#### 2. CSE-CIC-IDS2018 Dataset ⭐⭐⭐⭐⭐
- **Priority**: **HIGH** - Best for comprehensive evaluation
- **Description**: Communications Security Establishment & CIC collaborative dataset
- **Size**: ~16 million network flows (10 days of traffic)
- **Attack Types**: Brute Force, Heartbleed, Botnet, DoS, DDoS, Web Attack, Infiltration
- **Features**: 80+ network flow characteristics
- **Advantages**: Largest modern dataset, diverse attack scenarios
- **Download**: https://www.unb.ca/cic/datasets/ids-2018.html

#### 3. UNSW-NB15 Dataset ⭐⭐⭐⭐
- **Priority**: **MEDIUM** - Good for multi-class classification
- **Description**: University of New South Wales network intrusion dataset
- **Size**: ~2.5 million network flows
- **Attack Types**: 9 families (Fuzzers, Analysis, Backdoors, DoS, Exploits, etc.)
- **Features**: 49 network features
- **Advantages**: Modern attack types, balanced for multi-class problems
- **Download**: https://research.unsw.edu.au/projects/unsw-nb15-dataset

#### 4. NSL-KDD Dataset ⭐⭐⭐
- **Priority**: **MEDIUM** - Standard benchmark comparison
- **Description**: Network Security Laboratory - Improved KDD Cup 1999 dataset
- **Size**: ~148,517 records
- **Attack Types**: DoS, Probe, R2L, U2R
- **Features**: 41 network connection features
- **Advantages**: Standard benchmark, removes redundant records from original KDD
- **Download**: https://www.unb.ca/cic/datasets/nsl.html

## 📁 Current Directory Structure

```
data/
├── 02-14-2018.csv         # ✅ CIC-IDS2017 FTP-BruteForce dataset (ACTIVE)
├── README.md              # 📖 This documentation file
├── raw/                   # 📦 Original, unprocessed datasets
├── processed/             # 🔧 Cleaned and preprocessed data
├── samples/               # 🧪 Small sample datasets for quick testing
└── benchmarks/            # 📊 Standard benchmark datasets (NSL-KDD, etc.)
```

## 🔧 Dataset Integration Status

| Dataset | Status | Size | Use Case | Priority |
|---------|--------|------|----------|----------|
| **CIC-IDS2017** | ✅ **ACTIVE** | 10,345 samples | FTP-BruteForce detection | ✅ Complete |
| **CSE-CIC-IDS2018** | ⏳ Recommended | 16M+ samples | Comprehensive evaluation | 🔥 **HIGH** |
| **UNSW-NB15** | 💡 Suggested | 2.5M samples | Multi-class classification | 📈 Medium |
| **NSL-KDD** | 💡 Suggested | 148K samples | Benchmark comparison | 📊 Medium |

## 🚀 Quick Integration Guide

### Adding New Datasets

1. **Download** dataset from official source
2. **Place** in `data/raw/` directory
3. **Update** notebook preprocessing pipeline
4. **Validate** data quality and format
5. **Benchmark** against existing results

### Preprocessing Pipeline

All datasets undergo standardized preprocessing:
- ✅ **Duplicate Removal**: Remove redundant records
- ✅ **Missing Value Handling**: Imputation strategies
- ✅ **Feature Encoding**: Categorical to numerical conversion
- ✅ **Normalization**: Min-Max scaling (0-1 range)
- ✅ **Binary Classification**: Attack vs. Benign labeling
- ✅ **Stratified Splitting**: 70/30 train-test with class balance

## 📊 Performance Benchmarks

### CIC-IDS2017 (Current Dataset)
- **Classical Models**: 100% accuracy (SVM, RF, DT, LR)
- **Attack Type**: FTP-BruteForce (highly distinctive patterns)
- **Challenge**: Perfect classical performance sets high bar for quantum models

## 🎯 Recommended Next Steps

1. **Immediate**: Add CSE-CIC-IDS2018 for comprehensive evaluation
2. **Short-term**: Integrate UNSW-NB15 for multi-class scenarios  
3. **Long-term**: Include NSL-KDD for benchmark comparison
4. **Research**: Explore IoT-specific datasets for edge deployment

---

## 📝 Data Processing Notes

### Quality Assurance
- **Validation**: All datasets verified for integrity and completeness
- **Documentation**: Processing steps documented for reproducibility
- **Privacy**: Ensure compliance with data protection regulations
- **Versioning**: Track dataset versions and preprocessing changes

### Integration Requirements
- **Format**: CSV preferred, other formats supported with conversion
- **Size**: Large datasets processed in chunks for memory efficiency
- **Labels**: Support for binary and multi-class classification
- **Features**: Automatic feature type detection and encoding
