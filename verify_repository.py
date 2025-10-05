#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CogniThreat Repository Verification Script
==========================================

Verifies the repository structure and functionality before GitHub push.
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def verify_structure():
    """Verify expected directory structure exists."""
    print("🔍 Verifying Repository Structure...")
    
    required_files = [
        "main.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "CLEANUP_SUMMARY.md",
        "PRESENTATION_GUIDE.md",
        "EXECUTION_GUIDE.md",
        "src/__init__.py",
        "src/preprocessing.py",
        "src/hybrid_quantum_model.py",
        "src/baseline_dnn/cnn_lstm_baseline.py",
        "src/quantum_models/qcnn.py",
        "src/quantum_models/qlstm.py",
        "src/probabilistic_reasoning/fusion.py",
        "tests/test_qlstm.py",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"  ❌ Missing: {file}")
        else:
            print(f"  ✅ Found: {file}")
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required files!")
        return False
    
    print("\n✅ All required files present!")
    return True

def verify_no_duplicates():
    """Verify duplicate files have been removed."""
    print("\n🗑️ Verifying Duplicates Removed...")
    
    should_not_exist = [
        "final_quantum_training.py",
        "multi_day_quantum_training.py",
        "robust_quantum_training.py",
        "src/working_quantum_nids.py",
        "src/advanced_quantum_nids.py",
        "src/real_data_processor.py",
        "src/model_comparison_processor.py",
        "src/hybrid_quantum_trainer.py",
        "src/simple_classical_model.py",
        "src/classical_models.py",
    ]
    
    found = []
    for file in should_not_exist:
        if Path(file).exists():
            found.append(file)
            print(f"  ⚠️ Still exists: {file}")
        else:
            print(f"  ✅ Removed: {file}")
    
    if found:
        print(f"\n⚠️ Warning: {len(found)} duplicate files still exist!")
        return False
    
    print("\n✅ All duplicate files successfully removed!")
    return True

def verify_imports():
    """Verify main.py can import required modules."""
    print("\n🔌 Verifying Python Imports...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        print("  Testing: src.preprocessing")
        from src.preprocessing import preprocess_cicids_data
        print("  ✅ src.preprocessing imported")
        
        print("  Testing: src.baseline_dnn.cnn_lstm_baseline")
        from src.baseline_dnn.cnn_lstm_baseline import CNNLSTMBaseline
        print("  ✅ src.baseline_dnn.cnn_lstm_baseline imported")
        
        print("  Testing: src.hybrid_quantum_model")
        from src.hybrid_quantum_model import HybridQCNNQLSTM
        print("  ✅ src.hybrid_quantum_model imported")
        
        print("  Testing: src.hybrid_quantum_model_enhanced")
        from src.hybrid_quantum_model_enhanced import EnhancedHybridQCNNQLSTM
        print("  ✅ src.hybrid_quantum_model_enhanced imported")
        
        print("  Testing: src.probabilistic_reasoning")
        from src.probabilistic_reasoning import ProbabilisticPipeline
        print("  ✅ src.probabilistic_reasoning imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

def count_python_files():
    """Count Python files in the repository."""
    print("\n📊 Repository Statistics...")
    
    py_files = list(Path(".").rglob("*.py"))
    # Exclude .venv and __pycache__
    py_files = [f for f in py_files if ".venv" not in str(f) and "__pycache__" not in str(f)]
    
    print(f"  Total Python files: {len(py_files)}")
    
    # Count by directory
    src_files = [f for f in py_files if str(f).startswith("src")]
    test_files = [f for f in py_files if str(f).startswith("tests")]
    root_files = [f for f in py_files if "/" not in str(f) and "\\" not in str(f)]
    
    print(f"  Root level: {len(root_files)} files")
    print(f"  src/ module: {len(src_files)} files")
    print(f"  tests/ module: {len(test_files)} files")
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("CogniThreat Repository Verification")
    print("=" * 60)
    
    checks = [
        verify_structure(),
        verify_no_duplicates(),
        verify_imports(),
        count_python_files(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ ALL CHECKS PASSED - READY FOR GITHUB!")
    else:
        print("❌ SOME CHECKS FAILED - REVIEW REQUIRED")
    print("=" * 60)
    
    return all(checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
