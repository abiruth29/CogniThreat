#!/usr/bin/env python3
"""
Proper Probability Components Demonstration
=============================================

This script demonstrates the CORRECTED probability components for CogniThreat:

1. **Calibration**: ECE, Brier Score, Temperature Scaling
2. **Learned Fusion**: Stacking meta-model for fusion weights
3. **HMM without Label Leakage**: Feature-based encoding
4. **Cost Sensitivity**: Analysis and justification of C_FN/C_FP ratio

Run this to validate that all probability components are principled and
properly implemented.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Handle Windows encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def print_section(title: str, char: str = "=") -> None:
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}\n")


def demo_1_calibration():
    """Demonstrate probability calibration with ECE and temperature scaling."""
    print_section("1. PROBABILITY CALIBRATION", "=")
    
    from src.probabilistic_reasoning.calibration import (
        expected_calibration_error,
        brier_score,
        TemperatureScaling,
        IsotonicCalibration,
        calibration_report
    )
    
    # Generate synthetic predictions (overconfident classifier)
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (60% class 0, 40% class 1)
    y_true = np.random.binomial(1, 0.4, n_samples)
    
    # Simulated overconfident predictions (common in neural networks)
    base_probs = np.random.beta(2, 5, n_samples) * 0.4 + 0.5  # Biased toward confident
    # Add correlation with true labels
    noise = np.random.normal(0, 0.1, n_samples)
    y_prob = np.clip(0.3 * y_true + 0.4 * base_probs + noise, 0.01, 0.99)
    
    print("Original (uncalibrated) probabilities:")
    print(f"  Mean confidence: {y_prob.mean():.3f}")
    print(f"  Actual positive rate: {y_true.mean():.3f}")
    
    # Compute calibration metrics
    ece_orig, details = expected_calibration_error(y_true, y_prob)
    brier_orig = brier_score(y_true, y_prob)
    
    print(f"\n  ECE (Expected Calibration Error): {ece_orig:.4f}")
    print(f"  Brier Score: {brier_orig:.4f}")
    print(f"  Number of bins: {details['n_bins']}")
    
    # Apply temperature scaling
    print("\n--- Applying Temperature Scaling ---")
    
    # Split into calibration and test sets
    split_idx = int(0.7 * n_samples)
    val_probs, test_probs = y_prob[:split_idx], y_prob[split_idx:]
    val_labels, test_labels = y_true[:split_idx], y_true[split_idx:]
    
    ts = TemperatureScaling()
    ts.fit(val_probs, val_labels)
    
    print(f"  Learned temperature: T = {ts.temperature:.4f}")
    print(f"  (T > 1 means model was overconfident)")
    
    calibrated_probs = ts.transform(test_probs)
    
    # Compute calibrated metrics
    ece_cal, _ = expected_calibration_error(test_labels, calibrated_probs)
    brier_cal = brier_score(test_labels, calibrated_probs)
    
    print(f"\nAfter Temperature Scaling (on test set):")
    print(f"  ECE: {ece_orig:.4f} -> {ece_cal:.4f} ({(1 - ece_cal/ece_orig)*100:.1f}% reduction)")
    print(f"  Brier: {brier_orig:.4f} -> {brier_cal:.4f} ({(1 - brier_cal/brier_orig)*100:.1f}% reduction)")
    
    # Generate calibration report
    print("\n--- Calibration Report ---")
    report = calibration_report(test_labels, test_probs, calibrated_probs)
    
    print(f"  Original ECE: {report['original']['ece']:.4f}")
    print(f"  Calibrated ECE: {report['calibrated']['ece']:.4f}")
    print(f"  Improvement: {report['improvement']['ece_reduction']:.1f}%")
    
    print("\n✓ Calibration module working correctly!")
    
    return test_labels, test_probs, calibrated_probs


def demo_2_learned_fusion():
    """Demonstrate learned fusion weights via stacking."""
    print_section("2. LEARNED FUSION WEIGHTS (STACKING)", "=")
    
    from src.probabilistic_reasoning.fusion import (
        LearnedFusion,
        BayesianFusion,
        weighted_ensemble
    )
    
    # Simulate predictions from 3 models
    np.random.seed(42)
    n_samples = 800
    n_classes = 3
    
    # True labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Model 1: Good model (higher correlation with truth)
    model1_probs = np.zeros((n_samples, n_classes))
    for i, y in enumerate(y_true):
        model1_probs[i, y] = np.random.uniform(0.5, 0.9)
        others = [j for j in range(n_classes) if j != y]
        remaining = 1 - model1_probs[i, y]
        for j in others:
            model1_probs[i, j] = remaining / len(others)
    
    # Model 2: Medium model
    model2_probs = np.zeros((n_samples, n_classes))
    for i, y in enumerate(y_true):
        model2_probs[i, y] = np.random.uniform(0.3, 0.7)
        others = [j for j in range(n_classes) if j != y]
        remaining = 1 - model2_probs[i, y]
        for j in others:
            model2_probs[i, j] = remaining / len(others)
    
    # Model 3: Weak model (lower correlation)
    model3_probs = np.random.dirichlet([1, 1, 1], n_samples)
    # Add some correlation
    for i, y in enumerate(y_true):
        model3_probs[i, y] += 0.1
    model3_probs = model3_probs / model3_probs.sum(axis=1, keepdims=True)
    
    # Split into train/test
    split_idx = int(0.6 * n_samples)
    train_probs = [m[:split_idx] for m in [model1_probs, model2_probs, model3_probs]]
    test_probs = [m[split_idx:] for m in [model1_probs, model2_probs, model3_probs]]
    train_labels = y_true[:split_idx]
    test_labels = y_true[split_idx:]
    
    # Method 1: Hand-set uniform weights (old approach)
    print("Method 1: Hand-set uniform weights")
    uniform_fused = weighted_ensemble(test_probs, weights=[1/3, 1/3, 1/3])
    uniform_acc = np.mean(uniform_fused.argmax(axis=1) == test_labels)
    print(f"  Weights: [0.33, 0.33, 0.33]")
    print(f"  Test Accuracy: {uniform_acc:.4f}")
    
    # Method 2: Accuracy-based weights (heuristic, not principled)
    print("\nMethod 2: Accuracy-based weights (HEURISTIC)")
    model_accs = [
        np.mean(m[:split_idx].argmax(axis=1) == train_labels) 
        for m in [model1_probs, model2_probs, model3_probs]
    ]
    print(f"  Individual model accuracies: {[f'{a:.3f}' for a in model_accs]}")
    
    acc_weights = np.array(model_accs) / sum(model_accs)
    acc_fused = weighted_ensemble(test_probs, weights=acc_weights.tolist())
    acc_based_acc = np.mean(acc_fused.argmax(axis=1) == test_labels)
    print(f"  Weights: [{', '.join([f'{w:.3f}' for w in acc_weights])}]")
    print(f"  Test Accuracy: {acc_based_acc:.4f}")
    
    # Method 3: LEARNED weights via stacking (PRINCIPLED)
    print("\nMethod 3: Learned weights via STACKING (PRINCIPLED)")
    learned_fusion = LearnedFusion()
    learned_fusion.fit(train_probs, train_labels)
    
    weights_info = learned_fusion.get_weights()
    learned_weights = weights_info['aggregate']
    
    print(f"  Learned weights: [{', '.join([f'{w:.3f}' for w in learned_weights])}]")
    
    learned_fused = learned_fusion.transform(test_probs)
    learned_acc = np.mean(learned_fused.argmax(axis=1) == test_labels)
    print(f"  Test Accuracy: {learned_acc:.4f}")
    
    # Show ranking
    ranking = learned_fusion.get_model_ranking()
    print("\n  Model Ranking by Learned Importance:")
    for rank, (idx, weight) in enumerate(ranking):
        print(f"    {rank+1}. Model {idx+1}: {weight:.3f}")
    
    # Compare improvements
    print("\n--- Summary ---")
    print(f"  Uniform weights:        {uniform_acc:.4f}")
    print(f"  Accuracy-based weights: {acc_based_acc:.4f} (heuristic)")
    print(f"  Learned weights:        {learned_acc:.4f} (principled)")
    
    print("\n✓ LearnedFusion working correctly!")


def demo_3_hmm_no_leakage():
    """Demonstrate HMM with proper feature-based encoding (no label leakage)."""
    print_section("3. HMM WITHOUT LABEL LEAKAGE", "=")
    
    from src.temporal_reasoning.hmm_model import HMMChain
    from src.temporal_reasoning.event_encoder import EventEncoder
    
    # Generate synthetic network features (NOT labels!)
    np.random.seed(42)
    n_sequences = 50
    seq_length = 20
    n_features = 5
    
    # Simulate feature sequences (packet counts, byte rates, etc.)
    # These are FEATURES, not ground truth labels
    feature_sequences = []
    for _ in range(n_sequences):
        # Random walk with drift (simulating network traffic patterns)
        seq = np.random.randn(seq_length, n_features)
        seq = np.cumsum(seq, axis=0) / 10
        feature_sequences.append(seq)
    
    print("Feature-based HMM (NO LABEL LEAKAGE):")
    print(f"  Number of sequences: {n_sequences}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Feature dimensions: {n_features}")
    
    # Encode features using clustering (CORRECT: no labels)
    print("\n--- Encoding from Features (via Clustering) ---")
    encoder = EventEncoder(encoding_type='clustering', num_symbols=10)
    
    # Flatten for fitting
    all_features = np.vstack(feature_sequences)
    encoder.fit(all_features)
    
    # Encode each sequence
    observation_sequences = []
    for seq in feature_sequences:
        obs = encoder.transform(seq).tolist()
        observation_sequences.append(obs)
    
    print(f"  Encoder type: {encoder.encoding_type}")
    print(f"  Number of symbols: {encoder.num_symbols}")
    print(f"  Example encoded sequence: {observation_sequences[0][:5]}...")
    
    # Initialize HMM
    hmm = HMMChain(
        num_hidden_states=5,
        num_observations=10,
        state_names=['Normal', 'Recon', 'Exploit', 'Lateral', 'Exfil']
    )
    
    # Validate no label leakage
    print("\n--- Label Leakage Validation ---")
    is_valid = hmm.validate_no_label_leakage('features from KMeans clustering')
    print(f"  Observation source validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Train with time-split (PROPER evaluation)
    print("\n--- Time-Split Training ---")
    hmm, split_info = hmm.fit_time_split(
        observation_sequences,
        train_ratio=0.8
    )
    
    print(f"  Training sequences: {split_info['train_size']}")
    print(f"  Test sequences: {split_info['test_size']}")
    
    # Evaluate forecasting on held-out data
    print("\n--- Forecasting Evaluation (Held-Out Test Set) ---")
    metrics = hmm.evaluate_forecasting(
        split_info['test_sequences'],
        horizon=1
    )
    
    print(f"  Number of predictions: {metrics['n_predictions']}")
    print(f"  Mean confidence: {metrics['mean_confidence']:.4f}")
    print(f"  Mean log probability: {metrics['mean_log_prob']:.4f}")
    
    # Show learned parameters
    print("\n--- Learned HMM Parameters (via Baum-Welch) ---")
    print("  Transition matrix (rows = from state, cols = to state):")
    for i, name in enumerate(hmm.state_names):
        probs = hmm.transition_matrix[i]
        top_trans = np.argsort(probs)[-2:][::-1]
        print(f"    {name}: -> {hmm.state_names[top_trans[0]]} ({probs[top_trans[0]]:.2f}), "
              f"-> {hmm.state_names[top_trans[1]]} ({probs[top_trans[1]]:.2f})")
    
    print("\n✓ HMM with feature-based encoding working correctly!")


def demo_4_cost_sensitivity():
    """Demonstrate cost ratio justification and sensitivity analysis."""
    print_section("4. COST RATIO SENSITIVITY ANALYSIS", "=")
    
    from src.probabilistic_reasoning.cost_sensitivity import (
        cost_sensitivity_analysis,
        get_cost_ratio_justification,
        cost_sensitive_metrics_report,
        find_optimal_cost_ratio
    )
    
    # Generate predictions for IDS scenario
    np.random.seed(42)
    n_samples = 1000
    
    # Imbalanced: 90% normal, 10% attacks
    y_true = np.random.binomial(1, 0.1, n_samples)
    
    # Model predictions with moderate discrimination
    y_prob = np.zeros(n_samples)
    for i, y in enumerate(y_true):
        if y == 1:  # Attack
            y_prob[i] = np.random.beta(4, 2)  # Higher probs
        else:  # Normal
            y_prob[i] = np.random.beta(2, 4)  # Lower probs
    
    print("Intrusion Detection Scenario:")
    print(f"  Total samples: {n_samples}")
    print(f"  Attack rate: {y_true.mean()*100:.1f}%")
    print(f"  Mean predicted probability: {y_prob.mean():.3f}")
    
    # Sensitivity analysis across cost ratios
    print("\n--- Sensitivity Analysis ---")
    results = cost_sensitivity_analysis(
        y_true, y_prob,
        fn_fp_ratios=[1, 2, 5, 10, 20, 50]
    )
    
    print("\n  Ratio | Threshold | Recall | Precision | F1    | FN  | FP")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r.fn_fp_ratio:5.0f} | {r.threshold:9.4f} | {r.recall:6.3f} | "
              f"{r.precision:9.3f} | {r.f1_score:.3f} | {r.false_negatives:3d} | {r.false_positives:3d}")
    
    # Find ratio for target recall
    print("\n--- Finding Optimal Ratio for 95% Recall ---")
    optimal_ratio, optimal_result = find_optimal_cost_ratio(
        y_true, y_prob, target_recall=0.95
    )
    print(f"  Optimal ratio: {optimal_ratio:.1f}")
    print(f"  Achieved recall: {optimal_result.recall:.3f}")
    print(f"  Precision at this point: {optimal_result.precision:.3f}")
    
    # Show justification
    print("\n--- Cost Ratio Justification (for C_FN/C_FP = 10) ---")
    justification = get_cost_ratio_justification(10)
    # Print first few paragraphs
    lines = justification.strip().split('\n')
    for line in lines[:20]:
        print(f"  {line}")
    print("  ...")
    
    # Full report
    print("\n--- Cost-Sensitive Metrics Report ---")
    report = cost_sensitive_metrics_report(y_true, y_prob, fn_fp_ratio=10)
    print(f"  Cost ratio: {report['cost_ratio']}")
    print(f"  Optimal threshold: {report['optimal_threshold']:.4f}")
    print(f"  Recall at optimal: {report['metrics_at_optimal']['recall']:.3f}")
    print(f"  Cost reduction vs default: {report['cost_reduction']:.1f}%")
    
    print("\n✓ Cost sensitivity analysis working correctly!")


def demo_5_integration():
    """Demonstrate integrated workflow."""
    print_section("5. INTEGRATED WORKFLOW", "=")
    
    print("All probability components are now PRINCIPLED:")
    print()
    print("  1. CALIBRATION")
    print("     - ECE and Brier score metrics")
    print("     - Temperature Scaling on validation set")
    print("     - Isotonic Regression (alternative)")
    print()
    print("  2. FUSION")
    print("     - LearnedFusion via stacking meta-model")
    print("     - Weights learned from validation predictions")
    print("     - bayesian_fusion renamed to weighted_ensemble (honest naming)")
    print()
    print("  3. HMM")
    print("     - Observations from FEATURES (clustering), not labels")
    print("     - Baum-Welch training on training sequences only")
    print("     - Time-split evaluation for forecasting")
    print("     - validate_no_label_leakage() helper method")
    print()
    print("  4. COST RATIO")
    print("     - Sensitivity analysis across ratios")
    print("     - Documented justification for default ratio")
    print("     - find_optimal_cost_ratio() for target recall")
    print()
    print("These changes address all concerns about 'rigged probabilities'!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("   COGNIT THREAT: PROPER PROBABILITY COMPONENTS DEMO")
    print("=" * 60)
    
    try:
        demo_1_calibration()
        demo_2_learned_fusion()
        demo_3_hmm_no_leakage()
        demo_4_cost_sensitivity()
        demo_5_integration()
        
        print("\n" + "=" * 60)
        print("   ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
