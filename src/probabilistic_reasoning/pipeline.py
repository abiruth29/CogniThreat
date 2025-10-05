"""
Probabilistic Reasoning Integration Module

Provides high-level interface for integrating PR capabilities
into the CogniThreat pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .fusion import bayesian_fusion, ensemble_uncertainty
from .risk_inference import risk_score, create_cost_matrix, alert_prioritization
from .uncertainty import MCDropoutPredictor, predictive_entropy


class ProbabilisticPipeline:
    """
    Unified probabilistic reasoning pipeline for NIDS.
    
    Integrates fusion, uncertainty quantification,
    and risk-based decision making.
    """
    
    def __init__(
        self,
        fusion_method: str = 'weighted_average',
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 10.0
    ):
        """
        Initialize probabilistic pipeline.
        
        Args:
            fusion_method: 'weighted_average', 'product', or 'geometric_mean'
            false_positive_cost: Cost of false positive alerts
            false_negative_cost: Cost of false negative (missed attacks)
        """
        self.fusion_method = fusion_method
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        self.cost_matrix = None
        self.is_fitted = False
    
    def fit(
        self,
        y_true: np.ndarray,
        prob_list: List[np.ndarray],
        model_weights: Optional[List[float]] = None
    ) -> 'ProbabilisticPipeline':
        """
        Fit the probabilistic pipeline.
        
        Args:
            y_true: True labels
            prob_list: List of probability arrays from different models
            model_weights: Optional weights for model fusion
            
        Returns:
            self: Fitted pipeline
        """
        # Fuse model predictions
        fused_probs = bayesian_fusion(prob_list, weights=model_weights, method=self.fusion_method)
        
        # Create cost matrix
        n_classes = fused_probs.shape[1]
        self.cost_matrix = create_cost_matrix(
            n_classes=n_classes,
            false_positive_cost=self.fp_cost,
            false_negative_cost=self.fn_cost
        )
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        prob_list: List[np.ndarray],
        model_weights: Optional[List[float]] = None,
        return_diagnostics: bool = True
    ) -> Dict:
        """
        Make predictions with full probabilistic analysis.
        
        Args:
            prob_list: List of probability arrays from different models
            model_weights: Optional weights for model fusion
            return_diagnostics: If True, return detailed diagnostics
            
        Returns:
            Dictionary with predictions, probabilities, uncertainties, and risk scores
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")
        
        # Fuse predictions
        fused_probs = bayesian_fusion(prob_list, weights=model_weights, method=self.fusion_method)
        
        # Compute uncertainties
        total_uncertainty = ensemble_uncertainty(prob_list, return_components=False)
        
        # Compute risk scores
        risks = risk_score(fused_probs, self.cost_matrix)
        
        # Make predictions
        predictions = fused_probs.argmax(axis=1)
        confidences = fused_probs.max(axis=1)
        
        results = {
            'predictions': predictions,
            'probabilities': fused_probs,
            'confidences': confidences,
            'uncertainties': total_uncertainty,
            'risk_scores': risks
        }
        
        if return_diagnostics:
            # Additional diagnostics
            total_unc, aleatoric_unc, epistemic_unc = ensemble_uncertainty(
                prob_list, return_components=True
            )
            
            results.update({
                'aleatoric_uncertainty': aleatoric_unc,
                'epistemic_uncertainty': epistemic_unc,
                'predictive_entropy': predictive_entropy(fused_probs)
            })
        
        return results
    
    def evaluate(
        self,
        y_true: np.ndarray,
        prob_list: List[np.ndarray],
        model_weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Evaluate probabilistic predictions.
        
        Args:
            y_true: True labels
            prob_list: List of probability arrays
            model_weights: Optional model weights
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Fuse predictions
        fused_probs = bayesian_fusion(prob_list, weights=model_weights, method=self.fusion_method)
        
        # Compute metrics
        # Prediction accuracy
        predictions = fused_probs.argmax(axis=1)
        accuracy = (predictions == y_true).mean()
        
        # Uncertainty metrics
        uncertainties = ensemble_uncertainty(prob_list)
        mean_uncertainty = uncertainties.mean()
        
        return {
            'accuracy': accuracy,
            'mean_uncertainty': mean_uncertainty,
            'probabilities': fused_probs
        }
    
    def prioritize_alerts(
        self,
        prob_list: List[np.ndarray],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Prioritize alerts based on risk scores.
        
        Args:
            prob_list: List of probability arrays
            top_k: Return top-k highest risk samples
            threshold: Return samples above risk threshold
            
        Returns:
            Dictionary with prioritized alerts
        """
        # Get predictions and risk scores
        results = self.predict(prob_list, return_diagnostics=False)
        
        # Prioritize
        prioritized = alert_prioritization(
            risk_scores=results['risk_scores'],
            labels=results['predictions'],
            top_k=top_k,
            threshold=threshold
        )
        
        return prioritized


def quick_probabilistic_analysis(
    y_true: np.ndarray,
    prob_list: List[np.ndarray],
    model_names: Optional[List[str]] = None
) -> Dict:
    """
    Quick probabilistic analysis for model comparison.
    
    Args:
        y_true: True labels
        prob_list: List of probability arrays from different models
        model_names: Optional names for models
        
    Returns:
        Dictionary with comparative analysis
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(prob_list))]
    
    results = {}
    
    for name, probs in zip(model_names, prob_list):
        # Predictions
        preds = probs.argmax(axis=1)
        acc = (preds == y_true).mean()
        
        # Uncertainty
        unc = predictive_entropy(probs).mean()
        
        results[name] = {
            'accuracy': acc,
            'mean_uncertainty': unc
        }
    
    # Ensemble analysis
    fused = bayesian_fusion(prob_list)
    fused_preds = fused.argmax(axis=1)
    fused_acc = (fused_preds == y_true).mean()
    
    total_unc, alea, epis = ensemble_uncertainty(prob_list, return_components=True)
    
    results['Ensemble'] = {
        'accuracy': fused_acc,
        'mean_total_uncertainty': total_unc.mean(),
        'mean_aleatoric_uncertainty': alea.mean(),
        'mean_epistemic_uncertainty': epis.mean()
    }
    
    return results
