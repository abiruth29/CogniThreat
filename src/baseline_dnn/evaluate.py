"""
DNN Evaluation Module

Comprehensive evaluation metrics for the DNN baseline
including all metrics specified in Scientific Reports 2025:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Performance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, List, Optional, Tuple
import logging

class DNNEvaluator:
    """
    Comprehensive evaluator for DNN baseline performance
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize DNN Evaluator
        
        Args:
            class_names: List of class names for visualization
        """
        self.class_names = class_names
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_pred_proba: np.ndarray = None,
                         average: str = 'weighted') -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary containing all metrics
        """
        self.logger.info("Calculating performance metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # ROC-AUC (if probabilities provided)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # Binary classification
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    y_true_bin = label_binarize(y_true, classes=range(n_classes))
                    roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average=average, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC-AUC: {e}")
                roc_auc = None
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names or range(len(np.unique(y_true)))):
            per_class_metrics[str(class_name)] = {
                'precision': precision_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0] if i in y_true else 0,
                'recall': recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0] if i in y_true else 0,
                'f1': f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0] if i in y_true else 0,
                'support': np.sum(y_true == i)
            }
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_metrics': per_class_metrics,
            'n_samples': len(y_true),
            'n_classes': len(np.unique(y_true))
        }
        
        # Log key metrics
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-score: {f1:.4f}")
        if roc_auc is not None:
            self.logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             class_names: List[str] = None,
                             normalize: bool = False,
                             title: str = 'Confusion Matrix',
                             figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Class names for labels
            normalize: Whether to normalize values
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            cm_display = cm_normalized
        else:
            fmt = 'd'
            cm_display = confusion_matrix
        
        # Create heatmap
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names or range(confusion_matrix.shape[1]),
            yticklabels=class_names or range(confusion_matrix.shape[0]),
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       class_names: List[str] = None,
                       title: str = 'ROC Curves',
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            class_names: Class names
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Compute ROC curve for each class
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), colors):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                
                class_name = class_names[i] if class_names else f'Class {i}'
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self,
                             history: Dict[str, List],
                             title: str = 'Training History',
                             figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot training history (loss and accuracy)
        
        Args:
            history: Training history dictionary
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Plot training & validation loss
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self,
                                  metrics: Dict[str, Any],
                                  model_info: Dict[str, Any] = None,
                                  save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics: Metrics dictionary
            model_info: Model information
            save_path: Path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DNN BASELINE EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        # Model information
        if model_info:
            report_lines.append("\nMODEL INFORMATION:")
            report_lines.append("-" * 40)
            for key, value in model_info.items():
                report_lines.append(f"{key}: {value}")
        
        # Overall metrics
        report_lines.append("\nOVERALL METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision: {metrics['precision']:.4f}")
        report_lines.append(f"Recall:    {metrics['recall']:.4f}")
        report_lines.append(f"F1-score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            report_lines.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Per-class metrics
        report_lines.append("\nPER-CLASS METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        report_lines.append("-" * 60)
        
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            report_lines.append(
                f"{class_name:<15} "
                f"{class_metrics['precision']:<10.4f} "
                f"{class_metrics['recall']:<10.4f} "
                f"{class_metrics['f1']:<10.4f} "
                f"{class_metrics['support']:<10d}"
            )
        
        # Confusion matrix
        report_lines.append("\nCONFUSION MATRIX:")
        report_lines.append("-" * 40)
        cm = metrics['confusion_matrix']
        for row in cm:
            report_lines.append("  ".join(f"{val:6d}" for val in row))
        
        # Dataset information
        report_lines.append("\nDATASET INFORMATION:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total samples: {metrics['n_samples']:,}")
        report_lines.append(f"Number of classes: {metrics['n_classes']}")
        
        report_lines.append("\n" + "=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def compare_models(self, 
                      results_dict: Dict[str, Dict[str, Any]],
                      metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            results_dict: Dictionary of model results
            metrics: Metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in results and results[metric] is not None:
                    row[metric.replace('_', ' ').title()] = results[metric]
                else:
                    row[metric.replace('_', ' ').title()] = np.nan
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        return comparison_df
