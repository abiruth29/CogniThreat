"""
Explainable AI (XAI) Module for CogniThreat
Implements SHAP and LIME for model explainability in intrusion detection.

Author: Partner B
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# ML and XAI libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Explainability libraries
import shap
from lime import lime_tabular
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ExplainableAI:
    """
    Explainable AI class for intrusion detection model interpretability.
    
    Provides SHAP and LIME explanations, feature importance analysis,
    and trust metrics for AI-driven security decisions.
    """
    
    def __init__(self, model: Optional[Any] = None):
        """
        Initialize the ExplainableAI system.
        
        Args:
            model: Pre-trained model for explanation. If None, will train a demo model.
        """
        self.model = model
        self.scaler = StandardScaler()
        self.feature_names = []
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize SHAP
        shap.initjs()
    
    def generate_synthetic_network_data(
        self,
        n_samples: int = 2000,
        n_features: int = 10
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic network traffic data for XAI demonstration.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features per sample
            
        Returns:
            Tuple of (features DataFrame, labels array)
        """
        self.logger.info(f"Generating {n_samples} synthetic network samples...")
        
        # Define feature names representing network traffic characteristics
        self.feature_names = [
            'packet_size', 'flow_duration', 'tcp_flags', 'port_number',
            'protocol_type', 'service_type', 'flag_status', 'src_bytes',
            'dst_bytes', 'connection_count'
        ]
        
        # Generate base features
        np.random.seed(42)
        features = np.random.randn(n_samples, n_features)
        
        # Create realistic network traffic patterns
        # Normal traffic patterns
        normal_mask = np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
        
        # Attack patterns
        attack_types = ['dos', 'probe', 'u2r', 'r2l']
        attack_labels = np.zeros(n_samples)
        
        for i in range(n_samples):
            if not normal_mask[i]:  # Attack traffic
                attack_type = np.random.choice(attack_types)
                attack_labels[i] = 1
                
                if attack_type == 'dos':
                    # DoS: high packet size, short duration, many connections
                    features[i, 0] += np.random.normal(3, 0.5)  # packet_size
                    features[i, 1] += np.random.normal(-2, 0.3)  # flow_duration
                    features[i, 9] += np.random.normal(4, 0.8)  # connection_count
                    
                elif attack_type == 'probe':
                    # Probe: scanning multiple ports
                    features[i, 3] += np.random.normal(2, 0.4)  # port_number variation
                    features[i, 6] += np.random.normal(1.5, 0.3)  # flag_status
                    
                elif attack_type == 'u2r':
                    # User to Root: privilege escalation patterns
                    features[i, 4] += np.random.normal(1.8, 0.4)  # protocol_type
                    features[i, 5] += np.random.normal(2.2, 0.5)  # service_type
                    
                else:  # r2l
                    # Remote to Local: data exfiltration patterns
                    features[i, 7] += np.random.normal(-1.5, 0.4)  # src_bytes
                    features[i, 8] += np.random.normal(3.5, 0.7)  # dst_bytes
        
        # Create DataFrame
        df_features = pd.DataFrame(features, columns=self.feature_names)
        
        return df_features, attack_labels
    
    def prepare_model(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """
        Prepare and train a model for explanation if none provided.
        
        Args:
            X: Feature data
            y: Target labels
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        if self.model is None:
            self.logger.info("Training Random Forest model for demonstration...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(self.X_train_scaled, self.y_train)
        
        # Model performance
        train_score = self.model.score(self.X_train_scaled, self.y_train)
        test_score = self.model.score(self.X_test_scaled, self.y_test)
        
        self.logger.info(f"Model Performance - Train: {train_score:.4f}, Test: {test_score:.4f}")
    
    def setup_shap_explainer(self) -> None:
        """Setup SHAP explainer for the model."""
        self.logger.info("Setting up SHAP explainer...")
        
        # Use TreeExplainer for tree-based models, or Explainer for others
        if hasattr(self.model, 'estimators_'):  # Tree-based model
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            # For other models, use a sample background
            background = shap.sample(self.X_train_scaled, 100)
            self.shap_explainer = shap.Explainer(self.model.predict, background)
    
    def setup_lime_explainer(self) -> None:
        """Setup LIME explainer for the model."""
        self.logger.info("Setting up LIME explainer...")
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_scaled,
            feature_names=self.feature_names,
            class_names=['Normal', 'Attack'],
            mode='classification',
            discretize_continuous=True
        )
    
    def get_shap_explanations(self, X_explain: np.ndarray) -> shap.Explanation:
        """
        Get SHAP explanations for given samples.
        
        Args:
            X_explain: Samples to explain
            
        Returns:
            SHAP explanation object
        """
        if self.shap_explainer is None:
            self.setup_shap_explainer()
        
        return self.shap_explainer(X_explain)
    
    def get_lime_explanation(self, instance: np.ndarray, instance_idx: int = 0) -> Any:
        """
        Get LIME explanation for a single instance.
        
        Args:
            instance: Single instance to explain
            instance_idx: Index of the instance
            
        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            self.setup_lime_explainer()
        
        explanation = self.lime_explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        return explanation
    
    def plot_feature_importance(self) -> None:
        """Plot global feature importance from the model."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Create feature importance plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            
            ax.bar(range(len(importances)), importances[indices])
            ax.set_title('Global Feature Importance')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_shap_summary(self, shap_values: shap.Explanation, max_display: int = 10) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP explanation values
            max_display: Maximum features to display
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values.values,
            self.X_test_scaled,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_waterfall(self, shap_values: shap.Explanation, instance_idx: int = 0) -> None:
        """
        Plot SHAP waterfall plot for a single instance.
        
        Args:
            shap_values: SHAP explanation values
            instance_idx: Index of instance to plot
        """
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[instance_idx], show=False)
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_instance_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_lime_explanation(self, lime_exp: Any, instance_idx: int = 0) -> None:
        """
        Plot LIME explanation.
        
        Args:
            lime_exp: LIME explanation object
            instance_idx: Index of the explained instance
        """
        # Get explanation as list
        exp_list = lime_exp.as_list()
        
        # Separate positive and negative contributions
        features = [item[0] for item in exp_list]
        values = [item[1] for item in exp_list]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = ['red' if v < 0 else 'green' for v in values]
        
        bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Contribution')
        ax.set_title(f'LIME Explanation - Instance {instance_idx}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01 if width >= 0 else width - 0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{values[i]:.3f}',
                   ha='left' if width >= 0 else 'right',
                   va='center')
        
        plt.tight_layout()
        plt.savefig(f'lime_explanation_instance_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_trust_metrics(self, shap_values: shap.Explanation) -> Dict[str, float]:
        """
        Calculate trust metrics for explanations.
        
        Args:
            shap_values: SHAP explanation values
            
        Returns:
            Dictionary of trust metrics
        """
        # Feature consistency (how consistent are feature contributions)
        feature_std = np.std(shap_values.values, axis=0)
        consistency_score = 1 / (1 + np.mean(feature_std))
        
        # Prediction confidence (based on SHAP values magnitude)
        confidence_scores = np.abs(shap_values.values).sum(axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        # Explanation stability (correlation between similar instances)
        if len(shap_values.values) > 1:
            correlation_matrix = np.corrcoef(shap_values.values)
            stability_score = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        else:
            stability_score = 1.0
        
        trust_metrics = {
            'consistency_score': consistency_score,
            'average_confidence': avg_confidence,
            'stability_score': stability_score,
            'overall_trust': (consistency_score + stability_score) / 2
        }
        
        return trust_metrics
    
    def create_decision_boundary_viz(self, feature_x: int = 0, feature_y: int = 1) -> None:
        """
        Create decision boundary visualization for 2D feature space.
        
        Args:
            feature_x: Index of x-axis feature
            feature_y: Index of y-axis feature
        """
        # Select two features for visualization
        X_viz = self.X_test_scaled[:, [feature_x, feature_y]]
        
        # Create a mesh
        h = 0.02
        x_min, x_max = X_viz[:, 0].min() - 1, X_viz[:, 0].max() + 1
        y_min, y_max = X_viz[:, 1].min() - 1, X_viz[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        # Pad with zeros for other features
        full_mesh = np.zeros((mesh_points.shape[0], self.X_test_scaled.shape[1]))
        full_mesh[:, [feature_x, feature_y]] = mesh_points
        
        Z = self.model.predict_proba(full_mesh)[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
        
        # Plot data points
        scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=self.y_test, 
                            cmap='RdYlBu', edgecolors='black')
        
        plt.colorbar(label='Attack Probability')
        plt.xlabel(f'Feature: {self.feature_names[feature_x]}')
        plt.ylabel(f'Feature: {self.feature_names[feature_y]}')
        plt.title('Decision Boundary Visualization')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='blue', markersize=10, label='Normal'),
                          plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', markersize=10, label='Attack')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def demonstrate(self) -> None:
        """Demonstrate the complete XAI pipeline."""
        self.logger.info("Starting Explainable AI demonstration...")
        
        # Generate synthetic data
        X, y = self.generate_synthetic_network_data(n_samples=1500)
        
        # Prepare model
        self.prepare_model(X, y)
        
        # Setup explainers
        self.setup_shap_explainer()
        self.setup_lime_explainer()
        
        # Global explanations
        self.logger.info("Generating global explanations...")
        self.plot_feature_importance()
        
        # SHAP explanations
        self.logger.info("Generating SHAP explanations...")
        sample_size = min(100, len(self.X_test_scaled))
        shap_values = self.get_shap_explanations(self.X_test_scaled[:sample_size])
        
        self.plot_shap_summary(shap_values)
        self.plot_shap_waterfall(shap_values, instance_idx=0)
        
        # LIME explanations
        self.logger.info("Generating LIME explanations...")
        lime_exp = self.get_lime_explanation(self.X_test_scaled[0])
        self.plot_lime_explanation(lime_exp, instance_idx=0)
        
        # Trust metrics
        trust_metrics = self.calculate_trust_metrics(shap_values)
        self.logger.info("Trust Metrics:")
        for metric, value in trust_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        # Decision boundary
        self.create_decision_boundary_viz()
        
        self.logger.info("Explainable AI demonstration completed successfully!")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and demonstrate XAI
    xai = ExplainableAI()
    xai.demonstrate()
