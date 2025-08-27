
"""
Explainable AI (XAI) Module for CogniThreat
Contains SHAP, LIME implementations and interactive dashboard.
"""

from .explainer import ExplainableAI
from .dashboard import CogniThreatDashboard, launch_dashboard

__all__ = ['ExplainableAI', 'CogniThreatDashboard', 'launch_dashboard']
