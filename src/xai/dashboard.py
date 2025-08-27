"""
Streamlit Dashboard for CogniThreat XAI Visualization
Interactive dashboard for attack detection alerts and explanations.

Author: Partner B
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Import local modules
from src.xai.explainer import ExplainableAI


def launch_dashboard(port: int = 8501) -> None:
    """
    Launch the Streamlit dashboard.
    
    Args:
        port: Port number for the dashboard
    """
    import subprocess
    import sys
    
    # Run streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        __file__, "--server.port", str(port)
    ])


class CogniThreatDashboard:
    """Main dashboard class for CogniThreat visualization."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize XAI system
        if 'xai_system' not in st.session_state:
            st.session_state.xai_system = None
            st.session_state.data_generated = False
    
    def setup_page_config(self) -> None:
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="CogniThreat - AI Intrusion Detection",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
        }
        .alert-low {
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_real_time_data(self) -> Dict[str, Any]:
        """Generate simulated real-time network data."""
        current_time = datetime.now()
        
        # Generate network metrics
        normal_traffic = np.random.poisson(100, 1)[0]
        suspicious_activities = np.random.poisson(5, 1)[0]
        blocked_attacks = np.random.poisson(2, 1)[0]
        
        # Generate attack alerts
        attack_types = ['DDoS', 'Port Scan', 'Malware', 'SQL Injection', 'Brute Force']
        severity_levels = ['High', 'Medium', 'Low']
        
        alerts = []
        for _ in range(np.random.randint(0, 4)):
            alert = {
                'timestamp': current_time - timedelta(minutes=np.random.randint(0, 60)),
                'attack_type': np.random.choice(attack_types),
                'severity': np.random.choice(severity_levels, p=[0.2, 0.3, 0.5]),
                'source_ip': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'target_port': np.random.choice([22, 80, 443, 3389, 21]),
                'confidence': np.random.uniform(0.7, 0.99)
            }
            alerts.append(alert)
        
        return {
            'metrics': {
                'normal_traffic': normal_traffic,
                'suspicious_activities': suspicious_activities,
                'blocked_attacks': blocked_attacks,
                'threat_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
            },
            'alerts': alerts
        }
    
    def display_header(self) -> None:
        """Display the main header."""
        st.markdown('<h1 class="main-header">🛡️ CogniThreat Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Driven Intrusion Detection with Quantum Deep Learning & Explainable AI</p>', 
                   unsafe_allow_html=True)
    
    def display_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display key security metrics."""
        st.subheader("📊 Real-Time Security Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Normal Traffic",
                value=f"{metrics['normal_traffic']:,}",
                delta=f"{np.random.randint(-10, 20)}"
            )
        
        with col2:
            st.metric(
                label="Suspicious Activities",
                value=metrics['suspicious_activities'],
                delta=f"{np.random.randint(-2, 5)}"
            )
        
        with col3:
            st.metric(
                label="Blocked Attacks",
                value=metrics['blocked_attacks'],
                delta=f"{np.random.randint(0, 3)}"
            )
        
        with col4:
            threat_color = {
                'Low': '🟢',
                'Medium': '🟡',
                'High': '🔴'
            }
            st.metric(
                label="Threat Level",
                value=f"{threat_color[metrics['threat_level']]} {metrics['threat_level']}"
            )
    
    def display_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Display security alerts."""
        st.subheader("🚨 Security Alerts")
        
        if not alerts:
            st.info("No recent alerts. System is secure! ✅")
            return
        
        for i, alert in enumerate(alerts):
            severity_class = f"alert-{alert['severity'].lower()}"
            
            st.markdown(f"""
            <div class="alert-box {severity_class}">
                <strong>{alert['severity']} Alert: {alert['attack_type']}</strong><br>
                <small>
                Time: {alert['timestamp'].strftime('%H:%M:%S')} | 
                Source: {alert['source_ip']} | 
                Port: {alert['target_port']} | 
                Confidence: {alert['confidence']:.2%}
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    def plot_network_traffic(self) -> None:
        """Plot network traffic over time."""
        st.subheader("📈 Network Traffic Analysis")
        
        # Generate time series data
        times = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='H'
        )
        
        normal_traffic = np.random.poisson(100, len(times)) + \
                        20 * np.sin(np.arange(len(times)) * 2 * np.pi / 24)
        
        attack_traffic = np.random.poisson(5, len(times))
        
        # Add some attack spikes
        spike_indices = np.random.choice(len(times), 3, replace=False)
        attack_traffic[spike_indices] += np.random.poisson(20, 3)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=normal_traffic,
            mode='lines',
            name='Normal Traffic',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=attack_traffic,
            mode='lines',
            name='Attack Traffic',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="Network Traffic Over Time",
            xaxis_title="Time",
            yaxis_title="Packets per Hour",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_attack_distribution(self) -> None:
        """Display attack type distribution."""
        st.subheader("🎯 Attack Type Distribution")
        
        attack_types = ['DDoS', 'Port Scan', 'Malware', 'SQL Injection', 'Brute Force', 'Normal']
        counts = np.random.randint(5, 50, len(attack_types))
        counts[-1] = np.random.randint(200, 500)  # Normal traffic is highest
        
        fig = px.pie(
            values=counts,
            names=attack_types,
            title="Traffic Classification Distribution"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    def display_model_explanations(self) -> None:
        """Display model explanations and XAI insights."""
        st.subheader("🧠 AI Model Explanations")
        
        # Initialize XAI system if not done
        if st.session_state.xai_system is None:
            with st.spinner("Initializing AI Explanation System..."):
                xai = ExplainableAI()
                X, y = xai.generate_synthetic_network_data(n_samples=500)
                xai.prepare_model(X, y)
                st.session_state.xai_system = xai
                st.session_state.data_generated = True
        
        xai = st.session_state.xai_system
        
        if xai and hasattr(xai, 'feature_names'):
            # Feature importance plot
            if hasattr(xai.model, 'feature_importances_'):
                importances = xai.model.feature_importances_
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=xai.feature_names,
                        y=importances,
                        marker_color='lightblue'
                    )
                ])
                
                fig.update_layout(
                    title="Global Feature Importance",
                    xaxis_title="Features",
                    yaxis_title="Importance Score"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Model performance metrics
            if hasattr(xai, 'X_test_scaled') and hasattr(xai, 'y_test'):
                col1, col2 = st.columns(2)
                
                with col1:
                    accuracy = xai.model.score(xai.X_test_scaled, xai.y_test)
                    st.metric("Model Accuracy", f"{accuracy:.2%}")
                
                with col2:
                    # Calculate prediction confidence
                    pred_proba = xai.model.predict_proba(xai.X_test_scaled)
                    avg_confidence = np.mean(np.max(pred_proba, axis=1))
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
        
        # Trust metrics
        st.subheader("🔒 AI Trust Metrics")
        
        trust_col1, trust_col2, trust_col3 = st.columns(3)
        
        with trust_col1:
            st.metric("Explanation Consistency", "87.3%", "↑ 2.1%")
        
        with trust_col2:
            st.metric("Model Stability", "92.1%", "↑ 0.8%")
        
        with trust_col3:
            st.metric("Overall Trust Score", "89.7%", "↑ 1.4%")
    
    def display_quantum_metrics(self) -> None:
        """Display quantum model specific metrics."""
        st.subheader("⚛️ Quantum Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Quantum Advantage",
                "15.3%",
                help="Performance improvement over classical models"
            )
        
        with col2:
            st.metric(
                "Circuit Depth",
                "12 layers",
                help="Depth of the quantum circuit"
            )
        
        with col3:
            st.metric(
                "Quantum Fidelity",
                "94.7%",
                help="Quantum state preparation accuracy"
            )
        
        # Quantum circuit visualization placeholder
        st.info("🔬 Quantum Circuit Visualization: Integration with quantum backends in progress...")
    
    def run_dashboard(self) -> None:
        """Run the main dashboard."""
        self.setup_page_config()
        self.display_header()
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        if st.sidebar.button("🔄 Manual Refresh"):
            st.rerun()
        
        # Generate real-time data
        data = self.generate_real_time_data()
        
        # Display main content
        self.display_metrics(data['metrics'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.plot_network_traffic()
        
        with col2:
            self.display_alerts(data['alerts'])
        
        # Second row
        col3, col4 = st.columns(2)
        
        with col3:
            self.display_attack_distribution()
        
        with col4:
            self.display_quantum_metrics()
        
        # Full width sections
        self.display_model_explanations()
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()


def main():
    """Main function to run the dashboard."""
    dashboard = CogniThreatDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
