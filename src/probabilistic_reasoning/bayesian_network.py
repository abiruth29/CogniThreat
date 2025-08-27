"""
Bayesian Network Implementation for Network Intrusion Detection
================================================================

This module implements Bayesian networks for modeling probabilistic
relationships between network features and attack classifications.

Classes:
    CogniThreatBayesianNetwork: Main Bayesian network for intrusion detection
    NetworkSecurityBN: Specialized network security Bayesian model
    
Functions:
    create_intrusion_detection_bn: Factory function for creating configured BN
    
Author: CogniThreat Team
Date: August 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import K2Score, BicScore
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)


class CogniThreatBayesianNetwork:
    """
    Comprehensive Bayesian Network for Network Intrusion Detection.
    
    This class implements a probabilistic model that captures dependencies
    between network features and security threats using Bayesian inference.
    """
    
    def __init__(self,
                 structure: Optional[List[Tuple[str, str]]] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize CogniThreat Bayesian Network.
        
        Args:
            structure: List of edges defining network structure
            feature_names: Names of network features to include
        """
        self.feature_names = feature_names or self._get_default_features()
        self.structure = structure or self._create_default_structure()
        
        # Initialize pgmpy Bayesian Network
        self.model = BayesianNetwork(self.structure)
        self.inference_engine = None
        self.is_fitted = False
        
        # Create node categorizations
        self.network_nodes = [
            'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins'
        ]
        
        self.attack_nodes = [
            'dos_indicator', 'probe_indicator', 'u2r_indicator', 'r2l_indicator'
        ]
        
        self.context_nodes = [
            'time_of_day', 'day_of_week', 'network_load'
        ]
        
        # Target classification node
        self.target_node = 'attack_type'
        
        # Initialize CPD storage
        self.cpds = {}
        
    def _get_default_features(self) -> List[str]:
        """Get default network feature names."""
        return [
            'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
    
    def _create_default_structure(self) -> List[Tuple[str, str]]:
        """Create default Bayesian network structure for intrusion detection."""
        structure = []
        
        # Network feature dependencies
        structure.extend([
            ('protocol_type', 'service'),
            ('service', 'flag'),
            ('src_bytes', 'dst_bytes'),
            ('protocol_type', 'wrong_fragment'),
            ('service', 'urgent'),
            ('logged_in', 'num_failed_logins'),
            ('hot', 'num_compromised'),
        ])
        
        # Attack indicator dependencies
        structure.extend([
            ('src_bytes', 'dos_indicator'),
            ('dst_bytes', 'dos_indicator'),
            ('wrong_fragment', 'dos_indicator'),
            ('service', 'probe_indicator'),
            ('flag', 'probe_indicator'),
            ('num_failed_logins', 'u2r_indicator'),
            ('num_compromised', 'u2r_indicator'),
            ('root_shell', 'u2r_indicator'),
            ('hot', 'r2l_indicator'),
            ('num_file_creations', 'r2l_indicator'),
        ])
        
        # Context influences
        structure.extend([
            ('time_of_day', 'dos_indicator'),
            ('day_of_week', 'probe_indicator'),
            ('network_load', 'dos_indicator'),
        ])
        
        # Final attack classification
        structure.extend([
            ('dos_indicator', 'attack_type'),
            ('probe_indicator', 'attack_type'),
            ('u2r_indicator', 'attack_type'),
            ('r2l_indicator', 'attack_type'),
        ])
        
        return structure
    
    def discretize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Discretize continuous features for Bayesian network.
        
        Args:
            data: Raw network data
            
        Returns:
            Discretized data suitable for BN learning
        """
        discretized_data = data.copy()
        
        # Define discretization rules
        discretization_rules = {
            'src_bytes': [0, 100, 1000, 10000, float('inf')],
            'dst_bytes': [0, 100, 1000, 10000, float('inf')],
            'count': [0, 1, 5, 20, float('inf')],
            'srv_count': [0, 1, 5, 20, float('inf')],
            'serror_rate': [0, 0.1, 0.5, 1.0],
            'rerror_rate': [0, 0.1, 0.5, 1.0],
            'same_srv_rate': [0, 0.3, 0.7, 1.0],
            'diff_srv_rate': [0, 0.3, 0.7, 1.0],
        }
        
        # Apply discretization
        for feature, bins in discretization_rules.items():
            if feature in discretized_data.columns:
                discretized_data[feature] = pd.cut(
                    discretized_data[feature],
                    bins=bins,
                    labels=[f'{feature}_low', f'{feature}_medium', 
                           f'{feature}_high', f'{feature}_very_high'][:len(bins)-1],
                    include_lowest=True
                )
        
        # Create attack indicators
        if 'attack_type' in discretized_data.columns:
            discretized_data['dos_indicator'] = (
                discretized_data['attack_type'].str.contains('dos', case=False, na=False)
            ).astype(int)
            
            discretized_data['probe_indicator'] = (
                discretized_data['attack_type'].str.contains('probe|portsweep|ipsweep|nmap|satan', 
                                                           case=False, na=False)
            ).astype(int)
            
            discretized_data['u2r_indicator'] = (
                discretized_data['attack_type'].str.contains('u2r|buffer_overflow|loadmodule|perl|rootkit', 
                                                           case=False, na=False)
            ).astype(int)
            
            discretized_data['r2l_indicator'] = (
                discretized_data['attack_type'].str.contains('r2l|ftp_write|guess_passwd|imap|multihop|phf|spy|warezclient|warezmaster', 
                                                           case=False, na=False)
            ).astype(int)
        
        # Add context features
        if 'timestamp' in discretized_data.columns:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(discretized_data['timestamp']):
                discretized_data['timestamp'] = pd.to_datetime(discretized_data['timestamp'])
            
            discretized_data['time_of_day'] = discretized_data['timestamp'].dt.hour.map(
                lambda x: 'night' if x < 6 or x >= 22 else 'day' if 6 <= x < 18 else 'evening'
            )
            
            discretized_data['day_of_week'] = discretized_data['timestamp'].dt.dayofweek.map(
                lambda x: 'weekend' if x >= 5 else 'weekday'
            )
        else:
            # Use random assignment for demo
            discretized_data['time_of_day'] = np.random.choice(['night', 'day', 'evening'], len(discretized_data))
            discretized_data['day_of_week'] = np.random.choice(['weekend', 'weekday'], len(discretized_data))
        
        # Network load indicator
        if 'count' in discretized_data.columns:
            discretized_data['network_load'] = pd.cut(
                discretized_data['count'],
                bins=[0, 5, 20, float('inf')],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
        else:
            discretized_data['network_load'] = np.random.choice(['low', 'medium', 'high'], len(discretized_data))
        
        return discretized_data
    
    def fit(self, data: pd.DataFrame, estimator: str = 'MLE') -> None:
        """
        Learn Bayesian network parameters from data.
        
        Args:
            data: Training data
            estimator: Parameter estimation method ('MLE' or 'Bayes')
        """
        # Discretize data
        discretized_data = self.discretize_features(data)
        
        # Filter columns that exist in both structure and data
        available_nodes = set(discretized_data.columns)
        structure_nodes = set([node for edge in self.structure for node in edge])
        valid_nodes = available_nodes.intersection(structure_nodes)
        
        # Filter structure and data
        valid_structure = [
            (parent, child) for parent, child in self.structure
            if parent in valid_nodes and child in valid_nodes
        ]
        
        if valid_structure:
            self.model = BayesianNetwork(valid_structure)
            
            # Use only valid columns
            valid_data = discretized_data[list(valid_nodes)].dropna()
            
            # Learn parameters
            if estimator == 'MLE':
                estimator_obj = MaximumLikelihoodEstimator(self.model, valid_data)
            else:
                estimator_obj = BayesianEstimator(self.model, valid_data)
            
            # Fit CPDs
            for node in self.model.nodes():
                if node in valid_data.columns:
                    cpd = estimator_obj.estimate_cpd(node)
                    self.model.add_cpds(cpd)
                    self.cpds[node] = cpd
            
            # Verify model
            if self.model.check_model():
                # Initialize inference engine
                self.inference_engine = VariableElimination(self.model)
                self.is_fitted = True
            else:
                raise ValueError("Learned Bayesian network is not valid")
        else:
            raise ValueError("No valid nodes found in data for the defined structure")
    
    def predict_proba(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict attack probabilities given evidence.
        
        Args:
            evidence: Dictionary of observed feature values
            
        Returns:
            Probability distribution over attack types
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Filter evidence to valid nodes
        valid_evidence = {
            key: value for key, value in evidence.items()
            if key in self.model.nodes()
        }
        
        if not valid_evidence:
            return {'normal': 0.5, 'attack': 0.5}
        
        try:
            # Perform inference
            if self.target_node in self.model.nodes():
                query_result = self.inference_engine.query(
                    variables=[self.target_node],
                    evidence=valid_evidence
                )
                
                # Convert to probability dictionary
                prob_dict = {}
                for i, value in enumerate(query_result.state_names[self.target_node]):
                    prob_dict[value] = query_result.values[i]
                
                return prob_dict
            else:
                # Use attack indicators for prediction
                attack_indicators = [node for node in self.attack_nodes if node in self.model.nodes()]
                
                if attack_indicators:
                    total_attack_prob = 0.0
                    
                    for indicator in attack_indicators:
                        query_result = self.inference_engine.query(
                            variables=[indicator],
                            evidence=valid_evidence
                        )
                        
                        # Assuming binary indicator (0=normal, 1=attack)
                        if len(query_result.values) > 1:
                            total_attack_prob += query_result.values[1]
                    
                    # Normalize
                    avg_attack_prob = total_attack_prob / len(attack_indicators)
                    
                    return {
                        'normal': 1 - avg_attack_prob,
                        'attack': avg_attack_prob
                    }
                else:
                    return {'normal': 0.5, 'attack': 0.5}
                    
        except Exception as e:
            print(f"Inference error: {e}")
            return {'normal': 0.5, 'attack': 0.5}
    
    def get_network_structure_score(self, data: pd.DataFrame, score_type: str = 'bic') -> float:
        """
        Calculate structure learning score for the network.
        
        Args:
            data: Evaluation data
            score_type: Scoring method ('bic' or 'k2')
            
        Returns:
            Network structure score
        """
        discretized_data = self.discretize_features(data)
        
        if score_type == 'bic':
            scorer = BicScore(discretized_data)
        else:
            scorer = K2Score(discretized_data)
        
        return scorer.score(self.model)
    
    def visualize_network(self, save_path: str = 'bayesian_network.png') -> None:
        """
        Visualize the Bayesian network structure.
        
        Args:
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 10))
        
        # Create NetworkX graph
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())
        
        # Define node colors by type
        node_colors = []
        for node in G.nodes():
            if node in self.network_nodes:
                node_colors.append('lightblue')
            elif node in self.attack_nodes:
                node_colors.append('lightcoral')
            elif node in self.context_nodes:
                node_colors.append('lightgreen')
            elif node == self.target_node:
                node_colors.append('gold')
            else:
                node_colors.append('lightgray')
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=2000, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=15, label='Network Features'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=15, label='Attack Indicators'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=15, label='Context Features'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                      markersize=15, label='Target Classification')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title('CogniThreat Bayesian Network Structure', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on network structure.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating importance")
        
        # Calculate importance based on node connectivity
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())
        
        importance_scores = {}
        
        for node in G.nodes():
            # Combine in-degree and out-degree
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            # Weight by distance to target node
            try:
                if self.target_node in G.nodes():
                    distance = nx.shortest_path_length(G, node, self.target_node)
                    distance_weight = 1.0 / (distance + 1)
                else:
                    distance_weight = 1.0
            except nx.NetworkXNoPath:
                distance_weight = 0.1
            
            # Calculate importance score
            importance_scores[node] = (in_degree + out_degree * 2) * distance_weight
        
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        normalized_scores = {
            node: score / max_score for node, score in importance_scores.items()
        }
        
        return normalized_scores


class NetworkSecurityBN(CogniThreatBayesianNetwork):
    """
    Specialized Bayesian Network for Network Security Analysis.
    
    This class extends the base CogniThreat BN with specific security-focused
    features and domain knowledge.
    """
    
    def __init__(self):
        """Initialize Network Security Bayesian Network."""
        # Define security-specific structure
        security_structure = [
            # Basic network features
            ('protocol_type', 'service'),
            ('service', 'flag'),
            ('src_bytes', 'network_activity'),
            ('dst_bytes', 'network_activity'),
            
            # Security indicators
            ('wrong_fragment', 'packet_anomaly'),
            ('urgent', 'packet_anomaly'),
            ('hot', 'access_anomaly'),
            ('num_failed_logins', 'access_anomaly'),
            
            # Attack patterns
            ('network_activity', 'dos_likelihood'),
            ('packet_anomaly', 'probe_likelihood'),
            ('access_anomaly', 'privilege_escalation'),
            
            # Final classification
            ('dos_likelihood', 'threat_level'),
            ('probe_likelihood', 'threat_level'),
            ('privilege_escalation', 'threat_level'),
        ]
        
        super().__init__(structure=security_structure)
        
        # Override target node
        self.target_node = 'threat_level'
    
    def assess_threat_level(self, network_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess threat level based on network features.
        
        Args:
            network_features: Dictionary of network observations
            
        Returns:
            Threat level probabilities
        """
        return self.predict_proba(network_features)


def create_intrusion_detection_bn(data: Optional[pd.DataFrame] = None,
                                 structure_type: str = 'default') -> CogniThreatBayesianNetwork:
    """
    Factory function to create and configure intrusion detection Bayesian network.
    
    Args:
        data: Training data for parameter learning
        structure_type: Type of network structure ('default', 'security', 'custom')
        
    Returns:
        Configured Bayesian network
    """
    if structure_type == 'security':
        bn = NetworkSecurityBN()
    else:
        bn = CogniThreatBayesianNetwork()
    
    if data is not None:
        bn.fit(data)
    
    return bn


def demo_bayesian_network():
    """Demonstrate Bayesian network capabilities."""
    print("🧠 Bayesian Network Demonstration")
    print("=" * 50)
    
    # Generate synthetic network data
    n_samples = 1000
    
    synthetic_data = pd.DataFrame({
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'telnet', 'smtp'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(500, n_samples),
        'wrong_fragment': np.random.binomial(1, 0.1, n_samples),
        'urgent': np.random.binomial(1, 0.05, n_samples),
        'hot': np.random.binomial(1, 0.2, n_samples),
        'num_failed_logins': np.random.poisson(0.5, n_samples),
        'logged_in': np.random.binomial(1, 0.8, n_samples),
        'count': np.random.poisson(10, n_samples),
    })
    
    # Add attack types based on patterns
    attack_types = []
    for i in range(n_samples):
        if (synthetic_data.loc[i, 'src_bytes'] > 5000 and 
            synthetic_data.loc[i, 'wrong_fragment'] > 0):
            attack_types.append('dos')
        elif (synthetic_data.loc[i, 'num_failed_logins'] > 2 and 
              synthetic_data.loc[i, 'hot'] > 0):
            attack_types.append('u2r')
        elif synthetic_data.loc[i, 'service'] == 'ftp' and synthetic_data.loc[i, 'flag'] == 'REJ':
            attack_types.append('probe')
        else:
            attack_types.append('normal')
    
    synthetic_data['attack_type'] = attack_types
    
    # Create and train Bayesian network
    bn = create_intrusion_detection_bn(synthetic_data)
    
    print(f"📊 Bayesian Network Structure:")
    print(f"Nodes: {len(bn.model.nodes())}")
    print(f"Edges: {len(bn.model.edges())}")
    
    # Test inference
    test_evidence = {
        'protocol_type': 'tcp',
        'src_bytes': 'src_bytes_high',
        'wrong_fragment': 1,
        'time_of_day': 'night'
    }
    
    probabilities = bn.predict_proba(test_evidence)
    
    print(f"\n🔍 Inference Results for Evidence: {test_evidence}")
    for attack_type, prob in probabilities.items():
        print(f"{attack_type}: {prob:.4f}")
    
    # Feature importance
    importance = bn.get_feature_importance()
    
    print(f"\n📈 Top 5 Most Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, score in sorted_importance[:5]:
        print(f"{feature}: {score:.4f}")
    
    # Visualize network
    print(f"\n🎨 Generating network visualization...")
    bn.visualize_network('demo_bayesian_network.png')
    
    print("✅ Bayesian network demonstration completed successfully!")
    
    return bn


if __name__ == "__main__":
    demo_bayesian_network()
