"""
CogniThreat Architecture Diagrams Generator
============================================

This script generates publication-quality architecture diagrams for the 
CogniThreat research paper, including:
1. Overall System Architecture
2. Quantum CNN Architecture  
3. Quantum LSTM Architecture
4. Complete Pipeline Data Flow

Author: CogniThreat Team
Date: December 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Ellipse
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as path_effects
import numpy as np
import os

# Set style for publication quality
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def draw_box(ax, x, y, width, height, text, color='#E3F2FD', edge_color='#1565C0', 
             fontsize=9, text_color='black', alpha=0.9, bold=False):
    """Draw a rounded rectangle box with text."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor=edge_color, 
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True,
            path_effects=[path_effects.withStroke(linewidth=0, foreground='white')])
    
    return box


def draw_arrow(ax, start, end, color='#424242', style='->', connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                           connectionstyle=connectionstyle,
                           mutation_scale=12, linewidth=1.5)
    ax.add_patch(arrow)
    return arrow


def create_overall_architecture_diagram(save_path):
    """Create the overall system architecture diagram - clean vertical flow."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'CogniThreat: Hybrid Quantum-Classical Architecture', 
            ha='center', va='center', fontsize=16, weight='bold', color='#1565C0')
    
    # === ROW 1: INPUT ===
    draw_box(ax, 8, 10.2, 3.5, 0.8, 'Raw Network Traffic (CIC-IDS-2017)', 
             color='#FFECB3', edge_color='#FF8F00', fontsize=10, bold=True)
    
    # === ROW 2: PREPROCESSING ===
    draw_box(ax, 8, 8.8, 3.5, 0.8, 'Preprocessing: Feature Extraction,\nNormalization, Sequence (10x77)', 
             color='#E8F5E9', edge_color='#388E3C', fontsize=9)
    
    # === ROW 3: QUANTUM BRANCHES (side by side) ===
    # QCNN Branch
    draw_box(ax, 4.5, 7.0, 3.0, 1.2, 'QCNN Branch\n\nQuantum Convolution\n(4 qubits) + Pooling', 
             color='#BBDEFB', edge_color='#1565C0', fontsize=9, bold=True)
    
    # QLSTM Branch  
    draw_box(ax, 11.5, 7.0, 3.0, 1.2, 'QLSTM Branch\n\nQuantum LSTM Gates\n(4 qubits)', 
             color='#E1BEE7', edge_color='#7B1FA2', fontsize=9, bold=True)
    
    # === ROW 4: FUSION ===
    draw_box(ax, 8, 5.0, 4.0, 0.8, 'Fusion Layer (Concatenate + Dense)', 
             color='#FFF3E0', edge_color='#E65100', fontsize=10, bold=True)
    
    # === ROW 5: CLASSICAL ===
    draw_box(ax, 8, 3.5, 3.5, 0.8, 'Classical: Dense(128) -> Dense(64) -> Softmax', 
             color='#CFD8DC', edge_color='#455A64', fontsize=9)
    
    # === ROW 6: REASONING (side by side) ===
    draw_box(ax, 5, 2.0, 2.8, 0.8, 'Bayesian Uncertainty\n(MC Dropout x100)', 
             color='#FCE4EC', edge_color='#C2185B', fontsize=9)
    draw_box(ax, 11, 2.0, 2.8, 0.8, 'HMM Temporal\nForecasting', 
             color='#FCE4EC', edge_color='#C2185B', fontsize=9)
    
    # === ROW 7: OUTPUT ===
    draw_box(ax, 8, 0.6, 4.0, 0.8, 'FINAL DECISION + Risk Score + Forecast', 
             color='#C8E6C9', edge_color='#2E7D32', fontsize=10, bold=True)
    
    # === ARROWS - Simple vertical/diagonal flow ===
    # Input -> Preprocessing
    draw_arrow(ax, (8, 9.75), (8, 9.25))
    
    # Preprocessing -> QCNN (diagonal left)
    draw_arrow(ax, (6.2, 8.35), (5.2, 7.65))
    
    # Preprocessing -> QLSTM (diagonal right)
    draw_arrow(ax, (9.8, 8.35), (10.8, 7.65))
    
    # QCNN -> Fusion (diagonal to center)
    draw_arrow(ax, (5.2, 6.35), (6.8, 5.45))
    
    # QLSTM -> Fusion (diagonal to center)
    draw_arrow(ax, (10.8, 6.35), (9.2, 5.45))
    
    # Fusion -> Classical
    draw_arrow(ax, (8, 4.55), (8, 3.95))
    
    # Classical -> Bayesian (diagonal left)
    draw_arrow(ax, (6.2, 3.05), (5.5, 2.45))
    
    # Classical -> HMM (diagonal right)
    draw_arrow(ax, (9.8, 3.05), (10.5, 2.45))
    
    # Bayesian -> Output (diagonal to center)
    draw_arrow(ax, (5.5, 1.55), (6.8, 1.05))
    
    # HMM -> Output (diagonal to center)
    draw_arrow(ax, (10.5, 1.55), (9.2, 1.05))
    
    # Add note
    ax.text(8, 8.0, 'Parallel Quantum Processing', 
            ha='center', fontsize=9, style='italic', color='#5E35B1')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


def create_qcnn_architecture_diagram(save_path):
    """Create the Quantum CNN architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(6, 8.5, 'Quantum Convolutional Neural Network (QCNN) Architecture', 
            ha='center', va='center', fontsize=13, weight='bold', color='#1565C0')
    
    # === INPUT STAGE ===
    draw_box(ax, 1.5, 7, 2, 0.7, 'Input Features\n(10, 77)', 
             color='#FFECB3', edge_color='#FF8F00', fontsize=9)
    draw_box(ax, 1.5, 6, 2, 0.7, 'Reshape to\nSpatial (8×8×1)', 
             color='#FFF8E1', edge_color='#FF8F00', fontsize=9)
    
    ax.add_patch(Rectangle((0.3, 5.4), 2.4, 2.3, fill=False, 
                           edgecolor='#FF8F00', linestyle='--', linewidth=1))
    ax.text(1.5, 5.2, 'Input', ha='center', fontsize=9, style='italic', color='#FF8F00')
    
    # === DATA ENCODING ===
    draw_box(ax, 4, 7, 2.2, 0.7, 'Angle Encoding\nθ = π×(x+3)/6', 
             color='#E3F2FD', edge_color='#1565C0', fontsize=9)
    draw_box(ax, 4, 6, 2.2, 0.8, 'RY(θ) Gates\n|0⟩ → α|0⟩+β|1⟩', 
             color='#BBDEFB', edge_color='#1565C0', fontsize=9, bold=True)
    
    ax.add_patch(Rectangle((2.7, 5.3), 2.6, 2.4, fill=False, 
                           edgecolor='#1565C0', linestyle='--', linewidth=1))
    ax.text(4, 5.1, 'Data Encoding', ha='center', fontsize=9, 
            style='italic', color='#1565C0')
    
    # === ENTANGLEMENT ===
    draw_box(ax, 6.8, 6.5, 2.2, 0.8, 'CNOT Gates\nCircular Connectivity', 
             color='#E8F5E9', edge_color='#388E3C', fontsize=9, bold=True)
    draw_box(ax, 6.8, 5.5, 2.2, 0.6, 'Quantum\nCorrelation Capture', 
             color='#C8E6C9', edge_color='#388E3C', fontsize=8)
    
    ax.add_patch(Rectangle((5.5, 5.0), 2.6, 2.2, fill=False, 
                           edgecolor='#388E3C', linestyle='--', linewidth=1))
    ax.text(6.8, 4.8, 'Entanglement', ha='center', fontsize=9, 
            style='italic', color='#388E3C')
    
    # === VARIATIONAL LAYERS ===
    draw_box(ax, 9.8, 7, 2, 0.6, 'RX(θ₁) Rotation', 
             color='#F3E5F5', edge_color='#7B1FA2', fontsize=9)
    draw_box(ax, 9.8, 6.2, 2, 0.6, 'RY(θ₂) Rotation', 
             color='#E1BEE7', edge_color='#7B1FA2', fontsize=9)
    draw_box(ax, 9.8, 5.4, 2, 0.6, 'RZ(θ₃) Rotation', 
             color='#CE93D8', edge_color='#7B1FA2', fontsize=9)
    draw_box(ax, 9.8, 4.6, 2, 0.6, 'CRY Entangling', 
             color='#BA68C8', edge_color='#7B1FA2', fontsize=9)
    
    ax.add_patch(Rectangle((8.6, 4.1), 2.4, 3.5, fill=False, 
                           edgecolor='#7B1FA2', linestyle='--', linewidth=1))
    ax.text(9.8, 3.9, 'Variational (×2 layers)', ha='center', fontsize=9, 
            style='italic', color='#7B1FA2')
    
    # === MEASUREMENT ===
    draw_box(ax, 4, 3.2, 2.4, 0.7, 'Pauli-Z Expectation\n⟨Z⟩ = P(|0⟩)-P(|1⟩)', 
             color='#FCE4EC', edge_color='#C2185B', fontsize=9, bold=True)
    draw_box(ax, 4, 2.2, 2.4, 0.6, 'Filter Response\nAggregation', 
             color='#F8BBD0', edge_color='#C2185B', fontsize=8)
    
    ax.add_patch(Rectangle((2.6, 1.7), 2.8, 2.3, fill=False, 
                           edgecolor='#C2185B', linestyle='--', linewidth=1))
    ax.text(4, 1.5, 'Measurement', ha='center', fontsize=9, 
            style='italic', color='#C2185B')
    
    # === OUTPUT ===
    draw_box(ax, 7.5, 3.2, 2, 0.6, 'ReLU Activation', 
             color='#ECEFF1', edge_color='#455A64', fontsize=9)
    draw_box(ax, 7.5, 2.4, 2, 0.6, 'Quantum Pooling\n(2×2)', 
             color='#CFD8DC', edge_color='#455A64', fontsize=9)
    draw_box(ax, 7.5, 1.5, 2.2, 0.7, 'Feature Maps\n(80 quantum features)', 
             color='#B0BEC5', edge_color='#455A64', fontsize=9, bold=True)
    
    ax.add_patch(Rectangle((6.2, 0.95), 2.6, 2.6, fill=False, 
                           edgecolor='#455A64', linestyle='--', linewidth=1))
    ax.text(7.5, 0.75, 'Output', ha='center', fontsize=9, 
            style='italic', color='#455A64')
    
    # === FLOW ARROWS ===
    # Input internal
    draw_arrow(ax, (1.5, 6.6), (1.5, 6.4))
    
    # Input to Data Encoding
    draw_arrow(ax, (2.55, 6.5), (2.85, 6.5))
    
    # Data Encoding internal
    draw_arrow(ax, (4, 6.55), (4, 6.4))
    
    # Data Encoding to Entanglement
    draw_arrow(ax, (5.15, 6.5), (5.65, 6.5))
    
    # Entanglement internal
    draw_arrow(ax, (6.8, 6.05), (6.8, 5.85))
    
    # Entanglement to Variational
    draw_arrow(ax, (7.95, 6.0), (8.75, 6.0))
    
    # Variational internal (vertical)
    draw_arrow(ax, (9.8, 6.65), (9.8, 6.55))
    draw_arrow(ax, (9.8, 5.85), (9.8, 5.75))
    draw_arrow(ax, (9.8, 5.05), (9.8, 4.95))
    
    # Variational to Measurement (curved to go around boxes)
    ax.annotate('', xy=(5.25, 3.55), xytext=(8.75, 4.25),
                arrowprops=dict(arrowstyle='->', color='#424242', 
                               connectionstyle='arc3,rad=0.3', linewidth=1.5))
    
    # Measurement internal
    draw_arrow(ax, (4, 2.8), (4, 2.55))
    
    # Measurement to Output (curved)
    ax.annotate('', xy=(6.45, 3.2), xytext=(5.25, 2.5),
                arrowprops=dict(arrowstyle='->', color='#424242', 
                               connectionstyle='arc3,rad=-0.2', linewidth=1.5))
    
    # Output internal
    draw_arrow(ax, (7.5, 2.85), (7.5, 2.75))
    draw_arrow(ax, (7.5, 2.05), (7.5, 1.9))
    
    # === QUANTUM CIRCUIT INSET ===
    ax.text(10.5, 2.5, '4-Qubit Filter Circuit:', fontsize=9, weight='bold', color='#1565C0')
    circuit_text = """q0: -RY(x0)-o---RX-RY-RZ-<Z>
q1: -RY(x1)-X-o-RX-RY-RZ-<Z>
q2: -RY(x2)---X-o-RX-RY-RZ-<Z>
q3: -RY(x3)-----X-RX-RY-RZ-<Z>"""
    ax.text(10.5, 1.4, circuit_text, fontsize=7, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#FAFAFA', edgecolor='#1565C0', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


def create_qlstm_architecture_diagram(save_path):
    """Create QLSTM diagram - large boxes, no arrows, for manual editing in Paint."""
    fig, ax = plt.subplots(figsize=(20, 18))
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 16)
    ax.axis('off')
    
    # Title at top
    ax.text(10, 15.3, 'Quantum LSTM Cell: Detailed Architecture', 
            ha='center', va='center', fontsize=24, weight='bold', color='#7B1FA2')
    ax.text(10, 14.6, '4-Qubit Variational Quantum Circuits for Temporal Pattern Learning',
            ha='center', va='center', fontsize=14, style='italic', color='#666666')
    
    # ============ ROW 1: INPUTS (left side, large) ============
    ax.text(3, 13.5, 'INPUTS (at time t)', ha='center', fontsize=16, weight='bold', color='#424242')
    
    draw_box(ax, 3, 12.2, 4.5, 1.0, 'xt: Current Input\n(77-dim feature vector)', 
             color='#FFECB3', edge_color='#FF8F00', fontsize=12, bold=True)
    draw_box(ax, 3, 10.8, 4.5, 1.0, 'ht-1: Previous Hidden\n(64-dim from t-1)', 
             color='#E3F2FD', edge_color='#1565C0', fontsize=12, bold=True)
    draw_box(ax, 3, 9.4, 4.5, 1.0, 'ct-1: Previous Cell\n(64-dim memory)', 
             color='#E8F5E9', edge_color='#388E3C', fontsize=12, bold=True)
    
    # Concatenation box
    draw_box(ax, 3, 7.8, 4.5, 1.2, 'CONCATENATE\nvt = [ht-1 ; xt]\n(141-dim vector)', 
             color='#F3E5F5', edge_color='#7B1FA2', fontsize=12, bold=True)
    
    # ============ ROW 2: FOUR QUANTUM GATES (large, detailed) ============
    ax.text(13, 13.5, 'QUANTUM GATE CIRCUITS (4 qubits each)', 
            ha='center', fontsize=16, weight='bold', color='#7B1FA2')
    
    # Gate boxes - much larger
    gate_w, gate_h = 3.8, 2.2
    
    # FORGET GATE
    draw_box(ax, 9.5, 11.5, gate_w, gate_h, '', color='#FFCDD2', edge_color='#D32F2F', fontsize=10)
    ax.text(9.5, 12.3, 'FORGET GATE', ha='center', fontsize=14, weight='bold', color='#D32F2F')
    ax.text(9.5, 11.8, 'U_f(theta_f)', ha='center', fontsize=11, style='italic', color='#666')
    ax.text(9.5, 11.3, 'RY encode -> CNOT', ha='center', fontsize=10, color='#333')
    ax.text(9.5, 10.9, '-> RX, RY, RZ layers', ha='center', fontsize=10, color='#333')
    ax.text(9.5, 10.5, 'ft = sigmoid(<Z>)', ha='center', fontsize=11, weight='bold', color='#D32F2F')
    
    # INPUT GATE
    draw_box(ax, 14, 11.5, gate_w, gate_h, '', color='#C5E1A5', edge_color='#689F38', fontsize=10)
    ax.text(14, 12.3, 'INPUT GATE', ha='center', fontsize=14, weight='bold', color='#689F38')
    ax.text(14, 11.8, 'U_i(theta_i)', ha='center', fontsize=11, style='italic', color='#666')
    ax.text(14, 11.3, 'RY encode -> CNOT', ha='center', fontsize=10, color='#333')
    ax.text(14, 10.9, '-> RZ, RY, RX layers', ha='center', fontsize=10, color='#333')
    ax.text(14, 10.5, 'it = sigmoid(<Z>)', ha='center', fontsize=11, weight='bold', color='#689F38')
    
    # CANDIDATE GATE
    draw_box(ax, 9.5, 8.8, gate_w, gate_h, '', color='#B3E5FC', edge_color='#0288D1', fontsize=10)
    ax.text(9.5, 9.6, 'CANDIDATE GATE', ha='center', fontsize=14, weight='bold', color='#0288D1')
    ax.text(9.5, 9.1, 'U_g(theta_g)', ha='center', fontsize=11, style='italic', color='#666')
    ax.text(9.5, 8.6, 'H gates -> RY encode', ha='center', fontsize=10, color='#333')
    ax.text(9.5, 8.2, '-> CRY entangle', ha='center', fontsize=10, color='#333')
    ax.text(9.5, 7.8, 'gt = tanh(<Z>)', ha='center', fontsize=11, weight='bold', color='#0288D1')
    
    # OUTPUT GATE
    draw_box(ax, 14, 8.8, gate_w, gate_h, '', color='#E1BEE7', edge_color='#7B1FA2', fontsize=10)
    ax.text(14, 9.6, 'OUTPUT GATE', ha='center', fontsize=14, weight='bold', color='#7B1FA2')
    ax.text(14, 9.1, 'U_o(theta_o)', ha='center', fontsize=11, style='italic', color='#666')
    ax.text(14, 8.6, 'RY encode -> CNOT', ha='center', fontsize=10, color='#333')
    ax.text(14, 8.2, '-> RX, RY, RZ + ct-1', ha='center', fontsize=10, color='#333')
    ax.text(14, 7.8, 'ot = sigmoid(<Z>)', ha='center', fontsize=11, weight='bold', color='#7B1FA2')
    
    # ============ QUANTUM CIRCUIT STRUCTURE (center) ============
    ax.text(10, 6.5, 'VARIATIONAL QUANTUM CIRCUIT (per gate)', 
            ha='center', fontsize=14, weight='bold', color='#5E35B1')
    
    circuit_box = FancyBboxPatch((3, 4.0), 14, 2.2,
                                  boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor='#FAFAFA', edgecolor='#5E35B1', linewidth=2.5)
    ax.add_patch(circuit_box)
    
    circuit = """q0: |0> --RY(v0)--*-------RX(t0)--RY(t1)--RZ(t2)--<Z>
q1: |0> --RY(v1)--X--*----RX(t3)--RY(t4)--RZ(t5)--<Z>
q2: |0> --RY(v2)-----X--*-RX(t6)--RY(t7)--RZ(t8)--<Z>
q3: |0> --RY(v3)--------X-RX(t9)--RY(t10)-RZ(t11)-<Z>"""
    ax.text(10, 5.1, circuit, ha='center', va='center', fontsize=11, 
            family='monospace', color='#333')
    ax.text(10, 4.2, '|-- Encode --|-- Entangle --|---- Variational Layers ----|-- Measure --|',
            ha='center', fontsize=10, style='italic', color='#888')
    
    # ============ CELL STATE UPDATE (full width) ============
    ax.text(10, 3.5, 'CELL STATE UPDATE', ha='center', fontsize=14, weight='bold', color='#E65100')
    draw_box(ax, 10, 2.7, 16, 0.9, 'ct = ft * ct-1  +  it * gt', 
             color='#FFF3E0', edge_color='#E65100', fontsize=16, bold=True)
    
    # ============ HIDDEN STATE UPDATE (full width) ============
    ax.text(10, 2.0, 'HIDDEN STATE UPDATE', ha='center', fontsize=14, weight='bold', color='#E65100')
    draw_box(ax, 10, 1.3, 16, 0.8, 'ht = ot * tanh(ct)', 
             color='#FFE0B2', edge_color='#E65100', fontsize=16, bold=True)
    
    # ============ OUTPUTS (fed back to next timestep) ============
    ax.text(10, 0.6, 'OUTPUTS (fed back at t+1)', ha='center', fontsize=14, weight='bold', color='#00796B')
    draw_box(ax, 6, -0.3, 5, 0.8, 'ht: New Hidden State (64-dim)', 
             color='#B2DFDB', edge_color='#00796B', fontsize=12, bold=True)
    draw_box(ax, 14, -0.3, 5, 0.8, 'ct: New Cell State (64-dim)', 
             color='#80CBC4', edge_color='#00796B', fontsize=12, bold=True)
    
    # NO ARROWS - User will add them manually in Paint
    
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


def create_pipeline_flow_diagram(save_path):
    """Create the complete pipeline data flow diagram."""
    fig, ax = plt.subplots(figsize=(22, 8))  # Slightly wider figure
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title - larger and bolder
    ax.text(11, 7.3, 'CogniThreat: Complete Pipeline Data Flow', 
            ha='center', va='center', fontsize=20, weight='bold', color='#1565C0')
    
    # Stage parameters with adjusted x positions for 9 stages
    stages = [
        {'x': 1.2, 'title': 'Stage 1', 'name': 'Raw Input', 'content': 'Network Packet\n77 Features',
         'color': '#FFECB3', 'edge': '#FF8F00'},
        {'x': 3.6, 'title': 'Stage 2', 'name': 'Preprocessing', 'content': 'Normalize\n+ Sequence',
         'color': '#E8F5E9', 'edge': '#388E3C'},
        {'x': 6.0, 'title': 'Stage 3', 'name': 'QCNN', 'content': 'Encode\n+ Entangle',
         'color': '#E3F2FD', 'edge': '#1565C0'},
        {'x': 8.4, 'title': 'Stage 4', 'name': 'QLSTM', 'content': 'Quantum\nGates',
         'color': '#F3E5F5', 'edge': '#7B1FA2'},
        {'x': 10.8, 'title': 'Stage 5', 'name': 'Classification', 'content': 'Dense Layers\nSoftmax',
         'color': '#ECEFF1', 'edge': '#455A64'},
        {'x': 13.2, 'title': 'Stage 6', 'name': 'Calibration', 'content': 'Temperature\nScaling',
         'color': '#FFF9C4', 'edge': '#FBC02D'},
        {'x': 15.6, 'title': 'Stage 7', 'name': 'Fusion', 'content': 'Learned\nWeights\nMC Dropout',
         'color': '#FCE4EC', 'edge': '#C2185B'},
        {'x': 18.0, 'title': 'Stage 8', 'name': 'HMM', 'content': 'State Est.\nForecasting',
         'color': '#FFF3E0', 'edge': '#E65100'},
        {'x': 20.4, 'title': 'Stage 9', 'name': 'Decision', 'content': 'Risk Score\nAction',
         'color': '#C8E6C9', 'edge': '#2E7D32'},
    ]
    
    # Draw stages with larger boxes and fonts
    for stage in stages:
        # Stage number title - larger
        ax.text(stage['x'], 6.0, stage['title'], ha='center', fontsize=18, 
                color='#424242', weight='bold')
        # Stage name - large and bold
        ax.text(stage['x'], 5.0, stage['name'], ha='center', fontsize=18, 
                color=stage['edge'], weight='bold')
        # Main box - larger with bigger font
        box = FancyBboxPatch((stage['x'] - 1.0, 2.0), 2.0, 2.6,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=stage['color'], edgecolor=stage['edge'], 
                             linewidth=2.5, alpha=0.9)
        ax.add_patch(box)
        # Content text - larger
        ax.text(stage['x'], 3.3, stage['content'], ha='center', va='center', 
                fontsize=18, color='black', weight='normal', linespacing=1.3)
    
    # Draw arrows between stages - thicker
    for i in range(len(stages) - 1):
        arrow = FancyArrowPatch((stages[i]['x'] + 1.05, 3.3), 
                                (stages[i+1]['x'] - 1.05, 3.3), 
                                arrowstyle='-|>', color='#424242',
                                mutation_scale=15, linewidth=2.5)
        ax.add_patch(arrow)
    
    # Data transformation labels - larger
    transformations = [
        (2.4, 'Features'),
        (4.8, 'Encoding'),
        (7.2, 'Quantum Feat'),
        (9.6, 'Logits'),
        (12.0, 'Calibrate'),
        (14.4, 'Uncertainty'),
        (16.8, 'Timeline'),
        (19.2, 'Risk')
    ]
    
    for x, label in transformations:
        ax.text(x, 1.3, label, ha='center', fontsize=18, color='#555555', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#9E9E9E', alpha=0.9))
    
    # Sample data flow example - larger
    ax.text(11, 0.4, 'Ex: Port Scan -> Logits -> Calibrated P=99.9% -> Uncertainty=Low -> State=Recon -> Risk=Medium -> Alert', 
            ha='center', fontsize=18, style='italic', color='#D32F2F', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}")


def create_quantum_circuit_diagram(save_path):
    """Create a detailed quantum circuit diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Quantum Circuits in CogniThreat', 
            ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    
    # === QCNN FILTER CIRCUIT ===
    ax.text(3.5, 6.8, 'A. QCNN Filter Circuit (4 qubits)', fontsize=11, weight='bold', color='#1565C0')
    
    qcnn_circuit = """
    ┌─────────┐┌───────────────────────────────────┐┌─────────┐
q₀: ┤ RY(x₀) ├┤ ●──────────────────────────────── ├┤ RX(θ₁) ├─RY(θ₂)─RZ(θ₃)─●─────┤M⟩─⟨Z₀⟩
    └─────────┘│                                   └─────────┘                │
    ┌─────────┐│┌───┐                              ┌─────────┐                │
q₁: ┤ RY(x₁) ├┼┤ X ├●─────────────────────────────┤ RX(θ₄) ├─RY(θ₅)─RZ(θ₆)─X─CRY──┤M⟩─⟨Z₁⟩
    └─────────┘│└───┘│                             └─────────┘                  │
    ┌─────────┐│     │┌───┐                        ┌─────────┐                  │
q₂: ┤ RY(x₂) ├┼─────┼┤ X ├●───────────────────────┤ RX(θ₇) ├─RY(θ₈)─RZ(θ₉)────CRY──┤M⟩─⟨Z₂⟩
    └─────────┘│     │└───┘│                       └─────────┘                    │
    ┌─────────┐│     │     │┌───┐                  ┌──────────┐                   │
q₃: ┤ RY(x₃) ├┼─────┼─────┼┤ X ├──────────────────┤ RX(θ₁₀) ├─RY(θ₁₁)─RZ(θ₁₂)────┤M⟩─⟨Z₃⟩
    └─────────┘│     │     │└───┘                  └──────────┘
    ├─────────┤├─────┴─────┴─────┤├────────────────────────────────────┤├────────────┤
       Data       Entanglement              Variational Layer                 Measure
      Encoding        (CNOT)              (Trainable Parameters)             (Pauli-Z)
"""
    ax.text(0.2, 5.2, qcnn_circuit, fontsize=7, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.9))
    
    # === QLSTM GATE CIRCUIT ===
    ax.text(3.5, 1.8, 'B. QLSTM Forget Gate Circuit (4 qubits)', fontsize=11, weight='bold', color='#7B1FA2')
    
    qlstm_circuit = """
    ┌─────────────┐┌─────────────┐┌───┐┌─────────┐┌─────────┐┌─────────┐┌───────────┐
q₀: ┤ RY(xₜ[0])  ├┤ RY(hₜ₋₁[0]) ├┤ ● ├┤ RX(θ₁) ├┤ RY(θ₂) ├┤ RZ(θ₃) ├┤M⟩─⟨Z₀⟩→σ
    └─────────────┘└─────────────┘│   │└─────────┘└─────────┘└─────────┘
    ┌─────────────┐┌─────────────┐│┌──┴┐┌─────────┐┌─────────┐┌─────────┐
q₁: ┤ RY(xₜ[1])  ├┤ RY(hₜ₋₁[1]) ├┼┤ X ├┤ RX(θ₄) ├┤ RY(θ₅) ├┤ RZ(θ₆) ├┤M⟩─⟨Z₁⟩→σ
    └─────────────┘└─────────────┘│└───┘└─────────┘└─────────┘└─────────┘     ↓
    ...                           ...                                      fₜ = forget gate
                                                                          (sigmoid activation)
"""
    ax.text(0.2, 0.2, qlstm_circuit, fontsize=7, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor='#7B1FA2', alpha=0.9))
    
    # Legend
    ax.text(11, 5.5, 'Legend:', fontsize=10, weight='bold', color='#424242')
    legend_items = [
        ('RY(θ)', 'Rotation around Y-axis'),
        ('RX(θ)', 'Rotation around X-axis'),
        ('RZ(θ)', 'Rotation around Z-axis'),
        ('●─X', 'CNOT gate (entanglement)'),
        ('CRY', 'Controlled-RY gate'),
        ('⟨Z⟩', 'Pauli-Z expectation value'),
        ('σ', 'Sigmoid activation'),
    ]
    for i, (symbol, desc) in enumerate(legend_items):
        ax.text(11, 5.0 - i*0.4, f'{symbol}:', fontsize=8, weight='bold', 
                family='monospace', color='#1565C0')
        ax.text(12, 5.0 - i*0.4, desc, fontsize=8, color='#424242')
    
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {save_path}")


def main():
    """Generate architecture diagrams."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'project_report', 'diagrams')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎨 CogniThreat Architecture Diagram Generator")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")
    
    # Generate ONLY FIG 4 as requested
    print("Generating Pipeline Flow Diagram (Fig 4)...")
    
    # create_overall_architecture_diagram(...)
    # create_qcnn_architecture_diagram(...)
    # create_qlstm_architecture_diagram(...)
    
    create_pipeline_flow_diagram(
        os.path.join(output_dir, 'fig4_pipeline_dataflow.png'))
    
    # create_quantum_circuit_diagram(...)
    
    print("\n" + "=" * 60)
    print("✅ Diagram generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
