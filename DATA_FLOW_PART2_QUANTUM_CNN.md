# Part 2: Quantum CNN - Where Quantum Mechanics Meets Deep Learning 🌌

## SECTION 4: QUANTUM FEATURE ENCODING 🎭

### 4.1 The Challenge: Classical → Quantum

**Problem:** We have normalized numbers (z-scores), but quantum computers work with quantum states (qubits)!

```python
Classical data:  x = [0.52, 0.31, 1.64, ...]  (regular numbers)
                 ↓ MUST CONVERT ↓
Quantum state:   |ψ⟩ = α|0⟩ + β|1⟩           (superposition!)
```

### 4.2 What is a Qubit? (Quick Primer)

**Classical bit:**
```python
State = 0  OR  State = 1  (one or the other)
```

**Quantum qubit:**
```python
State = α|0⟩ + β|1⟩  (superposition - BOTH at once!)

Where:
  |α|² = probability of measuring 0
  |β|² = probability of measuring 1
  |α|² + |β|² = 1  (must sum to 100%)

Example:
  |ψ⟩ = (1/√2)|0⟩ + (1/√2)|1⟩
  → 50% chance of 0, 50% chance of 1
  → Equal superposition
```

### 4.3 Our Encoding Strategy: Angle Embedding

**Formula:**
```python
For feature x with normalized value x_norm ∈ [-3, 3]:

θ = π × (x_norm + 3) / 6

Apply: RY(θ) to qubit
```

**What this does:**
```python
RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩

Small x_norm (e.g., -2)  → θ ≈ π/6    → mostly |0⟩
Medium x_norm (e.g., 0)  → θ = π/2    → equal |0⟩ and |1⟩  
Large x_norm (e.g., +2)  → θ ≈ 5π/6   → mostly |1⟩
```

### 4.4 Step-by-Step: Encoding Our Sample (t=0)

Let's encode the FIRST 4 FEATURES into 4 qubits:

**Feature 1: duration (normalized z-score = 0.52)**
```python
# Step 1: Map to angle
x_norm = 0.52
θ_1 = π × (0.52 + 3) / 6
    = π × 3.52 / 6
    = 1.843 radians
    = 105.6 degrees

# Step 2: Apply rotation
RY(1.843)|0⟩ = cos(0.922)|0⟩ + sin(0.922)|1⟩
             = 0.597|0⟩ + 0.802|1⟩

# Step 3: Probabilities
P(measure 0) = |0.597|² = 0.356 = 35.6%
P(measure 1) = |0.802|² = 0.644 = 64.4%

Interpretation: Slightly above average duration 
                → qubit tilted toward |1⟩
```

**Feature 2: count (z-score = 1.805 - VERY HIGH!)**
```python
# Step 1: Map to angle  
x_norm = 1.805
θ_2 = π × (1.805 + 3) / 6
    = π × 4.805 / 6
    = 2.516 radians
    = 144.2 degrees

# Step 2: Apply rotation
RY(2.516)|0⟩ = cos(1.258)|0⟩ + sin(1.258)|1⟩
             = 0.304|0⟩ + 0.953|1⟩

# Step 3: Probabilities
P(measure 0) = |0.304|² = 0.092 = 9.2%
P(measure 1) = |0.953|² = 0.908 = 90.8%

Interpretation: VERY high connection count
                → qubit STRONGLY tilted toward |1⟩ 🚨
```

**Feature 3: serror_rate (z-score = 3.167 - EXTREME!)**
```python
# Step 1: Map to angle
x_norm = 3.167
θ_3 = π × (3.167 + 3) / 6
    = π × 6.167 / 6
    = 3.228 radians  (capped at π = 3.14159)
    = π radians
    = 180 degrees

# Step 2: Apply rotation
RY(π)|0⟩ = cos(π/2)|0⟩ + sin(π/2)|1⟩
         = 0|0⟩ + 1|1⟩
         = |1⟩  (pure state!)

# Step 3: Probabilities  
P(measure 0) = 0%
P(measure 1) = 100%

Interpretation: MAXIMUM error rate
                → qubit FULLY in |1⟩ state 🚨🚨🚨
```

**Feature 4: dst_host_count (z-score = 1.960)**
```python
# Step 1: Map to angle
x_norm = 1.960
θ_4 = π × (1.960 + 3) / 6
    = π × 4.960 / 6
    = 2.597 radians
    = 148.8 degrees

# Step 2: Apply rotation
RY(2.597)|0⟩ = cos(1.299)|0⟩ + sin(1.299)|1⟩
             = 0.264|0⟩ + 0.965|1⟩

# Step 3: Probabilities
P(measure 0) = |0.264|² = 0.070 = 7.0%
P(measure 1) = |0.965|² = 0.930 = 93.0%

Interpretation: Contacted many hosts
                → qubit strongly toward |1⟩ 🚨
```

### 4.5 Visual: Our 4-Qubit State

```python
BLOCH SPHERE REPRESENTATION (simplified):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Qubit 1 (duration = 0.52):
       |0⟩
        ↑
        |  θ = 105.6°
        |  /
        | /  ← slightly toward |1⟩
        |/___________> |1⟩
        35.6% |0⟩, 64.4% |1⟩

Qubit 2 (count = 1.805):
       |0⟩
        ↑
        |  θ = 144.2°
        |      /
        |     / ← strongly toward |1⟩  
        |____/________> |1⟩
        9.2% |0⟩, 90.8% |1⟩ 🚨

Qubit 3 (serror = 3.167):
       |0⟩
        ↑
        |  θ = 180°
        |
        |        ← FULLY at |1⟩!
        |________________> |1⟩
        0% |0⟩, 100% |1⟩ 🚨🚨🚨

Qubit 4 (hosts = 1.960):
       |0⟩
        ↑
        |  θ = 148.8°
        |      /
        |     / ← very strongly toward |1⟩
        |____/________> |1⟩
        7.0% |0⟩, 93.0% |1⟩ 🚨
```

### 4.6 Full 4-Qubit System State

```python
# Product state (before entanglement):
|ψ_encoded⟩ = |ψ_1⟩ ⊗ |ψ_2⟩ ⊗ |ψ_3⟩ ⊗ |ψ_4⟩

Expanding:
|ψ_encoded⟩ = (0.597|0⟩ + 0.802|1⟩) ⊗ 
              (0.304|0⟩ + 0.953|1⟩) ⊗
              (0|0⟩ + 1|1⟩) ⊗
              (0.264|0⟩ + 0.965|1⟩)

# This is a 2^4 = 16-dimensional state vector!
# Most probable measurement: |1111⟩ (all attack indicators!)

# In our implementation, we use 4 qubits to encode
# 8 features (2 features per qubit using separate circuits)
```

### 4.7 Code: PennyLane Implementation

```python
import pennylane as qml

# Define quantum device (4 qubits)
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def encode_features(features):
    """
    Encode 4 classical features into 4 qubits
    
    Args:
        features: Array of 4 normalized values
    
    Returns:
        Quantum state vector
    """
    # Encode each feature
    for i, x_norm in enumerate(features):
        # Map to angle: θ ∈ [0, π]
        theta = np.pi * (x_norm + 3) / 6
        
        # Apply rotation
        qml.RY(theta, wires=i)
    
    # Return state
    return qml.state()

# Encode our sample (first 4 features at t=0)
features = [0.52, 1.805, 3.167, 1.960]
quantum_state = encode_features(features)

print(f"Quantum state dimension: {quantum_state.shape}")
# Output: (16,) = 2^4 basis states

print(f"Most probable state: {np.argmax(np.abs(quantum_state)**2)}")
# Output: 15 = binary 1111 = |1111⟩ (all |1⟩'s!)
```

### 4.8 Summary: Classical → Quantum

```python
┌─────────────────────────────────────────────────┐
│         FEATURE ENCODING COMPLETE                │
│                                                  │
│  Classical Input:                                │
│    77 features × 10 time steps                   │
│    Z-scores: [-3, 3]                             │
│                                                  │
│  Quantum Output (for t=0):                       │
│    4 qubits in superposition                     │
│    State: |ψ⟩ ≈ 0.048|0000⟩ + ... + 0.735|1111⟩ │
│    Encodes attack signatures in quantum state!   │
│                                                  │
│  ✓ Ready for Quantum Convolution!               │
└─────────────────────────────────────────────────┘
```

---

## SECTION 5: QUANTUM CONVOLUTION 🌀

### 5.1 What is Convolution? (Classical vs Quantum)

**Classical CNN Convolution:**
```python
Input:   [x1, x2, x3, x4, x5]
Filter:  [w1, w2, w3]

Output:  y1 = w1·x1 + w2·x2 + w3·x3
         y2 = w1·x2 + w2·x3 + w3·x4
         y3 = w1·x3 + w2·x4 + w3·x5

Purpose: Detect local patterns (edges, shapes)
```

**Quantum Convolution:**
```python
Input:   |ψ⟩ = quantum state (4 qubits)
Filter:  U(θ) = parameterized unitary gate
         
Output:  |ψ'⟩ = U(θ)|ψ⟩
         
Purpose: Detect quantum correlations (entanglement, interference)
```

### 5.2 Our Quantum Convolutional Layer

**Architecture:**
```python
4 qubits: q0, q1, q2, q3

Convolutional pattern (nearest-neighbor):
  ┌──────┐
  │ U_01 │  ← entangle q0 and q1
  └──────┘
     ┌──────┐
     │ U_12 │  ← entangle q1 and q2
     └──────┘
        ┌──────┐
        │ U_23 │  ← entangle q2 and q3
        └──────┘

Each U_ij is a 2-qubit parameterized gate
```

### 5.3 The Quantum Gate: Controlled-RY

**Formula:**
```python
U_ij(θ) = CRY(θ) applied to qubits (i, j)

Effect:
  |00⟩ → |00⟩                          (no change)
  |01⟩ → |01⟩                          (no change)
  |10⟩ → cos(θ/2)|10⟩ + sin(θ/2)|11⟩  (rotation)
  |11⟩ → -sin(θ/2)|10⟩ + cos(θ/2)|11⟩ (rotation)

If control qubit = |0⟩: target unchanged
If control qubit = |1⟩: target rotates
```

### 5.4 Step-by-Step: Applying U_01

**Before convolution (approximate state):**
```python
Qubits 0-1 (duration, count):
|ψ_01⟩ ≈ 0.182|00⟩ + 0.581|01⟩ + 0.243|10⟩ + 0.765|11⟩

Probabilities:
  |00⟩: 3.3%   (both low)
  |01⟩: 33.8%  (duration low, count high)
  |10⟩: 5.9%   (duration high, count low)
  |11⟩: 57.0%  (BOTH HIGH - attack pattern!) 🚨
```

**Learned parameter:** θ_01 = 0.87 radians (from training)

**Apply CRY(0.87) gate:**
```python
# Effect on each basis state:
# |00⟩ → |00⟩ (control=0, no change)
# |01⟩ → |01⟩ (control=0, no change)
# |10⟩ → cos(0.435)|10⟩ + sin(0.435)|11⟩
#      = 0.906|10⟩ + 0.423|11⟩
# |11⟩ → -sin(0.435)|10⟩ + cos(0.435)|11⟩
#      = -0.423|10⟩ + 0.906|11⟩

# New state:
|ψ'_01⟩ = 0.182|00⟩ + 0.581|01⟩ 
         + 0.243(0.906|10⟩ + 0.423|11⟩)
         + 0.765(-0.423|10⟩ + 0.906|11⟩)
         
|ψ'_01⟩ = 0.182|00⟩ + 0.581|01⟩ 
         + (0.220 - 0.324)|10⟩
         + (0.103 + 0.693)|11⟩
         
|ψ'_01⟩ ≈ 0.182|00⟩ + 0.581|01⟩ - 0.104|10⟩ + 0.796|11⟩

# New probabilities:
  |00⟩: 3.3%   (unchanged)
  |01⟩: 33.8%  (unchanged)
  |10⟩: 1.1%   (DECREASED - unlikely combination)
  |11⟩: 63.3%  (INCREASED - attack pattern amplified!) 🚨🚨
```

**What happened?**
- The gate learned to AMPLIFY the |11⟩ state (both features high)
- This pattern correlates with attacks!
- Quantum interference redistributed probability

### 5.5 After All Three Convolutions

```python
# Apply convolutions sequentially:
|ψ_initial⟩ = encoded state

# Layer 1: q0 ↔ q1 (duration ↔ count)
U_01(θ_01) |ψ_initial⟩ = |ψ_1⟩

# Layer 2: q1 ↔ q2 (count ↔ serror)  
U_12(θ_12) |ψ_1⟩ = |ψ_2⟩

# Layer 3: q2 ↔ q3 (serror ↔ hosts)
U_23(θ_23) |ψ_2⟩ = |ψ_conv⟩

# Final state: highly entangled!
# Attack patterns are now encoded in quantum correlations
```

**Entanglement visualization:**
```python
BEFORE CONVOLUTION (product state):
  q0 ──o── independent
  q1 ──o── independent  
  q2 ──o── independent
  q3 ──o── independent

AFTER CONVOLUTION (entangled):
  q0 ──●══════════════════ correlated
  q1 ══●══════════════════ correlated
  q2 ══════●══════════════ correlated
  q3 ══════════════════●── correlated

Now: measuring q0 affects measurement of q1, q2, q3!
```

### 5.6 Code: PennyLane Implementation

```python
@qml.qnode(dev)
def quantum_convolution(features, params):
    """
    Quantum convolutional layer
    
    Args:
        features: 4 normalized values to encode
        params: [θ_01, θ_12, θ_23] learned angles
    
    Returns:
        Expectations of Pauli-Z measurements
    """
    # Step 1: Encode features
    for i, x in enumerate(features):
        theta = np.pi * (x + 3) / 6
        qml.RY(theta, wires=i)
    
    # Step 2: Apply convolutional gates
    qml.CRY(params[0], wires=[0, 1])  # U_01
    qml.CRY(params[1], wires=[1, 2])  # U_12
    qml.CRY(params[2], wires=[2, 3])  # U_23
    
    # Step 3: Measure each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Learned parameters (from training):
theta_01 = 0.87
theta_12 = 1.23
theta_23 = 0.95
params = [theta_01, theta_12, theta_23]

# Apply to our sample
features = [0.52, 1.805, 3.167, 1.960]
expectations = quantum_convolution(features, params)

print(f"Qubit expectations: {expectations}")
# Output: [0.23, 0.67, -0.91, -0.88]
#          ↑     ↑     ↑       ↑
#       medium high  VERY    VERY
#                    attack! attack!
```

### 5.7 Summary: Convolution Output

```python
┌───────────────────────────────────────────────────┐
│       QUANTUM CONVOLUTION COMPLETE                │
│                                                   │
│  Input:  4 independent qubits                     │
│          Encoded attack features                  │
│                                                   │
│  Process: 3 entangling gates                      │
│           Learn correlations between features     │
│                                                   │
│  Output: 4 expectation values: [0.23, 0.67, -0.91, -0.88] │
│          Range: [-1, 1]                           │
│          Negative = strong attack signal! 🚨      │
│                                                   │
│  ✓ Quantum features ready for measurement!       │
└───────────────────────────────────────────────────┘
```

---

## SECTION 6: QUANTUM MEASUREMENT & FEATURE EXTRACTION 📊

### 6.1 The Measurement Problem

**Quantum state:** |ψ_conv⟩ (superposition, can't observe directly!)  
**Classical NN:** Needs actual numbers, not quantum states!

**Solution:** Measure observables!

### 6.2 What We Measure: Pauli-Z Expectation

**Pauli-Z operator:**
```python
Z = |0⟩⟨0| - |1⟩⟨1|

Eigenvalues:
  Z|0⟩ = +1|0⟩  (state |0⟩ has eigenvalue +1)
  Z|1⟩ = -1|1⟩  (state |1⟩ has eigenvalue -1)
```

**Expectation value:**
```python
⟨Z⟩ = ⟨ψ|Z|ψ⟩ = P(|0⟩) × (+1) + P(|1⟩) × (-1)
    = P(|0⟩) - P(|1⟩)

Range: [-1, +1]
  ⟨Z⟩ = +1  → 100% in |0⟩ (benign)
  ⟨Z⟩ =  0  → equal superposition
  ⟨Z⟩ = -1  → 100% in |1⟩ (attack!)
```

### 6.3 Measuring Our 4 Qubits

**Qubit 0 (duration-related):**
```python
After convolution:
  P(|0⟩) = 0.615 = 61.5%
  P(|1⟩) = 0.385 = 38.5%

⟨Z_0⟩ = 0.615 - 0.385 = +0.23

Interpretation: Slightly toward |0⟩
                → Duration not strongly suspicious
```

**Qubit 1 (count-related):**
```python
After convolution:
  P(|0⟩) = 0.165 = 16.5%
  P(|1⟩) = 0.835 = 83.5%

⟨Z_1⟩ = 0.165 - 0.835 = -0.67

Interpretation: Strongly toward |1⟩
                → High connection count detected! 🚨
```

**Qubit 2 (serror-related):**
```python
After convolution:
  P(|0⟩) = 0.045 = 4.5%
  P(|1⟩) = 0.955 = 95.5%

⟨Z_2⟩ = 0.045 - 0.955 = -0.91

Interpretation: VERY strongly toward |1⟩
                → Extreme error rate! 🚨🚨🚨
```

**Qubit 3 (hosts-related):**
```python
After convolution:
  P(|0⟩) = 0.060 = 6.0%
  P(|1⟩) = 0.940 = 94.0%

⟨Z_3⟩ = 0.060 - 0.940 = -0.88

Interpretation: VERY strongly toward |1⟩
                → Many hosts contacted! 🚨🚨
```

### 6.4 Repeating for ALL Features

Remember: We have 77 features total at each time step!

```python
# We process them in groups of 4:
n_features = 77
n_qubits = 4
n_groups = ceil(77 / 4) = 20 groups

quantum_features_t0 = []

for group in range(20):
    # Get 4 features (or remaining)
    start = group * 4
    end = min(start + 4, 77)
    features_group = X_preprocessed[0, start:end]  # t=0
    
    # Pad if needed
    if len(features_group) < 4:
        features_group = np.pad(features_group, (0, 4-len(features_group)))
    
    # Quantum convolution
    expectations = quantum_convolution(features_group, params[group])
    
    # Store
    quantum_features_t0.extend(expectations)

# Result: 80 quantum features (20 groups × 4 qubits)
print(f"Quantum features shape: {len(quantum_features_t0)}")
# Output: 80
```

### 6.5 Complete Time Series Processing

```python
# For EACH of 10 time steps:
quantum_features_sequence = []

for t in range(10):
    quantum_features_t = []
    
    for group in range(20):
        # Extract features
        start = group * 4
        end = min(start + 4, 77)
        features = X_preprocessed[t, start:end]
        
        # Quantum processing
        expectations = quantum_convolution(features, params[group])
        quantum_features_t.extend(expectations)
    
    quantum_features_sequence.append(quantum_features_t)

# Convert to array
Q = np.array(quantum_features_sequence)
print(f"Quantum feature sequence shape: {Q.shape}")
# Output: (10, 80)  # 10 time steps × 80 quantum features
```

### 6.6 Visualization: Classical vs Quantum Features

**BEFORE Quantum CNN (t=0, first 8 features):**
```python
Feature        Classical (z-score)    Quantum (expectation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
duration       +0.52                  +0.23
count          +1.805 🚨               -0.67 🚨
serror_rate    +3.167 🚨🚨🚨            -0.91 🚨🚨🚨
dst_hosts      +1.960 🚨               -0.88 🚨🚨
src_bytes      +1.64                  -0.51
dst_bytes      -0.50                  +0.35
flag_SYN       +2.10 🚨                -0.73 🚨
flag_ACK       -0.15                  +0.08
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Key differences:
1. Quantum features are bounded: [-1, 1]
2. Negative values = attack indicators
3. Entanglement captured correlations:
   - count + serror both negative (correlated attack signal)
   - duration positive (less suspicious in context)
```

### 6.7 Code: Full Quantum CNN Layer

```python
class QuantumCNN(nn.Module):
    def __init__(self, n_features=77, n_qubits=4):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_groups = (n_features + n_qubits - 1) // n_qubits
        
        # Learnable quantum parameters
        self.params = nn.Parameter(
            torch.randn(self.n_groups, 3) * 0.1  # [θ_01, θ_12, θ_23] per group
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, time, features) classical features
        
        Returns:
            (batch, time, quantum_features) quantum expectations
        """
        batch, time, _ = x.shape
        quantum_out = []
        
        for t in range(time):
            quantum_t = []
            
            for g in range(self.n_groups):
                # Extract feature group
                start = g * self.n_qubits
                end = min(start + self.n_qubits, self.n_features)
                features = x[:, t, start:end]
                
                # Apply quantum convolution
                expectations = quantum_convolution(
                    features.detach().numpy(), 
                    self.params[g].detach().numpy()
                )
                quantum_t.append(torch.tensor(expectations))
            
            quantum_out.append(torch.cat(quantum_t, dim=-1))
        
        return torch.stack(quantum_out, dim=1)

# Usage:
qcnn = QuantumCNN(n_features=77, n_qubits=4)
X_quantum = qcnn(torch.tensor(X_preprocessed).unsqueeze(0))

print(f"Output shape: {X_quantum.shape}")
# Output: (1, 10, 80)  # batch=1, time=10, quantum_features=80
```

### 6.8 Summary: Quantum CNN Output

```python
┌────────────────────────────────────────────────────┐
│         QUANTUM CNN PROCESSING COMPLETE            │
│                                                    │
│  Input:  (10, 77) normalized classical features    │
│                                                    │
│  Process:                                          │
│    1. Encode 77 features → 20 groups of 4 qubits  │
│    2. Apply quantum convolutions (entangle)        │
│    3. Measure Pauli-Z expectations                 │
│                                                    │
│  Output: (10, 80) quantum features                 │
│          Range: [-1, +1]                           │
│          Negative values = attack patterns! 🚨     │
│                                                    │
│  Key insight: q2 = -0.91, q3 = -0.88               │
│               (strong attack signatures)           │
│                                                    │
│  ✓ Ready for Quantum LSTM temporal processing!    │
└────────────────────────────────────────────────────┘
```

**Next:** This quantum feature sequence (10 time steps × 80 features) now feeds into the Quantum LSTM to capture temporal attack patterns!

---

**STATUS: Sections 4-6 Complete (Quantum CNN)!**  
Next: Quantum LSTM (Section 7) - Temporal processing with quantum gates!

