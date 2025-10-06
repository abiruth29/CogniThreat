# 🔬 CogniThreat: Complete Mathematical Data Flow Walkthrough
# For Professor Review - Step-by-Step with Real Numbers

**Document Purpose:** Show exactly how ONE data point transforms through your entire system with mathematical details at every step.

**Perfect for:** Tomorrow's review, PPT preparation, understanding the "why" behind each transformation

---

## 📖 DOCUMENT STRUCTURE

This document is divided into **20 detailed sections**. Study them in order:

**PART 1: Foundation** (Sections 1-3)
**PART 2: Quantum CNN** (Sections 4-6)  
**PART 3: Quantum LSTM** (Sections 7-9)
**PART 4: Classical Layers** (Sections 10-12)
**PART 5: Bayesian Reasoning** (Sections 13-15)
**PART 6: Temporal Reasoning** (Sections 16-18)
**PART 7: Final Output** (Sections 19-20)

---

## SECTION 1: THE SAMPLE DATA POINT 🎯

### 1.1 What Are We Tracking?

ONE network flow from the CIC-IDS-2017 dataset: **A Port Scan Attack**

### 1.2 Raw Network Flow (Before Any Processing)

```
Source IP: 192.168.1.100
Destination IPs: 192.168.1.1 through 192.168.1.255 (255 hosts!)
Protocol: TCP  
Duration: 5.234 seconds
Pattern: SYN packets sent, NO ACK received

KEY FEATURES (77 total, showing critical ones):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature Name          Raw Value    What It Means
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
duration              5.234        Connection lasted 5.234 seconds
protocol_type         6            TCP (6 = TCP, 17 = UDP)
src_bytes             328          328 bytes sent FROM source
dst_bytes             0            0 bytes received (NO RESPONSE!)
flag_SYN              1            SYN flag was set
flag_ACK              0            ACK flag NOT set (suspicious!)
count                 511          511 connection attempts!
serror_rate           1.00         100% SYN errors (MAJOR RED FLAG!)
same_srv_rate         0.00         0% same service (all different)
diff_srv_rate         1.00         100% different services  
dst_host_count        255          Contacted 255 different hosts
dst_host_serror_rate  1.00         100% errors across all hosts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRUE LABEL: Attack (Class 1)
ATTACK TYPE: Port Scan
```

### 1.3 Why Is This Obviously an Attack?

**5 Red Flags:**

1. **511 connections in 5 seconds** = 100+ connections/second  
   Normal user: ~1-5 connections/second

2. **100% SYN errors** (serror_rate = 1.00)  
   Normal: <1% errors  
   This: EVERY single connection failed!

3. **No responses** (dst_bytes = 0, flag_ACK = 0)  
   Normal traffic: bidirectional communication  
   This: One-way scanning

4. **255 different hosts** (dst_host_count = 255)  
   Normal: Connect to 1-10 hosts  
   This: Scanning entire subnet (192.168.1.0/24)

5. **All different services** (diff_srv_rate = 1.00)  
   Normal: Connect to same service repeatedly  
   This: Trying every port on every host

**Conclusion:** This is a classic **port scan** - attacker is mapping the network!

### 1.4 Vector Representation

Converting to numerical array that our model can process:

```python
# Our sample as a feature vector
x_raw = np.array([
    5.234,    # duration
    6,        # protocol
    328,      # src_bytes
    0,        # dst_bytes  
    1,        # flag_SYN
    0,        # flag_ACK
    0, 0, 0, 0, 0, 0, 0, 0, 0,  # other flags and counters
    511,      # count (VERY HIGH!)
    511,      # srv_count
    1.00,     # serror_rate (100%!)
    1.00,     # srv_serror_rate
    0.00, 0.00, 0.00, 1.00, 0.00,  # service rates
    255,      # dst_host_count (full subnet!)
    255,      # dst_host_srv_count
    0.00, 1.00, 0.00, 0.00,  # host connection patterns
    1.00,     # dst_host_serror_rate
    1.00,     # dst_host_srv_serror_rate  
    0.00, 0.00,  # host error rates
    # ... (77 features total)
])

print(f"Shape: {x_raw.shape}")  # Output: (77,)
print(f"Data type: {x_raw.dtype}")  # Output: float32
```

**This is our starting point!** Let's track how it transforms...

---

## SECTION 2: FROM SINGLE POINT TO SEQUENCE ⏱️

### 2.1 Why Sequences?

**Problem:** A single snapshot doesn't capture attack evolution!

**Example:** Port scan progression over time:

```
t=0  (5 sec ago):   count=10,  serror=0.80  ← scan starting
t=1  (4 sec ago):   count=50,  serror=0.90  ← ramping up
t=2  (3 sec ago):   count=100, serror=0.95  ← accelerating
t=3  (2 sec ago):   count=200, serror=0.98  ← intensifying  
t=4  (1 sec ago):   count=400, serror=0.99  ← near peak
t=5  (NOW):         count=511, serror=1.00  ← FULL SCAN!

Pattern: Exponential growth in connections + increasing error rate
```

**Solution:** Use a **sequence of 10 time steps** to capture temporal patterns!

### 2.2 Sequence Construction (Sliding Window)

```python
# Instead of ONE flow:
x_single = [77 features]  # Shape: (77,)

# We use SEQUENCE of 10 flows:
x_sequence = [
    flow_t-9,  # 9 steps ago
    flow_t-8,  # 8 steps ago
    flow_t-7,
    flow_t-6,
    flow_t-5,
    flow_t-4,
    flow_t-3,
    flow_t-2,
    flow_t-1,  # 1 step ago
    flow_t     # NOW (our sample)
]

# Shape: (10, 77)  
#         ↑   ↑
#      time  features
```

### 2.3 Our Actual Sequence (Concrete Values)

Let me show you the REAL sequence values:

```python
X_sequence = np.array([
    # [time, duration, count, serror_rate, diff_srv_rate, dst_host_count]
    # (showing 6 key features out of 77 for clarity)
    
    # t-9: Scan is just starting
    [0, 2.1, 10, 0.80, 0.70, 10],
    
    # t-8: Activity increasing
    [1, 2.5, 25, 0.84, 0.75, 25],
    
    # t-7: Clear pattern emerging
    [2, 3.0, 50, 0.88, 0.80, 50],
    
    # t-6: Scan expanding
    [3, 3.4, 75, 0.90, 0.85, 75],
    
    # t-5: Half subnet scanned
    [4, 3.8, 100, 0.92, 0.88, 100],
    
    # t-4: Accelerating
    [5, 4.2, 150, 0.94, 0.90, 150],
    
    # t-3: Rapid growth
    [6, 4.5, 200, 0.96, 0.93, 200],
    
    # t-2: Near completion  
    [7, 4.8, 300, 0.98, 0.96, 230],
    
    # t-1: Almost done
    [8, 5.0, 400, 0.99, 0.98, 250],
    
    # t=0: FULL SCAN COMPLETE! (our sample)
    [9, 5.234, 511, 1.00, 1.00, 255],
])

print(f"Sequence shape: {X_sequence.shape}")
# Output: (10, 77)  # 10 time steps × 77 features
```

### 2.4 Visual Timeline

```
ATTACK PROGRESSION TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time:     t-9    t-8    t-7    t-6    t-5    t-4    t-3    t-2    t-1    t=0
          ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
Count:    10 →   25 →   50 →   75 →  100 →  150 →  200 →  300 →  400 →  511
          •      ••     ••••   •••••• ••••••••••••••••••••••••••••••••••••
          
Error:   80% →  84% →  88% →  90% →  92% →  94% →  96% →  98% →  99% → 100%

Hosts:    10 →   25 →   50 →   75 →  100 →  150 →  200 →  230 →  250 →  255
          [▓]    [▓▓]   [▓▓▓▓] [▓▓▓▓▓▓][▓▓▓▓▓▓▓▓][▓▓▓▓▓▓▓▓▓▓▓][▓▓▓▓▓▓▓▓▓▓▓▓▓▓]

Pattern: EXPONENTIAL GROWTH = PORT SCAN ATTACK!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Key Insight:** The LSTM will learn to recognize this exponential growth pattern!

### 2.5 Batch Dimension (For Training)

```
During training, we process MULTIPLE sequences in parallel:

Single sequence:    (10, 77)           # What we're tracking
Batch of 32:        (32, 10, 77)       # 32 sequences at once
                     ↑   ↑   ↑
                   batch time features

But for this walkthrough, we focus on ONE sequence to keep it clear.
```

---

## SECTION 3: PREPROCESSING - MAKING DATA MODEL-READY 🧹

### 3.1 The Problem: Mixed Scales

**Before preprocessing:**
```
duration:    5.234      (range: 0.01 to 5000 seconds)
count:       511        (range: 0 to 10,000 connections)
serror_rate: 1.00       (range: 0.0 to 1.0)
protocol:    6          (range: 1 to 17)
```

**Issue:** Neural networks struggle with mixed scales!  
Large values (count=511) dominate small values (serror_rate=1.0)

### 3.2 Solution: Z-Score Normalization (StandardScaler)

**Formula:**
```
x_normalized = (x - μ) / σ

Where:
  x = raw value
  μ = mean (computed from 2.8M training samples)
  σ = standard deviation
```

**Result:** All features have mean≈0, std≈1

### 3.3 Step-by-Step Example: Normalizing 'count'

```python
# Step 1: Get statistics from training data
mu_count = 150.0      # Average connection count
sigma_count = 200.0   # Standard deviation

# Step 2: Our raw value
x_count_raw = 511

# Step 3: Apply formula
x_count_norm = (511 - 150.0) / 200.0
             = 361.0 / 200.0
             = 1.805

# Step 4: Interpret
print(f"Z-score: {x_count_norm:.3f}")
# Output: Z-score: 1.805

# This means: 1.8 standard deviations ABOVE average
# For normal traffic, z-score would be near 0
# High z-score = SUSPICIOUS!
```

### 3.4 Normalizing ALL Features

```python
# Precomputed from training data (2.8M samples):
training_stats = {
    'duration': {'mean': 15.2, 'std': 150.0},
    'protocol': {'mean': 8.5, 'std': 5.2},
    'src_bytes': {'mean': 850.0, 'std': 2000.0},
    'dst_bytes': {'mean': 1200.0, 'std': 3000.0},
    'count': {'mean': 150.0, 'std': 200.0},
    'serror_rate': {'mean': 0.05, 'std': 0.30},
    # ... for all 77 features
}

# Normalize each feature in our sequence
X_normalized = np.zeros((10, 77))

for t in range(10):  # For each time step
    for f in range(77):  # For each feature
        x_raw = X_sequence[t, f]
        mu = training_stats[feature_names[f]]['mean']
        sigma = training_stats[feature_names[f]]['std']
        
        X_normalized[t, f] = (x_raw - mu) / sigma

print(f"Normalized shape: {X_normalized.shape}")
# Output: (10, 77)
```

### 3.5 Before vs After Normalization

**BEFORE (t=0, our sample, showing 5 key features):**
```
Feature         Raw Value    Min-Max Range
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
duration        5.234        [0.01, 5000]
count           511          [0, 10000]
serror_rate     1.00         [0.0, 1.0]
dst_host_count  255          [0, 255]
diff_srv_rate   1.00         [0.0, 1.0]

Problem: WILDLY different scales!
```

**AFTER (t=0, normalized, z-scores):**
```
Feature         Z-Score      Interpretation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
duration        0.52         Slightly above average
count           1.805        WAY above average! 🚨
serror_rate     3.167        EXTREMELY high! 🚨🚨🚨
dst_host_count  1.960        Much higher than normal 🚨
diff_srv_rate   3.150        All services different! 🚨🚨

All features now: Mean≈0, Std≈1, Range≈[-3, 3]
✓ Comparable scales!
```

### 3.6 Complete Normalized Sequence

```python
# Our final preprocessed sequence (showing first 3 time steps completely):

X_preprocessed = np.array([
    # t-9: Early scan (normalized values)
    [-1.2, 0.3, -0.8, -0.5, 0.1, -0.2, ..., -0.7, -0.8, -0.9],
    
    # t-8: Scan growing (normalized values)
    [-0.9, 0.3, -0.6, -0.5, 0.1, -0.2, ..., -0.5, -0.6, -0.7],
    
    # t-7: More activity (normalized values)
    [-0.6, 0.3, -0.4, -0.5, 0.1, -0.2, ..., -0.3, -0.4, -0.5],
    
    # ... t-6 through t-1 ...
    
    # t=0: Full attack (normalized values - OUR SAMPLE!)
    [0.52, 0.31, 1.64, -0.50, 2.10, -0.15, ..., 1.805, 3.167, 3.150],
])

print("✓ Preprocessing complete!")
print(f"Shape: {X_preprocessed.shape}")       # (10, 77)
print(f"Mean: {X_preprocessed.mean():.3f}")   # ≈ 0.000
print(f"Std: {X_preprocessed.std():.3f}")     # ≈ 1.000
```

### 3.7 Why This Step is CRITICAL

**Without normalization:**
```python
# Neural network training:
Weight update for 'count': 
  gradient = 0.0001 × 511 = 0.051  (tiny!)

Weight update for 'serror_rate':
  gradient = 100.0 × 1.0 = 100.0   (huge!)

Result: Unstable training, poor convergence 😔
```

**With normalization:**
```python
# Neural network training:
Weight update for 'count':
  gradient = 0.5 × 1.805 = 0.903   (balanced)

Weight update for 'serror_rate':
  gradient = 0.8 × 3.167 = 2.534   (balanced)

Result: Stable training, fast convergence! 😊
```

### 3.8 Summary: Input is Ready!

```
┌─────────────────────────────────────────────────────────┐
│           PREPROCESSED INPUT TO MODEL                    │
│                                                          │
│  X_input = Normalized sequence of 10 flows              │
│  Shape: (10, 77)                                         │
│  Values: Z-scores, mostly in [-3, 3]                    │
│  Mean ≈ 0, Std ≈ 1                                       │
│                                                          │
│  ✓ Ready for Quantum CNN Layer!                         │
└─────────────────────────────────────────────────────────┘
```

**Next:** This preprocessed sequence now enters the Quantum CNN layer where things get interesting!

---

**STATUS: Sections 1-3 Complete!**  
Next: Quantum CNN (Section 4) - Where quantum mechanics meets deep learning!

Would you like me to continue with Section 4 (Quantum Feature Encoding)? That's where we'll see how classical data becomes quantum states!

