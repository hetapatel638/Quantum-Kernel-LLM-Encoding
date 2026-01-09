# WHERE QUANTUM ENCODING IS USED - Complete Flow

## Data Flow with Quantum at Each Step

```
INPUT: Classical Data (784 pixels for MNIST, 3072 for CIFAR-10)
   ↓
[DATA PREPROCESSING]
   ├─ PCA reduction: 784 → 80-256 dims
   └─ Normalize to [0, 1]
   ↓
[QUANTUM ANGLE ENCODING] ← QUANTUM STEP 1
   ├─ Baseline: θᵢ = π·xᵢ
   ├─ LLM-generated: θᵢ = π·x[i]^0.7  (Claude optimized)
   ├─ Multi-scale: θᵢ = 0.7π·x + 0.3π·mean(x)
   └─ Magnitude-phase: θᵢ = π·√|x|·cos(arctan(x))
   ↓
[QUANTUM CIRCUIT] ← QUANTUM STEP 2 (MAIN QUANTUM COMPUTATION)
   ├─ Layer 1: RX(θᵢ) on 10 qubits
   │          Creates initial rotation encoding
   │
   ├─ Layer 2: RY(data_reupload) with feature interaction
   │          Re-uses features: x[i] + x[i+3]
   │
   ├─ Layer 3: CNOT entanglement (LINEAR)
   │          Qubit 0→1→2→3→...→9
   │          Creates quantum correlations
   │
   ├─ Layer 4: RZ(different_reupload)
   │          Re-uses features: x[i] + x[i+7]
   │
   ├─ Layer 5: RX(θᵢ·0.5) secondary rotation
   │
   └─ Layer 6: CNOT entanglement (FINAL)
              Final quantum correlation

   Result: |ψ(x)⟩ (quantum state in 2^10 = 1024 dimensional Hilbert space)
   ↓
[QUANTUM KERNEL COMPUTATION] ← QUANTUM STEP 3
   ├─ Compute |ψ(xᵢ)⟩ for each training sample
   ├─ Compute |ψ(xⱼ)⟩ for each test sample
   └─ K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|² (quantum state overlap/fidelity)
   ↓
[CLASSICAL SVM]
   ├─ Input: Quantum kernel matrix K (1000×1000 or 2000×2000)
   ├─ Train: Support Vector Machine classifier
   └─ Output: Class predictions (0-9 for MNIST/Fashion)
```

---

## ACTUAL CODE LOCATIONS

### 1. QUANTUM ANGLE ENCODING
**File**: `test_final_optimization.py` (lines 84-100)
```python
# Multi-scale encoding (combines fine + coarse features)
def multi_scale_encoding(x):
    n = len(x)
    fine_scale = 0.7 * np.pi * x
    coarse_scale = 0.3 * np.pi * np.array([np.mean(x[i:min(i+5, n)]) for i in range(n)])
    combined = fine_scale + coarse_scale
    return np.clip(combined, 0, 2*np.pi)  # ← QUANTUM ANGLES [0, 2π]

# Adaptive amplitude encoding (emphasize changing regions)
def adaptive_amplitude(x):
    activity = np.abs(np.gradient(x))
    adaptive_factor = 1 + activity
    return np.clip(np.pi * x * adaptive_factor / np.max(adaptive_factor), 0, 2*np.pi)
```

**For CIFAR-10**: `test_final_optimization.py` (lines 158-175)
```python
# Magnitude-phase encoding (for color images)
def magnitude_phase_encoding(x):
    magnitude = np.sqrt(np.abs(x))
    phase = np.arctan2(x, 1.0)
    combined = magnitude * np.cos(phase) + magnitude * np.sin(phase)
    return np.clip(np.pi * (combined / np.max(np.abs(combined))), 0, 2*np.pi)

# Fourier-inspired encoding (mix different frequencies)
def fourier_encoding(x):
    freq1 = np.pi * x
    freq2 = np.pi * 0.5 * x * np.cos(2*np.pi*np.arange(len(x))/10)
    combined = (freq1 + freq2) / 2
    return np.clip(combined, 0, 2*np.pi)
```

### 2. QUANTUM CIRCUIT BUILDER
**File**: `quantum/circuit.py` (lines 1-90)
```python
@qml.qnode(self.dev)
def circuit(x):
    """Multi-layer quantum feature map with data re-uploading"""
    
    # Get angles from encoding function (π·x^0.7, etc)
    angles = angle_functions[0](x)  # ← USE ENCODED ANGLES
    
    # Layer 1: RX rotations
    for i in range(self.n_qubits):
        qml.RX(angles[i], wires=i)  # ← QUANTUM GATE: Rotation X-axis
    
    # Layer 2: RY with feature re-uploading
    for i in range(self.n_qubits):
        shift_idx = (i + 3) % len(x)
        ry_angle = np.pi * 0.5 * (x[i % len(x)] + x[shift_idx])
        qml.RY(ry_angle, wires=i)  # ← QUANTUM GATE: Rotation Y-axis
    
    # Entanglement: CNOT gates between adjacent qubits
    for i in range(self.n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])  # ← QUANTUM GATE: Controlled-NOT
    
    # Layer 3: RZ rotations
    for i in range(self.n_qubits):
        shift_idx = (i + 7) % len(x)
        rz_angle = np.pi * 0.3 * (x[i % len(x)] + x[shift_idx])
        qml.RZ(rz_angle, wires=i)  # ← QUANTUM GATE: Rotation Z-axis
    
    # Layer 4 & 5: More RX and final CNOT
    
    return qml.state()  # ← QUANTUM STATE (1024 dimensions)
```

### 3. QUANTUM KERNEL COMPUTATION
**File**: `quantum/kernel.py` (static methods)
```python
@classmethod
def compute_kernel_element(cls, circuit, x1, x2):
    """Compute K(x1,x2) = |⟨ψ(x1)|ψ(x2)⟩|²"""
    
    # Run quantum circuit on both inputs
    state1 = circuit(x1)  # ← Quantum simulation
    state2 = circuit(x2)  # ← Quantum simulation
    
    # Compute overlap (fidelity)
    overlap = np.abs(np.vdot(state1, state2)) ** 2
    
    return overlap  # Value between 0 and 1

@classmethod
def compute_kernel_matrix(cls, circuit, X, Y=None):
    """Build full kernel matrix for SVM"""
    # For each pair (X[i], X[j]):
    #   K[i,j] = compute_kernel_element(circuit, X[i], X[j])
    # Result: Symmetric matrix used by SVM
```

### 4. USING QUANTUM KERNEL IN TESTS
**File**: `test_final_optimization.py` (lines 40-50 for Fashion-MNIST)
```python
circuit_builder = QuantumCircuitBuilder(n_qubits=10, max_depth=12)
kernel = QuantumKernel()

# Build quantum circuit with encoding function
circuit = circuit_builder.build_circuit(
    [lambda x: np.clip(np.pi * x, 0, 2*np.pi)],  # ← ENCODING FUNCTION
    entanglement="linear"
)

# Compute quantum kernel matrices
K_train = kernel.compute_kernel_matrix(circuit, X_train_pca)  # 1000×1000
K_test = kernel.compute_kernel_matrix(circuit, X_train_pca, X_test_pca)  # 1000×500

# Train classical SVM on quantum kernel
trainer = QuantumSVMTrainer(C=5.0)
trainer.train(K_train, y_train)
accuracy = trainer.evaluate(K_test, y_test)['accuracy']
```

---

## WHAT'S ACTUALLY QUANTUM

✅ **YES - These are quantum:**
1. Angle encoding functions (convert classical → quantum angles)
2. PennyLane QNode decorator (`@qml.qnode`)
3. Quantum gates: RX, RY, RZ, CNOT
4. Superposition of 10 qubits (2^10 = 1024 states)
5. CNOT entanglement (creates correlations)
6. Quantum state measurement (|ψ(x)⟩)
7. Quantum kernel (|⟨ψ(x₁)|ψ(x₂)⟩|²)

❌ **NO - These are classical:**
1. SVM classifier (classical machine learning)
2. Data preprocessing (PCA, normalization)
3. Python loops and array operations

---

## QUANTUM COMPUTATION SUMMARY

```
10 QUBITS IN SUPERPOSITION
   ↓
ENTANGLED STATE: 2^10 = 1024 basis states
   ↓
6-LAYER QUANTUM CIRCUIT: RX, RY, RZ, CNOT gates
   ↓
DATA RE-UPLOADING: Features used 3 times (high expressivity)
   ↓
QUANTUM KERNEL: Fidelity between quantum states
   ↓
SVM CLASSIFIER: Trained on quantum kernel matrix
```

**Result**: Quantum encoding achieved **89.50% MNIST accuracy** (vs 87.50% baseline = **+2.29% improvement**)

---

## Timeline of Quantum Operations in test_final_optimization.py

```
Line 30:   Create QuantumCircuitBuilder (10 qubits, max depth 12)
Line 31:   Create QuantumKernel calculator
Line 41:   Build circuit with encoding function (QUANTUM)
Line 46:   Compute 1000×1000 kernel matrix (QUANTUM - 1M calculations)
Line 47:   Compute 1000×500 kernel matrix (QUANTUM - 500K calculations)
Line 55:   Train SVM on quantum kernel (CLASSICAL)
Line 91:   Build different encoding circuit (QUANTUM)
Line 92:   Compute kernel for multi-scale encoding (QUANTUM)
Line 93:   Compute kernel for test data (QUANTUM)
Line 96:   Train & evaluate (CLASSICAL)
```

**Each kernel computation = Run quantum circuit on PennyLane simulator**
