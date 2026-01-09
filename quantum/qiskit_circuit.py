"""
Level 4: Multi-Layer Quantum Circuit with Data Re-uploading
Architecture: RX (primary) → RY (shifted) → CNOT → RZ (mixed) → RX (modified) → CNOT
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import Callable


class QiskitCircuitBuilder:
    """Build multi-layer Qiskit quantum circuits with data re-uploading"""
    
    def __init__(self, n_qubits: int = 10, entanglement: str = "linear", depth: int = 12):
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        self.depth = depth  # Number of layers (default 12, can increase to 14-16)
    
    def build_feature_map(self, encoding_func: Callable) -> Callable:
        """
        Build multi-layer quantum feature map
        
        Architecture:
        - Layer 1: RX(θᵢ) - Primary encoding with LLM angles
        - Layer 2: RY(0.5·(xᵢ + xᵢ₊₁)) - Shifted data
        - Entanglement 1: Linear CNOT chain
        - Layer 3: RZ(0.3·(xᵢ + xᵢ₊₁)) - Feature mixing
        - Layer 4: RX(θᵢ × 0.5) - Modified angles
        - Entanglement 2: Linear CNOT chain
        
        Args:
            encoding_func: Function that takes data point and returns angle array
        
        Returns:
            Function that creates QuantumCircuit
        """
        
        def feature_map_circuit(x: np.ndarray) -> QuantumCircuit:
            """Create multi-layer feature map circuit for data point x"""
            qc = QuantumCircuit(self.n_qubits)
            
            # Get angles from LLM encoding function
            angles = encoding_func(x)
            
            # Ensure angles is 1D array of length n_qubits
            if isinstance(angles, (float, int)):
                angles = np.full(self.n_qubits, angles)
            elif isinstance(angles, np.ndarray):
                if angles.ndim == 0:
                    angles = np.full(self.n_qubits, angles.item())
                elif len(angles) < self.n_qubits:
                    # Tile to match number of qubits
                    angles = np.tile(angles, (self.n_qubits // len(angles)) + 1)[:self.n_qubits]
                elif len(angles) > self.n_qubits:
                    angles = angles[:self.n_qubits]
            else:
                angles = np.full(self.n_qubits, float(angles))
            
            # Ensure angles are in [0, 2π]
            angles = np.clip(angles, 0, 2*np.pi)
            
            # Pool full feature vector into n_qubits blocks for re-uploading layers
            x_vec = np.asarray(x, dtype=float).flatten()
            if x_vec.size == 0:
                x_vec = np.zeros(self.n_qubits, dtype=float)
            if x_vec.size < self.n_qubits:
                x_vec = np.pad(x_vec, (0, self.n_qubits - x_vec.size))

            if x_vec.size == self.n_qubits:
                x_pooled = x_vec
            else:
                blocks = np.array_split(x_vec, self.n_qubits)
                x_pooled = np.array(
                    [float(np.mean(b)) if len(b) else 0.0 for b in blocks],
                    dtype=float
                )

            # Multiple repetitions of circuit for deeper expressivity
            num_repetitions = max(1, min(self.depth // 2, 5))  # Allow up to 5 reps
            
            for rep in range(num_repetitions):
                # ===== Layer: RX with LLM-generated angles =====
                for i in range(self.n_qubits):
                    qc.rx(float(angles[i]), i)
                
                # ===== Layer: RY with shifted data (data re-uploading) =====
                for i in range(self.n_qubits - 1):
                    # Use pooled feature pairs to incorporate all PCA components
                    shifted = 0.5 * (float(x_pooled[i]) + float(x_pooled[i + 1]))
                    shifted = np.clip(shifted * np.pi, 0, 2*np.pi)  # Scale to [0, 2π]
                    qc.ry(float(shifted), i)
                
                # ===== Entanglement: Linear CNOT chain =====
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                
                # ===== Layer: RZ with feature mixing =====
                for i in range(self.n_qubits - 1):
                    mixed = 0.3 * (float(x_pooled[i]) + float(x_pooled[i + 1]))
                    mixed = np.clip(mixed * np.pi, 0, 2*np.pi)
                    qc.rz(float(mixed), i)
                
                # ===== Layer: RX with modified angles (0.5 scaling) =====
                for i in range(self.n_qubits):
                    modified = (0.5 * float(angles[i])) * (1 + rep * 0.1)  # Vary with repetition
                    qc.rx(float(modified), i)
                
                # ===== Entanglement: Linear CNOT chain with variation =====
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
            
            return qc
        
        return feature_map_circuit
    
    def build_measurement_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Add measurement to circuit"""
        qc_measured = qc.copy()
        c = ClassicalRegister(self.n_qubits, 'c')
        qc_measured.add_register(c)
        qc_measured.measure(range(self.n_qubits), c)
        return qc_measured
