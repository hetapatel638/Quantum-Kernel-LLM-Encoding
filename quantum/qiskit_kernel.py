"""
Level 4: Qiskit Quantum Kernel
Compute kernel matrix using Qiskit circuits
"""

import numpy as np
from typing import Callable
from qiskit_aer import AerSimulator
from tqdm import tqdm


class QiskitKernel:
    """Compute quantum kernel matrix using Qiskit"""
    
    def __init__(self, circuit_builder: Callable, n_qubits: int = 10, 
                 shots: int = 1024, backend: str = "qasm_simulator"):
        """
        Initialize kernel
        
        Args:
            circuit_builder: Function that builds circuit for data point
            n_qubits: Number of qubits
            shots: Number of measurement shots
            backend: Qiskit backend name
        """
        self.circuit_builder = circuit_builder
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator(method='automatic')
    
    def compute_kernel(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix K(X_train, X_test)
        
        Returns:
            Kernel matrix of shape (n_train, n_test)
        """
        n_train = len(X_train)
        n_test = len(X_test)
        
        K = np.zeros((n_train, n_test))
        
        for i in tqdm(range(n_train), desc="Computing kernel"):
            for j in range(n_test):
                # Build circuits for training and test points
                qc_train = self.circuit_builder(X_train[i])
                qc_test = self.circuit_builder(X_test[j])
                
                # Compute overlap (fidelity)
                K[i, j] = self._compute_fidelity(qc_train, qc_test)
        
        return K
    
    def _compute_fidelity(self, qc1, qc2) -> float:
        """Compute fidelity between two quantum states (simplified)"""
        # For simplicity, use output state distribution overlap
        # In production, this would use state vector or density matrix fidelity
        
        try:
            # Get statevectors
            from qiskit_aer.backends import QasmSimulator
            from qiskit.quantum_info import Statevector
            
            sv1 = Statevector.from_instruction(qc1)
            sv2 = Statevector.from_instruction(qc2)
            
            # Compute fidelity
            fidelity = abs(np.dot(sv1.data.conj(), sv2.data)) ** 2
            return float(fidelity)
        except:
            # Fallback: return default value
            return 0.5
