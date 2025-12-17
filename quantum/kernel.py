import numpy as np
from typing import Callable
from tqdm import tqdm
from functools import lru_cache


class QuantumKernel:
    """Compute quantum kernel matrices"""
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def _cached_circuit_output(circuit: Callable, x_tuple: tuple) -> np.ndarray:
        """Cache circuit outputs to avoid redundant computation"""
        return circuit(np.array(x_tuple))
    
    @classmethod
    def compute_kernel_element(cls, circuit: Callable, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute single kernel element: K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|²
        
        Args:
            circuit: Quantum circuit (QNode)
            x1: First input vector
            x2: Second input vector
            
        Returns:
            Kernel value
        """
        # Convert to tuple for caching (arrays aren't hashable)
        x1_tuple = tuple(x1.tolist())
        x2_tuple = tuple(x2.tolist())
        
        state1 = cls._cached_circuit_output(circuit, x1_tuple)
        state2 = cls._cached_circuit_output(circuit, x2_tuple)
        
        # Compute overlap
        overlap = np.abs(np.vdot(state1, state2)) ** 2
        
        return overlap
    
    @classmethod
    def compute_kernel_matrix(cls, circuit: Callable, X: np.ndarray, Y: np.ndarray = None, subsample: int = None) -> np.ndarray:
        """
        Compute full kernel matrix
        
        Args:
            circuit: Quantum circuit
            X: Training data (n_samples, n_features)
            Y: Test data (optional, defaults to X)
            subsample: If set, only compute for first N samples (for speed)
            
        Returns:
            Kernel matrix (n_samples_X, n_samples_Y)
        """
        if Y is None:
            Y = X
            symmetric = True
        else:
            symmetric = False
        
        # Subsample for faster computation
        if subsample and subsample < X.shape[0]:
            print(f"⚡ Fast mode: Using {subsample} samples instead of {X.shape[0]}")
            X = X[:subsample]
            if symmetric:
                Y = Y[:subsample]
        
        n_X = X.shape[0]
        n_Y = Y.shape[0]
        
        K = np.zeros((n_X, n_Y))
        
        print(f"Computing kernel matrix: {n_X}×{n_Y}")
        
        # Compute kernel elements
        for i in tqdm(range(n_X), desc="Kernel computation"):
            for j in range(n_Y):
                if symmetric and j < i:
                    K[i, j] = K[j, i]  # Use symmetry
                else:
                    K[i, j] = cls.compute_kernel_element(circuit, X[i], Y[j])
        
        return K
    
    @staticmethod
    def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel-target alignment (quality metric)
        
        Args:
            K: Kernel matrix
            y: Labels
            
        Returns:
            Alignment score [0, 1]
        """
        # Construct ideal kernel (labels match = 1, else = 0)
        y_matrix = y.reshape(-1, 1) == y.reshape(1, -1)
        Y_ideal = y_matrix.astype(float)
        
        # Frobenius inner product
        alignment = np.sum(K * Y_ideal) / (np.linalg.norm(K, 'fro') * np.linalg.norm(Y_ideal, 'fro'))
        
        return alignment


# Test
if __name__ == "__main__":
    from quantum.circuit import QuantumCircuitBuilder
    
    # Build simple circuit
    builder = QuantumCircuitBuilder(n_qubits=4)
    
    def angle_func(x):
        return np.pi * x[0]
    
    circuit = builder.build_circuit([angle_func], entanglement=None)
    
    # Test kernel computation
    X_test = np.random.rand(5, 10)
    y_test = np.array([0, 0, 1, 1, 0])
    
    kernel_computer = QuantumKernel()
    K = kernel_computer.compute_kernel_matrix(circuit, X_test)
    
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel matrix:\n{K}")
    
    alignment = kernel_computer.kernel_target_alignment(K, y_test)
    print(f"Kernel-target alignment: {alignment:.3f}")