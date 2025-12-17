import numpy as np
from typing import Callable


def manual_baseline_kernel(X, Y=None):
    """
    Compute baseline quantum kernel using simple rotation encoding: θᵢ = π·xᵢ
    
    Args:
        X: First dataset (n_samples, n_features)
        Y: Second dataset (optional, defaults to X)
        
    Returns:
        Kernel matrix (n_samples_X, n_samples_Y)
    """
    from quantum.circuit import build_quantum_circuit
    from quantum.kernel import QuantumKernel
    
    n_features = X.shape[1]
    
    # Create baseline encoding function
    def baseline_encoding(x):
        return [np.pi * x[i] for i in range(len(x))]
    
    # Build circuit
    circuit = build_quantum_circuit(n_features, baseline_encoding)
    
    # Compute kernel matrix
    if Y is None:
        Y = X
    
    return QuantumKernel.compute_kernel_matrix(circuit, X, Y)


class ManualBaseline:
    """Simple rotation encoding baseline"""
    
    @staticmethod
    def simple_rotation(n_features: int) -> Callable:
        """
        θᵢ = π · xᵢ
        
        Args:
            n_features: Number of PCA features
            
        Returns:
            Function that takes x and returns list of angles
        """
        def encoding_function(x: np.ndarray) -> list:
            """Simple rotation encoding"""
            return [np.pi * x[i] for i in range(min(len(x), n_features))]
        
        return encoding_function
    
    @staticmethod
    def get_description() -> dict:
        """Get baseline description for reporting"""
        return {
            "name": "Manual Simple Rotation",
            "formula": "θᵢ = π · xᵢ",
            "template_family": "linear",
            "complexity": "minimal"
        }


# Test
if __name__ == "__main__":
    baseline = ManualBaseline()
    encode = baseline.simple_rotation(n_features=10)
    
    # Test on dummy data
    x_test = np.random.rand(10)
    angles = encode(x_test)
    
    print(f"Input: {x_test[:3]}")
    print(f"Angles: {angles[:3]}")
    print(f"Description: {baseline.get_description()}")