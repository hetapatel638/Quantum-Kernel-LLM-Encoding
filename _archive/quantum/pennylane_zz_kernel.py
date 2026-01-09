"""
ZZ Feature Map implementation in PennyLane with trainable parameters.

This replicates the Qiskit ML ZZFeatureMap approach using PennyLane,
avoiding version conflicts while gaining the benefits of entangled encodings.
"""

import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from scipy.optimize import minimize


def zz_feature_map(x, params, n_qubits, reps=2):
    """
    Implement ZZ Feature Map with trainable rotation layer.
    
    This replicates Qiskit's ZZFeatureMap architecture:
    1. Trainable RY rotations on all qubits
    2. H gates for superposition
    3. Data encoding with RZ(2*x[i])
    4. ZZ entanglement with RZZ(2*x[i]*x[j]) on adjacent pairs
    5. Repeat for 'reps' layers
    
    Args:
        x: Input features (n_qubits,)
        params: Trainable parameters (n_qubits,)
        n_qubits: Number of qubits
        reps: Number of ZZ layers
    """
    # Trainable rotation layer (optimized by kernel alignment)
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    
    # ZZ Feature Map layers
    for rep in range(reps):
        # Hadamard gates
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Data encoding with RZ
        for i in range(min(n_qubits, len(x))):
            qml.RZ(2.0 * x[i], wires=i)
        
        # ZZ entanglement (linear connectivity)
        for i in range(n_qubits - 1):
            if i < len(x) - 1:
                # RZZ gate: exp(-i * theta/2 * Z_i * Z_j)
                qml.IsingZZ(2.0 * x[i] * x[i+1], wires=[i, i+1])


class PennyLaneZZKernel:
    """
    Trainable ZZ quantum kernel using PennyLane.
    
    Based on Qiskit Tutorial 08 architecture but implemented in PennyLane
    to avoid API version conflicts.
    """
    
    def __init__(self, n_features, n_qubits=None, reps=2, device='default.qubit'):
        """
        Initialize ZZ quantum kernel.
        
        Args:
            n_features: Number of input features
            n_qubits: Number of qubits (defaults to n_features)
            reps: Number of ZZ feature map repetitions
            device: PennyLane device
        """
        self.n_features = n_features
        self.n_qubits = n_qubits if n_qubits else n_features
        self.reps = reps
        self.device = device
        
        # Initialize trainable parameters (random)
        self.params = np.random.uniform(0, 2*np.pi, self.n_qubits)
        self.is_trained = False
        
        # Create PennyLane device and quantum node
        self.dev = qml.device(self.device, wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2, params):
            """Quantum kernel circuit: |<φ(x1)|φ(x2)>|^2"""
            # Prepare state |φ(x1)>
            zz_feature_map(x1, params, self.n_qubits, self.reps)
            
            # Apply adjoint of |φ(x2)>
            qml.adjoint(zz_feature_map)(x2, params, self.n_qubits, self.reps)
            
            # Return probability of all zeros (overlap measurement)
            return qml.probs(wires=range(self.n_qubits))
        
        self.circuit = kernel_circuit
    
    def kernel_function(self, x1, x2):
        """
        Compute kernel between two samples.
        
        Returns |<φ(x1)|φ(x2)>|^2
        """
        probs = self.circuit(x1, x2, self.params)
        # Probability of all zeros state = |<0|U†(x2)U(x1)|0>|^2 = |<φ(x1)|φ(x2)>|^2
        return probs[0]
    
    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute kernel matrix between two datasets.
        
        Args:
            X1: First dataset (n_samples1, n_features)
            X2: Second dataset (n_samples2, n_features). If None, uses X1.
            
        Returns:
            Kernel matrix (n_samples1, n_samples2)
        """
        if X2 is None:
            X2 = X1
        
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel_function(X1[i], X2[j])
        
        return K
    
    def kernel_alignment_loss(self, params, X_train, y_train):
        """
        Compute kernel alignment loss for training.
        
        Measures how well the kernel matrix aligns with the ideal kernel
        (perfect separation between classes).
        """
        self.params = params
        K = self.compute_kernel_matrix(X_train)
        
        # Ideal kernel: 1 if same class, 0 if different
        n = len(y_train)
        K_ideal = np.outer(y_train, y_train)
        K_ideal = (K_ideal == np.max(K_ideal)).astype(float)
        
        # Frobenius inner product
        alignment = np.trace(K @ K_ideal)
        
        # Normalize
        K_norm = np.linalg.norm(K, 'fro')
        K_ideal_norm = np.linalg.norm(K_ideal, 'fro')
        
        if K_norm > 0 and K_ideal_norm > 0:
            alignment /= (K_norm * K_ideal_norm)
        
        # Return negative (we want to maximize alignment)
        return -alignment
    
    def train(self, X_train, y_train, maxiter=30, method='Powell'):
        """
        Train kernel parameters using kernel alignment.
        
        Args:
            X_train: Training features
            y_train: Training labels
            maxiter: Maximum optimization iterations
            method: Scipy optimization method (Powell works better than COBYLA)
            
        Returns:
            Optimization result
        """
        print(f"Training ZZ quantum kernel with kernel alignment...")
        print(f"  Initial parameters: {self.params}")
        
        # Use subset for faster training
        n_train_subset = min(len(X_train), 80)
        indices = np.random.choice(len(X_train), n_train_subset, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        print(f"  Using {n_train_subset} samples for kernel training")
        
        # Optimize parameters
        result = minimize(
            fun=lambda p: self.kernel_alignment_loss(p, X_subset, y_subset),
            x0=self.params,
            method=method,
            options={'maxiter': maxiter, 'disp': False}
        )
        
        self.params = result.x
        self.is_trained = True
        
        print(f"  Training completed:")
        print(f"  Final parameters: {self.params}")
        print(f"  Final loss: {result.fun:.6f}")
        print(f"  Iterations: {getattr(result, 'nit', 'N/A')}")
        print(f"  Success: {result.success}")
        
        return result


def test_zz_kernel():
    """Quick test of ZZ kernel on toy dataset."""
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Create toy dataset
    X, y = make_classification(n_samples=20, n_features=4, n_informative=4, 
                                n_redundant=0, random_state=42)
    
    # Split train/test
    X_train, X_test = X[:15], X[15:]
    y_train, y_test = y[:15], y[15:]
    
    # Create kernel
    kernel = PennyLaneZZKernel(n_features=4, reps=2)
    
    # Train kernel
    kernel.train(X_train, y_train, maxiter=10)
    
    # Compute kernel matrices
    K_train = kernel.compute_kernel_matrix(X_train)
    K_test = kernel.compute_kernel_matrix(X_test, X_train)
    
    # Train SVM
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)
    
    # Predict
    y_pred = svm.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest accuracy: {acc:.4f}")


if __name__ == "__main__":
    test_zz_kernel()
