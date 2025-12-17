"""
Qiskit-based trainable quantum kernel for improved accuracy.

This module uses Qiskit ML's ZZFeatureMap and trainable quantum kernels
to achieve better performance than PennyLane baseline encodings.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.state_fidelities import ComputeUncompute
try:
    from qiskit.primitives import Sampler
except ImportError:
    from qiskit.primitives import StatevectorSampler as Sampler


class QiskitTrainableKernel:
    """
    Trainable quantum kernel using Qiskit ML framework.
    
    Based on Tutorial 08: Quantum Kernel Training
    https://qiskit-community.github.io/qiskit-machine-learning/tutorials/08_quantum_kernel_trainer.html
    """
    
    def __init__(self, n_features, n_qubits=None):
        """
        Initialize trainable quantum kernel.
        
        Args:
            n_features: Number of input features (PCA dimensions)
            n_qubits: Number of qubits (defaults to n_features)
        """
        self.n_features = n_features
        self.n_qubits = n_qubits if n_qubits else n_features
        
        # Create trainable rotation layer
        self.training_params = ParameterVector("θ", self.n_qubits)
        self.rotation_layer = self._build_rotation_layer()
        
        # Create ZZFeatureMap for data encoding
        self.zz_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement="linear"
        )
        
        # Compose into full feature map
        self.feature_map = self.rotation_layer.compose(self.zz_map)
        
        # Initialize quantum kernel (will be set after training)
        self.quantum_kernel = None
        self.is_trained = False
        
    def _build_rotation_layer(self):
        """Build trainable rotation layer for each qubit."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Apply RY rotation to each qubit with trainable parameters
        for i in range(self.n_qubits):
            qc.ry(self.training_params[i], i)
            
        return qc
    
    def train(self, X_train, y_train, optimizer=None, maxiter=20):
        """
        Train the quantum kernel on labeled data.
        
        Uses Quantum Kernel Alignment (QKA) to optimize trainable parameters.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            optimizer: Qiskit optimizer (defaults to SPSA)
            maxiter: Maximum optimizer iterations
            
        Returns:
            Training results dictionary
        """
        from qiskit_machine_learning.optimizers import SPSA
        
        # Initialize sampler and fidelity
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        
        # Initialize trainable kernel
        trainable_kernel = TrainableFidelityQuantumKernel(
            feature_map=self.feature_map,
            training_parameters=self.training_params,
            fidelity=fidelity
        )
        
        # Set up optimizer
        if optimizer is None:
            optimizer = SPSA(
                maxiter=maxiter,
                learning_rate=0.05,
                perturbation=0.05
            )
        
        # Initialize kernel trainer with SVC loss (Quantum Kernel Alignment)
        trainer = QuantumKernelTrainer(
            quantum_kernel=trainable_kernel,
            loss="svc_loss",
            optimizer=optimizer,
            initial_point=np.random.uniform(0, 2*np.pi, self.n_qubits)
        )
        
        # Train the kernel
        print(f"Training quantum kernel on {len(X_train)} samples...")
        results = trainer.fit(X_train, y_train)
        
        # Store trained kernel
        self.quantum_kernel = results.quantum_kernel
        self.is_trained = True
        
        print(f"Kernel training completed:")
        print(f"  Optimal value: {results.optimal_value:.4f}")
        print(f"  Optimizer evals: {results.optimizer_evals}")
        
        return {
            "optimal_parameters": results.optimal_parameters,
            "optimal_value": results.optimal_value,
            "optimizer_evals": results.optimizer_evals
        }
    
    def evaluate(self, X_test, X_train=None):
        """
        Evaluate kernel matrix on test data.
        
        Args:
            X_test: Test features
            X_train: Training features (if None, computes test-test kernel)
            
        Returns:
            Kernel matrix (n_test, n_train) or (n_test, n_test)
        """
        if not self.is_trained:
            raise ValueError("Kernel must be trained before evaluation")
        
        if X_train is None:
            # Compute test-test kernel matrix
            return self.quantum_kernel.evaluate(x_vec=X_test)
        else:
            # Compute test-train kernel matrix
            return self.quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)


class QiskitZZKernel:
    """
    Fixed ZZFeatureMap quantum kernel (no training).
    
    Based on Tutorial 03: Quantum Kernel Machine Learning
    Uses standard ZZ entanglement without trainable parameters.
    """
    
    def __init__(self, n_features, reps=2):
        """
        Initialize ZZ feature map kernel.
        
        Args:
            n_features: Number of input features
            reps: Number of ZZ feature map repetitions
        """
        self.n_features = n_features
        self.reps = reps
        
        # Create ZZFeatureMap
        self.feature_map = ZZFeatureMap(
            feature_dimension=n_features,
            reps=reps,
            entanglement="linear"
        )
        
        # Initialize sampler and fidelity
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        
        # Initialize quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity
        )
        
    def evaluate(self, X_test, X_train=None):
        """
        Evaluate kernel matrix.
        
        Args:
            X_test: Test features
            X_train: Training features (if None, computes test-test kernel)
            
        Returns:
            Kernel matrix
        """
        if X_train is None:
            return self.quantum_kernel.evaluate(x_vec=X_test)
        else:
            return self.quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)


def build_qiskit_kernel(n_features, kernel_type="trainable", **kwargs):
    """
    Factory function to build Qiskit quantum kernels.
    
    Args:
        n_features: Number of input features (PCA dimensions)
        kernel_type: "trainable" or "zz"
        **kwargs: Additional arguments for kernel construction
        
    Returns:
        QiskitTrainableKernel or QiskitZZKernel instance
    """
    if kernel_type == "trainable":
        return QiskitTrainableKernel(n_features, **kwargs)
    elif kernel_type == "zz":
        return QiskitZZKernel(n_features, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
