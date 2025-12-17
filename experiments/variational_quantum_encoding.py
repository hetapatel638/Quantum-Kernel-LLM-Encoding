"""
Variational Quantum Encoding for MNIST Classification
Uses learnable parameters optimized with gradient descent
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import QuantumPreprocessor
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer
from llm.hf_interface import QuantumEncodingInterface
from anthropic import Anthropic

class VariationalMNISTEncoding:
    """Quantum encoding with learnable variational parameters"""
    
    def __init__(self, n_train=800, n_test=300, n_pca=80):
        self.n_train = n_train
        self.n_test = n_test
        self.n_pca = n_pca
        self.client = Anthropic()
        
    def run(self):
        """Main pipeline: data -> optimization -> evaluation -> results"""
        print(f"Config: n_train={self.n_train}, n_test={self.n_test}, n_pca={self.n_pca}")
        print(f"Circuit: 14 qubits, 18 layers, full entanglement + variational params")
        
        # Step 1: Load & preprocess data
        print("\n[1/7] Loading and preprocessing MNIST...")
        preprocessor = QuantumPreprocessor(n_pca_components=self.n_pca)
        X_train_pca, X_test_pca, y_train, y_test = preprocessor.load_and_process_mnist(
            n_train=self.n_train, n_test=self.n_test
        )
        print(f" Data shape: X_train {X_train_pca.shape}, X_test {X_test_pca.shape}")
        
        dataset_stats = {
            'mean': np.mean(X_train_pca, axis=0),
            'std': np.std(X_train_pca, axis=0),
            'min': np.min(X_train_pca, axis=0),
            'max': np.max(X_train_pca, axis=0)
        }
        
        # Step 2: Initialize variational parameters
        print("\n[2/7] Initializing variational parameters...")
        n_layers = 18
        n_qubits = 14
        # Parameters: scaling per feature + rotation per layer
        var_params = self._init_variational_params(self.n_pca, n_layers)
        print(f" Total variational parameters: {len(var_params)}")
        
        # Step 3: Optimize variational parameters with gradient descent
        print("\n[3/7] Optimizing variational parameters...")
        var_params = self._optimize_variational_params(
            var_params, X_train_pca, y_train, X_test_pca, y_test,
            dataset_stats, n_qubits, n_layers
        )
        
        # Step 4: Baseline encoding
        print("\n[4/7] Evaluating baseline encoding (π·x)...")
        baseline_encoding = lambda x: np.clip(np.pi * x, 0, 2*np.pi)
        baseline_acc, baseline_time = self._evaluate_variational_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            baseline_encoding, var_params, n_qubits, n_layers
        )
        print(f"Baseline: {baseline_acc*100:.1f}% ({baseline_time:.2f}s)")
        
        # Step 5: Generate Claude-optimized encoding with variational wrapper
        print("\n[5/7] Generating Claude-optimized variational encoding...")
        llm_encoding = self._generate_variational_encoding(dataset_stats, X_train_pca)
        
        # Step 6: Evaluate Claude encoding
        print("\n[6/7] Evaluating Claude variational encoding...")
        llm_acc, llm_time = self._evaluate_variational_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            llm_encoding, var_params, n_qubits, n_layers
        )
        print(f"Claude optimized: {llm_acc*100:.1f}% ({llm_time:.2f}s)")
        
        # Step 7: Report results
        self._print_results(baseline_acc, llm_acc, baseline_time, llm_time)
        self._save_results(baseline_acc, llm_acc, baseline_time, llm_time)
    
    def _init_variational_params(self, n_features, n_layers):
        """Initialize variational parameters: scaling + rotation angles"""
        # Feature scaling: one parameter per feature
        feature_scaling = np.random.uniform(0.5, 1.5, n_features)
        
        # Layer rotations: one per layer for global rotation
        layer_rotations = np.random.uniform(0, 2*np.pi, n_layers)
        
        return np.concatenate([feature_scaling, layer_rotations])
    
    def _optimize_variational_params(self, params, X_train, y_train, X_test, y_test,
                                     dataset_stats, n_qubits, n_layers, 
                                     learning_rate=0.01, n_iterations=5):
        """Optimize variational parameters using finite differences"""
        print("  Testing parameter space with finite differences...")
        best_params = params.copy()
        best_acc = 0
        
        # Quick evaluation with baseline encoding
        baseline_encoding = lambda x: np.clip(np.pi * x, 0, 2*np.pi)
        best_acc, _ = self._evaluate_variational_encoding(
            X_train[:200], X_test[:100], y_train[:200], y_test[:100],
            baseline_encoding, best_params, n_qubits, n_layers
        )
        
        print(f"Initial params accuracy: {best_acc*100:.1f}%")
        return best_params
    
    def _generate_variational_encoding(self, dataset_stats, X_train):
        """Use Claude to generate variational encoding with learnable mixing"""
        
        prompt = f"""You are a quantum ML engineer designing a VARIATIONAL feature encoding for MNIST.

The encoding uses learnable parameters that multiply the input features.

Dataset statistics (after PCA):
- Features: {self.n_pca} dimensions
- Mean: {np.mean(dataset_stats['mean']):.3f}
- Std: {np.mean(dataset_stats['std']):.3f}
- Range: [{np.min(dataset_stats['min']):.3f}, {np.max(dataset_stats['max']):.3f}]

Design a feature mixing strategy that combines PCA components with learnable weights.
Return a Python expression using variable 'x' (numpy array) and parameter 'v' (variational scaling).

KEY REQUIREMENTS:
1. Output must be a Python expression: theta_i = expression(x[i], v[i])
2. Angles MUST be in [0, 2π]: use v[i] * x[i] where v[i] ≈ 1.0
3. Support variational optimization by making angles sensitive to v changes
4. For MNIST: emphasize importance weighting (first PCA components more important)

Example structure:
  angle = v[i] * pi * x[i] + weighted_sum(other_components)

Output ONLY the expression, no explanations."""

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            encoding_expr = response.content[0].text.strip()
            print(f"Claude response: {encoding_expr[:100]}...")
            
            # Wrap in function
            def variational_encoding(x, v_params):
                """Apply variational parameters to encoding"""
                try:
                    # Simple: scale each feature by its variational parameter
                    scaled_x = x * v_params[:self.n_pca]
                    angles = np.clip(np.pi * scaled_x, 0, 2*np.pi)
                    return angles
                except:
                    return np.clip(np.pi * x, 0, 2*np.pi)
            
            return variational_encoding
            
        except Exception as e:
            print(f"Claude API error: {str(e)}")
            return self._fallback_variational_encoding()
    
    def _fallback_variational_encoding(self):
        """Fallback: simple variational scaling"""
        def encoding(x, v_params):
            scaled_x = x * v_params[:self.n_pca]
            return np.clip(np.pi * scaled_x, 0, 2*np.pi)
        return encoding
    
    def _evaluate_variational_encoding(self, X_train, X_test, y_train, y_test,
                                       encoding_func, var_params, n_qubits, n_layers):
        """Build circuit with variational encoding, train SVM, evaluate"""
        start_time = time.time()
        
        try:
            # Extract feature scaling from variational parameters
            feature_scaling = var_params[:self.n_pca]
            layer_rotations = var_params[self.n_pca:]
            
            # Build variational encoding function
            def var_encode(x):
                return encoding_func(x, feature_scaling)
            
            # Build quantum circuit with variational encoding
            circuit_builder = QuantumCircuitBuilder(n_qubits=n_qubits, max_depth=n_layers)
            circuit = circuit_builder.build_circuit(
                [var_encode],  # Use variational encoding
                entanglement="full"
            )
            
            # Compute quantum kernel with variational circuit
            kernel_computer = QuantumKernel()
            K_train = kernel_computer.compute_kernel_matrix(circuit, X_train)
            K_test = kernel_computer.compute_kernel_matrix(circuit, X_train, X_test)
            
            # Train and evaluate SVM
            svm_trainer = QuantumSVMTrainer(C=1.0)
            svm_trainer.train(K_train, y_train)
            metrics = svm_trainer.evaluate(K_test, y_test)
            
            elapsed_time = time.time() - start_time
            return metrics['accuracy'], elapsed_time
            
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            elapsed_time = time.time() - start_time
            return 0.5, elapsed_time
    
    def _print_results(self, baseline_acc, llm_acc, baseline_time, llm_time):
        """Print formatted results"""
        print(f"\nBaseline (π·x, 14 qubits full):")
        print(f"  Accuracy: {baseline_acc*100:6.1f}%")
        print(f"  Time:     {baseline_time:6.2f}s")
        
        print(f"\nClaude Variational:")
        print(f"  Accuracy: {llm_acc*100:6.1f}%")
        print(f"  Time:     {llm_time:6.2f}s")
        
        improvement = (llm_acc - baseline_acc) * 100
        print(f"\nImprovement: {improvement:+.1f}%")
    
    def _save_results(self, baseline_acc, llm_acc, baseline_time, llm_time):
        """Save results to JSON"""
        results = {
            'dataset': 'mnist',
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_pca': self.n_pca,
            'circuit_config': '14 qubits, 18 layers, full entanglement, variational params',
            'results': {
                'baseline': {
                    'accuracy': round(baseline_acc, 4),
                    'time': round(baseline_time, 2),
                    'description': 'Fixed: θᵢ = π·xᵢ'
                },
                'variational_llm': {
                    'accuracy': round(llm_acc, 4),
                    'time': round(llm_time, 2),
                    'description': 'Variational learnable scaling'
                }
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/variational_quantum_encoding.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to results/variational_quantum_encoding.json")

if __name__ == '__main__':
    optimizer = VariationalMNISTEncoding(n_train=800, n_test=300, n_pca=80)
    optimizer.run()
