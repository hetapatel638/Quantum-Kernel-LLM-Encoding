import sys
sys.path.insert(0, '/Users/husky95/Desktop/Innovation')

import numpy as np
import json
import time
from pathlib import Path
from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer
import pennylane as qml


class AdvancedMNISTOptimization:
    """Advanced MNIST optimization with multiple strategies"""
    
    def __init__(self, n_train=800, n_test=300, n_pca=80):
        self.n_train = n_train
        self.n_test = n_test
        self.n_pca = n_pca
        self.kernel = QuantumKernel()
        self.results = {}
        
    def run(self):
        """Run comprehensive MNIST optimization"""        
        # Load data
        print(f"\nLoading MNIST: {self.n_train} train, {self.n_test} test")
        loader = DatasetLoader()
        X_train, X_test, y_train, y_test = loader.load_dataset(
            'mnist', self.n_train, self.n_test
        )
        
        print(f"Preprocessing: {X_train.shape[1]} → {self.n_pca} PCA dims")
        preprocessor = QuantumPreprocessor(n_components=self.n_pca)
        X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
        print(f"  Train shape: {X_train_pca.shape}, Test shape: {X_test_pca.shape}")
        
        # Strategy 1: Baseline
        print("STRATEGY 1: BASELINE (θᵢ = π·xᵢ, 10 qubits, linear)")
        baseline_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=lambda x: np.clip(np.pi * x, 0, 2*np.pi),
            name="Baseline",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['baseline'] = baseline_acc
        
        # Strategy 2: Full entanglement
        print("STRATEGY 2: FULL ENTANGLEMENT (10 qubits, all-to-all)")
        full_ent_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=lambda x: np.clip(np.pi * x, 0, 2*np.pi),
            name="Full Entanglement",
            n_qubits=10,
            entanglement="full"
        )
        self.results['full_entanglement'] = full_ent_acc
        
        # Strategy 3: Power scaling
        print("STRATEGY 3: POWER SCALING (θᵢ = π·xᵢ^0.7)")
        power_scaling_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=lambda x: np.clip(np.pi * np.power(np.abs(x), 0.7), 0, 2*np.pi),
            name="Power Scaling (0.7)",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['power_scaling'] = power_scaling_acc
        
        # Strategy 4: Phase modulation
        print("STRATEGY 4: PHASE MODULATION (θᵢ = π·xᵢ + 0.5π·sin(2πxᵢ))")
        phase_mod_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=lambda x: np.clip(
                np.pi * x + 0.5 * np.pi * np.sin(2 * np.pi * x),
                0, 2*np.pi
            ),
            name="Phase Modulation",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['phase_modulation'] = phase_mod_acc
        
        # Strategy 5: Adaptive weighting
        print("STRATEGY 5: ADAPTIVE WEIGHTING (importance-based)")
        
        def adaptive_encoding(x):
            # Weight by variance
            variance = np.std(X_train_pca, axis=0)
            variance_normalized = variance / np.max(variance)
            adaptive_angles = np.pi * x * (0.5 + 1.5 * variance_normalized)
            return np.clip(adaptive_angles, 0, 2*np.pi)
        
        adaptive_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=adaptive_encoding,
            name="Adaptive Weighting",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['adaptive_weighting'] = adaptive_acc
        
        # Strategy 6: Dual-scale encoding
        print("STRATEGY 6: DUAL-SCALE ENCODING (fine + coarse)")
        
        def dual_scale_encoding(x):
            fine_scale = 0.8 * np.pi * x
            coarse = np.zeros_like(x)
            for i in range(len(x)):
                if i == 0:
                    coarse[i] = np.mean(x[:2]) if len(x) > 1 else x[0]
                elif i == len(x) - 1:
                    coarse[i] = np.mean(x[-2:])
                else:
                    coarse[i] = np.mean(x[i-1:i+2])
            coarse_scale = 0.2 * np.pi * coarse
            combined = fine_scale + coarse_scale
            return np.clip(combined, 0, 2*np.pi)
        
        dual_scale_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=dual_scale_encoding,
            name="Dual-Scale",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['dual_scale'] = dual_scale_acc
        
        # Strategy 7: Log scaling
        print("STRATEGY 7: LOG SCALING (θᵢ = π·log(1 + xᵢ))")
        log_scale_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=lambda x: np.clip(np.pi * np.log(1 + x), 0, 2*np.pi),
            name="Log Scaling",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['log_scaling'] = log_scale_acc
        
        # Strategy 8: Combined weighted encoding
        print("STRATEGY 8: COMBINED WEIGHTED (linear + phase)")
        
        def combined_weighted(x):
            variance = np.std(X_train_pca, axis=0)
            variance_normalized = variance / np.max(variance)
            linear_part = 0.6 * np.pi * x
            phase_part = 0.4 * np.pi * np.sin(2 * np.pi * x * variance_normalized)
            combined = linear_part + phase_part
            return np.clip(combined, 0, 2*np.pi)
        
        combined_acc = self._test_encoding(
            X_train_pca, X_test_pca, y_train, y_test,
            encoding_func=combined_weighted,
            name="Combined Weighted",
            n_qubits=10,
            entanglement="linear"
        )
        self.results['combined_weighted'] = combined_acc
        
        # Print summary
        self._print_summary()
        
        # Save results
        self._save_results()
    
    def _test_encoding(self, X_train, X_test, y_train, y_test, encoding_func, 
                       name, n_qubits=10, entanglement="linear"):
        """Test single encoding configuration"""
        print(f"\n  Testing {name}...")
        
        try:
            start = time.time()
            
            # Build circuit with specified qubits
            circuit_builder = QuantumCircuitBuilder(n_qubits=n_qubits, max_depth=12)
            circuit = circuit_builder.build_circuit(
                [encoding_func],
                entanglement=entanglement
            )
            
            # Compute kernel matrices
            print(f"    Computing kernel matrices (train)...")
            K_train = self.kernel.compute_kernel_matrix(circuit, X_train)
            
            print(f"    Computing kernel matrices (test)...")
            K_test = self.kernel.compute_kernel_matrix(circuit, X_train, X_test)
            
            # Train SVM
            print(f"    Training SVM...")
            trainer = QuantumSVMTrainer(C=1.0)
            trainer.train(K_train, y_train)
            
            # Evaluate
            metrics = trainer.evaluate(K_test, y_test)
            accuracy = metrics['accuracy']
            elapsed = time.time() - start
            
            print(f"    ✓ {name}: {accuracy*100:.2f}% ({elapsed:.1f}s)")
            
            return {
                'name': name,
                'accuracy': accuracy,
                'config': f"{n_qubits} qubits, {entanglement} entanglement",
                'time': elapsed
            }
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return {'name': name, 'accuracy': 0, 'error': str(e)}
    
    def _print_summary(self):
        """Print results summary"""
        print("RESULTS SUMMARY")
        # Sort by accuracy
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items() if isinstance(v, dict)],
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        print(f"\n{'Rank':<5} {'Strategy':<30} {'Accuracy':<12} {'Config':<25}")
        
        for rank, (key, result) in enumerate(sorted_results, 1):
            acc_str = f"{result['accuracy']*100:.2f}%"
            config = result.get('config', 'N/A')
            name = result['name']
            print(f"{rank:<5} {name:<30} {acc_str:<12} {config:<25}")
        
        # Best result
        if sorted_results:
            best_name = sorted_results[0][1]['name']
            best_acc = sorted_results[0][1]['accuracy']
            print("\n" + "=" * 70)
            print(f"BEST: {best_name}")
            print(f"Accuracy: {best_acc*100:.2f}%")
            if 'baseline' in self.results:
                baseline = self.results['baseline']['accuracy']
                improvement = (best_acc - baseline) * 100
                print(f"Improvement over baseline: +{improvement:.2f}%")
    
    def _save_results(self):
        """Save results to JSON"""
        output = {
            'config': {
                'n_train': self.n_train,
                'n_test': self.n_test,
                'n_pca': self.n_pca,
            },
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_path = Path('/Users/husky95/Desktop/Innovation/results/advanced_mnist_optimization.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    optimizer = AdvancedMNISTOptimization(
        n_train=800,   # Optimized: balance speed vs data
        n_test=300,    # Optimized test set
        n_pca=80       # Same PCA as before
    )
    optimizer.run()
