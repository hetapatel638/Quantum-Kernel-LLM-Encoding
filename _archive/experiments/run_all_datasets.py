#!/usr/bin/env python3
"""
Run Multi-Layer Quantum Encoding Experiments on All Datasets
MNIST, Fashion-MNIST, and CIFAR-10
"""

import numpy as np
import json
import time
from pathlib import Path
import sys

# Add parent directory to path (adjust based on your structure)
# sys.path.append('/path/to/your/Innovation/directory')

from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from llm.hf_interface import LLMInterface
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer
from config import CONFIG


class MultiDatasetExperiment:
    """Run experiments across multiple datasets"""
    
    def __init__(self, n_train=500, n_test=200, n_pca=80):
        self.n_train = n_train
        self.n_test = n_test
        self.n_pca = n_pca
        self.all_results = {}
    
    def run_single_dataset(self, dataset_name):
        """Run experiment on a single dataset"""
        
        results = {
            'dataset': dataset_name,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_pca': self.n_pca
        }
        
        try:
            # Step 1: Load and preprocess
            print(f"Step 1: Loading {dataset_name}...")
            loader = DatasetLoader()
            X_train, X_test, y_train, y_test = loader.load_dataset(
                dataset_name, self.n_train, self.n_test
            )
            
            print(f"Step 2: Preprocessing (PCA {X_train.shape[1]} → {self.n_pca})...")
            preprocessor = QuantumPreprocessor(n_components=self.n_pca)
            X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
            dataset_stats = preprocessor.get_stats(X_train_pca)
            
            results['original_dim'] = X_train.shape[1]
            results['pca_variance_explained'] = float(np.sum(preprocessor.pca.explained_variance_ratio_))
            
            # Step 2: Baseline
            print(f"\nStep 3: Computing BASELINE (θᵢ = π·xᵢ)...")
            baseline_acc, baseline_time = self._evaluate_encoding(
                X_train_pca, X_test_pca, y_train, y_test,
                encoding_name="BASELINE",
                encoding_func=lambda x: np.clip(np.pi * x, 0, 2*np.pi),
                dataset_name=dataset_name
            )
            
            results['baseline'] = {
                'accuracy': float(baseline_acc),
                'time': float(baseline_time),
                'description': 'Simple linear: θᵢ = π·xᵢ'
            }
            
            # Step 3: Generate LLM encoding
            print(f"\nStep 4: Generating LLM-OPTIMIZED encoding with Claude AI...")
            llm_func, llm_desc = self._generate_optimized_encoding(
                dataset_stats, X_train_pca, dataset_name
            )
            
            # Step 4: Evaluate LLM
            print(f"\nStep 5: Computing LLM-GENERATED encoding accuracy...")
            llm_acc, llm_time = self._evaluate_encoding(
                X_train_pca, X_test_pca, y_train, y_test,
                encoding_name="LLM-GENERATED",
                encoding_func=llm_func,
                dataset_name=dataset_name
            )
            
            results['llm_generated'] = {
                'accuracy': float(llm_acc),
                'time': float(llm_time),
                'description': llm_desc
            }
            
            # Calculate improvement
            improvement = llm_acc - baseline_acc
            improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
            
            results['improvement'] = {
                'absolute': float(improvement),
                'relative_percent': float(improvement_pct)
            }
            
            # Step 5: Report
            self._print_results(dataset_name, baseline_acc, llm_acc, 
                              baseline_time, llm_time, improvement_pct)
            
            results['status'] = 'success'
            
        except Exception as e:
            print(f"\nError running {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _generate_optimized_encoding(self, dataset_stats, X_train, dataset_name):
        """Generate dataset-specific optimized encoding"""
        
        # Dataset-specific characteristics
        dataset_info = {
            'mnist': {
                'description': 'Handwritten digits (0-9)',
                'properties': 'stroke patterns, local correlations, edge detection',
                'challenge': 'Similar stroke patterns across digits (e.g., 3 vs 8)'
            },
            'fashion_mnist': {
                'description': 'Fashion items (10 categories)',
                'properties': 'texture patterns, global structure, shape boundaries',
                'challenge': 'High intra-class variation (different styles of same item)'
            },
            'cifar10': {
                'description': 'Natural images (10 object categories)',
                'properties': 'color channels, multi-scale features, spatial structure',
                'challenge': 'Complex backgrounds, occlusion, scale variation'
            }
        }
        
        info = dataset_info.get(dataset_name, dataset_info['mnist'])
        
        prompt = f"""You are a quantum machine learning engineer. Design a quantum feature encoding for {info['description']} classification using {self.n_pca} PCA components.

CRITICAL: Your encoding MUST generate exactly {self.n_pca} angles (one per PCA component).

Dataset: {dataset_name.upper()}
Description: {info['description']}
Key Properties: {info['properties']}
Challenge: {info['challenge']}

PCA Structure:
- Original dimension: {dataset_stats['n_features']} → {self.n_pca} PCA components
- Variance explained: {np.sum(dataset_stats.get('variance_per_feature', [])):.2%}
- First few components capture most variance
- Later components capture fine details and noise

Baseline Encoding: θᵢ = π·xᵢ (simple linear)

Your Task: Create a BETTER encoding that exploits {self.n_pca} dimensions intelligently for {dataset_name}.

Smart Strategies for {dataset_name}:
1. AMPLITUDE SCALING by importance:
   - Early PCA components (higher variance) → higher amplitude weights
   - Later components (lower variance) → lower amplitude weights

2. NONLINEAR ENHANCEMENT:
   - Power scaling to enhance separability
   - Example: θᵢ = π·xᵢ^0.8

3. FEATURE CORRELATION:
   - Adjacent PCA components often correlated → use neighbor info
   - Example: θᵢ = π·xᵢ + 0.1·π·(xᵢ₋₁ + xᵢ₊₁)/2

4. PHASE STRUCTURING:
   - Add position-based phase to distinguish qubits
   - Example: θᵢ = π·xᵢ + 0.2·π·(i/{self.n_pca})

QUANTUM CIRCUIT CONTEXT:
Your angles θᵢ feed into a multi-layer quantum circuit:
- {self.n_pca//8} qubits (10 for practical simulation)
- Multi-layer: RX(θᵢ) → RY(shifted) → CNOT → RZ(mixed) → RX(scaled) → CNOT
- Data re-uploading through 4 rotation axes
- Output: quantum state |ψ(x)⟩ for kernel computation

CONSTRAINTS:
1. Generate EXACTLY {self.n_pca} angles
2. All angles MUST be in [0, 2π]
3. Return Python list comprehension using len(x)={self.n_pca}
4. Use only: numpy (np), x (input), range, len, i (loop index)
5. NO imports, NO external functions

GOAL: Beat baseline by designing angles that exploit {dataset_name}'s structure.

Return JSON with VALID Python code:
{{
    "function": "[np.clip(...) for i in range(len(x))]",
    "strategy": "Brief strategy name for {dataset_name}",
    "reasoning": "Why this exploits {dataset_name} properties"
}}"""
        
        print(f"\nQuerying Claude API for {dataset_name}-specific encoding...")
        llm = LLMInterface()
        
        try:
            response = llm.generate(prompt, temperature=0.95, max_new_tokens=1024)
            parsed = llm.parse_json_response(response)
            
            if parsed is None:
                print(f"Claude parsing failed for {dataset_name}, using fallback...")
                return self._fallback_encoding(dataset_name), f"Fallback encoding ({dataset_name})"
            
            func_str = parsed.get('function', '')
            strategy = parsed.get('strategy', 'Unknown')
            reasoning = parsed.get('reasoning', '')
            
            print(f"Claude Generated Strategy: {strategy}")
            print(f"Reasoning: {reasoning[:100]}...")
            
            # Test the function
            try:
                test_x = np.random.rand(self.n_pca)
                namespace = {'np': np, 'numpy': np, 'range': range, 'len': len, 'x': test_x}
                test_angles = eval(func_str, {"__builtins__": {}}, namespace)
                
                if isinstance(test_angles, list):
                    test_angles = np.array(test_angles)
                
                if len(test_angles) != self.n_pca:
                    print(f"Wrong number of angles: {len(test_angles)} != {self.n_pca}")
                    return self._fallback_encoding(dataset_name), f"Fallback (wrong size)"
                
                if np.all(test_angles >= 0) and np.all(test_angles <= 2*np.pi):
                    print(f"Function is valid (angles in [0, 2π])")
                    
                    func_string = func_str
                    def angle_func(x):
                        namespace = {'np': np, 'numpy': np, 'range': range, 'len': len, 'x': x}
                        return eval(func_string, {"__builtins__": {}}, namespace)
                    
                    return angle_func, f"Claude Optimized ({dataset_name}): {strategy}"
                else:
                    print(f"Angles out of range")
                    return self._fallback_encoding(dataset_name), f"Fallback (out of range)"
            
            except Exception as e:
                print(f"Function execution failed: {e}")
                return self._fallback_encoding(dataset_name), f"Fallback (exec error)"
        
        except Exception as e:
            print(f"API error: {e}")
            return self._fallback_encoding(dataset_name), f"Fallback (API error)"
    
    def _fallback_encoding(self, dataset_name):
        """Dataset-specific fallback encoding"""
        def encoding(x):
            n = len(x)
            
            # Base weights by dataset
            if dataset_name == 'cifar10':
                # CIFAR-10: emphasize early components (color/structure)
                weights = np.array([1.0 if i < n//4 else 0.7 if i < n//2 else 0.4 for i in range(n)])
            elif dataset_name == 'fashion_mnist':
                # Fashion-MNIST: balanced weights (texture + shape)
                weights = np.array([0.9 if i < n//3 else 0.7 if i < 2*n//3 else 0.5 for i in range(n)])
            else:  # MNIST
                # MNIST: strong emphasis on early components (edges/strokes)
                weights = np.array([0.9 if i < n//3 else 0.6 if i < 2*n//3 else 0.3 for i in range(n)])
            
            # Base encoding
            angles = np.pi * x * weights
            
            # Add neighbor influence
            for i in range(n):
                neighbor_influence = 0.15 * (x[(i-1) % n] + x[(i+1) % n]) / 2
                angles[i] += neighbor_influence
            
            # Nonlinear scaling
            angles = np.pi * (np.clip(angles / np.pi, 0, 1) ** 0.8)
            
            return np.clip(angles, 0, 2*np.pi)
        
        return encoding
    
    def _evaluate_encoding(self, X_train, X_test, y_train, y_test, 
                          encoding_name, encoding_func, dataset_name):
        """Evaluate an encoding function"""
        start = time.time()
        
        try:
            # Build circuit
            circuit_builder = QuantumCircuitBuilder(n_qubits=10)  # Fixed 10 qubits
            circuit = circuit_builder.build_circuit([encoding_func], entanglement="linear")
            
            # Compute kernels
            print(f"  Computing quantum kernels for {dataset_name}...")
            kernel_computer = QuantumKernel()
            K_train = kernel_computer.compute_kernel_matrix(circuit, X_train)
            K_test = kernel_computer.compute_kernel_matrix(circuit, X_train, X_test)
            
            # Train SVM
            print(f"  Training SVM...")
            svm_trainer = QuantumSVMTrainer(C=1.0)
            svm_trainer.train(K_train, y_train)
            
            # Evaluate
            metrics = svm_trainer.evaluate(K_test, y_test)
            accuracy = metrics['accuracy']
            
            elapsed = time.time() - start
            
            print(f"{encoding_name} ({dataset_name}): {accuracy:.4f} ({accuracy*100:.1f}%)")
            
            return accuracy, elapsed
        
        except Exception as e:
            print(f"Error evaluating {encoding_name} on {dataset_name}: {e}")
            return 0.0, 0.0
    
    def _print_results(self, dataset_name, baseline_acc, llm_acc, 
                      baseline_time, llm_time, improvement_pct):
        print(f"RESULTS: {dataset_name.upper()} - BASELINE vs LLM-GENERATED")
        print(f"Baseline (θᵢ=π·xᵢ):        {baseline_acc:.4f} ({baseline_acc*100:.2f}%) in {baseline_time:.1f}s")
        print(f"LLM-Generated (Claude):     {llm_acc:.4f} ({llm_acc*100:.2f}%) in {llm_time:.1f}s")
        print(f"Improvement: {(llm_acc-baseline_acc):+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement_pct >= 3:
            print(f"LLM BEATS BASELINE by 3%+ (SIGNIFICANT)")
        elif improvement_pct >= 1:
            print(f"LLM improves baseline")
        elif improvement_pct >= 0:
            print(f"LLM matches baseline")
        else:
            print(f"LLM underperforms")
    
    def run_all_datasets(self):
        datasets = ['mnist', 'fashion_mnist', 'cifar10']

        print("MULTI-DATASET QUANTUM ENCODING EXPERIMENT")
        print("Multi-Layer Circuit + LLM Optimization")
        print(f"Config: n_train={self.n_train}, n_test={self.n_test}, n_pca={self.n_pca}")

        
        for dataset in datasets:
            results = self.run_single_dataset(dataset)
            self.all_results[dataset] = results
            
            # Save intermediate results
            self._save_results()
        
        # Final summary
        self._print_summary()
    
    def _save_results(self):
        """Save all results to JSON"""
        output_file = Path("results/multi_dataset_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "experiment": "multi_dataset_quantum_encoding",
            "architecture": "multi_layer_circuit",
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_pca": self.n_pca,
            "results": self.all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def _print_summary(self):
        print("FINAL SUMMARY: ALL DATASETS")
        print(f"\n{'Dataset':<20} {'Baseline':<12} {'LLM':<12} {'Improvement':<15} {'Status'}")
        
        for dataset_name, results in self.all_results.items():
            if results.get('status') == 'success':
                baseline = results['baseline']['accuracy']
                llm = results['llm_generated']['accuracy']
                imp = results['improvement']['relative_percent']
                
                status = "Better" if imp > 0 else "Same" if imp == 0 else "Worse"
                
                print(f"{dataset_name:<20} {baseline:<12.4f} {llm:<12.4f} {imp:>+6.2f}%         {status}")
            else:
                print(f"{dataset_name:<20} {'FAILED':<12} {'-':<12} {'-':<15} ✗")
        


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multi-dataset quantum encoding experiments")
    parser.add_argument("--n_train", type=int, default=500, help="Training samples per dataset")
    parser.add_argument("--n_test", type=int, default=200, help="Test samples per dataset")
    parser.add_argument("--n_pca", type=int, default=80, help="PCA dimensions")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=['all', 'mnist', 'fashion_mnist', 'cifar10'],
                       help="Which dataset(s) to run")
    
    args = parser.parse_args()
    
    experiment = MultiDatasetExperiment(
        n_train=args.n_train,
        n_test=args.n_test,
        n_pca=args.n_pca
    )
    
    if args.dataset == 'all':
        experiment.run_all_datasets()
    else:
        results = experiment.run_single_dataset(args.dataset)
        experiment.all_results[args.dataset] = results
        experiment._save_results()
        print("\n Single dataset experiment complete!")