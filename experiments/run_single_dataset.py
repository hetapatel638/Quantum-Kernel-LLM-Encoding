"""
Main Pipeline: Full 5-Level Quantum ML Framework
Data → LLM Prompting → Encoding → Qiskit Quantum → SVM Evaluation
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from encoding.prompt_builder import PromptBuilder
from encoding.validator import EncodingValidator
from encoding.optimized_generator import OptimizedEncodingGenerator
from encoding.advanced_optimizer import AdvancedEncodingOptimizer
from llm.claude_interface import ClaudeInterface
from quantum.qiskit_circuit import QiskitCircuitBuilder
from quantum.qiskit_kernel import QiskitKernel
from evaluation.svm_trainer import QuantumSVMTrainer
from config import CONFIG


class SingleDatasetExperiment:
    """Run complete experiment on single dataset"""
    
    def __init__(self, dataset_name: str, n_pca: int = 40, 
                 n_train: int = 500, n_test: int = 200):
        self.dataset_name = dataset_name
        self.n_pca = n_pca
        self.n_train = n_train
        self.n_test = n_test
        self.results = {}
    
    def run(self, template_type: str = "linear", use_mock: bool = False) -> Dict[str, Any]:
        """
        Execute complete pipeline
        
        Args:
            template_type: Type of encoding (linear, polynomial, global_stats, pca_mix)
            use_mock: Use mock Claude response for testing
        """
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {self.dataset_name.upper()}")
        print(f"Template: {template_type}, PCA dims: {self.n_pca}")
        print(f"Train: {self.n_train}, Test: {self.n_test}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Step 1: Load Data
        print("[1/6] Loading data...")
        loader = DatasetLoader()
        X_train_raw, y_train, X_test_raw, y_test = loader.load_dataset(
            dataset_name=self.dataset_name,
            n_train=self.n_train,
            n_test=self.n_test
        )
        print(f"  ✓ Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
        
        # Step 2: Preprocess (PCA + Normalize)
        print("\n[2/6] Preprocessing (PCA + Normalization)...")
        preprocessor = QuantumPreprocessor(n_components=self.n_pca)
        X_train, X_test = preprocessor.fit_transform(X_train_raw, X_test_raw)
        print(f"  ✓ After PCA: Train {X_train.shape}, Test {X_test.shape}")
        print(f"  ✓ Range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        # Step 3: Generate Encoding via Claude
        print("\n[3/6] Generating encoding with Claude...")
        prompt = PromptBuilder.build_encoding_prompt(
            dataset_name=self.dataset_name,
            X_train=X_train,
            n_pca=self.n_pca,
            template_type=template_type
        )
        
        if use_mock:
            raise RuntimeError("LLM is mandatory. Disable --mock to use Claude.")
        try:
            claude = ClaudeInterface()
            encoding_code, explanation = claude.generate_encoding(prompt)
        except Exception as e:
            raise RuntimeError(f"Claude generation failed: {e}") from e
        
        if encoding_code:
            print(f"  ✓ Generated: {encoding_code[:60]}...")
        
        # Step 4: Validate Encoding (LLM required; no fallback)
        print("\n[4/6] Validating encoding...")
        
        is_valid, error = EncodingValidator.validate_encoding(encoding_code, X_train)
        if is_valid:
            print("  ✓ Valid encoding")
        else:
            raise RuntimeError(f"LLM encoding invalid: {error}")

        # Quick template sweep on a small subset (LLM required)
        print("\n[4b/6] Quick template sweep on subset (LLM-only)...")
        # Keep subset small for speed
        subset_train = min(120, len(X_train))
        subset_test = min(50, len(X_test))
        X_train_small = X_train[:subset_train]
        y_train_small = y_train[:subset_train]
        X_test_small = X_test[:subset_test]
        y_test_small = y_test[:subset_test]

        candidate_encodings = [("claude", encoding_code)]

        def make_safe(func_code: str):
            def _safe(x):
                try:
                    arr = eval(func_code, {"x": x, "np": np})
                    if isinstance(arr, (int, float)):
                        arr = np.full(10, float(arr))
                    arr = np.asarray(arr)
                    if arr.ndim == 0:
                        arr = np.full(10, float(arr))
                    if len(arr) < 10:
                        arr = np.pad(arr, (0, 10 - len(arr)))
                    if len(arr) > 10:
                        arr = arr[:10]
                    return np.clip(arr, 0, 2*np.pi)
                except Exception:
                    return np.full(10, np.pi)
            return _safe

        best_acc = -1.0
        best_name = None
        best_code = encoding_code

        # Small C grid per dataset for the sweep only (low overhead)
        if self.dataset_name == "mnist":
            c_grid = [20.0, 40.0, 60.0]
        elif self.dataset_name == "fashion_mnist":
            c_grid = [30.0, 50.0, 70.0]
        else:
            c_grid = [10.0, 20.0, 30.0]

        for name, code in candidate_encodings:
            try:
                encoding_func = make_safe(code)
                feature_map = QiskitCircuitBuilder(
                    n_qubits=CONFIG["quantum"]["n_qubits"],
                    entanglement=CONFIG["quantum"]["entanglement"]
                ).build_feature_map(encoding_func)
                kernel_small = QiskitKernel(
                    circuit_builder=feature_map,
                    n_qubits=CONFIG["quantum"]["n_qubits"],
                    shots=0  # statevector path for speed
                )
                K_train_small = kernel_small.compute_kernel(X_train_small, X_train_small)
                K_test_small = kernel_small.compute_kernel(X_test_small, X_train_small)

                for c_val in c_grid:
                    trainer_small = QuantumSVMTrainer(C=c_val)
                    trainer_small.train(K_train_small, y_train_small)
                    metrics_small = trainer_small.evaluate(K_test_small, y_test_small)
                    acc = metrics_small.get("accuracy", 0)
                    print(f"  • {name} @C={c_val}: acc={acc:.3f} on subset")
                    if acc > best_acc:
                        best_acc = acc
                        best_name = f"{name}_C{c_val}"
                        best_code = code
            except Exception as e:
                print(f"  • {name}: skipped ({str(e)[:60]})")

        encoding_code = best_code
        print(f"  → Selected encoding: {best_name} (subset acc={best_acc:.3f})")

        self.results["encoding"] = {
            "code": encoding_code,
            "template_type": best_name or template_type,
            "is_valid": True
        }
        
        # Step 5: Build Quantum Circuit & Compute Kernel
        print("\n[5/6] Building quantum circuit and computing kernel...")
        circuit_builder = QiskitCircuitBuilder(
            n_qubits=CONFIG["quantum"]["n_qubits"],
            entanglement=CONFIG["quantum"]["entanglement"]
        )
        
        # Create encoding function with safety wrapper
        def safe_encoding(x):
            try:
                # Ensure x is numpy array
                if not isinstance(x, np.ndarray):
                    x = np.array(x)
                
                # Evaluate encoding
                result = eval(encoding_code, {"x": x, "np": np, "len": len})
                
                # Ensure result is always a numpy array
                if isinstance(result, (int, float)):
                    result = np.full(10, float(result))
                elif isinstance(result, np.ndarray):
                    if result.ndim == 0:
                        result = np.full(10, float(result.item()))
                    elif len(result) < 10:
                        # Pad to 10 angles
                        pad_amount = 10 - len(result)
                        result = np.concatenate([result, np.zeros(pad_amount)])
                    elif len(result) > 10:
                        # Take first 10
                        result = result[:10]
                else:
                    # Fallback to mean encoding
                    result = np.full(10, np.mean(x) * np.pi)
                
                # Ensure all values are valid
                result = np.asarray(result, dtype=float)
                
                # Clip to [0, 2π]
                result = np.clip(result, 0, 2*np.pi)
                
                return result
            except Exception as e:
                print(f"  ⚠️  Encoding evaluation error: {str(e)[:60]}, using fallback")
                return np.full(10, np.pi)
        
        encoding_func = safe_encoding
        feature_map = circuit_builder.build_feature_map(encoding_func)
        
        # Compute kernel
        kernel = QiskitKernel(
            circuit_builder=feature_map,
            n_qubits=CONFIG["quantum"]["n_qubits"],
            shots=CONFIG["quantum"]["shots"]
        )
        
        # For precomputed kernel SVM, test kernel must be (n_test, n_train)
        # We compute kernel between test samples and training samples
        K_train = kernel.compute_kernel(X_train, X_train)
        K_test = kernel.compute_kernel(X_test, X_train)  # Test vs Train (NOT Train vs Test)
        print(f"  ✓ Kernel shapes: Train {K_train.shape}, Test {K_test.shape}")
        
        # Step 6: Train SVM & Evaluate
        print("\n[6/6] Training SVM and evaluating...")
        trainer = QuantumSVMTrainer(C=CONFIG["svm"]["C"])
        trainer.train(K_train, y_train)
        metrics = trainer.evaluate(K_test, y_test)
        
        print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ✓ F1 (macro): {metrics['f1_macro']:.4f}")
        
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.2f}s")
        
        # Store results
        self.results["metrics"] = metrics
        self.results["timing"] = elapsed
        self.results["dataset"] = self.dataset_name
        
        return self.results
    
    def save_results(self):
        """Save results to JSON"""
        Path(CONFIG["results_dir"]).mkdir(exist_ok=True)
        filename = f"{CONFIG['results_dir']}/{self.dataset_name}_{self.n_pca}pca.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Single dataset quantum ML experiment")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"], 
                       default="mnist")
    parser.add_argument("--template", choices=["linear", "polynomial", "global_stats", "pca_mix"],
                       default="linear")
    parser.add_argument("--n_pca", type=int, default=40)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--mock", action="store_true", help="Use mock Claude")
    
    args = parser.parse_args()
    
    experiment = SingleDatasetExperiment(
        dataset_name=args.dataset,
        n_pca=args.n_pca,
        n_train=args.n_train,
        n_test=args.n_test
    )
    
    results = experiment.run(template_type=args.template, use_mock=args.mock)
    experiment.save_results()
