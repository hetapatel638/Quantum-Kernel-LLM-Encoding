"""
MNIST classification using PennyLane ZZ Feature Map.

This implements the Qiskit ZZFeatureMap architecture in PennyLane
with trainable parameters for kernel alignment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import time
import json

from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from quantum.pennylane_zz_kernel import PennyLaneZZKernel


def run_zz_experiment(
    dataset_name="mnist",
    n_train=500,
    n_test=200,
    pca_dims=40,
    train_kernel=True,
    kernel_training_iters=20,
    zz_reps=2,
    random_seed=42
):
    """
    Run ZZ quantum kernel experiment.
    
    Args:
        dataset_name: Dataset name
        n_train: Number of training samples
        n_test: Number of test samples
        pca_dims: PCA dimensions
        train_kernel: Whether to train kernel parameters
        kernel_training_iters: Number of training iterations
        zz_reps: Number of ZZ feature map repetitions
        random_seed: Random seed
        
    Returns:
        Results dictionary
    """
    np.random.seed(random_seed)
    
    print("ZZ FEATURE MAP QUANTUM KERNEL EXPERIMENT")
    print()
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"PCA dimensions: {pca_dims}")
    print(f"ZZ repetitions: {zz_reps}")
    print(f"Train kernel: {train_kernel}")
    if train_kernel:
        print(f"Training iterations: {kernel_training_iters}")
    print("=" * 60)
    
    # Load data
    print("\nLoading dataset...")
    loader = DatasetLoader()
    X_train_raw, X_test_raw, y_train, y_test = loader.load_dataset(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test
    )
    
    # Preprocess
    print(f"Applying PCA (n_components={pca_dims})...")
    preprocessor = QuantumPreprocessor(n_components=pca_dims)
    X_train, X_test = preprocessor.fit_transform(X_train_raw, X_test_raw)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Initialize ZZ kernel
    print(f"\nInitializing ZZ quantum kernel...")
    kernel = PennyLaneZZKernel(n_features=pca_dims, reps=zz_reps)
    
    # Train kernel if requested
    training_time = 0.0
    if train_kernel:
        print("\nTraining kernel with kernel alignment...")
        train_start = time.time()
        kernel.train(X_train, y_train, maxiter=kernel_training_iters)
        training_time = time.time() - train_start
        print(f"Kernel training time: {training_time:.2f}s")
    
    # Compute kernel matrices
    print("\nComputing kernel matrices...")
    kernel_start = time.time()
    K_train = kernel.compute_kernel_matrix(X_train)
    K_test = kernel.compute_kernel_matrix(X_test, X_train)
    kernel_time = time.time() - kernel_start
    
    print(f"Kernel computation time: {kernel_time:.2f}s")
    print(f"Train kernel shape: {K_train.shape}")
    print(f"Test kernel shape: {K_test.shape}")
    
    # Train SVM
    print("\nTraining SVM classifier...")
    svm_start = time.time()
    svm = SVC(kernel="precomputed")
    svm.fit(K_train, y_train)
    svm_time = time.time() - svm_start
    
    # Evaluate
    print("Evaluating on test set...")
    y_pred = svm.predict(K_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print("\nRESULTS")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total time: {training_time + kernel_time + svm_time:.2f}s")
    print("=" * 60)
    
    return {
        "dataset": dataset_name,
        "n_train": n_train,
        "n_test": n_test,
        "pca_dims": pca_dims,
        "kernel_type": "zz_pennylane",
        "zz_reps": zz_reps,
        "trained": train_kernel,
        "training_iters": kernel_training_iters if train_kernel else 0,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "kernel_time": float(kernel_time),
        "svm_time": float(svm_time),
        "total_time": float(training_time + kernel_time + svm_time)
    }


def compare_with_baseline(
    dataset_name="mnist",
    n_train=500,
    n_test=200,
    pca_dims=40,
    random_seed=42
):
    """
    Compare ZZ kernel with baseline.
    """
    print("\nCOMPARING ZZ KERNEL WITH BASELINE")
    print()
    
    # Load data once
    loader = DatasetLoader()
    X_train_raw, X_test_raw, y_train, y_test = loader.load_dataset(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test
    )
    
    preprocessor = QuantumPreprocessor(n_components=pca_dims)
    X_train, X_test = preprocessor.fit_transform(X_train_raw, X_test_raw)
    
    results = {}
    
    # Test 1: Baseline
    print("\n### TEST 1: Baseline (θ = π·x) ###")
    from quantum.circuit import build_quantum_circuit
    from quantum.kernel import QuantumKernel
    
    baseline_start = time.time()
    
    # Simple baseline encoding
    def baseline_encoding(x):
        return [np.pi * x[i] for i in range(len(x))]
    
    baseline_circuit = build_quantum_circuit(pca_dims, baseline_encoding)
    K_train_baseline = QuantumKernel.compute_kernel_matrix(baseline_circuit, X_train)
    K_test_baseline = QuantumKernel.compute_kernel_matrix(baseline_circuit, X_test, X_train)
    
    svm_baseline = SVC(kernel="precomputed")
    svm_baseline.fit(K_train_baseline, y_train)
    y_pred_baseline = svm_baseline.predict(K_test_baseline)
    
    baseline_time = time.time() - baseline_start
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average="weighted")
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Baseline Time: {baseline_time:.2f}s")
    
    results["baseline"] = {
        "accuracy": baseline_acc,
        "f1_score": baseline_f1,
        "total_time": baseline_time
    }
    
    # Test 2: ZZ kernel (untrained)
    print("\n### TEST 2: ZZ Kernel (UNTRAINED) ###")
    results["zz_untrained"] = run_zz_experiment(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test,
        pca_dims=pca_dims,
        train_kernel=False,
        random_seed=random_seed
    )
    
    # Test 3: ZZ kernel (trained)
    print("\n### TEST 3: ZZ Kernel (TRAINED) ###")
    results["zz_trained"] = run_zz_experiment(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test,
        pca_dims=pca_dims,
        train_kernel=True,
        kernel_training_iters=20,
        random_seed=random_seed
    )
    
    # Print comparison
    print("\nCOMPARISON TABLE")
    print()
    print(f"{'Kernel':<25} {'Accuracy':<12} {'F1 Score':<12} {'Time (s)':<12}")
    
    for name, res in results.items():
        acc = res['accuracy']
        f1 = res['f1_score']
        time_val = res['total_time']
        print(f"{name:<25} {acc:<12.4f} {f1:<12.4f} {time_val:<12.2f}")
    
    print("=" * 60)
    
    # Calculate improvements
    baseline_acc = results["baseline"]["accuracy"]
    zz_trained_acc = results["zz_trained"]["accuracy"]
    improvement = ((zz_trained_acc - baseline_acc) / baseline_acc) * 100
    
    print(f"\nImprovement over baseline: {improvement:+.2f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ZZ quantum kernel experiment")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--pca_dims", type=int, default=40)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--kernel_iters", type=int, default=20)
    parser.add_argument("--zz_reps", type=int, default=2)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_with_baseline(
            dataset_name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            pca_dims=args.pca_dims,
            random_seed=args.seed
        )
    else:
        results = run_zz_experiment(
            dataset_name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            pca_dims=args.pca_dims,
            train_kernel=not args.no_train,
            kernel_training_iters=args.kernel_iters,
            zz_reps=args.zz_reps,
            random_seed=args.seed
        )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
