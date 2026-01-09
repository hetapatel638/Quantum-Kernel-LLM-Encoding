"""
MNIST classification using Qiskit trainable quantum kernels.

This experiment uses Qiskit ML's trainable quantum kernels with ZZFeatureMap
to achieve better accuracy than PennyLane baseline encodings.
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
from quantum.qiskit_kernel import QiskitTrainableKernel, QiskitZZKernel


def run_qiskit_experiment(
    dataset_name="mnist",
    n_train=500,
    n_test=200,
    pca_dims=40,
    kernel_type="trainable",
    train_kernel=True,
    kernel_training_iters=20,
    random_seed=42
):
    """
    Run quantum kernel experiment using Qiskit ML.
    
    Args:
        dataset_name: "mnist", "fashion_mnist", or "cifar10"
        n_train: Number of training samples
        n_test: Number of test samples
        pca_dims: PCA dimensions (must be <= n_qubits)
        kernel_type: "trainable" (trainable kernel) or "zz" (fixed ZZ kernel)
        train_kernel: Whether to train kernel parameters (only for trainable)
        kernel_training_iters: Number of SPSA optimization iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with experiment results
    """
    np.random.seed(random_seed)
    
    print("=" * 60)
    print("QISKIT QUANTUM KERNEL EXPERIMENT")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"PCA dimensions: {pca_dims}")
    print(f"Kernel type: {kernel_type}")
    if kernel_type == "trainable":
        print(f"Train kernel: {train_kernel}")
        print(f"Training iterations: {kernel_training_iters}")
    print("=" * 60)
    
    # Load and preprocess data
    print("\nLoading dataset...")
    loader = DatasetLoader()
    X_train_raw, y_train, X_test_raw, y_test = loader.load_dataset(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test,
        random_seed=random_seed
    )
    
    print(f"Applying PCA (n_components={pca_dims})...")
    preprocessor = QuantumPreprocessor(n_components=pca_dims)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train labels: {np.unique(y_train)}")
    print(f"Test labels: {np.unique(y_test)}")
    
    # Initialize quantum kernel
    print(f"\nInitializing Qiskit {kernel_type} quantum kernel...")
    if kernel_type == "trainable":
        kernel = QiskitTrainableKernel(n_features=pca_dims)
    elif kernel_type == "zz":
        kernel = QiskitZZKernel(n_features=pca_dims, reps=2)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Train kernel (if trainable)
    training_time = 0.0
    if kernel_type == "trainable" and train_kernel:
        print("\nTraining quantum kernel with Quantum Kernel Alignment...")
        train_start = time.time()
        
        training_results = kernel.train(
            X_train=X_train,
            y_train=y_train,
            maxiter=kernel_training_iters
        )
        
        training_time = time.time() - train_start
        print(f"Kernel training time: {training_time:.2f}s")
    
    # Compute kernel matrices
    print("\nComputing kernel matrices...")
    kernel_start = time.time()
    
    if kernel_type == "trainable" and train_kernel:
        K_train = kernel.evaluate(X_train, X_train)
        K_test = kernel.evaluate(X_test, X_train)
    else:
        # For fixed kernel or untrained trainable kernel, use quantum_kernel directly
        if kernel_type == "trainable":
            # Use untrained kernel (random initial parameters)
            K_train = kernel.quantum_kernel.evaluate(x_vec=X_train)
            K_test = kernel.quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
        else:
            K_train = kernel.evaluate(X_train, X_train)
            K_test = kernel.evaluate(X_test, X_train)
    
    kernel_time = time.time() - kernel_start
    print(f"Kernel computation time: {kernel_time:.2f}s")
    print(f"Train kernel matrix shape: {K_train.shape}")
    print(f"Test kernel matrix shape: {K_test.shape}")
    
    # Train SVM classifier
    print("\nTraining SVM classifier...")
    svm_start = time.time()
    
    svm = SVC(kernel="precomputed")
    svm.fit(K_train, y_train)
    
    svm_time = time.time() - svm_start
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = svm.predict(K_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total time: {training_time + kernel_time + svm_time:.2f}s")
    print("=" * 60)
    
    # Prepare results dictionary
    results = {
        "dataset": dataset_name,
        "n_train": n_train,
        "n_test": n_test,
        "pca_dims": pca_dims,
        "kernel_type": kernel_type,
        "trained": train_kernel if kernel_type == "trainable" else False,
        "kernel_training_iters": kernel_training_iters if kernel_type == "trainable" else 0,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "f1_score": float(f1),
        "training_time": float(training_time),
        "kernel_time": float(kernel_time),
        "svm_time": float(svm_time),
        "total_time": float(training_time + kernel_time + svm_time)
    }
    
    return results


def compare_kernels(
    dataset_name="mnist",
    n_train=500,
    n_test=200,
    pca_dims=40,
    random_seed=42
):
    """
    Compare different Qiskit quantum kernel approaches.
    
    Tests:
    1. Fixed ZZ kernel (no training)
    2. Trainable kernel (with training)
    3. Trainable kernel (without training - random initialization)
    
    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 60)
    print("COMPARING QISKIT QUANTUM KERNELS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Fixed ZZ kernel
    print("\n### TEST 1: Fixed ZZ Kernel ###")
    results["zz_fixed"] = run_qiskit_experiment(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test,
        pca_dims=pca_dims,
        kernel_type="zz",
        random_seed=random_seed
    )
    
    # Test 2: Trainable kernel (trained)
    print("\n### TEST 2: Trainable Kernel (TRAINED) ###")
    results["trainable_trained"] = run_qiskit_experiment(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test,
        pca_dims=pca_dims,
        kernel_type="trainable",
        train_kernel=True,
        kernel_training_iters=20,
        random_seed=random_seed
    )
    
    # Test 3: Trainable kernel (untrained)
    print("\n### TEST 3: Trainable Kernel (UNTRAINED) ###")
    results["trainable_untrained"] = run_qiskit_experiment(
        dataset_name=dataset_name,
        n_train=n_train,
        n_test=n_test,
        pca_dims=pca_dims,
        kernel_type="trainable",
        train_kernel=False,
        random_seed=random_seed
    )
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Kernel Type':<30} {'Accuracy':<12} {'F1 Score':<12} {'Total Time':<12}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<30} {res['accuracy']:<12.4f} {res['f1_score']:<12.4f} {res['total_time']:<12.2f}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qiskit quantum kernel experiment")
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--pca_dims", type=int, default=40)
    parser.add_argument("--kernel_type", type=str, default="trainable",
                       choices=["trainable", "zz"])
    parser.add_argument("--no_train", action="store_true",
                       help="Skip kernel training (use random initialization)")
    parser.add_argument("--kernel_iters", type=int, default=20,
                       help="Number of kernel training iterations")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison of all kernel types")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison
        results = compare_kernels(
            dataset_name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            pca_dims=args.pca_dims,
            random_seed=args.seed
        )
    else:
        # Run single experiment
        results = run_qiskit_experiment(
            dataset_name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            pca_dims=args.pca_dims,
            kernel_type=args.kernel_type,
            train_kernel=not args.no_train,
            kernel_training_iters=args.kernel_iters,
            random_seed=args.seed
        )
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
