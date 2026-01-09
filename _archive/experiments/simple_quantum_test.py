"""
Simple quantum kernel test without circuit building.

Direct test of baseline and ZZ feature map kernels on MNIST.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import PCA
import time

from data.loader import DatasetLoader
from quantum.pennylane_zz_kernel import PennyLaneZZKernel


def normalize_data(X_train, X_test):
    """Normalize to [0,1]"""
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test = (X_test - X_min) / (X_max - X_min + 1e-8)
    return np.clip(X_train, 0, 1), np.clip(X_test, 0, 1)


def baseline_kernel_matrix(X1, X2=None):
    """
    Compute baseline kernel: theta_i = pi * x_i
    
    Kernel = |<psi(x1)|psi(x2)>|^2
    where psi encodes angles theta_i = pi * x_i
    """
    if X2 is None:
        X2 = X1
    
    n1, n_features = X1.shape
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            # Angles for encoding
            angles1 = np.pi * X1[i]
            angles2 = np.pi * X2[j]
            
            # Quantum state overlap: |<psi(x1)|psi(x2)>|^2
            # For simple rotations: cos(angle_diff)^n_features
            angle_diffs = angles1 - angles2
            overlap = np.prod(np.cos(angle_diffs / 2) ** 2)
            K[i, j] = overlap
    
    return K


def test_mnist_quantum_kernels():
    """Test baseline vs ZZ kernel on MNIST"""
    
    print("QUANTUM KERNEL COMPARISON ON MNIST")
    print()
    
    np.random.seed(42)
    
    n_train = 150
    n_test = 50
    pca_dims = 15
    
    print(f"Loading MNIST: {n_train} train, {n_test} test")
    loader = DatasetLoader()
    X_train_raw, X_test_raw, y_train, y_test = loader.load_dataset("mnist", n_train, n_test)
    
    print(f"Applying PCA: {X_train_raw.shape[1]} -> {pca_dims}")
    pca = PCA(n_components=pca_dims)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)
    
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    X_train, X_test = normalize_data(X_train_pca, X_test_pca)
    print(f"Normalized to [0,1]")
    print()
    
    results = {}
    
    print("TEST 1: BASELINE KERNEL (theta=pi*x)")
    print()
    
    baseline_start = time.time()
    K_train_baseline = baseline_kernel_matrix(X_train)
    K_test_baseline = baseline_kernel_matrix(X_test, X_train)
    baseline_kernel_time = time.time() - baseline_start
    
    print(f"Kernel matrix shapes: train {K_train_baseline.shape}, test {K_test_baseline.shape}")
    
    svm_start = time.time()
    svm_baseline = SVC(kernel="precomputed")
    svm_baseline.fit(K_train_baseline, y_train)
    y_pred_baseline = svm_baseline.predict(K_test_baseline)
    baseline_svm_time = time.time() - svm_start
    
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average="weighted")
    baseline_total = baseline_kernel_time + baseline_svm_time
    
    print(f"Accuracy: {baseline_acc:.4f}")
    print(f"F1 Score: {baseline_f1:.4f}")
    print(f"Time: {baseline_total:.2f}s (kernel {baseline_kernel_time:.2f}s + SVM {baseline_svm_time:.2f}s)")
    print()
    
    results["baseline"] = {
        "accuracy": baseline_acc,
        "f1_score": baseline_f1,
        "time": baseline_total
    }
    
    print("TEST 2: ZZ FEATURE MAP KERNEL (untrained)")
    print()
    
    zz_start = time.time()
    zz_kernel = PennyLaneZZKernel(n_features=pca_dims, reps=2)
    
    zz_kernel_start = time.time()
    K_train_zz = zz_kernel.compute_kernel_matrix(X_train)
    K_test_zz = zz_kernel.compute_kernel_matrix(X_test, X_train)
    zz_kernel_time = time.time() - zz_kernel_start
    
    print(f"Kernel matrix shapes: train {K_train_zz.shape}, test {K_test_zz.shape}")
    
    svm_zz_start = time.time()
    svm_zz = SVC(kernel="precomputed")
    svm_zz.fit(K_train_zz, y_train)
    y_pred_zz = svm_zz.predict(K_test_zz)
    zz_svm_time = time.time() - svm_zz_start
    
    zz_acc = accuracy_score(y_test, y_pred_zz)
    zz_f1 = f1_score(y_test, y_pred_zz, average="weighted")
    zz_total = time.time() - zz_start
    
    print(f"Accuracy: {zz_acc:.4f}")
    print(f"F1 Score: {zz_f1:.4f}")
    print(f"Time: {zz_total:.2f}s (kernel {zz_kernel_time:.2f}s + SVM {zz_svm_time:.2f}s)")
    print()
    
    results["zz_untrained"] = {
        "accuracy": zz_acc,
        "f1_score": zz_f1,
        "time": zz_total
    }
    
    print("TEST 3: ZZ FEATURE MAP KERNEL (trained)")
    print()
    
    zz_trained_start = time.time()
    zz_kernel_trained = PennyLaneZZKernel(n_features=pca_dims, reps=2)
    
    print("Training kernel with kernel alignment...")
    train_result = zz_kernel_trained.train(X_train, y_train, maxiter=20, method='Powell')
    training_time = train_result.fun if hasattr(train_result, 'fun') else 0
    
    zz_kernel_start = time.time()
    K_train_zz_tr = zz_kernel_trained.compute_kernel_matrix(X_train)
    K_test_zz_tr = zz_kernel_trained.compute_kernel_matrix(X_test, X_train)
    zz_kernel_time_tr = time.time() - zz_kernel_start
    
    print(f"Kernel matrix shapes: train {K_train_zz_tr.shape}, test {K_test_zz_tr.shape}")
    
    svm_zz_tr_start = time.time()
    svm_zz_tr = SVC(kernel="precomputed")
    svm_zz_tr.fit(K_train_zz_tr, y_train)
    y_pred_zz_tr = svm_zz_tr.predict(K_test_zz_tr)
    zz_svm_time_tr = time.time() - svm_zz_tr_start
    
    zz_tr_acc = accuracy_score(y_test, y_pred_zz_tr)
    zz_tr_f1 = f1_score(y_test, y_pred_zz_tr, average="weighted")
    zz_tr_total = time.time() - zz_trained_start
    
    print(f"Accuracy: {zz_tr_acc:.4f}")
    print(f"F1 Score: {zz_tr_f1:.4f}")
    print(f"Time: {zz_tr_total:.2f}s")
    print()
    
    results["zz_trained"] = {
        "accuracy": zz_tr_acc,
        "f1_score": zz_tr_f1,
        "time": zz_tr_total
    }
    
    print("SUMMARY")
    print()
    print(f"{'Kernel':<25} {'Accuracy':<12} {'F1 Score':<12} {'Time (s)':<12}")
    print()
    
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:<12.4f} {res['f1_score']:<12.4f} {res['time']:<12.2f}")
    
    print()
    print("COMPARISON")
    print()
    
    baseline_acc = results["baseline"]["accuracy"]
    zz_tr_acc = results["zz_trained"]["accuracy"]
    improvement = ((zz_tr_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"ZZ Trained accuracy: {zz_tr_acc:.4f}")
    print(f"Improvement: {improvement:+.2f}%")
    print()
    
    return results


if __name__ == "__main__":
    test_mnist_quantum_kernels()
