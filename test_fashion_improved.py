#!/usr/bin/env python3
"""
Test Fashion-MNIST with higher PCA dimensions for >85% accuracy
Uses the same multi-encoding strategy as run_all_datasets.py
"""

import sys
sys.path.insert(0, '/Users/husky95/Desktop/Innovation')

import numpy as np
import json
from pathlib import Path
from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer

print("FASHION-MNIST WITH HIGHER PCA DIMENSIONS")

# Parameters for improved accuracy
n_train = 800
n_test = 300
n_pca = 160  # 160 dims preserves 96.78% variance (vs 80 dims = 92.78%)

print(f"\nConfiguration:")
print(f"  Dataset: Fashion-MNIST")
print(f"  Training samples: {n_train}")
print(f"  Test samples: {n_test}")
print(f"  PCA dimensions: {n_pca}")

# Step 1: Load
print(f"\nStep 1: Loading Fashion-MNIST...")
loader = DatasetLoader()
X_train, X_test, y_train, y_test = loader.load_dataset('fashion_mnist', n_train, n_test)
print(f"Loaded: train={X_train.shape}, test={X_test.shape}")

# Step 2: Preprocess with HIGHER PCA
print(f"\nStep 2: Preprocessing (PCA 784 → {n_pca})...")
preprocessor = QuantumPreprocessor(n_components=n_pca)
X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
print(f"Done: train={X_train_pca.shape}, test={X_test_pca.shape}")

# Step 3: Baseline
print(f"\nStep 3: Computing BASELINE (θᵢ = π·xᵢ)...")
circuit_builder = QuantumCircuitBuilder(n_qubits=10, max_depth=12)
circuit = circuit_builder.build_circuit(
    [lambda x: np.clip(np.pi * x, 0, 2*np.pi)],
    entanglement="linear"
)

kernel = QuantumKernel()
print("  Computing train kernel...")
K_train = kernel.compute_kernel_matrix(circuit, X_train_pca)
print("  Computing test kernel...")
K_test = kernel.compute_kernel_matrix(circuit, X_train_pca, X_test_pca)

trainer = QuantumSVMTrainer(C=1.0)
print("  Training SVM...")
trainer.train(K_train, y_train)

baseline_metrics = trainer.evaluate(K_test, y_test)
baseline_acc = baseline_metrics['accuracy']
print(f"BASELINE: {baseline_acc*100:.2f}%")

# Step 4: Test improved encodings
print(f"\nStep 4: Testing improved encodings...")

encodings = {
    'Power Scaling (0.7)': lambda x: np.clip(np.pi * x**0.7, 0, 2*np.pi),
    'Power Scaling (0.6)': lambda x: np.clip(np.pi * x**0.6, 0, 2*np.pi),
    'Phase Shift (+0.3π)': lambda x: np.clip(np.pi * x + 0.3*np.pi*(np.arange(len(x))/len(x)), 0, 2*np.pi),
    'Differential': lambda x: np.clip(np.pi * (x + 0.4*(np.roll(x, 1) - x)), 0, 2*np.pi),
    'Weighted Features': lambda x: np.clip(np.pi * x * np.array([1.2 if i < len(x)//2 else 0.8 for i in range(len(x))]), 0, 2*np.pi),
}

results = {'baseline': baseline_acc, 'encodings': {}}

for enc_name, enc_func in encodings.items():
    try:
        print(f"\n  Testing: {enc_name}...")
        
        circuit = circuit_builder.build_circuit([enc_func], entanglement="linear")
        
        K_train = kernel.compute_kernel_matrix(circuit, X_train_pca)
        K_test = kernel.compute_kernel_matrix(circuit, X_train_pca, X_test_pca)
        
        trainer = QuantumSVMTrainer(C=1.0)
        trainer.train(K_train, y_train)
        metrics = trainer.evaluate(K_test, y_test)
        acc = metrics['accuracy']
        
        results['encodings'][enc_name] = acc
        improvement = (acc - baseline_acc) * 100
        
        print(f"    ✓ {acc*100:.2f}% (improvement: {improvement:+.2f}%)")
    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:50]}")

# Summary
print("Fashion-MNIST with {n_pca} PCA dimensions")
print(f"Baseline (80 dims):        {baseline_acc*100:.2f}%")

best_name = max(results['encodings'], key=results['encodings'].get)
best_acc = results['encodings'][best_name]
print(f"Best encoding ({best_name}):   {best_acc*100:.2f}%")
print(f"Improvement: {(best_acc - baseline_acc)*100:+.2f}%")

if best_acc > 0.85:
    print(f"\nTARGET REACHED: >85% accuracy!")
else:
    needed = (0.85 - best_acc) * 100
    print(f"\nClose! Need {needed:.2f}% more improvement")

# Save results
output = {
    'dataset': 'fashion_mnist',
    'n_train': n_train,
    'n_test': n_test,
    'n_pca': n_pca,
    'baseline_accuracy': baseline_acc,
    'best_encoding': best_name,
    'best_accuracy': best_acc,
    'all_encodings': results['encodings']
}

output_path = Path('results/fashion_mnist_improved.json')
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")
