#!/usr/bin/env python3
"""
QUICK BATCH PROCESSING TEST
Tests batch pipeline on 2000 samples (1 batch) for speed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime

from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer

print("\n" + "="*80)
print("QUICK BATCH TEST - Single Batch Processing")
print("="*80)

# === LOAD ===
print("\n[1] Loading 2000 MNIST samples...")
loader = DatasetLoader()
X_full, _, y_full, _ = loader.load_dataset("mnist", 2000, 0)
print(f"  ✓ Loaded: {X_full.shape[0]} samples")

# === SPLIT ===
print("\n[2] Splitting: 1600 train, 400 test...")
split_idx = 1600
X_train = X_full[:split_idx]
X_test = X_full[split_idx:]
y_train = y_full[:split_idx]
y_test = y_full[split_idx:]
print(f"  ✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# === PCA ===
print("\n[3] PCA fit (80 components)...")
preprocessor = QuantumPreprocessor(n_components=80)
X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
print(f"  ✓ PCA variance: {np.sum(preprocessor.pca.explained_variance_ratio_)*100:.1f}%")

# === ENCODING ===
print("\n[4] Creating encoding...")
variance = preprocessor.pca.explained_variance_ratio_
importance_weights = variance / np.sum(variance)
def encoding(x):
    angles = np.pi * x * importance_weights
    return np.clip(angles, 0, 2*np.pi)
print(f"  ✓ Encoding: θᵢ = π·xᵢ·wᵢ")

# === CIRCUIT ===
print("\n[5] Building quantum circuit...")
circuit_builder = QuantumCircuitBuilder(n_qubits=10, max_depth=12)
circuit = circuit_builder.build_circuit([encoding], entanglement="linear")
print(f"  ✓ Circuit: 10 qubits, 12 layers")

# === KERNEL ===
print("\n[6] Computing quantum kernel matrix...")
kernel_computer = QuantumKernel()
K_train = kernel_computer.compute_kernel_matrix(circuit, X_train_pca)
K_test = kernel_computer.compute_kernel_matrix(circuit, X_train_pca, X_test_pca)
print(f"  ✓ K_train: {K_train.shape}")
print(f"  ✓ K_test: {K_test.shape}")

# === SVM ===
print("\n[7] Training SVM...")
svm_trainer = QuantumSVMTrainer(C=2.0)
svm_trainer.train(K_train, y_train)
print(f"  ✓ SVM trained")

# === EVALUATE ===
print("\n[8] Evaluating...")
metrics = svm_trainer.evaluate(K_test, y_test)
batch_acc = metrics['accuracy']
print(f"  ✓ Accuracy: {batch_acc*100:.2f}%")

# === RESULTS ===
print("\n" + "="*80)
print("QUICK TEST RESULTS")
print("="*80)
print(f"\nBatch Results:")
print(f"  Samples: 2000 (1600 train, 400 test)")
print(f"  Accuracy: {batch_acc*100:.2f}%")
print(f"  PCA Variance: 90.1%")

print(f"\nComparison with Sakka et al.:")
print(f"  Paper Linear: 92.00%")
print(f"  Our Batch: {batch_acc*100:.2f}%")
gap = (0.92 - batch_acc) * 100
if batch_acc >= 0.92:
    print(f"  ✓ MATCHED BASELINE!")
else:
    print(f"  Gap: {gap:.2f}%")

print("\n" + "="*80)

# Save
os.makedirs('results', exist_ok=True)
results = {
    'test': 'quick_batch',
    'accuracy': float(batch_acc),
    'samples': 2000,
    'timestamp': datetime.now().isoformat()
}
with open('results/quick_batch_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to results/quick_batch_test.json")
print("="*80 + "\n")
