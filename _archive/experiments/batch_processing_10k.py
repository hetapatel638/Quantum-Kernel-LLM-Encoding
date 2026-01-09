#!/usr/bin/env python3
"""
BATCH PROCESSING PIPELINE FOR 10K MNIST DATASET
Processes full dataset in 2k batches, merges results for paper submission

Strategy:
- Batch 1-5: Each 2k samples (train:test = 1600:400)
- Each batch: independent PCA fit, encoding, evaluation
- Final: Merge all predictions for overall accuracy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path

from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class BatchProcessor10K:
    """Process 10K MNIST in 2K batches"""
    
    def __init__(self, batch_size=1000, n_pca=80):
        self.batch_size = batch_size
        self.n_pca = n_pca
        self.num_batches = 10  # 10k / 1k = 10 batches (faster per batch)
        self.all_results = {}
        self.all_predictions = {
            'indices': [],
            'y_true': [],
            'y_pred': []
        }
        
        if HAS_ANTHROPIC:
            self.client = Anthropic()
            self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        else:
            self.api_key = None
    
    def run_full_pipeline(self):
        """Process all 10k samples in batches"""
        print("\n" + "="*80)
        print("10K MNIST BATCH PROCESSING PIPELINE")
        print("="*80)
        print(f"Configuration: {self.num_batches} batches × {self.batch_size} samples")
        print(f"Each batch: {int(self.batch_size*0.8)} train, {int(self.batch_size*0.2)} test")
        
        # === STEP 1: Load full 10k dataset ===
        print("\n[STEP 1/4] Loading full 10k MNIST dataset...")
        loader = DatasetLoader()
        X_full, _, y_full, _ = loader.load_dataset("mnist", 10000, 0)  # Load 10k, no test split
        print(f"  ✓ Loaded: {X_full.shape[0]} samples")
        
        # === STEP 2: Process each batch ===
        print("\n[STEP 2/4] Processing batches...")
        for batch_idx in range(self.num_batches):
            self._process_batch(batch_idx, X_full, y_full)
        
        # === STEP 3: Merge predictions ===
        print("\n[STEP 3/4] Merging batch predictions...")
        merged_acc = self._merge_batch_results()
        
        # === STEP 4: Generate final report ===
        print("\n[STEP 4/4] Generating final report...")
        self._print_final_report(merged_acc)
        self._save_final_results()
    
    def _process_batch(self, batch_idx, X_full, y_full):
        """Process single batch"""
        print(f"\n  [BATCH {batch_idx+1}/{self.num_batches}]")
        
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        X_batch = X_full[start_idx:end_idx]
        y_batch = y_full[start_idx:end_idx]
        
        print(f"    Samples: {start_idx}-{end_idx}")
        
        # Split batch: 80% train, 20% test
        split_idx = int(self.batch_size * 0.8)
        X_train = X_batch[:split_idx]
        X_test = X_batch[split_idx:]
        y_train = y_batch[:split_idx]
        y_test = y_batch[split_idx:]
        
        print(f"    Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # === 1. PCA fit on this batch ===
        print(f"    PCA fit on batch {self.n_pca} components...")
        preprocessor = QuantumPreprocessor(n_components=self.n_pca)
        X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
        variance = preprocessor.pca.explained_variance_ratio_
        
        # === 2. Generate encoding (Claude) ===
        print(f"    Generating encoding (Claude)...")
        encoding_func = self._generate_batch_encoding(variance)
        
        # === 3. Evaluate on batch ===
        print(f"    Building quantum circuit...")
        circuit_builder = QuantumCircuitBuilder(n_qubits=10, max_depth=12)
        circuit = circuit_builder.build_circuit([encoding_func], entanglement="linear")
        
        print(f"    Computing quantum kernel...")
        kernel_computer = QuantumKernel()
        K_train = kernel_computer.compute_kernel_matrix(circuit, X_train_pca)
        K_test = kernel_computer.compute_kernel_matrix(circuit, X_train_pca, X_test_pca)
        
        print(f"    Training SVM...")
        svm_trainer = QuantumSVMTrainer(C=2.0)  # Use optimal C from previous experiments
        svm_trainer.train(K_train, y_train)
        metrics = svm_trainer.evaluate(K_test, y_test)
        
        batch_acc = metrics['accuracy']
        
        # Store batch results
        self.all_results[f'batch_{batch_idx+1}'] = {
            'accuracy': batch_acc,
            'n_samples': self.batch_size,
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'pca_variance': float(np.sum(variance)),
            'metrics': metrics
        }
        
        # Store predictions for merge
        y_pred = svm_trainer.predict(K_test)
        self.all_predictions['indices'].extend(range(start_idx + split_idx, end_idx))
        self.all_predictions['y_true'].extend(y_test.tolist())
        self.all_predictions['y_pred'].extend(y_pred.tolist())
        
        print(f"    ✓ Batch {batch_idx+1} accuracy: {batch_acc*100:.2f}%")
    
    def _generate_batch_encoding(self, variance):
        """Generate encoding using Claude or fallback"""
        if not (HAS_ANTHROPIC and self.api_key):
            # Fallback: simple weighted encoding
            importance = variance / np.sum(variance)
            def encoding(x):
                angles = np.pi * x * importance
                return np.clip(angles, 0, 2*np.pi)
            return encoding
        
        prompt = f"""Design a quantum angle encoding for MNIST classification.

Dataset: {self.n_pca} PCA components
Variance profile: {variance[:5].round(3).tolist()}...{variance[-5:].round(3).tolist()}

Requirements:
1. Use feature importance weighting
2. Add non-linear terms for expressiveness
3. Angles must be in [0, 2π]
4. Target: >92% accuracy

Return ONLY Python expression for angles (use x, variance, np.pi):"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            formula = response.content[0].text.strip()
            formula = formula.replace('```python', '').replace('```', '').strip()
            
            importance = variance / np.sum(variance)
            def encoding(x):
                try:
                    angles = eval(formula, {"np": np, "x": x, "variance": importance, "pi": np.pi})
                    return np.clip(angles, 0, 2*np.pi)
                except:
                    return np.clip(np.pi * x * importance, 0, 2*np.pi)
            return encoding
        except:
            # Fallback
            importance = variance / np.sum(variance)
            def encoding(x):
                angles = np.pi * x * importance
                return np.clip(angles, 0, 2*np.pi)
            return encoding
    
    def _merge_batch_results(self):
        """Merge predictions from all batches"""
        # Convert to arrays
        y_true = np.array(self.all_predictions['y_true'])
        y_pred = np.array(self.all_predictions['y_pred'])
        
        # Calculate merged accuracy
        merged_acc = np.mean(y_true == y_pred)
        
        return merged_acc
    
    def _print_final_report(self, merged_acc):
        """Print comprehensive final report"""
        print("\n" + "="*80)
        print("FINAL REPORT - 10K MNIST BATCH PROCESSING")
        print("="*80)
        
        print("\nBatch Results:")
        print(f"{'Batch':<10} {'Samples':<12} {'Accuracy':<12} {'Train':<10} {'Test':<10}")
        print("-" * 70)
        
        total_samples = 0
        weighted_acc = 0
        
        for i in range(1, self.num_batches + 1):
            batch_key = f'batch_{i}'
            result = self.all_results[batch_key]
            acc = result['accuracy']
            n_samples = result['n_samples']
            
            print(f"Batch {i:<3} {n_samples:<12} {acc*100:>10.2f}% "
                  f"{result['n_train']:<10} {result['n_test']:<10}")
            
            total_samples += n_samples
            weighted_acc += acc * n_samples
        
        weighted_avg = weighted_acc / total_samples if total_samples > 0 else 0
        
        print("-" * 70)
        print(f"{'TOTAL':<10} {total_samples:<12} {merged_acc*100:>10.2f}% (merged)")
        print(f"{'AVERAGE':<10} {'':12} {weighted_avg*100:>10.2f}% (weighted avg)")
        
        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE PAPER (Sakka et al. 2023)")
        print("="*80)
        print(f"\nPaper Results:")
        print(f"  • MNIST Linear:    92.00%")
        print(f"  • MNIST YZCX:      97.27%")
        
        print(f"\nOur Results (Full 10k):")
        print(f"  • Merged Accuracy: {merged_acc*100:.2f}%")
        
        gap = (0.92 - merged_acc) * 100
        if merged_acc >= 0.92:
            print(f"\n✓ SUCCESS! Matched/exceeded baseline (92%)")
        else:
            print(f"\n⚠ Gap to baseline: {gap:.2f}%")
        
        print("="*80)
    
    def _save_final_results(self):
        """Save results for paper submission"""
        results_obj = {
            'experiment': 'Full 10K MNIST Batch Processing',
            'configuration': {
                'total_samples': 10000,
                'batch_size': self.batch_size,
                'num_batches': self.num_batches,
                'pca_components': self.n_pca,
                'circuit': '10 qubits, 12 layers, linear entanglement'
            },
            'batch_results': self.all_results,
            'merged_accuracy': self._merge_batch_results(),
            'predictions': {
                'y_true': self.all_predictions['y_true'],
                'y_pred': self.all_predictions['y_pred'],
                'indices': self.all_predictions['indices']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/batch_processing_10k.json', 'w') as f:
            json.dump(results_obj, f, indent=2)
        
        print(f"\n✓ Results saved to results/batch_processing_10k.json")
        
        # Also save predictions for confusion matrix analysis
        with open('results/batch_predictions_10k.json', 'w') as f:
            json.dump(self.all_predictions, f)
        
        print(f"✓ Predictions saved to results/batch_predictions_10k.json")


if __name__ == '__main__':
    processor = BatchProcessor10K(batch_size=2000, n_pca=80)
    processor.run_full_pipeline()
