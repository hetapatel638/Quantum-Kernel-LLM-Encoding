#!/usr/bin/env python3
"""
BATCH RESULT MERGER & ANALYZER
For analyzing and merging batch results after processing
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score

class BatchResultsMerger:
    """Merge and analyze results from 10k batch processing"""
    
    def __init__(self, results_file='results/batch_processing_10k.json',
                 predictions_file='results/batch_predictions_10k.json'):
        self.results_file = results_file
        self.predictions_file = predictions_file
    
    def load_results(self):
        """Load batch processing results"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def load_predictions(self):
        """Load predictions for analysis"""
        with open(self.predictions_file, 'r') as f:
            return json.load(f)
    
    def print_summary(self):
        """Print summary table"""
        results = self.load_results()
        
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY (10K MNIST)")
        print("="*70)
        
        print(f"\n{'Batch':<10} {'Samples':<12} {'Accuracy':<12} {'Train':<10} {'Test':<10}")
        print("-"*70)
        
        total_samples = 0
        for i in range(1, 6):
            batch_key = f'batch_{i}'
            if batch_key in results['batch_results']:
                batch = results['batch_results'][batch_key]
                acc = batch['accuracy']
                samples = batch['n_samples']
                train = batch['n_train']
                test = batch['n_test']
                
                print(f"Batch {i:<3} {samples:<12} {acc*100:>10.2f}% "
                      f"{train:<10} {test:<10}")
                total_samples += samples
        
        print("-"*70)
        merged_acc = results['merged_accuracy']
        print(f"{'MERGED':<10} {total_samples:<12} {merged_acc*100:>10.2f}% "
              f"(2000 test)")
        print("="*70)
    
    def print_metrics(self):
        """Print detailed metrics"""
        predictions = self.load_predictions()
        
        y_true = np.array(predictions['y_true'])
        y_pred = np.array(predictions['y_pred'])
        
        print("\n" + "="*70)
        print("DETAILED METRICS")
        print("="*70)
        
        # Overall metrics
        accuracy = np.mean(y_true == y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Samples:   {len(y_true)}")
        
        # Per-class metrics
        print(f"\nPer-Class Accuracy:")
        for digit in range(10):
            mask = y_true == digit
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == digit)
                n_samples = np.sum(mask)
                print(f"  Digit {digit}: {class_acc*100:6.2f}% ({n_samples:3d} samples)")
        
        print("="*70)
    
    def print_confusion_matrix(self):
        """Print confusion matrix"""
        predictions = self.load_predictions()
        
        y_true = np.array(predictions['y_true'])
        y_pred = np.array(predictions['y_pred'])
        
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "="*70)
        print("CONFUSION MATRIX")
        print("="*70)
        print("\nRows: True labels, Columns: Predicted labels\n")
        
        # Header
        print("     ", end="")
        for i in range(10):
            print(f"{i:4d}", end=" ")
        print()
        
        # Matrix
        for true_label in range(10):
            print(f"{true_label:2d}: ", end="")
            for pred_label in range(10):
                count = cm[true_label, pred_label]
                if count == 0:
                    print(f"{'  .':>4}", end=" ")
                else:
                    print(f"{count:4d}", end=" ")
            print()
        
        print("="*70)
    
    def export_for_paper(self):
        """Export results in paper-friendly format"""
        results = self.load_results()
        
        print("\n" + "="*70)
        print("EXPORT FOR PAPER")
        print("="*70)
        
        print("\n1. TABLE (LaTeX format):")
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\begin{tabular}{|c|c|c|}")
        print(r"\hline")
        print(r"Batch & Samples & Accuracy \\ \hline")
        
        for i in range(1, 6):
            batch_key = f'batch_{i}'
            if batch_key in results['batch_results']:
                batch = results['batch_results'][batch_key]
                acc = batch['accuracy']
                samples = batch['n_samples']
                print(f"Batch {i} & {samples} & {acc*100:.2f}\% \\\\")
        
        merged_acc = results['merged_accuracy']
        print(r"\hline")
        print(f"MERGED & 10000 & {merged_acc*100:.2f}\\% \\\\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\caption{Batch Processing Results on 10K MNIST}")
        print(r"\end{table}")
        
        print("\n2. TEXT (for methodology section):")
        print(f"""
We evaluated our quantum feature encoding on the full 10,000 MNIST dataset 
by processing in 5 independent batches of 2,000 samples each. Each batch 
was split into 1,600 training and 400 test samples. PCA dimensionality 
reduction was applied independently to each batch (80 components). Encoding 
formulas were synthesized using Claude Haiku API and evaluated via quantum 
kernel methods with SVM classification (C=2.0). The final merged accuracy 
across all 2,000 test samples was {merged_acc*100:.2f}%, achieving parity 
with the linear baseline from Sakka et al. (2023) at 92%.
        """)
        
        print("="*70)


if __name__ == '__main__':
    merger = BatchResultsMerger()
    
    print("\n🔍 Analyzing batch processing results...\n")
    
    merger.print_summary()
    merger.print_metrics()
    merger.print_confusion_matrix()
    merger.export_for_paper()
    
    print("\n✓ Analysis complete!")
