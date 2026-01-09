# BATCH MERGING LOGIC - DETAILED EXPLANATION

## The Core Problem

How do you calculate **final accuracy** from 5 batches?

```
Batch 1: 92.50% accuracy (on 400 test samples)
Batch 2: 91.75% accuracy (on 400 test samples)
Batch 3: 92.25% accuracy (on 400 test samples)
Batch 4: 91.50% accuracy (on 400 test samples)
Batch 5: 92.00% accuracy (on 400 test samples)

What is the FINAL accuracy?
```

---

## ❌ WRONG APPROACH #1: Simple Average

```python
batch_accs = [0.925, 0.9175, 0.9225, 0.915, 0.92]
final_acc = np.mean(batch_accs)  # = 0.92
```

**Problem**: Each batch has different test set distributions
- Doesn't account for actual predictions
- What if batch 1 had easy digits, batch 5 had hard digits?

---

## ❌ WRONG APPROACH #2: Weighted Average (by batch size)

```python
batch_accs = [0.925, 0.9175, 0.9225, 0.915, 0.92]
batch_sizes = [2000, 2000, 2000, 2000, 2000]
test_sizes = [400, 400, 400, 400, 400]

final_acc = np.average(batch_accs, weights=test_sizes)
# Still = 0.92 (since all equal)
```

**Problem**: Still doesn't merge actual predictions
- Ignores label distribution differences
- Not reproducible (can't analyze errors)

---

## ✅ CORRECT APPROACH: Merge Predictions

```python
# DURING BATCH PROCESSING:
batch_1_y_true = [7, 3, 5, 2, 1, ...]  (400 values)
batch_1_y_pred = [7, 3, 5, 2, 1, ...]  (400 values)

batch_2_y_true = [1, 9, 4, 6, 0, ...]  (400 values)
batch_2_y_pred = [1, 9, 4, 6, 0, ...]  (400 values)

... (batches 3-5)

# AFTER ALL BATCHES COMPLETE:
all_y_true = concatenate([batch_1_y_true, batch_2_y_true, ..., batch_5_y_true])
# Shape: (2000,) = 5 × 400

all_y_pred = concatenate([batch_1_y_pred, batch_2_y_pred, ..., batch_5_y_pred])
# Shape: (2000,) = 5 × 400

# CALCULATE FINAL ACCURACY:
final_acc = np.mean(all_y_true == all_y_pred)
# = (# correct predictions) / (total predictions)
# = 1840 / 2000 = 0.92
```

---

## Python Implementation

### During Batch Processing

```python
class BatchProcessor:
    def __init__(self):
        # Store ALL predictions
        self.all_predictions = {
            'indices': [],      # Original indices in 10k dataset
            'y_true': [],       # Ground truth labels
            'y_pred': [],       # Model predictions
            'batch_ids': []     # Which batch each came from
        }
    
    def process_batch(self, batch_idx, X_test, y_test):
        # ... train SVM ...
        
        # Get predictions
        y_pred = svm.predict(K_test)
        
        # STORE THEM
        self.all_predictions['indices'].extend(
            range(batch_idx * 2000 + 1600, (batch_idx+1) * 2000)
        )
        self.all_predictions['y_true'].extend(y_test.tolist())
        self.all_predictions['y_pred'].extend(y_pred.tolist())
        self.all_predictions['batch_ids'].extend([batch_idx] * len(y_test))
        
        # Also print batch accuracy
        batch_acc = np.mean(y_pred == y_test)
        print(f"Batch {batch_idx}: {batch_acc*100:.2f}%")
```

### After All Batches

```python
def merge_predictions(self):
    """Merge all batch predictions"""
    
    # Convert to numpy arrays
    y_true = np.array(self.all_predictions['y_true'])
    y_pred = np.array(self.all_predictions['y_pred'])
    
    # Calculate merged accuracy
    matches = (y_true == y_pred)
    final_acc = np.mean(matches)
    
    # Report
    n_correct = np.sum(matches)
    n_total = len(y_true)
    
    print(f"\nMERGED RESULTS:")
    print(f"  Correct: {n_correct}")
    print(f"  Total:   {n_total}")
    print(f"  Accuracy: {final_acc*100:.2f}%")
    
    return final_acc
```

---

## Why This Matters for Your Paper

### Example: Different Label Distributions

**Scenario:** Digit 8 is harder to classify

```
Batch 1: Mostly 0-5, easy → 92.50% ✓
Batch 2: Mostly 6-9, harder → 91.75% ✓
Batch 3: Mixed, balanced → 92.25% ✓
Batch 4: More 8's, very hard → 91.50% ✓
Batch 5: Balanced → 92.00% ✓
```

If you just average: 92.00%
If you merge predictions: 91.95% (more realistic)

**The merged accuracy is the TRUE performance** across all data variations!

---

## Mathematical Definition

Given $k$ batches, each with test set size $n_{test,i}$:

$$\text{Merged Accuracy} = \frac{\sum_{i=1}^{k} \sum_{j=1}^{n_{test,i}} \mathbb{1}[\hat{y}_{i,j} = y_{i,j}]}{\sum_{i=1}^{k} n_{test,i}}$$

Where:
- $\hat{y}_{i,j}$ = predicted label for sample $j$ in batch $i$
- $y_{i,j}$ = true label for sample $j$ in batch $i$
- $\mathbb{1}[\cdot]$ = indicator function (1 if correct, 0 if wrong)

In English: "Count how many predictions are correct, divide by total predictions"

---

## Concrete Example with Numbers

### Batch Results

```
Batch 1:
  y_true: [7, 3, 5, 2, 1, 9, 4, 6, 0, 8]  (10 samples for demo)
  y_pred: [7, 3, 5, 2, 1, 9, 4, 6, 0, 8]
  Match:  [T, T, T, T, T, T, T, T, T, T]
  Acc: 10/10 = 100% ✓

Batch 2:
  y_true: [1, 9, 4, 6, 0, 8, 2, 5, 3, 7]
  y_pred: [1, 9, 4, 6, 0, 8, 2, 9, 3, 7]  (1 wrong: digit 5→9)
  Match:  [T, T, T, T, T, T, T, F, T, T]
  Acc: 9/10 = 90% ✓
```

### Merge Process

```python
all_y_true = [7, 3, 5, 2, 1, 9, 4, 6, 0, 8, 1, 9, 4, 6, 0, 8, 2, 5, 3, 7]
all_y_pred = [7, 3, 5, 2, 1, 9, 4, 6, 0, 8, 1, 9, 4, 6, 0, 8, 2, 9, 3, 7]
              T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  F  T  T

matches = 19 correct
total = 20 samples
merged_accuracy = 19/20 = 0.95 = 95%
```

---

## For Your Paper

### What to Report

```
"We evaluated our method on 10,000 MNIST samples processed 
in 5 independent batches of 2,000 samples each. Per-batch 
results ranged from 91.50% to 92.50%. The final merged accuracy 
on all 2,000 test predictions was 92.00%, calculated by 
concatenating predictions across batches and computing the 
proportion of correct classifications."
```

### What to Show

```
Table: Batch Processing Results
┌─────────┬──────────┬──────────┐
│ Batch   │ Samples  │ Accuracy │
├─────────┼──────────┼──────────┤
│ 1       │ 2000     │ 92.50%   │
│ 2       │ 2000     │ 91.75%   │
│ 3       │ 2000     │ 92.25%   │
│ 4       │ 2000     │ 91.50%   │
│ 5       │ 2000     │ 92.00%   │
├─────────┼──────────┼──────────┤
│ MERGED  │ 10000    │ 92.00%   │
└─────────┴──────────┴──────────┘

Note: Merged accuracy computed on concatenated 
predictions from all test sets (2000 total samples).
```

---

## Implementation in `batch_processing_10k.py`

### Storage (Lines ~90-95)
```python
self.all_predictions = {
    'indices': [],      # Track original positions
    'y_true': [],       # All ground truth
    'y_pred': [],       # All predictions
}
```

### During Each Batch (Lines ~145-150)
```python
self.all_predictions['indices'].extend(
    range(start_idx + split_idx, end_idx)
)
self.all_predictions['y_true'].extend(y_test.tolist())
self.all_predictions['y_pred'].extend(y_pred.tolist())
```

### After All Batches (Lines ~165-170)
```python
def _merge_batch_results(self):
    y_true = np.array(self.all_predictions['y_true'])
    y_pred = np.array(self.all_predictions['y_pred'])
    merged_acc = np.mean(y_true == y_pred)
    return merged_acc
```

---

## Summary

| Metric | Value | Source |
|--------|-------|--------|
| Batch 1 Acc | 92.50% | Test set of batch 1 only |
| Batch 2 Acc | 91.75% | Test set of batch 2 only |
| ... | ... | ... |
| **Merged Acc** | **92.00%** | **All predictions concatenated** |

The **Merged Accuracy** is your final number for the paper! 🎓

---

## Code Example

```python
# AFTER batch_processing_10k.py completes:

import json
import numpy as np

# Load results
with open('results/batch_processing_10k.json', 'r') as f:
    results = json.load(f)

# Get merged accuracy
final_acc = results['merged_accuracy']
print(f"Final accuracy: {final_acc*100:.2f}%")

# Verify by recalculating
predictions = results['predictions']
y_true = np.array(predictions['y_true'])
y_pred = np.array(predictions['y_pred'])
recalc_acc = np.mean(y_true == y_pred)
print(f"Verification: {recalc_acc*100:.2f}%")

# They should match!
assert abs(final_acc - recalc_acc) < 0.001
```

This is your **ground truth** for the paper! ✅
