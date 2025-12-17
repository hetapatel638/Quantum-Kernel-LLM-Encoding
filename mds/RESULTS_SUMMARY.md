# Quantum Encoding Experiment Results

## Summary of Changes Made

### 1. Prompt Engineering Improvements
- Removed all emojis and decorative elements
- Made language natural and student-like
- Added 3 clear strategy templates (PCA-mixed, Polynomial, Adaptive)
- Increased temperature from 0.7 to 0.95 for diversity
- Added explicit failure modes to avoid baseline copying

### 2. Technical Fixes
- Added `max`/`min` support in validator namespace
- Added `max`/`min` support in circuit execution namespace  
- Implemented kernel caching with lru_cache
- Fixed diversity validation (std > 0.3 check)

### 3. Multi-Trial Testing Infrastructure
- Created `run_multi_trial.py` for automated comparison
- Tests 5 independent trials with different Claude encodings
- Saves comprehensive summary with best/worst trials

## Experimental Results

### Large Dataset Test (Most Reliable)
```
Dataset: MNIST
Training: 500 samples
Testing: 200 samples  
PCA Dimensions: 40

BASELINE (theta_i = pi * x[i]):
  Accuracy: 84.50%
  Training: 92.80%

CLAUDE LLM (PCA-mixed strategy):
  Function: [np.clip(np.pi * (0.5*x[i] + 0.3*x[(i+3)] + 0.2*x[(i+7)]), 0, 2*np.pi)]
  Accuracy: 75.50%
  Training: 82.60%

DIFFERENCE: -10.65% (Claude WORSE)
```

### Multi-Trial Results (5 trials, 200/50 split)
```
Trial 1: -7.0%  (80% vs 86% baseline)
Trial 2: +5.3%  (80% vs 76% baseline) <- ONLY SUCCESS
Trial 3: -12.8% (68% vs 78% baseline)
Trial 4: -9.1%  (80% vs 88% baseline)
Trial 5: -34.1% (54% vs 82% baseline)

Average: -11.6% worse than baseline
Success Rate: 1/5 (20%)
```

## Key Findings

### Why Claude Underperforms

1. **Information Dilution**: Mixing PCA components (0.5*x[i] + 0.3*x[i+3] + 0.2*x[i+7]) spreads information across features, reducing signal strength

2. **Quantum Kernel Mismatch**: The quantum kernel approach benefits from direct feature-to-qubit mapping, not feature mixing

3. **Training vs Test Gap**: Claude encodings show larger generalization gap
   - Baseline: 92.8% train -> 84.5% test (8.3% gap)
   - Claude: 82.6% train -> 75.5% test (7.1% gap but lower overall)

4. **High Variance**: Small test sets (50 samples) show high variance
   - Baseline varies: 76-88%
   - Claude varies: 54-80%

### What Actually Works

The simple baseline encoding **theta_i = pi * x[i]** is actually optimal for this quantum kernel approach because:
- Direct PCA-to-qubit mapping preserves maximum variance
- No information loss from coefficient weighting
- Each qubit encodes one principal component directly

## Conclusion

**Current Status**: Claude generates valid, diverse encodings but they underperform baseline by ~10% on average.

**The Disappointment**: The baseline is actually well-designed for quantum kernels. LLM-generated "improvements" hurt performance because they dilute information.

**Recommendation**: Either:
1. Accept that baseline is optimal for this architecture
2. Change the architecture (different entanglement, more qubits, different kernel)
3. Try completely different encoding families (amplitude encoding, basis encoding)

The prompt engineering is working - Claude generates diverse strategies. The problem is the strategies themselves don't improve this specific quantum kernel method.
