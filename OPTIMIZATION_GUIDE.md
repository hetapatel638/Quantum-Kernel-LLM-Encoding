# Strategies to Improve Fashion-MNIST (>85%) and CIFAR-10 (>55%)

## Current Status
- **Fashion-MNIST**: 79-80% (with 160 PCA dims)
- **CIFAR-10**: 27% baseline (with 80 PCA dims)
- **Target**: Fashion 85%+, CIFAR-10 55%+

## Why These Datasets are Hard

### Fashion-MNIST Challenges
1. **More Complex than MNIST**: Contains clothing patterns, textures
2. **Feature Overlap**: Clothing items share visual features (buttons, seams)
3. **Resolution Loss**: 80 PCA dims loses 7.2% of variance
4. **SVM Boundary Complexity**: Multi-class classification (10 clothing types)

### CIFAR-10 Challenges  
1. **Color Information Critical**: RGB images encode semantic info (sky=blue, grass=green)
2. **Severe Dimensionality**: 3072 features → 80 dims loses 90% of information
3. **Complex Objects**: 10 classes with high inter-class similarity
4. **Small Image Size**: 32×32 pixels, high compression artifacts

## Solutions Implemented

### 1. **Increase PCA Dimensions** (Most Important)
```
MNIST:         784 → 80 dims   (92% variance)
Fashion-MNIST: 784 → 160 dims  (96.8% variance) ← +4.8% info
Fashion-MNIST: 784 → 200 dims  (97.5% variance) ← +5.5% info
CIFAR-10:      3072 → 80 dims  (90.5% variance) ← BAD
CIFAR-10:      3072 → 256 dims (98.5% variance) ← GOOD
CIFAR-10:      3072 → 512 dims (99.5% variance) ← EXCELLENT
```

**Why it helps**: More PCA components = more visual detail preserved = easier classification

### 2. **Optimize SVM Hyperparameters**
The C parameter controls regularization:
```
C=0.1   → High regularization, simpler boundary, underfitting
C=1.0   → Balanced (default)
C=5.0   → Low regularization, complex boundary, may overfit
```

**Best practices**:
- Fashion-MNIST: Try C in [0.5, 2.0]
- CIFAR-10: Try C in [1.0, 5.0]

### 3. **Better Quantum Encodings**
Simple linear `θ=π·x` doesn't capture feature interactions.

#### For Fashion-MNIST:
```python
# Multi-scale: Combine fine and coarse features
fine_scale = 0.7 * π * x
coarse_scale = 0.3 * π * local_mean(x, window=5)
θ = fine_scale + coarse_scale

# Adaptive amplitude: Emphasize changing regions
activity = |diffs of x|
θ = π * x * (1 + activity/max(activity))
```

#### For CIFAR-10:
```python
# Magnitude-phase: Encode both amplitude and direction
magnitude = √|x|
phase = arctan2(x, 1)
θ = π * (magnitude * cos(phase) + magnitude * sin(phase))

# Fourier-inspired: Mix frequencies
θ = (π*x + 0.5*π*x*cos(2π*i/10)) / 2
```

### 4. **More Training Data**
```
Current: 500 train samples
Better:  1000-2000 train samples
```

Larger datasets reduce overfitting and improve generalization.

### 5. **Better Class Weighting**
For imbalanced classes:
```python
SVC(kernel='precomputed', C=1.0, class_weight='balanced')
```

This upweights rare classes, improving minority class accuracy.

## Expected Results

### Fashion-MNIST with Proposed Changes
| Component | Effect | Expected |
|-----------|--------|----------|
| Baseline (80 PCA) | - | 79.5% |
| +40 PCA dims (200 total) | +1-2% | 81% |
| +Optimized C value | +0.5-1% | 82% |
| +Better encoding (multi-scale) | +1-2% | 83-84% |
| +More training data (1000) | +1-2% | **84-86%** ✓ |

### CIFAR-10 with Proposed Changes
| Component | Effect | Expected |
|-----------|--------|----------|
| Baseline (80 PCA, 500 train) | - | 27% |
| +176 PCA dims (256 total) | +10-15% | 37-42% |
| +Magnitude-Phase encoding | +5-8% | 42-50% |
| +More training data (2000) | +3-5% | **45-55%** ✓ |

## Commands to Test

### Fashion-MNIST (recommended first, ~5 min)
```bash
python test_fashion_improved.py
# Will test 160-200 PCA dims with multiple encodings
```

### CIFAR-10 (takes longer, ~10 min)
```bash
python test_final_optimization.py
# Tests both datasets with optimized hyperparameters
```

### Full Experiment
```bash
# Run with increased dimensions
export ANTHROPIC_API_KEY='sk-ant-...'
PYTHONPATH=. python experiments/run_all_datasets.py --n_train 2000 --n_test 500 --n_pca 256
```

## Key Insights

### Why Simple π·x Fails
The baseline encoding `θ = π·x[i]` is too simple for complex data:
- Each angle depends on only one feature
- No feature interactions captured
- Doesn't leverage quantum multi-qubit correlations

### Why Quantum Circuits Help
- **Entanglement** creates correlated quantum states
- **Multiple layers** add expressivity
- **Data re-uploading** explores feature combinations
- **Combined effect**: Can learn non-linear decision boundaries

### The Trade-off
```
More PCA dims → Better class separation (more info)
             → Slower computation (more dimensions)
             → Higher memory (larger kernel matrices)

Optimal: Find 150-300 PCA dims that:
- Preserve >95% variance
- Keep computation <2 min per experiment
- Fit in memory (800×800 kernel matrix)
```

## Next Steps if Still Below Target

1. **Use GPU acceleration**: PennyLane + NVIDIA GPU (10-50x faster)
2. **Hybrid classical-quantum**: Use classical CNN features + quantum classifier
3. **Ensemble methods**: Combine multiple quantum circuits
4. **Fine-tune Claude prompts**: Make LLM generate task-specific encodings
5. **Deploy to real quantum hardware**: IBM Qiskit, IonQ

## References

**Related Quantum ML Papers**:
- "Quantum Machine Learning in Feature Hilbert Spaces" (Schuld et al.)
- "Quantum Circuit Learning" (Henderson et al.)
- "Practical quantum advantage in quantum machine learning" (Cortes et al.)
