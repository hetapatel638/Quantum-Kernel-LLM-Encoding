## Project Overview
This is a **Quantum Machine Learning Framework** that synthesizes feature encoding strategies using LLMs and evaluates them on quantum circuits. The goal is to discover novel encoding templates that encode classical data into quantum angles for better classification.

### Core Data Flow
1. **Data Loading** (`data/loader.py`) → Load MNIST, Fashion-MNIST, CIFAR-10
2. **Preprocessing** (`data/preprocessor.py`) → Apply PCA reduction, normalize to [0,1]
3. **LLM Encoding Synthesis** (`llm/hf_interface.py`, `encoding/prompt_builder.py`) → Generate encoding formulas
4. **Encoding Validation** (`encoding/validator.py`) → Validate template syntax & constraints
5. **Quantum Circuit** (`quantum/circuit.py`, `quantum/kernel.py`) → Build PennyLane circuits
6. **SVM Evaluation** (`evaluation/svm_trainer.py`) → Train SVM classifier, compute metrics
7. **Comparison & Reporting** (`experiments/`, `visualization/results_table.py`)

## Architecture Patterns

### Configuration-Driven Design
All runtime parameters (datasets, models, templates, quantum config) are defined in `config.py`. Key structure:
- **datasets**: MNIST/Fashion-MNIST/CIFAR-10 with PCA ranges [10, 40, 80]
- **reference_results**: Baseline accuracies from Sakka et al. paper (e.g., MNIST YZCX quantum: 0.9727)
- **llm**: Uses `google/gemma-2b` or `t5-small` locally on macOS/Colab
- **templates**: Four families to synthesize: `linear`, `polynomial`, `global_stats`, `pca_mix`
- **quantum**: 10 qubits, max depth 12, linear entanglement

### Template Families (Key Encoding Patterns)
All defined in `encoding/templates.py` as static methods returning Python code strings:
- **Linear**: `θᵢ = Σ αⱼxⱼ` (high interpretability, best for MNIST/Fashion)
- **Polynomial**: `θᵢ = Σ αⱼxⱼ + Σ βⱼₖxⱼxₖ` (degree 2, O(n²) complexity)
- **Global Stats**: `θᵢ = δ·mean(x) + ε·std(x)` (statistical aggregation, best for CIFAR-10)
- **PCA Mix**: `θᵢ = Σ ωⱼ·PCⱼ` (max 4 components for dimensionality)

Each template enforces constraints (e.g., linear coefficients summed in absolute value ≤ 1).

### LLM-Driven Synthesis Pattern
The LLM doesn't train models—it generates *encoding formulas* as Python code:
1. `encoding/prompt_builder.py` creates prompts with dataset statistics and template schema
2. `llm/hf_interface.py` calls HuggingFace model locally
3. LLM outputs Python expressions like `0.8*x[0] + 0.2*x[1]`
4. `encoding/validator.py` parses and validates against template constraints

### Data Normalization & PCA
- Input data normalized to [0,1] after PCA (not standard -1,1 range)
- PCA components stored in `QuantumPreprocessor.pca` for test data transformation
- Angles clipped to [0, 2π] using `np.clip(..., 0, 2*np.pi)` in all templates

## Critical Workflows

### Running Single Dataset Experiment
```bash
python experiments/run_single_dataset.py --dataset mnist --n_train 10000 --n_test 10000 --pca_dims 40
```
This runs the full pipeline: load → preprocess → synthesize templates → evaluate → report metrics.

### Comparing Templates Across Datasets
```bash
python experiments/compare_templates.py
```
Evaluates all four template families on all three datasets with reference baselines.

### Quick Validation
Test individual components:
```python
from encoding.templates import EncodingTemplates
from encoding.prompt_builder import PromptBuilder
from encoding.validator import TemplateValidator

# Generate a linear template
template_code = EncodingTemplates.linear_template([0.8, 0.2], n_features=2)
# Validate: check syntax, coefficient constraints
is_valid = TemplateValidator.validate_linear(template_code)
```

## Project-Specific Conventions

### No Aspirational Patterns
- **SVM always one-vs-rest**: Binary classifiers for 10-way MNIST/Fashion (see `evaluation/svm_trainer.py`)
- **Always evaluate on test set**: Cross-validation is only for hyperparameter tuning (CV folds in config)
- **Template outputs are Python expressions**: Never raw formulas—must be evaluable as `eval(template_code, {"x": np.array(...)})`

### Angle Encoding Constraint
All angles **must** be in [0, 2π] for quantum rotation gates. Templates enforce this via `np.clip(..., 0, 2*np.pi)`.

### Reproducibility
- `config.py` includes `random_seed: 42` for all experiments
- PCA fit on training data only; applied to test data using stored transform
- Baseline reference results from peer-reviewed paper (Sakka et al.)

## Integration Points

### Quantum Backend
- **Framework**: PennyLane with Qiskit simulator
- **Circuit**: Fixed 10 qubits, encoded angles → rotation gates → entangling layer → measurement
- **Kernel Function**: `quantum/kernel.py` computes Gram matrix from circuit outputs

### Baseline Comparison
- `baselines/manual_baseline.py` provides simple rotation baseline: `θᵢ = π·xᵢ`
- All LLM-synthesized templates benchmarked against this and reference results in `config.py`

### Experiment Tracking
- Results saved as CSV/JSON with columns: dataset, template_family, pca_dims, accuracy, f1_score, training_time
- Visualization in `visualization/results_table.py` compares against reference baselines

## Key Files to Review First
1. `config.py` — Understand all runtime parameters
2. `encoding/templates.py` — See how template code is generated
3. `data/preprocessor.py` — Learn PCA + normalization pipeline
4. `encoding/prompt_builder.py` — Understand LLM prompt structure
5. `evaluation/svm_trainer.py` — See SVM training and metric computation

## Common Debugging
- **LLM output invalid Python**: Check `encoding/validator.py` for syntax errors
- **SVM accuracy 0.5**: Likely encoding angles out of [0, 2π] or NaN values
- **PCA mismatch test/train**: Ensure test PCA transform uses `QuantumPreprocessor.pca` fitted on training
