# 5-Level Quantum ML Framework

## Architecture Overview

```
Level 1: DATA PIPELINE
├── data/loader.py          → Load MNIST, Fashion-MNIST, CIFAR-10
└── data/preprocessor.py    → PCA + Normalization to [0,1]

Level 2: LLM PROMPT ENGINEERING
├── llm/claude_interface.py       → Call Claude API
└── encoding/prompt_builder.py    → Generate smart prompts with dataset stats

Level 3: ENCODING SYNTHESIS & VALIDATION
├── encoding/templates.py         → Template families
└── encoding/validator.py         → Validate generated encoding

Level 4: QUANTUM CIRCUIT & KERNEL (QISKIT)
├── quantum/qiskit_circuit.py     → Build Qiskit parameterized circuits
└── quantum/qiskit_kernel.py      → Compute Gram matrix

Level 5: TRAINING & EVALUATION
└── evaluation/svm_trainer.py     → Train SVM, compute metrics

ORCHESTRATION:
├── config.py              → All parameters
├── main.py               → CLI entry point
└── experiments/          → Dataset runners
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Claude API Key
```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

## Quick Start

### Single Dataset Experiment
```bash
# MNIST with linear encoding, 40 PCA dims
python main.py --mode single --dataset mnist --template linear --n_pca 40

# Fashion-MNIST with polynomial encoding
python main.py --mode single --dataset fashion_mnist --template polynomial --n_pca 40

# CIFAR-10 with global_stats encoding
python main.py --mode single --dataset cifar10 --template global_stats --n_pca 40
```

### All Three Datasets
```bash
# Run all datasets with linear encoding
python main.py --mode multi --template linear --n_pca 40

# Run all datasets with polynomial encoding
python main.py --mode multi --template polynomial --n_pca 80
```

### Testing (Mock Mode)
```bash
# Test without Claude API (uses fallback encoding)
python main.py --mode single --dataset mnist --mock
```

## Configuration

Edit `config.py` to modify:
- **Datasets**: MNIST, Fashion-MNIST, CIFAR-10
- **PCA dimensions**: [10, 40, 80]
- **Qiskit quantum**: n_qubits=10, max_depth=12
- **SVM parameters**: C=1.0
- **Claude model**: claude-3-5-sonnet-20241022

## Pipeline Flow

```
1. Load Data
   └→ MNIST/Fashion-MNIST/CIFAR-10 (raw pixels)

2. Preprocess
   └→ PCA reduction (n_pca dimensions)
   └→ Normalize to [0,1]

3. Generate Encoding
   └→ Claude creates formula via prompt
   └→ Template: linear, polynomial, global_stats, or pca_mix

4. Validate Encoding
   └→ Check syntax & output range [0, 2π]

5. Build Quantum Circuit
   └→ Qiskit circuit with RY rotations
   └→ Entanglement layer (linear or full)

6. Compute Kernel
   └→ Quantum fidelity between data points

7. Train SVM
   └→ SVM on precomputed kernel matrix

8. Evaluate
   └→ Accuracy, F1, Precision, Recall
```

## Output

Results saved to `results/` as JSON:
```json
{
  "dataset": "mnist",
  "encoding": {
    "code": "0.5*np.mean(x) + 0.3*np.std(x)",
    "template_type": "global_stats",
    "is_valid": true
  },
  "metrics": {
    "accuracy": 0.8725,
    "f1_macro": 0.8610,
    "f1_weighted": 0.8710,
    "precision_macro": 0.8650,
    "recall_macro": 0.8600
  },
  "timing": 234.5
}
```

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Global configuration |
| `data/loader.py` | Dataset loading |
| `data/preprocessor.py` | PCA + normalization |
| `llm/claude_interface.py` | Claude API calls |
| `encoding/prompt_builder.py` | Prompt generation |
| `encoding/templates.py` | Template definitions |
| `encoding/validator.py` | Encoding validation |
| `quantum/qiskit_circuit.py` | Qiskit circuit builder |
| `quantum/qiskit_kernel.py` | Kernel computation |
| `evaluation/svm_trainer.py` | SVM training & evaluation |
| `experiments/run_single_dataset.py` | Single dataset pipeline |
| `experiments/run_all_datasets.py` | Multi-dataset runner |
| `main.py` | CLI entry point |

## References

- **Claude API**: https://docs.anthropic.com/
- **Qiskit**: https://qiskit.org/
- **Quantum Kernels**: https://qiskit-machine-learning.readthedocs.io/

---

**Created**: December 26, 2025  
**Architecture**: 5-Level design with clear separation of concerns  
**LLM**: Claude (Anthropic)  
**Quantum**: Qiskit
