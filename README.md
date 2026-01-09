# Hybrid Quantum–Classical Image Classification  
### LLM-Guided Nonlinear Encoding with Quantum Kernel Estimation

This repository contains a reference implementation of a **hybrid quantum–classical image classification pipeline** based on **quantum kernel methods** and **multi-layer data re-uploading circuits**.

The project explores how **nonlinear data encoding strategies**, including those generated using a **Large Language Model (LLM)**, can be integrated into quantum kernel estimation frameworks in a controlled and reproducible manner. The emphasis of this work is on **methodological design and feasibility**, rather than exhaustive performance benchmarking.


## Overview

Quantum kernel methods embed classical data into a high-dimensional quantum feature space using parameterized quantum circuits. The effectiveness of these methods depends critically on how classical features are encoded into circuit parameters.

This implementation combines:
- classical dimensionality reduction using PCA,
- structured nonlinear feature encoding,
- multi-layer data re-uploading quantum circuits, and
- kernel-based classification using a classical Support Vector Machine (SVM).

An optional LLM-based component is used to synthesize candidate nonlinear encoding functions under strict safety and reproducibility constraints.

## Key Characteristics

- Hybrid quantum–classical learning pipeline
- PCA-based preprocessing with feature normalization
- Fixed-architecture data re-uploading quantum circuit
- Fidelity-based quantum kernel estimation using Qiskit
- Optional LLM-guided nonlinear encoding with deterministic fallback
- Designed for clarity, reproducibility, and academic evaluation

## Pipeline Description

The implemented workflow consists of the following stages:

1. **Data Loading**  
   Image datasets (MNIST, Fashion-MNIST, or CIFAR-10) are loaded and flattened into classical feature vectors.

2. **Preprocessing**  
   Features are reduced using Principal Component Analysis (typically 80 components) and normalized to the range [0, 1].

3. **Encoding Generation**  
   Nonlinear mappings from classical features to circuit parameters are generated either:
   - via an LLM (if enabled), or
   - using a deterministic fallback encoder.

4. **Quantum Feature Map Construction**  
   A parameterized quantum circuit with 10 qubits is constructed using a multi-layer data re-uploading design with interleaved entangling operations.

5. **Quantum Kernel Estimation**  
   Pairwise fidelities between quantum states are computed to form a kernel (Gram) matrix.

6. **Classical Classification**  
   A Support Vector Machine is trained using the precomputed quantum kernel.

7. **Evaluation**  
   Classification accuracy and standard metrics are reported.

## Repository Structure

QML/
├── llm_qke_reuploading.ipynb # Main implementation (Google Colab notebook)
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore
└── _archive/ # Notes, drafts, and experimental material

yaml
Copy code

Only the main notebook is required to run the reference implementation.  
Archived files are retained for transparency but are not part of the core pipeline.

## Setup

### Recommended Environment
Google Colab (Python 3.10 or later)

### Install Dependencies
```bash
pip install -r requirements.txt
Optional: LLM API Key
```bash

### Copy code
export ANTHROPIC_API_KEY="your_api_key_here"
If no API key is provided, the pipeline automatically uses a deterministic nonlinear encoding strategy.

Running the Implementation
Open and execute the notebook:

### Copy code
llm_qke_reuploading.ipynb
The default configuration runs an experiment on the MNIST dataset using:
      80 PCA components,
      10 qubits,
      5 data re-uploading layers,
      fidelity-based quantum kernel estimation,
      SVM classification with a precomputed kernel.
Other datasets can be enabled by modifying the configuration cell within the notebook.

### Quantum Circuit Design
The quantum feature map uses a fixed circuit architecture to ensure reproducibility across experiments. Data re-uploading is implemented by alternating between data-dependent rotations and entangling operations over multiple layers.

This design increases the expressive capacity of the induced quantum kernel while maintaining controlled circuit depth and structure.

### LLM-Guided Encoding
When enabled, the LLM generates nonlinear expressions that map classical features to rotation angles. All generated expressions are:

restricted to a predefined set of mathematical operations,

validated for numerical stability,

cached to support reproducibility.

The LLM is used as a tool for structured hypothesis generation, rather than as an unconstrained optimizer.

### Experimental Scope
For initial validation, experiments are typically conducted on MNIST due to its computational efficiency. Support for Fashion-MNIST and CIFAR-10 is included, but these datasets are more computationally demanding.

The methodology is designed to scale to additional datasets in future work.

### References
Qiskit Machine Learning Documentation
https://qiskit-machine-learning.readthedocs.io/

Havlíček et al., Supervised learning with quantum-enhanced feature spaces, Nature (2019)

Schuld & Killoran, Quantum machine learning in feature Hilbert spaces (2019)

Anthropic Claude API Documentation
https://docs.anthropic.com/
