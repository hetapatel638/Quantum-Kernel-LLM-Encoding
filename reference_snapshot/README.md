# LLM-Guided Nonlinear Encoding for Quantum Kernel Methods

This repository contains the reference implementation for the paper:

**"LLM-Guided Nonlinear Encoding in Quantum Kernel Methods"**

## Overview

We propose a hybrid quantum–classical image classification pipeline in which a large language model (LLM) synthesizes and iteratively refines nonlinear data encodings within a fixed, multi-layer data re-uploading quantum circuit.

The method is evaluated using quantum kernel estimation and a classical support vector machine.

## Key Features

- PCA-based dimensionality reduction (80 components)
- Fixed 10-qubit, 5-layer data re-uploading circuit
- Nonlinear LLM-generated encoding functions
- Controlled refinement loop (≤ 3 iterations)
- Quantum kernel estimation via Qiskit
- Classical SVM classifier

## Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10

Due to the quadratic scaling of quantum kernel evaluation, experiments are conducted on balanced subsets of each dataset, following standard practice in quantum kernel learning.

## Installation

```bash
pip install -r requirements.txt
```
