"""
Configuration for Quantum Machine Learning Framework
5-Level Architecture with Claude LLM Prompting + Qiskit Quantum
"""

CONFIG = {
    # Level 1: Data Configuration
    "datasets": {
        "mnist": {
            "n_classes": 10,
            "n_features": 784,
            "pca_dims": [10, 40, 80],
            "n_train": 500,
            "n_test": 200
        },
        "fashion_mnist": {
            "n_classes": 10,
            "n_features": 784,
            "pca_dims": [10, 40, 80],
            "n_train": 500,
            "n_test": 200
        },
        "cifar10": {
            "n_classes": 10,
            "n_features": 3072,
            "pca_dims": [10, 40, 80],
            "n_train": 500,
            "n_test": 200
        }
    },
    
    # Level 2: LLM Prompting Configuration
    "llm": {
        "model": "claude-3-5-haiku-20241022",  # Fast Haiku model (available & cost-effective)
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.7,
        "max_tokens": 500,
        "fallback_models": ["claude-3-haiku-20240307"]  # Older Haiku as fallback
    },
    
    # Level 3: Encoding Templates
    "encoding": {
        "templates": ["linear", "polynomial", "global_stats", "pca_mix"],
        "validation": {
            "max_coefficients": 1.0,  # Sum of absolute coefficients
            "angle_range": [0, 6.283185307179586]  # [0, 2π]
        }
    },
    
    # Level 4: Qiskit Quantum Circuit
    "quantum": {
        "n_qubits": 10,
        "max_depth": 12,
        "entanglement": "linear",
        "backend": "qasm_simulator",
        "shots": 1024
    },
    
    # Level 5: SVM Evaluation
    "svm": {
        "C": 50.0,  # High C for better quantum kernel separation
        "kernel": "precomputed",
        "random_state": 42
    },
    
    # General
    "random_seed": 42,
    "results_dir": "results/",
    "verbose": True
}
