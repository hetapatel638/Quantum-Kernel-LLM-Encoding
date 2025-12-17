CONFIG = {
    # Datasets
    "datasets": {
        "mnist": {
            "name": "MNIST",
            "n_train": 10000,
            "n_test": 10000,
            "original_dim": 784,
            "pca_range": [10, 40, 80],  # Try multiple
            "characteristics": "stroke patterns, local correlations"
        },
        "fashion_mnist": {
            "name": "Fashion-MNIST", 
            "n_train": 10000,
            "n_test": 10000,
            "original_dim": 784,
            "pca_range": [10, 40, 80],
            "characteristics": "texture, global structure, edges"
        },
        "cifar10": {
            "name": "CIFAR-10",
            "n_train": 10000,
            "n_test": 10000,
            "original_dim": 3072,  # 32x32x3
            "pca_range": [10, 40, 80],
            "characteristics": "color channels, multi-scale features"
        }
    },
    
    # Baseline reference results (from Sakka et al. Table 1)
    "reference_results": {
        "mnist": {
            "rbf_kernel": 0.9765,
            "polynomial_kernel": 0.9667,
            "linear_kernel": 0.9385,
            "yzcx_quantum": 0.9727,
            "npqc_quantum": 0.9644,
            "zz_quantum": 0.9255,
            "sakka_generated": 0.9731
        },
        "fashion_mnist": {
            "rbf_kernel": 0.8864,
            "polynomial_kernel": 0.8702,
            "linear_kernel": 0.8437,
            "yzcx_quantum": 0.8778,
            "npqc_quantum": 0.8749,
            "zz_quantum": 0.8252,
            "sakka_generated": 0.8835
        },
        "cifar10": {
            "rbf_kernel": 0.5669,
            "polynomial_kernel": 0.5375,
            "linear_kernel": 0.4087,
            "npqc_quantum": 0.4903,
            "yzcx_quantum": 0.4753,
            "zz_quantum": 0.3907,
            "sakka_generated": 0.5290
        }
    },
    
    # LLM Configuration
    "llm": {
        "model_name": "google-t5/t5-small",  # Standard T5-small from HuggingFace
        "use_local": True,  # Run locally on MacBook/Colab
        "temperature": 0.7,
        "max_new_tokens": 256,
        "device": "auto"  # Auto-detect CPU/GPU
    },
    
    # Template Families
    "templates": {
        "linear": {
            "description": "θᵢ = Σ αⱼxⱼ",
            "constraint": "sum of |α| ≤ 1",
            "best_for": ["mnist", "fashion_mnist"]
        },
        "polynomial": {
            "description": "θᵢ = Σ αⱼxⱼ + Σ βⱼₖxⱼxₖ",
            "degree": 2,
            "best_for": ["mnist"]
        },
        "global_stats": {
            "description": "θᵢ = δ·mean(x) + ε·std(x)",
            "best_for": ["fashion_mnist", "cifar10"]
        },
        "pca_mix": {
            "description": "θᵢ = Σ ωⱼ·PCⱼ",
            "max_components": 4,
            "best_for": ["cifar10"]
        }
    },
    
    # Quantum Circuit
    "quantum": {
        "n_qubits": 10,
        "max_depth": 12,
        "max_gates": 50,
        "backend": "default.qubit",
        "entanglement": "linear"  # or None for rotation-only
    },
    
    # Evaluation
    "evaluation": {
        "svm_C": 1.0,
        "svm_gamma": "scale",
        "cv_folds": 5,
        "random_seed": 42
    },
    
    # Iteration
    "refinement": {
        "max_iterations": 5,
        "convergence_threshold": 0.01  # Stop if improvement < 1%
    }
}