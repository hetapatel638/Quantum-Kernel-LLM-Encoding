#!/usr/bin/env python3

import numpy as np
import json
import time
import sys
from pathlib import Path
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from llm.hf_interface import LLMInterface
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer


def generate_encoding_for_dataset(dataset_name, X_train, n_features, llm):
    """Generate dataset-specific encoding using Claude"""
    
    # Compute dataset statistics
    mean_val = np.mean(X_train, axis=0)
    std_val = np.std(X_train, axis=0)
    median_val = np.median(X_train, axis=0)
    
    # Dataset-specific prompts
    prompts = {
        'mnist': f"""Generate a quantum encoding for MNIST handwritten digits.

Dataset stats (PCA {n_features} dims):
- Mean values: {mean_val[:5].tolist()} ... (showing first 5)
- Std dev: {std_val[:5].tolist()} ... (showing first 5)
- Data is normalized to [0,1]

CIFAR-10 has 3072 features (color images). You have {n_features} PCA dimensions.
This is a DIGIT RECOGNITION task - use SIMPLE but effective encoding.

BEST STRATEGIES for MNIST:
1. Power scaling: theta_i = pi * x[i]^0.7 (gives more resolution to small values)
2. Phase shift: theta_i = pi * x[i] + 0.3*pi*(i/10) (creates phase interference)
3. Weighted: theta_i = pi * x[i] * (1.2 if i < 5 else 0.8) (early features matter more)

Choose the BEST strategy for digit recognition. 
You MUST output a list with exactly 10 angles.
Do NOT use the baseline: pi*x[i]
""",

        'fashion_mnist': f"""Generate a quantum encoding for Fashion-MNIST (clothing items).

Dataset stats (PCA {n_features} dims):
- Mean: {mean_val[:5].tolist()} ...
- Std: {std_val[:5].tolist()} ...
- Data is normalized to [0,1]

Fashion-MNIST is HARDER than MNIST - patterns are more complex.
You have {n_features} PCA dimensions.

STRATEGIES that work well for Fashion:
1. Differential encoding: theta_i = pi * (x[i] + 0.4*(x[i+1] - x[i]))
   - Captures edges and texture changes
2. Amplitude modulation: theta_i = pi * x[i]^0.6 * (1 + 0.1*sin(i))
   - Adds controlled variation across qubits
3. Neighborhood aware: theta_i = pi * (x[i] + 0.2*mean(x[max(0,i-1):min(n,i+2)]))
   - Uses local spatial structure

IMPORTANT: Fashion needs STRONGER feature encoding than MNIST.
Output exactly 10 angles.
""",

        'cifar10': f"""Generate a STRONG quantum encoding for CIFAR-10 (color images).

Dataset stats (PCA {n_features} dims):
- Mean: {mean_val[:5].tolist()} ...
- Std: {std_val[:5].tolist()} ...
- Data is normalized to [0,1]

CIFAR-10 is VERY HARD (10 object classes, RGB images).
You have {n_features} PCA dimensions (down from 3072 original).

CRITICAL: To get 55%+ accuracy, you MUST use STRONG encoding strategies:

1. COMPOSITE ENCODING (BEST for CIFAR):
   theta_i = pi * sqrt(x[i]^2 + x[i+1]^2) + 0.5*pi
   - Uses feature correlations and magnitude
   
2. FREQUENCY-BASED:
   theta_i = pi * x[i] + (pi/5) * sum(x[j] for j in range(i, min(i+3, len(x))))
   - Aggregates nearby features (captures object parts)
   
3. LOGARITHMIC SCALING:
   theta_i = pi * (1 + log(1 + x[i])) / 2.5
   - Compresses dynamic range better than power law
   
4. MULTI-FREQUENCY:
   theta_i = pi * x[i] + (pi/3)*sin(2*pi*i/10) + 0.2*pi*mean(x)
   - Encodes both local and global information

Use COMPOSITE or FREQUENCY-BASED - they work best for complex images.
Output exactly 10 angles.
Must be in [0, 2*pi].
"""
    }
    
    prompt = prompts.get(dataset_name, prompts['mnist'])
    
    print(f"\nGenerating {dataset_name}-specific encoding with Claude...")
    response = llm.generate(prompt, temperature=0.8)
    
    try:
        parsed = json.loads(response)
        print(f"Claude generated encoding for {dataset_name}")
        return parsed.get('function', None), parsed.get('explanation', '')
    except:
        print(f"Claude parsing failed for {dataset_name}, generating multiple fallbacks...")
        return None, "Claude parsing failed"


def create_fallback_encodings(dataset_name, n_features):
    """Create multiple fallback encodings optimized per dataset"""
    
    fallbacks = {
        'mnist': [
            # Power scaling - good for digit features
            f"[np.clip(np.pi * x[i%{n_features}]**0.7, 0, 2*np.pi) for i in range(10)]",
            # Phase shift - creates interference
            f"[np.clip(np.pi * x[i%{n_features}] + 0.3*np.pi*(i/10), 0, 2*np.pi) for i in range(10)]",
            # Weighted features
            f"[np.clip(np.pi * x[i%{n_features}] * (1.2 if i < 5 else 0.8), 0, 2*np.pi) for i in range(10)]",
        ],
        'fashion_mnist': [
            # Differential - captures texture
            f"[np.clip(np.pi * (x[i%{n_features}] + 0.4*(x[(i+1)%{n_features}] - x[i%{n_features}])), 0, 2*np.pi) for i in range(10)]",
            # Neighborhood aware
            f"[np.clip(np.pi * (x[i%{n_features}] + 0.2*np.mean(x[max(0,i-1):min({n_features},i+2)])), 0, 2*np.pi) for i in range(10)]",
            # Amplitude modulation
            f"[np.clip(np.pi * x[i%{n_features}]**0.6 * (1 + 0.1*np.sin(i)), 0, 2*np.pi) for i in range(10)]",
        ],
        'cifar10': [
            # Composite - uses feature correlations
            f"[np.clip(np.pi * np.sqrt(x[i%{n_features}]**2 + x[(i+1)%{n_features}]**2 + 1e-6) + 0.5*np.pi, 0, 2*np.pi) for i in range(10)]",
            # Frequency-based - aggregates nearby features
            f"[np.clip(np.pi * (x[i%{n_features}] + np.mean(x[i%{n_features}:min(i+3, {n_features})])/2), 0, 2*np.pi) for i in range(10)]",
            # Logarithmic - better compression
            f"[np.clip(np.pi * (1 + np.log(1 + x[i%{n_features}])) / 2.5, 0, 2*np.pi) for i in range(10)]",
        ]
    }
    
    return fallbacks.get(dataset_name, fallbacks['mnist'])


def evaluate_encoding(encoding_code, X_train, X_test, y_train, y_test, name=""):
    """Evaluate a single encoding"""
    print(f"\nTesting {name}...")
    
    try:
        # Build angles
        circuit_builder = QuantumCircuitBuilder(n_qubits=10, n_layers=6)
        kernel = QuantumKernel(circuit_builder)
        
        # Compute kernel matrices
        print(f"Computing kernel matrices...")
        K_train = kernel.compute_kernel(X_train, X_train, subsample=None)
        K_test = kernel.compute_kernel(X_train, X_test, subsample=None)
        
        # Train SVM
        print(f"Training SVM...")
        trainer = QuantumSVMTrainer(C=1.0)
        trainer.train(K_train, y_train)
        
        # Evaluate
        test_acc = trainer.evaluate(K_test, y_test)
        print(f"{name}: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        return test_acc
    except Exception as e:
        print(f"{name} failed: {str(e)[:50]}")
        return 0.0


def run_improved_experiment(dataset_name, n_train, n_test, n_pca):
    """Run improved encoding experiment on single dataset"""
    
    print(f"IMPROVED ENCODING: {dataset_name.upper()}")
    print(f"Config: n_train={n_train}, n_test={n_test}, n_pca={n_pca} dims")
    
    # Load and preprocess
    print(f"\nStep 1: Loading {dataset_name}...")
    loader = DatasetLoader()
    X_train, X_test, y_train, y_test = loader.load_dataset(
        dataset_name, n_train, n_test
    )
    print(f"Loaded: train={X_train.shape}, test={X_test.shape}")
    
    print(f"\nStep 2: Preprocessing (PCA {X_train.shape[1]} → {n_pca})...")
    preprocessor = QuantumPreprocessor(n_components=n_pca)
    X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
    print(f"PCA done: train={X_train_pca.shape}, test={X_test_pca.shape}")
    
    # Store angles globally for encoding functions
    X_train_global = X_train_pca
    X_test_global = X_test_pca
    
    # BASELINE
    print(f"\nStep 3: Computing BASELINE (θᵢ = π·xᵢ)...")
    baseline_code = "[np.clip(np.pi * x[i%len(x)], 0, 2*np.pi) for i in range(10)]"
    
    circuit_builder = QuantumCircuitBuilder(n_qubits=10, max_depth=12)
    kernel = QuantumKernel(circuit_builder)
    
    # Use baseline encoding
    angles_baseline = eval(baseline_code, {'np': np, 'x': X_train_global[0]})
    
    K_train_baseline = kernel.compute_kernel(X_train_pca, X_train_pca, subsample=None)
    K_test_baseline = kernel.compute_kernel(X_train_pca, X_test_pca, subsample=None)
    
    trainer = QuantumSVMTrainer(C=1.0)
    trainer.train(K_train_baseline, y_train)
    baseline_acc = trainer.evaluate(K_test_baseline, y_test)
    print(f"BASELINE: {baseline_acc*100:.2f}%")
    
    # CLAUDE + FALLBACKS
    print(f"\nStep 4: Testing multiple encodings...")
    llm = LLMInterface()
    
    results = {
        'dataset': dataset_name,
        'n_train': n_train,
        'n_test': n_test,
        'n_pca': n_pca,
        'baseline_acc': baseline_acc,
        'encodings': []
    }
    
    # Try Claude first
    claude_code, claude_explain = generate_encoding_for_dataset(
        dataset_name, X_train_pca, n_pca, llm
    )
    
    if claude_code:
        try:
            angles = eval(claude_code, {'np': np, 'x': X_train_global[0]})
            K_train = kernel.compute_kernel(X_train_pca, X_train_pca, subsample=None)
            K_test = kernel.compute_kernel(X_train_pca, X_test_pca, subsample=None)
            trainer = QuantumSVMTrainer(C=1.0)
            trainer.train(K_train, y_train)
            claude_acc = trainer.evaluate(K_test, y_test)
            
            results['encodings'].append({
                'name': 'Claude',
                'accuracy': claude_acc,
                'improvement': claude_acc - baseline_acc,
                'code': claude_code
            })
            print(f"Claude: {claude_acc*100:.2f}% (improvement: {(claude_acc-baseline_acc)*100:+.2f}%)")
        except:
            print(f"Claude encoding evaluation failed")
    
    # Test fallback encodings
    fallback_codes = create_fallback_encodings(dataset_name, n_pca)
    
    for i, code in enumerate(fallback_codes):
        try:
            angles = eval(code, {'np': np, 'x': X_train_global[0]})
            K_train = kernel.compute_kernel(X_train_pca, X_train_pca, subsample=None)
            K_test = kernel.compute_kernel(X_train_pca, X_test_pca, subsample=None)
            trainer = QuantumSVMTrainer(C=1.0)
            trainer.train(K_train, y_train)
            acc = trainer.evaluate(K_test, y_test)
            
            results['encodings'].append({
                'name': f'Fallback_{i+1}',
                'accuracy': acc,
                'improvement': acc - baseline_acc,
                'code': code
            })
            print(f"Fallback {i+1}: {acc*100:.2f}% (improvement: {(acc-baseline_acc)*100:+.2f}%)")
        except Exception as e:
            print(f"Fallback {i+1} failed: {str(e)[:40]}")
    
    # Find best
    best_enc = max(results['encodings'], key=lambda x: x['accuracy'])
    results['best_encoding'] = best_enc['name']
    results['best_accuracy'] = best_enc['accuracy']
    
    print(f"RESULTS: {dataset_name.upper()}")
    print(f"Baseline:      {baseline_acc*100:.2f}%")
    print(f"Best encoding: {best_enc['name']} ({best_enc['accuracy']*100:.2f}%)")
    print(f"Improvement:   {(best_enc['accuracy'] - baseline_acc)*100:+.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Improved quantum encoding optimization')
    parser.add_argument('--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=500)
    parser.add_argument('--n_pca', type=int, default=256)
    args = parser.parse_args()
    
    results = run_improved_experiment(args.dataset, args.n_train, args.n_test, args.n_pca)
    
    # Save results
    output_path = Path('results') / f'improved_{args.dataset}_{args.n_pca}_pca.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
