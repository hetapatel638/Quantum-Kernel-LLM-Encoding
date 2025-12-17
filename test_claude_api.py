"""
Direct test of Claude API for quantum encoding generation.

This tests the actual pipeline:
1. Claude generates encodings
2. Baseline computes accuracy
3. Compare improvement
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from data.loader import DatasetLoader
from llm.hf_interface import LLMInterface


def test_claude_api():
    """Test that Claude API is available and working"""
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Please set: export ANTHROPIC_API_KEY='your_key_here'")
        return False
    
    print("TESTING CLAUDE API")
    print()
    print(f"API Key found: {api_key[:20]}...")
    print()
    
    try:
        llm = LLMInterface()
        
        if not llm.use_claude:
            print("WARNING: Claude not available, using mock")
            return False
        
        print("Claude API is available")
        print()
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def normalize_to_01(X_train, X_test):
    """Normalize to [0,1]"""
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)
    return np.clip(X_train_norm, 0, 1), np.clip(X_test_norm, 0, 1)


def baseline_kernel(X1, X2=None):
    """Baseline: theta_i = pi * x_i"""
    if X2 is None:
        X2 = X1
    
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            angles1 = np.pi * X1[i]
            angles2 = np.pi * X2[j]
            angle_diffs = angles1 - angles2
            K[i, j] = np.prod(np.cos(angle_diffs / 2) ** 2)
    
    return K


def claude_kernel(encoding_func, X1, X2=None):
    """Compute kernel using Claude-generated encoding"""
    if X2 is None:
        X2 = X1
    
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            try:
                namespace = {
                    'np': np,
                    'x': X1[i],
                    'range': range,
                    'len': len,
                    'min': min,
                    'max': max,
                    'mean': np.mean,
                    'std': np.std,
                    'sum': np.sum
                }
                angles1 = eval(encoding_func, {'__builtins__': {}}, namespace)
                if not isinstance(angles1, list):
                    angles1 = [angles1]
                
                namespace['x'] = X2[j]
                angles2 = eval(encoding_func, {'__builtins__': {}}, namespace)
                if not isinstance(angles2, list):
                    angles2 = [angles2]
                
                # Pad to same length
                max_len = max(len(angles1), len(angles2))
                angles1 = np.array(angles1 + [0] * (max_len - len(angles1)))
                angles2 = np.array(angles2 + [0] * (max_len - len(angles2)))
                
                angle_diffs = angles1 - angles2
                K[i, j] = np.prod(np.cos(angle_diffs / 2) ** 2)
            except Exception as e:
                K[i, j] = 0.0
    
    return K


def main():
    """Run full Claude API test"""
    
    print("CLAUDE API QUANTUM ENCODING TEST")
    print()
    
    if not test_claude_api():
        print("Cannot proceed without Claude API")
        return
    
    print()
    np.random.seed(42)
    
    n_train = 120
    n_test = 40
    pca_dims = 20
    
    print("Loading MNIST data...")
    loader = DatasetLoader()
    X_train_raw, X_test_raw, y_train, y_test = loader.load_dataset("mnist", n_train, n_test)
    
    print("Applying PCA...")
    pca = PCA(n_components=pca_dims)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)
    
    X_train, X_test = normalize_to_01(X_train_pca, X_test_pca)
    
    print(f"Data ready: train {X_train.shape}, test {X_test.shape}")
    print()
    
    print("TEST 1: BASELINE KERNEL")
    print()
    
    K_train_baseline = baseline_kernel(X_train)
    K_test_baseline = baseline_kernel(X_test, X_train)
    
    svm_baseline = SVC(kernel="precomputed", C=1.0)
    svm_baseline.fit(K_train_baseline, y_train)
    y_pred_baseline = svm_baseline.predict(K_test_baseline)
    
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print()
    
    print("TEST 2: CLAUDE-GENERATED ENCODING")
    print()
    
    prompt = f"""Generate a quantum encoding function for MNIST classification with {pca_dims} features.

The function should:
1. Take input array x of length {pca_dims}
2. Return a list/array of {pca_dims} angles in range [0, 2*pi]
3. Perform better than baseline theta_i = pi * x[i] (baseline accuracy: {baseline_acc:.2%})

Requirements:
- Output as Python code: [expression for i in range({pca_dims})]
- All angles must be clipped to [0, 2*pi]
- Ensure high diversity across angles

Return JSON:
{{
    "function": "code here",
    "explanation": "how it works"
}}
"""
    
    print("Querying Claude API...")
    llm = LLMInterface()
    
    if not llm.use_claude:
        print("Claude not available, skipping")
        return
    
    response = llm.generate(prompt, temperature=0.95)
    
    try:
        import json
        parsed = json.loads(response)
        encoding_func = parsed.get("function")
        explanation = parsed.get("explanation")
    except:
        print("Failed to parse response")
        print(response)
        return
    
    print(f"Claude generated function:")
    print(f"  {encoding_func[:100]}...")
    print()
    print(f"Explanation: {explanation[:200]}...")
    print()
    
    print("Evaluating Claude encoding...")
    
    try:
        K_train_claude = claude_kernel(encoding_func, X_train)
        K_test_claude = claude_kernel(encoding_func, X_test, X_train)
        
        svm_claude = SVC(kernel="precomputed", C=1.0)
        svm_claude.fit(K_train_claude, y_train)
        y_pred_claude = svm_claude.predict(K_test_claude)
        
        claude_acc = accuracy_score(y_test, y_pred_claude)
        
        print(f"Claude Accuracy: {claude_acc:.4f}")
        print()
    except Exception as e:
        print(f"Error evaluating Claude encoding: {e}")
        return
    
    print("COMPARISON")
    print()
    print(f"Baseline: {baseline_acc:.4f}")
    print(f"Claude:   {claude_acc:.4f}")
    print(f"Diff:     {claude_acc - baseline_acc:+.4f}")
    improvement = ((claude_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    print(f"Improvement: {improvement:+.2f}%")
    print()
    
    if claude_acc > baseline_acc:
        print("SUCCESS: Claude encoding outperformed baseline!")
    else:
        print("Note: Claude encoding did not improve over baseline yet")
        print("This is expected - quantum encoding is challenging")
        print("Key is that Claude API is working and generating valid functions")


if __name__ == "__main__":
    main()
