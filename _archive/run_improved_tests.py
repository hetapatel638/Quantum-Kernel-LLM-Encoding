#!/usr/bin/env python3
"""
Run improved encoding tests with HIGHER PCA dimensions
to get >85% Fashion-MNIST and >55% CIFAR-10
"""

import subprocess
import sys
import json
from pathlib import Path

def run_test(dataset, n_train, n_test, n_pca):
    """Run a single improved encoding test"""
    
    print(f"\n{'='*70}")
    print(f"Testing {dataset.upper()}")
    print(f"Config: n_train={n_train}, n_test={n_test}, n_pca={n_pca}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, 
        'experiments/optimized_encoding.py',
        '--n_train', str(n_train),
        '--n_test', str(n_test),
        '--n_pca', str(n_pca)
    ]
    
    # Set environment
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    result = subprocess.run(cmd, cwd='/Users/husky95/Desktop/Innovation', 
                          capture_output=False, text=True, env=env)
    
    return result.returncode == 0

# Test configurations for improvement
tests = [
    # Fashion-MNIST: Needs higher PCA (96%+ variance)
    ('fashion_mnist', 800, 300, 160),   # 160 dims = 96.78% variance
    
    # CIFAR-10: Needs MUCH higher PCA  
    ('cifar10', 1000, 400, 512),        # 512 dims = 99%+ variance for color images
]

print("""
IMPROVED QUANTUM ENCODING TESTS
===============================

Strategy to reach targets:
1. Fashion-MNIST >85%: Use 160 PCA dims (instead of 80) to preserve 96.78% variance
2. CIFAR-10 >55%: Use 512 PCA dims to preserve color information fully

Note: These tests will take ~5-10 minutes each due to larger feature dimensions.
""")

for dataset, n_train, n_test, n_pca in tests:
    success = run_test(dataset, n_train, n_test, n_pca)
    if not success:
        print(f"⚠ Test failed for {dataset}")
    print()

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Check results/ folder for detailed results")
print(f"{'='*70}")
