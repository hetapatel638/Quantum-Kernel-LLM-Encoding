"""
Level 1: Data Loader
Handles MNIST, Fashion-MNIST, CIFAR-10 loading
"""

import numpy as np
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from typing import Tuple


class DatasetLoader:
    """Load and return datasets in consistent format"""
    
    @staticmethod
    def load_dataset(dataset_name: str, n_train: int, n_test: int, 
                     random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
        """
        Load dataset: MNIST, Fashion-MNIST, or CIFAR-10
        
        Returns:
            (X_train, y_train, X_test, y_test) as flattened arrays
        """
        np.random.seed(random_seed)
        
        if dataset_name == "mnist":
            (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
        elif dataset_name == "fashion_mnist":
            (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()
        elif dataset_name == "cifar10":
            (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
            y_train_full = y_train_full.flatten()
            y_test_full = y_test_full.flatten()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Flatten images
        X_train_full = X_train_full.reshape(X_train_full.shape[0], -1).astype(np.float32)
        X_test_full = X_test_full.reshape(X_test_full.shape[0], -1).astype(np.float32)
        
        # Sample if needed
        if n_train < len(X_train_full):
            idx_train = np.random.choice(len(X_train_full), n_train, replace=False)
            X_train = X_train_full[idx_train]
            y_train = y_train_full[idx_train]
        else:
            X_train, y_train = X_train_full, y_train_full
        
        if n_test < len(X_test_full):
            idx_test = np.random.choice(len(X_test_full), n_test, replace=False)
            X_test = X_test_full[idx_test]
            y_test = y_test_full[idx_test]
        else:
            X_test, y_test = X_test_full, y_test_full
        
        return X_train, y_train, X_test, y_test
