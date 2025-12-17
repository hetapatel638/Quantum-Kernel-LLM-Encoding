import numpy as np
from sklearn.datasets import fetch_openml
from tensorflow.keras.datasets import cifar10
from typing import Tuple


class DatasetLoader:
    """Load and prepare datasets for quantum encoding experiments"""
    
    @staticmethod
    def load_mnist(n_train: int = 10000, n_test: int = 10000) -> Tuple:
        """Load MNIST dataset"""
        print("Loading MNIST...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
        
        # Convert to numpy and normalize
        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.int32)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        # Split
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:n_train + n_test]
        y_test = y[n_train:n_train + n_test]
        
        print(f"MNIST loaded: train={X_train.shape}, test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_fashion_mnist(n_train: int = 10000, n_test: int = 10000) -> Tuple:
        """Load Fashion-MNIST dataset"""
        print("Loading Fashion-MNIST...")
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, parser='auto')
        
        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.int32)
        
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:n_train + n_test]
        y_test = y[n_train:n_train + n_test]
        
        print(f"Fashion-MNIST loaded: train={X_train.shape}, test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_cifar10(n_train: int = 10000, n_test: int = 10000) -> Tuple:
        """Load CIFAR-10 dataset"""
        print("Loading CIFAR-10...")
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # Flatten images and normalize
        X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        # Subsample
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]
        
        print(f"CIFAR-10 loaded: train={X_train.shape}, test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    @classmethod
    def load_dataset(cls, dataset_name: str, n_train: int = 10000, n_test: int = 10000):
        """Universal loader"""
        loaders = {
            "mnist": cls.load_mnist,
            "fashion_mnist": cls.load_fashion_mnist,
            "cifar10": cls.load_cifar10
        }
        
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return loaders[dataset_name](n_train, n_test)


# Test
if __name__ == "__main__":
    loader = DatasetLoader()
    X_train, X_test, y_train, y_test = loader.load_dataset("mnist", 1000, 200)
    print(f"Test passed: {X_train.shape}, {y_train.shape}")