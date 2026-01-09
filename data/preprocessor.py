"""
Level 1: Data Preprocessor
PCA reduction + Normalization to [0,1]
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple


class QuantumPreprocessor:
    """Preprocess data: PCA reduction + normalization to [0,1]"""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = None
        self.data_min = None
        self.data_max = None
    
    def fit_transform(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA on training data, apply to both train & test
        Normalize to [0,1]
        """
        # Apply PCA
        self.pca = PCA(n_components=self.n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        # Normalize to [0,1]
        self.data_min = X_train_pca.min(axis=0)
        self.data_max = X_train_pca.max(axis=0)
        
        # Avoid division by zero
        self.data_max[self.data_max == self.data_min] += 1.0
        
        X_train_norm = (X_train_pca - self.data_min) / (self.data_max - self.data_min)
        X_test_norm = (X_test_pca - self.data_min) / (self.data_max - self.data_min)
        
        # Clip to [0,1]
        X_train_norm = np.clip(X_train_norm, 0, 1)
        X_test_norm = np.clip(X_test_norm, 0, 1)
        
        return X_train_norm, X_test_norm
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted PCA and normalization"""
        if self.pca is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        X_pca = self.pca.transform(X)
        X_norm = (X_pca - self.data_min) / (self.data_max - self.data_min)
        X_norm = np.clip(X_norm, 0, 1)
        return X_norm
