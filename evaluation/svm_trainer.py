"""
Level 5: SVM Trainer & Evaluation
Train SVM classifier and compute metrics
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict


class QuantumSVMTrainer:
    """Train SVM on quantum kernel matrix"""
    
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.svm = None
    
    def train(self, K_train: np.ndarray, y_train: np.ndarray):
        """
        Train SVM on kernel matrix
        
        Args:
            K_train: Kernel matrix (n_samples, n_samples)
            y_train: Training labels
        """
        self.svm = SVC(kernel='precomputed', C=self.C, random_state=self.random_state)
        self.svm.fit(K_train, y_train)
    
    def evaluate(self, K_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate SVM on test kernel matrix
        
        Returns:
            Dictionary with metrics
        """
        if self.svm is None:
            raise ValueError("SVM not trained. Call train() first.")
        
        y_pred = self.svm.predict(K_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        }
        
        return metrics
