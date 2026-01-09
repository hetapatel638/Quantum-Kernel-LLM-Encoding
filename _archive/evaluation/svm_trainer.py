import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict


class QuantumSVMTrainer:
    """Train SVM classifier on precomputed quantum kernels"""
    
    def __init__(self, C: float = 1.0):
        """
        Initialize SVM trainer
        
        Args:
            C: Regularization parameter
        """
        self.C = C
        self.model = None
    
    def train(self, K_train: np.ndarray, y_train: np.ndarray):
        """
        Train SVM on kernel matrix
        
        Args:
            K_train: Precomputed kernel matrix (n_samples, n_samples)
            y_train: Training labels
        """
        print(f"Training SVM with C={self.C}...")
        
        self.model = SVC(kernel='precomputed', C=self.C)
        self.model.fit(K_train, y_train)
        
        # Training accuracy
        train_pred = self.model.predict(K_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"Training accuracy: {train_acc:.4f}")
    
    def evaluate(self, K_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate on test set
        
        Args:
            K_test: Kernel matrix between train and test (n_train, n_test)
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predict
        y_pred = self.model.predict(K_test.T)  # Transpose for sklearn
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted')
        }
        
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1-score: {metrics['f1_score']:.4f}")
        
        return metrics


# Test
if __name__ == "__main__":
    # Generate dummy kernel matrix
    n_train = 100
    n_test = 20
    
    # Random kernel (for testing)
    K_train = np.random.rand(n_train, n_train)
    K_train = (K_train + K_train.T) / 2  # Make symmetric
    K_test = np.random.rand(n_train, n_test)
    
    y_train = np.random.randint(0, 2, n_train)
    y_test = np.random.randint(0, 2, n_test)
    
    # Train and evaluate
    trainer = QuantumSVMTrainer(C=1.0)
    trainer.train(K_train, y_train)
    metrics = trainer.evaluate(K_test, y_test)
    
    print("\nMetrics:", metrics)