
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple


class QuantumPreprocessor:
    """Prepare data for quantum encoding"""
    
    def __init__(self, n_components: int = 80):
        self.n_components = n_components
        self.pca = None
        self.feature_min = None
        self.feature_max = None
    
    def fit_transform(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """
        Apply PCA and normalize to [0, 1]
        
        Args:
            X_train: Training data (n_samples, original_dim)
            X_test: Test data (n_samples, original_dim)
        
        Returns:
            X_train_processed, X_test_processed
        """
        print(f"Applying PCA: {X_train.shape[1]} → {self.n_components} dims")
        
        # Fit PCA on training data
        self.pca = PCA(n_components=self.n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        # Report variance explained
        var_explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA variance explained: {var_explained:.2%}")
        
        # Normalize to [0, 1] based on training data range
        self.feature_min = X_train_pca.min(axis=0)
        self.feature_max = X_train_pca.max(axis=0)
        
        X_train_norm = (X_train_pca - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        X_test_norm = (X_test_pca - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        
        # Clip to [0, 1] to handle outliers
        X_train_norm = np.clip(X_train_norm, 0, 1)
        X_test_norm = np.clip(X_test_norm, 0, 1)
        
        print(f"Normalized to [0, 1]: train={X_train_norm.shape}, test={X_test_norm.shape}")
        
        return X_train_norm, X_test_norm
    
    def get_stats(self, X: np.ndarray) -> dict:
        """Get dataset statistics for LLM prompts"""
        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "mean_value": float(np.mean(X)),
            "std_value": float(np.std(X)),
            "min_value": float(np.min(X)),
            "max_value": float(np.max(X)),
            "variance_per_feature": np.var(X, axis=0).tolist()[:5]  # First 5 for prompt
        }


# Test
if __name__ == "__main__":
    from data.loader import DatasetLoader
    
    loader = DatasetLoader()
    X_train, X_test, y_train, y_test = loader.load_dataset("mnist", 1000, 200)
    
    preprocessor = QuantumPreprocessor(n_components=10)
    X_train_proc, X_test_proc = preprocessor.fit_transform(X_train, X_test)
    
    stats = preprocessor.get_stats(X_train_proc)
    print("Dataset stats:", stats)