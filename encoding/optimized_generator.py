"""
Optimized Encoding Generator
Analyzes data to generate better quantum encodings
"""

import numpy as np


class OptimizedEncodingGenerator:
    """Generate optimized encodings based on data statistics"""
    
    @staticmethod
    def generate_optimized_encoding(X_train: np.ndarray, n_qubits: int = 10) -> str:
        """
        Analyze training data and generate optimized encoding
        Uses variance-based feature selection
        
        Args:
            X_train: Training data (n_samples, n_features)
            n_qubits: Number of qubits (default 10)
        
        Returns:
            Python code string for encoding
        """
        n_features = X_train.shape[1]
        
        # Compute feature statistics
        feature_variance = np.var(X_train, axis=0)
        feature_mean = np.mean(X_train, axis=0)
        
        # Normalize variance to [0.1, 0.5] for scaling
        var_normalized = feature_variance / (np.max(feature_variance) + 1e-8)
        var_normalized = 0.1 + 0.4 * var_normalized  # Range [0.1, 0.5]
        
        # Get top variance features
        top_indices = np.argsort(feature_variance)[::-1][:n_qubits]
        top_indices = sorted(top_indices)  # Keep order
        
        # Generate encoding code
        lines = ["np.array(["]
        
        for i in range(n_qubits):
            if i < len(top_indices):
                feat_idx = top_indices[i]
                coeff = var_normalized[feat_idx]
                
                # Create angle formula with better scaling
                # Use combination of feature + variance weighting
                if feat_idx < n_features - 1:
                    # Mix with next feature for diversity
                    next_feat = top_indices[i + 1] if i + 1 < len(top_indices) else feat_idx
                    angle_formula = f"    {coeff:.3f}*x[{feat_idx}]*np.pi + {coeff*0.5:.3f}*x[{next_feat}]*np.pi"
                else:
                    angle_formula = f"    {coeff:.3f}*x[{feat_idx}]*np.pi"
            else:
                # Fallback for remaining qubits
                angle_formula = f"    0.15*x[{i%n_features}]*np.pi"
            
            if i < n_qubits - 1:
                lines.append(angle_formula + ",")
            else:
                lines.append(angle_formula)
        
        lines.append("])")
        
        code = "\n".join(lines)
        return code
    
    @staticmethod
    def generate_simple_strong_encoding(X_train: np.ndarray) -> str:
        """
        Generate simple but strong encoding (proven to work)
        Uses feature scaling strategy
        """
        n_features = X_train.shape[1]
        
        # Compute feature scales
        x_min = np.min(X_train, axis=0)
        x_max = np.max(X_train, axis=0)
        x_range = x_max - x_min + 1e-8
        
        # Get feature importance (variance)
        feature_var = np.var(X_train, axis=0)
        var_weights = feature_var / (np.sum(feature_var) + 1e-8)
        
        lines = ["np.array(["]
        
        for i in range(10):
            feat_idx = i % n_features
            weight = var_weights[feat_idx]
            
            # Use variance-weighted scaling
            coeff = (0.2 + 0.3 * weight) * np.pi
            
            angle_formula = f"    {coeff:.3f}*(x[{feat_idx}] - {x_min[feat_idx]:.3f}) / {x_range[feat_idx]:.3f}"
            
            if i < 9:
                lines.append(angle_formula + ",")
            else:
                lines.append(angle_formula)
        
        lines.append("])")
        
        return "\n".join(lines)
