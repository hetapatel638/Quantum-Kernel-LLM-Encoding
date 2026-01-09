"""
Level 2: Prompt Builder
Create intelligent prompts with dataset statistics for encoding generation
"""

import numpy as np
from typing import Dict, Any


class PromptBuilder:
    """Build prompts for Claude to generate quantum encodings"""
    
    @staticmethod
    def build_encoding_prompt(dataset_name: str, X_train: np.ndarray, 
                            n_pca: int, template_type: str) -> str:
        """
        Create a prompt with dataset statistics to guide Claude
        
        Args:
            dataset_name: Name of dataset (mnist, fashion_mnist, cifar10)
            X_train: Training data (normalized to [0,1])
            n_pca: Number of PCA dimensions
            template_type: Type of encoding (linear, polynomial, global_stats, pca_mix)
        
        Returns:
            Prompt string for Claude
        """
        
        # Compute dataset statistics
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        stats = {
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "mean_val": float(np.mean(mean)),
            "std_val": float(np.mean(std)),
            "min_val": float(X_train.min()),
            "max_val": float(X_train.max())
        }
        
        prompts = {
            "linear": PromptBuilder._linear_prompt(dataset_name, stats, n_pca),
            "polynomial": PromptBuilder._polynomial_prompt(dataset_name, stats, n_pca),
            "global_stats": PromptBuilder._global_stats_prompt(dataset_name, stats, n_pca),
            "pca_mix": PromptBuilder._pca_mix_prompt(dataset_name, stats, n_pca)
        }
        
        return prompts.get(template_type, prompts["linear"])
    
    @staticmethod
    def _linear_prompt(dataset_name: str, stats: Dict[str, Any], n_pca: int) -> str:
        return f"""
Generate Python code for quantum encoding.

Input: x (array of {n_pca} normalized features in [0,1])
Output: Array of 10 angles in [0, 2π]

Format: np.array([angle1, angle2, ..., angle10])

Simple strategy:
1. Use 2-3 features per angle
2. Scale each by 0.1-0.3
3. Multiply by π to get [0, 3π] range
4. Use np.clip to enforce [0, 2π]

Example:
np.array([
    0.2*x[0]*np.pi,
    0.3*x[1]*np.pi,
    0.2*x[2]*np.pi,
    0.3*x[3]*np.pi,
    0.2*x[4]*np.pi,
    0.3*x[5]*np.pi,
    0.2*x[6]*np.pi,
    0.3*x[7]*np.pi,
    0.2*x[8]*np.pi,
    0.3*x[9]*np.pi
])

Rules:
- Return valid Python expression only
- Must return exactly 10 angles
- All angles in [0, 2π]
- No comments or explanation
"""
    
    @staticmethod
    def _polynomial_prompt(dataset_name: str, stats: Dict[str, Any], n_pca: int) -> str:
        return f"""
Generate a POLYNOMIAL quantum encoding (degree 2) for {dataset_name} classification.

Dataset Statistics:
- Samples: {stats['n_samples']}
- Features (PCA dims): {n_pca}
- Data range: [{stats['min_val']:.3f}, {stats['max_val']:.3f}]

Return ONLY a Python expression that:
1. Takes input `x` (numpy array)
2. Returns angles in [0, 2π]
3. Uses format: theta_i = Σ(α_j * x_j) + Σ(β_jk * x_j * x_k)

Constraints:
- Include interaction terms (x_j * x_k)
- Sum of absolute coefficients ≤ 1.0
- All angles in [0, 2π]
- Keep degree ≤ 2

OUTPUT ONLY the Python expression.
"""
    
    @staticmethod
    def _global_stats_prompt(dataset_name: str, stats: Dict[str, Any], n_pca: int) -> str:
        return f"""
Generate a GLOBAL STATISTICS quantum encoding for {dataset_name} classification.

Dataset Statistics:
- Samples: {stats['n_samples']}
- Features: {n_pca}
- Data range: [{stats['min_val']:.3f}, {stats['max_val']:.3f}]

Return ONLY a Python expression that:
1. Takes input `x` (numpy array)
2. Returns angles in [0, 2π]
3. Aggregates using global statistics: mean(x), std(x), etc.
4. Format: theta_i = δ·mean(x) + ε·std(x) + ζ·max(x)

Constraints:
- Use np.mean(), np.std(), np.max()
- All angles in [0, 2π]
- Coefficients sum ≤ 1.0

OUTPUT ONLY the Python expression.
"""
    
    @staticmethod
    def _pca_mix_prompt(dataset_name: str, stats: Dict[str, Any], n_pca: int) -> str:
        return f"""
Generate a PCA_MIX quantum encoding for {dataset_name} classification.

Dataset Statistics:
- Samples: {stats['n_samples']}
- PCA dimensions: {n_pca}
- Data range: [{stats['min_val']:.3f}, {stats['max_val']:.3f}]

Return ONLY a Python expression that:
1. Takes input `x` (numpy array of PCA-reduced features)
2. Returns angles in [0, 2π]
3. Mix primary and derived features: theta_i = Σ(ω_j * x_j)
4. Use limited components (max 4) for stability

Constraints:
- Focus on top variance components
- Use at most 4 principal components
- Coefficients sum ≤ 1.0
- All angles in [0, 2π]

OUTPUT ONLY the Python expression.
"""
