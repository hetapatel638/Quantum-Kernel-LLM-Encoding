"""
Level 3: Encoding Templates
Standard template families for quantum encoding
"""

import numpy as np
from typing import Callable


class EncodingTemplates:
    """Standard encoding template families"""
    
    @staticmethod
    def linear_template(coefficients: np.ndarray) -> Callable:
        """Linear encoding: θ_i = Σ α_j * x_j"""
        def encode(x: np.ndarray) -> np.ndarray:
            angles = np.dot(coefficients, x)
            return np.clip(angles, 0, 2*np.pi)
        return encode
    
    @staticmethod
    def polynomial_template(lin_coeffs: np.ndarray, 
                           poly_coeffs: np.ndarray) -> Callable:
        """Polynomial encoding: θ_i = Σ α_j * x_j + Σ β_jk * x_j * x_k"""
        def encode(x: np.ndarray) -> np.ndarray:
            n = len(x)
            angles = np.dot(lin_coeffs, x)
            
            # Add polynomial terms
            for j in range(n):
                for k in range(j, n):
                    if j < len(poly_coeffs) and k < len(poly_coeffs):
                        angles += poly_coeffs[j] * x[j] * x[k]
            
            return np.clip(angles, 0, 2*np.pi)
        return encode
    
    @staticmethod
    def global_stats_template(mean_coeff: float, std_coeff: float, 
                             max_coeff: float) -> Callable:
        """Global statistics encoding: θ_i = δ·mean(x) + ε·std(x) + ζ·max(x)"""
        def encode(x: np.ndarray) -> np.ndarray:
            angle = (mean_coeff * np.mean(x) + 
                    std_coeff * np.std(x) + 
                    max_coeff * np.max(x))
            return np.clip(angle, 0, 2*np.pi)
        return encode
    
    @staticmethod
    def pca_mix_template(coefficients: np.ndarray) -> Callable:
        """PCA mix encoding: θ_i = Σ ω_j * PC_j (limited to top components)"""
        def encode(x: np.ndarray) -> np.ndarray:
            # Use only top n_components
            n_use = min(4, len(coefficients), len(x))
            angles = np.dot(coefficients[:n_use], x[:n_use])
            return np.clip(angles, 0, 2*np.pi)
        return encode
