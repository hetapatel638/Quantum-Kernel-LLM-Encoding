import numpy as np
from typing import Dict, List


class EncodingTemplates:
    """Define and validate encoding template families"""
    
    @staticmethod
    def linear_template(coefficients: List[float], n_features: int) -> str:
        """
        Template: θᵢ = Σ αⱼxⱼ
        
        Args:
            coefficients: List of weights αⱼ
            n_features: Number of features to use
            
        Returns:
            Python code string
        """
        terms = [f"{coefficients[j]:.4f} * x[{j}]" for j in range(min(len(coefficients), n_features))]
        return f"np.clip({' + '.join(terms)}, 0, 2*np.pi)"
    
    @staticmethod
    def polynomial_template(linear_coefs: List[float], interaction_pairs: List[tuple], n_features: int) -> str:
        """
        Template: θᵢ = Σ αⱼxⱼ + Σ βⱼₖxⱼxₖ (degree 2)
        
        Args:
            linear_coefs: Linear term coefficients
            interaction_pairs: List of (j, k, β) for interaction terms
            n_features: Number of features
            
        Returns:
            Python code string
        """
        # Linear terms
        linear_terms = [f"{linear_coefs[j]:.4f} * x[{j}]" 
                       for j in range(min(len(linear_coefs), n_features))]
        
        # Interaction terms
        interaction_terms = [f"{beta:.4f} * x[{j}] * x[{k}]" 
                           for j, k, beta in interaction_pairs 
                           if j < n_features and k < n_features]
        
        all_terms = linear_terms + interaction_terms
        return f"np.clip({' + '.join(all_terms)}, 0, 2*np.pi)"
    
    @staticmethod
    def global_stats_template(mean_coef: float, std_coef: float, local_coef: float = 0.0, index: int = 0) -> str:
        """
        Template: θᵢ = δ·mean(x) + ε·std(x) + γ·xᵢ
        
        Args:
            mean_coef: Weight for mean
            std_coef: Weight for std
            local_coef: Weight for local feature (optional)
            index: Which feature index to use for local term
            
        Returns:
            Python code string
        """
        terms = [
            f"{mean_coef:.4f} * np.mean(x)",
            f"{std_coef:.4f} * np.std(x)"
        ]
        
        if local_coef != 0.0:
            terms.append(f"{local_coef:.4f} * x[{index}]")
        
        return f"np.clip({' + '.join(terms)}, 0, 2*np.pi)"
    
    @staticmethod
    def pca_mix_template(pca_weights: List[float], max_components: int = 4) -> str:
        """
        Template: θᵢ = Σ ωⱼ·xⱼ (using first K PCA components)
        
        Args:
            pca_weights: Weights for PCA components
            max_components: Maximum number of components to use
            
        Returns:
            Python code string
        """
        n_use = min(len(pca_weights), max_components)
        terms = [f"{pca_weights[j]:.4f} * x[{j}]" for j in range(n_use)]
        return f"np.clip({' + '.join(terms)}, 0, 2*np.pi)"
    
    @staticmethod
    def get_template_description() -> Dict:
        """Get description of all templates"""
        return {
            "linear": {
                "formula": "θᵢ = Σ αⱼxⱼ",
                "constraints": "sum(|α|) ≤ 1",
                "complexity": "O(n)",
                "interpretability": "high"
            },
            "polynomial": {
                "formula": "θᵢ = Σ αⱼxⱼ + Σ βⱼₖxⱼxₖ",
                "constraints": "degree ≤ 2",
                "complexity": "O(n²)",
                "interpretability": "medium"
            },
            "global_stats": {
                "formula": "θᵢ = δ·mean(x) + ε·std(x) + γ·xᵢ",
                "constraints": "statistical aggregation",
                "complexity": "O(n)",
                "interpretability": "high"
            },
            "pca_mix": {
                "formula": "θᵢ = Σ ωⱼ·PCⱼ",
                "constraints": "max 4 components",
                "complexity": "O(k), k≤4",
                "interpretability": "medium"
            }
        }


# Test
if __name__ == "__main__":
    templates = EncodingTemplates()
    
    # Test linear
    linear_code = templates.linear_template([0.8, 0.2], n_features=2)
    print(f"Linear: {linear_code}")
    
    # Test polynomial
    poly_code = templates.polynomial_template([0.5, 0.3], [(0, 1, 0.2)], n_features=2)
    print(f"Polynomial: {poly_code}")
    
    # Test global stats
    global_code = templates.global_stats_template(1.0, 0.5, 0.2, 0)
    print(f"Global: {global_code}")
    
    # Test PCA mix
    pca_code = templates.pca_mix_template([0.6, 0.3, 0.1], max_components=3)
    print(f"PCA: {pca_code}")