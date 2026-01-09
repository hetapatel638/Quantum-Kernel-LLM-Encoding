
from typing import Dict

class PromptBuilder:
    """Construct structured prompts for angle encoding generation"""
    
    def __init__(self, dataset_name: str, dataset_stats: Dict, template_family: str):
        self.dataset_name = dataset_name
        self.dataset_stats = dataset_stats
        self.template_family = template_family
    
    def build_base_prompt(self) -> str:
        """Core instruction prompt"""
        return f"""Generate a Python function that maps {self.dataset_stats['n_features']} features to an angle in [0, 2π].

Template: {self.template_family}
{self._get_template_description()}

Dataset: {self.dataset_name} with {self.dataset_stats['n_features']} PCA features
Value range: [{self.dataset_stats['min_value']:.3f}, {self.dataset_stats['max_value']:.3f}]

Requirements:
- Output in range [0, 2π] using np.clip()
- Single Python expression
- Use only: x[i], np.mean(x), np.std(x), arithmetic
- No imports, loops, or conditionals

Output as JSON:
{{"function": "np.clip(your_expression, 0, 2*np.pi)", "explanation": "brief explanation", "template_family": "{self.template_family}"}}

Example:
{{"function": "np.clip(0.8 * x[0] + 0.2 * np.mean(x), 0, 2*np.pi)", "explanation": "Weighted combination", "template_family": "linear"}}

Generate the function:"""
    
    def _get_dataset_characteristics(self) -> str:
        """Dataset-specific hints"""
        characteristics = {
            "mnist": """
**Dataset Characteristics**:
- Handwritten digits (0-9)
- Key patterns: Stroke continuity, local pixel correlations
- Hint: Neighboring feature interactions (x[i]*x[i+1]) capture stroke flow
""",
            "fashion_mnist": """
**Dataset Characteristics**:
- Clothing items (shirts, shoes, bags, etc.)
- Key patterns: Texture, edges, global shape
- Hint: Balance local features with global statistics (mean/std)
""",
            "cifar10": """
**Dataset Characteristics**:
- Natural color images (animals, vehicles)
- Key patterns: Color channels, multi-scale textures
- Hint: First few PCA components capture color info - weight them heavily
- Note: This is the hardest dataset, quantum baselines only reach ~50% accuracy
"""
        }
        
        return characteristics.get(self.dataset_name, "")
    
    def _get_template_description(self) -> str:
        """Template-specific instructions"""
        descriptions = {
            "linear": """
**Linear Template Rules**:
- Form: θ = Σ αᵢ·xᵢ
- Constraint: Sum of |α| should be ≤ 1
- Example: θ = 0.7*x[0] + 0.3*x[1]
- Can include global stats: θ = 0.5*x[0] + 0.3*np.mean(x)
""",
            "polynomial": """
**Polynomial Template Rules**:
- Form: θ = Σ αᵢ·xᵢ + Σ βᵢⱼ·xᵢ·xⱼ
- Degree: Maximum 2 (quadratic)
- Example: θ = 0.5*x[0] + 0.3*x[1] + 0.2*x[0]*x[1]
- Captures feature interactions
""",
            "global_stats": """
**Global Statistics Template Rules**:
- Form: θ = δ·mean(x) + ε·std(x) + γ·xᵢ
- Use: np.mean(x), np.std(x)
- Example: θ = 1.2*np.mean(x) + 0.5*np.std(x) + 0.3*x[0]
- Captures dataset-level patterns
""",
            "pca_mix": """
**PCA Mixing Template Rules**:
- Form: θ = Σ ωᵢ·xᵢ (first 4 components only)
- Constraint: Only use x[0], x[1], x[2], x[3]
- Example: θ = 0.5*x[0] + 0.3*x[1] + 0.15*x[2] + 0.05*x[3]
- First PCs contain most information
"""
        }
        
        return descriptions.get(self.template_family, "")
    
    def build_refinement_prompt(self, previous_function: str, previous_accuracy: float, baseline_accuracy: float) -> str:
        """Prompt for iterative refinement"""
        improvement = previous_accuracy - baseline_accuracy
        
        feedback = "improved" if improvement > 0 else "decreased"
        
        return f"""The previous function achieved {previous_accuracy:.2%} accuracy (baseline: {baseline_accuracy:.2%}).
Performance {feedback} by {abs(improvement):.2%}.

**Previous function**:
{previous_function}

**Task**: Generate an IMPROVED function that:
{'- Builds on successful patterns from previous function' if improvement > 0 else '- Takes a different approach to boost accuracy'}
- Maintains template family: {self.template_family}
- Stays within all constraints

Generate the improved function in JSON format:
{{
    "function": "improved Python expression",
    "explanation": "what changed and why",
    "template_family": "{self.template_family}"
}}"""


# Test
if __name__ == "__main__":
    stats = {
        "n_features": 10,
        "mean_value": 0.5,
        "std_value": 0.2,
        "min_value": 0.0,
        "max_value": 1.0
    }
    
    builder = PromptBuilder("mnist", stats, "linear")
    prompt = builder.build_base_prompt()
    print(prompt)
    print("\n" + "="*80 + "\n")
    
    refine_prompt = builder.build_refinement_prompt(
        "0.8 * x[0] + 0.2 * x[1]",
        0.89,
        0.87
    )
    print(refine_prompt)