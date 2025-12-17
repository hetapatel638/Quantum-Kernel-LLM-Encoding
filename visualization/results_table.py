import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict


class ResultsVisualizer:
    """Create visualizations for thesis"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def create_comparison_table(self, results: List[Dict]) -> pd.DataFrame:
        """
        Create Table 1 format comparison
        
        Args:
            results: List of experiment results
            
        Returns:
            DataFrame in publication format
        """
        from config import CONFIG
        
        # Organize by dataset
        table_data = []
        
        for dataset in ["mnist", "fashion_mnist", "cifar10"]:
            dataset_results = [r for r in results if r.get('dataset') == dataset]
            
            if not dataset_results:
                continue
            
            # Find best result
            best = max(dataset_results, key=lambda x: x['comparison']['llm_accuracy'])
            
            # Get baseline references
            ref = CONFIG['reference_results'][dataset]
            
            row = {
                "Dataset": dataset.replace("_", " ").title(),
                "Linear SVM": f"{ref['linear_kernel']:.2%}",
                "Polynomial SVM": f"{ref['polynomial_kernel']:.2%}",
                "RBF SVM": f"{ref['rbf_kernel']:.2%}",
                "ZZ Quantum": f"{ref['zz_quantum']:.2%}",
                "NPQC Quantum": f"{ref['npqc_quantum']:.2%}",
                "YZCX Quantum": f"{ref['yzcx_quantum']:.2%}",
                "Sakka et al.": f"{ref['sakka_generated']:.2%}",
                "AQED-Hybrid": f"{best['comparison']['llm_accuracy']:.2%}",
                "Template": best['template_family'],
                "Depth": best['llm_generated']['complexity']['circuit_depth']
            }
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save
        df.to_csv(self.results_dir / "table1_comparison.csv", index=False)
        df.to_latex(self.results_dir / "table1_comparison.tex", index=False)
        
        return df
    
    def plot_accuracy_comparison(self, results: List[Dict]):
        """Create bar plot comparing methods"""
        from config import CONFIG
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        datasets = ["mnist", "fashion_mnist", "cifar10"]
        dataset_names = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
        
        for idx, (dataset, name) in enumerate(zip(datasets, dataset_names)):
            ax = axes[idx]
            
            # Get reference and our results
            ref = CONFIG['reference_results'][dataset]
            dataset_results = [r for r in results if r.get('dataset') == dataset]
            
            if dataset_results:
                best = max(dataset_results, key=lambda x: x['comparison']['llm_accuracy'])
                our_acc = best['comparison']['llm_accuracy']
            else:
                our_acc = 0
            
            # Prepare data
            methods = ['RBF\n(Classical)', 'YZCX\n(Quantum)', 'Sakka\net al.', 'AQED-Hybrid\n(Ours)']
            accuracies = [
                ref['rbf_kernel'],
                ref['yzcx_quantum'],
                ref['sakka_generated'],
                our_acc
            ]
            colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71']
            
            # Plot
            bars = ax.bar(methods, accuracies, color=colors, alpha=0.8)
            
            # Styling
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / "accuracy_comparison.pdf", bbox_inches='tight')
        print(f"💾 Saved: {self.results_dir / 'accuracy_comparison.png'}")
    
    def plot_complexity_vs_accuracy(self, results: List[Dict]):
        """Scatter plot: circuit depth vs accuracy"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dataset in ["mnist", "fashion_mnist", "cifar10"]:
            dataset_results = [r for r in results 
                             if r.get('dataset') == dataset 
                             and r.get('status') == 'success']
            
            if not dataset_results:
                continue
            
            depths = [r['llm_generated']['complexity']['circuit_depth'] 
                     for r in dataset_results]
            accuracies = [r['comparison']['llm_accuracy'] 
                        for r in dataset_results]
            
            ax.scatter(depths, accuracies, label=dataset.replace("_", " ").title(),
                      s=100, alpha=0.7)
        
        # Add Sakka et al. reference (high depth)
        from config import CONFIG
        for dataset in ["mnist", "fashion_mnist", "cifar10"]:
            ref = CONFIG['reference_results'][dataset]
            ax.scatter([50], [ref['sakka_generated']], marker='x', s=200, 
                      label=f"Sakka ({dataset})", linewidths=3)
        
        ax.set_xlabel('Circuit Depth', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Circuit Complexity vs Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "complexity_vs_accuracy.png", dpi=300)
        print(f"💾 Saved: {self.results_dir / 'complexity_vs_accuracy.png'}")
    
    def create_template_heatmap(self, results: List[Dict]):
        """Heatmap showing template effectiveness per dataset"""
        # Prepare data
        data = []
        for dataset in ["mnist", "fashion_mnist", "cifar10"]:
            row = []
            for template in ["linear", "polynomial", "global_stats", "pca_mix"]:
                result = next((r for r in results 
                             if r.get('dataset') == dataset 
                             and r.get('template_family') == template
                             and r.get('status') == 'success'), None)
                
                if result:
                    row.append(result['comparison']['llm_accuracy'])
                else:
                    row.append(0)
            data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df = pd.DataFrame(
            data,
            index=["MNIST", "Fashion-MNIST", "CIFAR-10"],
            columns=["Linear", "Polynomial", "Global Stats", "PCA Mix"]
        )
        
        sns.heatmap(df, annot=True, fmt='.2%', cmap='RdYlGn', 
                   vmin=0.4, vmax=1.0, ax=ax, cbar_kws={'label': 'Accuracy'})
        
        ax.set_title('Template Family Effectiveness by Dataset', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "template_heatmap.png", dpi=300)
        print(f"💾 Saved: {self.results_dir / 'template_heatmap.png'}")


# Test
if __name__ == "__main__":
    # Load example results
    import json
    from pathlib import Path
    
    results = []
    results_dir = Path("results/individual")
    
    if results_dir.exists():
        for file in results_dir.glob("*.json"):
            with open(file) as f:
                results.append(json.load(f))
    
    if results:
        viz = ResultsVisualizer()
        
        # Generate all visualizations
        table = viz.create_comparison_table(results)
        print("\n📊 Comparison Table:")
        print(table)
        
        viz.plot_accuracy_comparison(results)
        viz.plot_complexity_vs_accuracy(results)
        viz.create_template_heatmap(results)
        
        print("\n✅ All visualizations generated!")
    else:
        print("No results found. Run experiments first.")