"""
Multi-Dataset Experiment Runner
Run experiments on MNIST, Fashion-MNIST, and CIFAR-10
"""

import json
from pathlib import Path
from experiments.run_single_dataset import SingleDatasetExperiment
from config import CONFIG


class MultiDatasetExperiment:
    """Run experiments across all three datasets"""
    
    def __init__(self, n_pca: int = 40, n_train: int = 500, n_test: int = 200):
        self.n_pca = n_pca
        self.n_train = n_train
        self.n_test = n_test
        self.all_results = {}
    
    def run_all_datasets(self, template_type: str = "linear", use_mock: bool = False):
        """Run experiment on all three datasets"""
        datasets = ["mnist", "fashion_mnist", "cifar10"]
        
        print(f"\n{'='*70}")
        print(f"MULTI-DATASET EXPERIMENT")
        print(f"Template: {template_type}, PCA: {self.n_pca}")
        print(f"{'='*70}")
        
        for dataset in datasets:
            print(f"\n\nRunning {dataset.upper()}...")
            experiment = SingleDatasetExperiment(
                dataset_name=dataset,
                n_pca=self.n_pca,
                n_train=self.n_train,
                n_test=self.n_test
            )
            
            results = experiment.run(template_type=template_type, use_mock=use_mock)
            self.all_results[dataset] = results
            experiment.save_results()
        
        self._print_summary()
        return self.all_results
    
    def _print_summary(self):
        """Print summary table"""
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Dataset':<20} {'Accuracy':<15} {'F1 (macro)':<15}")
        print("-" * 70)
        
        for dataset, results in self.all_results.items():
            acc = results["metrics"]["accuracy"]
            f1 = results["metrics"]["f1_macro"]
            print(f"{dataset:<20} {acc:<15.4f} {f1:<15.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-dataset experiment")
    parser.add_argument("--template", choices=["linear", "polynomial", "global_stats", "pca_mix"],
                       default="linear")
    parser.add_argument("--n_pca", type=int, default=40)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--mock", action="store_true")
    
    args = parser.parse_args()
    
    experiment = MultiDatasetExperiment(
        n_pca=args.n_pca,
        n_train=args.n_train,
        n_test=args.n_test
    )
    
    experiment.run_all_datasets(template_type=args.template, use_mock=args.mock)
