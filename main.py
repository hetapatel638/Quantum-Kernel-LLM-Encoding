import argparse
from pathlib import Path

from experiments.run_single_dataset import SingleDatasetExperiment
from experiments.run_all_datasets import MultiDatasetExperiment
from visualization.results_table import ResultsVisualizer


def main():
    parser = argparse.ArgumentParser(description="AQED-Hybrid: Automated Quantum Encoding Design")
    
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "multi", "visualize"],
                       help="Execution mode")
    
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "fashion_mnist", "cifar10"],
                       help="Dataset to use (for single mode)")
    
    parser.add_argument("--template", type=str, default="linear",
                       choices=["linear", "polynomial", "global_stats", "pca_mix"],
                       help="Template family (for single mode)")
    
    parser.add_argument("--n_pca", type=int, default=10,
                       help="Number of PCA components")
    
    parser.add_argument("--n_train", type=int, default=1000,
                       help="Number of training samples")
    
    parser.add_argument("--n_test", type=int, default=200,
                       help="Number of test samples")
    
    parser.add_argument("--mock_llm", action="store_true",
                       help="Use mock LLM for testing")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        print("Running single dataset experiment...\n ")
        experiment = SingleDatasetExperiment(
            dataset_name=args.dataset,
            template_family=args.template,
            n_pca=args.n_pca,
            n_train=args.n_train,
            n_test=args.n_test
        )
        
        results = experiment.run(use_mock_llm=args.mock_llm)
        experiment.save_results()
        
    elif args.mode == "multi":
        print("\n Running multi-dataset evaluation...")
        experiment = MultiDatasetExperiment(
            n_pca=args.n_pca,
            n_train=args.n_train,
            n_test=args.n_test
        )
        
        results = experiment.run_all(use_mock_llm=args.mock_llm)
        experiment.generate_template_analysis()
        
    elif args.mode == "visualize":
        print("\n Generating visualizations...")
        
        # Load all results
        import json
        results = []
        results_dir = Path("results/individual")
        
        if results_dir.exists():
            for file in results_dir.glob("*.json"):
                with open(file) as f:
                    results.append(json.load(f))
        
        if results:
            viz = ResultsVisualizer()
            viz.create_comparison_table(results)
            viz.plot_accuracy_comparison(results)
            viz.plot_complexity_vs_accuracy(results)
            viz.create_template_heatmap(results)
            print("\n Visualizations complete!")
        else:
            print(" No results found. Run experiments first.")


if __name__ == "__main__":
    main()