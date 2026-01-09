"""
Main CLI Entry Point
5-Level Quantum ML Framework with Claude LLM Prompting + Qiskit
"""

import argparse
from experiments.run_single_dataset import SingleDatasetExperiment
from experiments.run_all_datasets import MultiDatasetExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Quantum ML Framework: Claude Prompting + Qiskit Circuits"
    )
    
    parser.add_argument("--mode", choices=["single", "multi"],
                       default="single",
                       help="Run single dataset or all datasets")
    
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"],
                       default="mnist",
                       help="Dataset (for single mode)")
    
    parser.add_argument("--template", choices=["linear", "polynomial", "global_stats", "pca_mix"],
                       default="linear",
                       help="Encoding template type")
    
    parser.add_argument("--n_pca", type=int, default=40,
                       help="PCA dimensions")
    
    parser.add_argument("--n_train", type=int, default=500,
                       help="Number of training samples")
    
    parser.add_argument("--n_test", type=int, default=200,
                       help="Number of test samples")
    
    parser.add_argument("--mock", action="store_true",
                       help="Use mock Claude response (for testing)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        print(f"\n{'='*70}")
        print("SINGLE DATASET EXPERIMENT")
        print(f"{'='*70}")
        
        experiment = SingleDatasetExperiment(
            dataset_name=args.dataset,
            n_pca=args.n_pca,
            n_train=args.n_train,
            n_test=args.n_test
        )
        
        results = experiment.run(template_type=args.template, use_mock=args.mock)
        experiment.save_results()
    
    elif args.mode == "multi":
        print(f"\n{'='*70}")
        print("MULTI-DATASET EXPERIMENT")
        print(f"{'='*70}")
        
        experiment = MultiDatasetExperiment(
            n_pca=args.n_pca,
            n_train=args.n_train,
            n_test=args.n_test
        )
        
        experiment.run_all_datasets(template_type=args.template, use_mock=args.mock)


if __name__ == "__main__":
    main()
