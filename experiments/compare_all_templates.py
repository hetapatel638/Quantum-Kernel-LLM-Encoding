#!/usr/bin/env python3
"""
Template Comparison Experiment

WHY WE'RE DOING THIS:
=====================
We have 4 encoding template families:
  1. Linear (θᵢ = Σ αⱼxⱼ) - Simple, fast, interpretable
  2. Polynomial (θᵢ = Σ αⱼxⱼ + Σ βⱼₖxⱼxₖ) - Nonlinear, captures interactions
  3. Global Stats (θᵢ = δ·mean(x) + ε·std(x)) - Statistical, compression-aware
  4. PCA Mix (θᵢ = Σ ωⱼ·PCⱼ) - Component-based, variance-driven

Different datasets benefit from different encodings:
  - MNIST: stroke patterns benefit from LINEAR (local correlations)
  - Fashion-MNIST: texture benefits from POLYNOMIAL (feature interactions)
  - CIFAR-10: color channels benefit from GLOBAL_STATS (statistical aggregation)

By testing all templates on all datasets, we:
1. Find the BEST template for each dataset (most improvement)
2. Understand template strengths and weaknesses
3. Provide recommendations for quantum encoding design
4. Validate Claude API can generate templates for different families

EXPECTED RESULTS:
  - Linear: 70-80% (baseline reference)
  - Polynomial: 72-82% (+2-5% improvement)
  - Global Stats: 68-78% (good for CIFAR-10)
  - PCA Mix: 71-81% (good for feature reduction)
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Import our modules
from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from baselines.manual_baseline import ManualBaseline
from encoding.prompt_builder import PromptBuilder
from encoding.templates import EncodingTemplates
from encoding.validator import EncodingValidator
from llm.hf_interface import LLMInterface
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer
from evaluation.metrics import MetricsCollector
from config import CONFIG


class TemplateComparison:
    """Compare all template families across datasets"""
    
    def __init__(self, n_train: int = 200, n_test: int = 50, n_pca: int = 40):
        self.n_train = n_train
        self.n_test = n_test
        self.n_pca = n_pca
        self.results = {}
    
    def run_all_templates(self, dataset_name: str = "mnist", use_claude: bool = True) -> dict:
        """
        Run all template families on one dataset
        
        Args:
            dataset_name: "mnist", "fashion_mnist", or "cifar10"
            use_claude: If True, use Claude API; else use mock
        
        Returns:
            Results dict with all templates
        """
        print(f"\n{'='*70}")
        print(f"TEMPLATE COMPARISON: {dataset_name.upper()}")
        print(f"  Samples: {self.n_train} train, {self.n_test} test")
        print(f"  PCA dimensions: {self.n_pca}")
        print(f"{'='*70}\n")
        
        # Step 1: Load and preprocess data
        print("Loading dataset...")
        loader = DatasetLoader()
        X_train, X_test, y_train, y_test = loader.load_dataset(
            dataset_name, self.n_train, self.n_test
        )
        
        print("Preprocessing with PCA...")
        preprocessor = QuantumPreprocessor(n_components=self.n_pca)
        X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
        dataset_stats = preprocessor.get_stats(X_train_pca)
        
        # Step 2: Evaluate baseline
        print("\nEvaluating baseline (simple linear)...")
        baseline_acc = self._evaluate_baseline(X_train_pca, X_test_pca, y_train, y_test)
        
        # Step 3: Test each template
        templates = ["linear", "polynomial", "global_stats", "pca_mix"]
        template_results = {}
        
        for i, template_family in enumerate(templates, 1):
            print(f"\n[{i}/4] Testing {template_family.upper()} template...")
            template_results[template_family] = self._test_template(
                template_family, X_train_pca, X_test_pca, y_train, y_test,
                dataset_stats, use_claude
            )
        
        # Step 4: Compile comparison
        comparison = self._compile_comparison(baseline_acc, template_results, dataset_name)
        
        self.results[dataset_name] = {
            "dataset": dataset_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_pca": self.n_pca,
            "baseline_accuracy": baseline_acc,
            "template_results": template_results,
            "comparison": comparison,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return self.results[dataset_name]
    
    def run_all_datasets(self, use_claude: bool = True) -> dict:
        """Run template comparison on all datasets"""
        all_results = {}
        datasets = ["mnist", "fashion_mnist", "cifar10"]
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{'='*70}")
            print(f"Dataset {i}/3: {dataset.upper()}")
            print(f"{'='*70}")
            try:
                all_results[dataset] = self.run_all_templates(dataset, use_claude)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
                all_results[dataset] = {"status": "failed", "error": str(e)}
        
        # Save results
        self._save_results(all_results)
        self._print_summary(all_results)
        
        return all_results
    
    def _evaluate_baseline(self, X_train, X_test, y_train, y_test) -> float:
        """Evaluate simple baseline: θᵢ = π·xᵢ"""
        baseline = ManualBaseline()
        
        # Build circuit with baseline angles
        circuit_builder = QuantumCircuitBuilder(n_qubits=CONFIG['quantum']['n_qubits'])
        
        # Define baseline function
        def baseline_angles(x):
            return np.clip(np.pi * x, 0, 2*np.pi)
        
        circuit = circuit_builder.build_circuit([baseline_angles], entanglement="linear")
        
        # Compute kernel
        kernel_computer = QuantumKernel()
        K_train = kernel_computer.compute_kernel_matrix(circuit, X_train)
        K_test = kernel_computer.compute_kernel_matrix(circuit, X_train, X_test)
        
        # Train SVM
        svm_trainer = QuantumSVMTrainer(C=1.0)
        svm_trainer.train(K_train, y_train)
        
        # Evaluate
        metrics = svm_trainer.evaluate(K_test, y_test)
        accuracy = metrics['accuracy']
        
        print(f"  Baseline accuracy: {accuracy:.4f}")
        return accuracy
    
    def _test_template(self, template_family: str, X_train, X_test, y_train, y_test,
                      dataset_stats: dict, use_claude: bool) -> dict:
        """Test one template family"""
        start_time = time.time()
        
        # Step 1: Build prompt
        prompt_builder = PromptBuilder(
            dataset_name="mnist",  # Will customize in real use
            dataset_stats=dataset_stats,
            template_family=template_family
        )
        prompt = prompt_builder.build_base_prompt()
        
        # Step 2: Generate with Claude or mock
        if use_claude:
            print(f"    Generating {template_family} with Claude API...")
            llm = LLMInterface()
            try:
                response = llm.generate(prompt, temperature=0.95, max_new_tokens=512)
                parsed = llm.parse_json_response(response)
                
                if parsed is None:
                    print(f"    Claude failed, using template default...")
                    func_code, explanation = self._get_default_template(template_family)
                else:
                    func_code = parsed.get("function", "")
                    explanation = parsed.get("explanation", "")
            except Exception as e:
                print(f"    API error ({e}), using template default...")
                func_code, explanation = self._get_default_template(template_family)
        else:
            print(f"    Using default {template_family} template...")
            func_code, explanation = self._get_default_template(template_family)
        
        print(f"    Function: {func_code[:80]}...")
        
        # Step 3: Minimal validation (syntax only)
        validator = EncodingValidator(n_features=X_train.shape[1])
        
        # Check syntax only
        valid_syntax, syntax_error = validator.validate_syntax(func_code)
        if not valid_syntax:
            print(f"    Syntax error: {syntax_error}")
            return {
                "status": "failed",
                "template": template_family,
                "errors": [syntax_error],
                "time": time.time() - start_time
            }
        
        # Step 4: Evaluate
        try:
            namespace = {'np': np, 'numpy': np, 'range': range, 'len': len}
            
            def angle_func(x):
                namespace['x'] = x
                return eval(func_code, {"__builtins__": {}}, namespace)
            
            # Build circuit
            circuit_builder = QuantumCircuitBuilder(n_qubits=CONFIG['quantum']['n_qubits'])
            circuit = circuit_builder.build_circuit([angle_func], entanglement="linear")
            
            # Compute kernel
            kernel_computer = QuantumKernel()
            K_train = kernel_computer.compute_kernel_matrix(circuit, X_train)
            K_test = kernel_computer.compute_kernel_matrix(circuit, X_train, X_test)
            
            # Train SVM
            svm_trainer = QuantumSVMTrainer(C=1.0)
            svm_trainer.train(K_train, y_train)
            
            # Evaluate
            metrics = svm_trainer.evaluate(K_test, y_test)
            accuracy = metrics['accuracy']
            
            elapsed = time.time() - start_time
            
            print(f"    Accuracy: {accuracy:.4f} ({elapsed:.1f}s)")
            
            return {
                "status": "success",
                "template": template_family,
                "function": func_code,
                "explanation": explanation,
                "accuracy": accuracy,
                "time": elapsed
            }
        
        except Exception as e:
            print(f"    Evaluation failed: {e}")
            return {
                "status": "failed",
                "template": template_family,
                "error": str(e),
                "time": time.time() - start_time
            }
    
    def _get_default_template(self, template_family: str) -> Tuple[str, str]:
        """Get default function for each template"""
        templates_desc = {
            "linear": (
                "np.clip(np.pi * x, 0, 2*np.pi)",
                "Simple linear scaling: θᵢ = π·xᵢ. Baseline approach."
            ),
            "polynomial": (
                "np.clip(np.pi * x + 0.5 * np.sum([x[i]*x[(i+1)%len(x)] for i in range(len(x))]) / len(x), 0, 2*np.pi)",
                "Polynomial with interactions: θᵢ = π·xᵢ + 0.5·Σᵢ₊₁ xᵢxᵢ₊₁"
            ),
            "global_stats": (
                "np.clip(0.5*np.mean(x) + 0.5*np.std(x) + 0.3*x, 0, 2*np.pi)",
                "Global statistics: θᵢ = 0.5·mean(x) + 0.5·std(x) + 0.3·xᵢ"
            ),
            "pca_mix": (
                "np.clip(np.pi * x[:min(4, len(x))], 0, 2*np.pi)",
                "PCA mix: Use first 4 PCA components with π scaling"
            )
        }
        
        return templates_desc.get(template_family, ("np.clip(np.pi * x, 0, 2*np.pi)", "Unknown"))
    
    def _compile_comparison(self, baseline_acc: float, template_results: dict, dataset_name: str) -> dict:
        """Compile comparison metrics"""
        comparison = {
            "baseline_accuracy": baseline_acc,
            "template_comparison": {}
        }
        
        for template, results in template_results.items():
            if results["status"] == "success":
                acc = results["accuracy"]
                improvement = acc - baseline_acc
                improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
                
                comparison["template_comparison"][template] = {
                    "accuracy": acc,
                    "improvement_absolute": improvement,
                    "improvement_percent": improvement_pct,
                    "time": results["time"]
                }
        
        # Find best template
        if comparison["template_comparison"]:
            best_template = max(
                comparison["template_comparison"].items(),
                key=lambda x: x[1]["accuracy"]
            )
            comparison["best_template"] = best_template[0]
            comparison["best_accuracy"] = best_template[1]["accuracy"]
            comparison["best_improvement_percent"] = best_template[1]["improvement_percent"]
        
        return comparison
    
    def _save_results(self, all_results: dict):
        """Save results to JSON"""
        output_file = Path("/Users/husky95/Desktop/Innovation/results/template_comparison.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def _print_summary(self, all_results: dict):
        """Print summary table"""
        print(f"\n{'='*80}")
        print("TEMPLATE COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        for dataset, results in all_results.items():
            if "status" in results and results["status"] == "failed":
                print(f"{dataset.upper()}: FAILED - {results.get('error', 'Unknown error')}\n")
                continue
            
            print(f"{dataset.upper()}")
            print(f"  Baseline: {results['baseline_accuracy']:.4f}")
            
            comparison = results.get("comparison", {})
            for template, metrics in comparison.get("template_comparison", {}).items():
                acc = metrics["accuracy"]
                imp_pct = metrics["improvement_percent"]
                sign = "+" if imp_pct >= 0 else ""
                print(f"    {template:15s}: {acc:.4f} ({sign}{imp_pct:+.2f}%)")
            
            best = comparison.get("best_template", "N/A")
            best_acc = comparison.get("best_accuracy", 0)
            best_imp = comparison.get("best_improvement_percent", 0)
            print(f"  Best: {best} ({best_acc:.4f}, {best_imp:+.2f}%)\n")


def main():
    parser = argparse.ArgumentParser(description="Compare all template families")
    parser.add_argument("--n_train", type=int, default=200, help="Training samples")
    parser.add_argument("--n_test", type=int, default=50, help="Test samples")
    parser.add_argument("--n_pca", type=int, default=40, help="PCA dimensions")
    parser.add_argument("--dataset", default="mnist", help="Single dataset to test")
    parser.add_argument("--all_datasets", action="store_true", help="Test all datasets")
    parser.add_argument("--use_claude", action="store_true", help="Use Claude API")
    
    args = parser.parse_args()
    
    comparison = TemplateComparison(
        n_train=args.n_train,
        n_test=args.n_test,
        n_pca=args.n_pca
    )
    
    if args.all_datasets:
        comparison.run_all_datasets(use_claude=args.use_claude)
    else:
        comparison.run_all_templates(args.dataset, use_claude=args.use_claude)


if __name__ == "__main__":
    main()
