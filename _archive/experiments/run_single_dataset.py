import numpy as np
import json
import time
from pathlib import Path

# Import our modules
from data.loader import DatasetLoader
from data.preprocessor import QuantumPreprocessor
from baselines.manual_baseline import ManualBaseline
from encoding.prompt_builder import PromptBuilder
from encoding.validator import EncodingValidator
from llm.hf_interface import LLMInterface
from quantum.circuit import QuantumCircuitBuilder
from quantum.kernel import QuantumKernel
from evaluation.svm_trainer import QuantumSVMTrainer
from evaluation.metrics import MetricsCollector
from config import CONFIG


class SingleDatasetExperiment:
    """Run experiment on one dataset with one template"""
    
    def __init__(self, dataset_name: str = "mnist", template_family: str = "linear", 
                 n_pca: int = 10, n_train: int = 1000, n_test: int = 200):
        """
        Initialize experiment
        
        Args:
            dataset_name: "mnist", "fashion_mnist", or "cifar10"
            template_family: "linear", "polynomial", "global_stats", or "pca_mix"
            n_pca: Number of PCA components
            n_train: Number of training samples (use small for Day 1)
            n_test: Number of test samples
        """
        self.dataset_name = dataset_name
        self.template_family = template_family
        self.n_pca = n_pca
        self.n_train = n_train
        self.n_test = n_test
        
        self.results = {}
    
    def run(self, use_mock_llm: bool = True) -> dict:
        """
        Run complete pipeline
        
        Args:
            use_mock_llm: If True, use mock LLM for testing (faster)
            
        Returns:
            Results dictionary
        """

        print(f"STARTING EXPERIMENT: {self.dataset_name} - {self.template_family}")
        
        
        start_time = time.time()
        
        # Step 1: Load data
        print("Step 1: Loading data...")
        loader = DatasetLoader()
        X_train, X_test, y_train, y_test = loader.load_dataset(
            self.dataset_name, self.n_train, self.n_test
        )
        
        # Step 2: Preprocess with PCA
        print("\nStep 2: Applying PCA...")
        preprocessor = QuantumPreprocessor(n_components=self.n_pca)
        X_train_pca, X_test_pca = preprocessor.fit_transform(X_train, X_test)
        
        dataset_stats = preprocessor.get_stats(X_train_pca)
        
        # Step 3: Baseline evaluation
        print("\nStep 3: Evaluating baseline...")
        baseline_results = self._evaluate_baseline(X_train_pca, X_test_pca, y_train, y_test)
        
        # Step 4: LLM generation
        print("\nStep 4: Generating encoding function with LLM...")
        llm_function, llm_explanation = self._generate_with_llm(dataset_stats, use_mock_llm)
        
        # Step 5: Validate function
        print("\nStep 5: Validating generated function...")
        is_valid, errors = self._validate_function(llm_function)
        
        if not is_valid:
            print(f"Validation failed: {errors}")
            return {"status": "failed", "errors": errors}
        
        print("Validation passed!")
        
        # Step 6: Evaluate LLM-generated encoding
        print("\nStep 6: Evaluating LLM-generated encoding...")
        llm_results = self._evaluate_encoding(llm_function, X_train_pca, X_test_pca, y_train, y_test)
        
        # Step 7: Compare results
        print("\nStep 7: Comparing results...")
        comparison = self._compare_results(baseline_results, llm_results)
        
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            "dataset": self.dataset_name,
            "template_family": self.template_family,
            "n_pca": self.n_pca,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "baseline": baseline_results,
            "llm_generated": llm_results,
            "llm_function": llm_function,
            "llm_explanation": llm_explanation,
            "comparison": comparison,
            "total_time": total_time,
            "status": "success"
        }
        
        self.results = final_results
        self._print_summary(final_results)
        
        return final_results
    
    def _evaluate_baseline(self, X_train, X_test, y_train, y_test) -> dict:
        """Evaluate manual baseline encoding"""
        baseline = ManualBaseline()
        encode_func = baseline.simple_rotation(self.n_pca)
        
        return self._evaluate_encoding(encode_func, X_train, X_test, y_train, y_test)
    
    def _generate_with_llm(self, dataset_stats: dict, use_mock: bool = True) -> tuple:
        """Generate encoding function using LLM"""
        # Build prompt
        prompt_builder = PromptBuilder(self.dataset_name, dataset_stats, self.template_family)
        prompt = prompt_builder.build_base_prompt()
        
        # Query LLM (will auto-detect Claude API)
        from config import CONFIG
        llm = LLMInterface(model_name=CONFIG['llm']['model_name'], use_local=CONFIG['llm']['use_local'])
        
        response = llm.generate(prompt, temperature=0.95, max_new_tokens=512)  # High temp for exploration
        
        # Parse response
        parsed = llm.parse_json_response(response)
        
        if parsed is None:
            print("Failed to parse LLM response, using fallback...")
            return "np.clip(0.8 * x[0] + 0.2 * np.mean(x), 0, 2*np.pi)", "Fallback function"
        
        function_code = parsed.get("function", "")
        explanation = parsed.get("explanation", "No explanation provided")
        
        print(f"\nGenerated function: {function_code}")
        print(f"Explanation: {explanation}")
        
        return function_code, explanation
    
    def _validate_function(self, function_code: str) -> tuple:
        """Validate generated function"""
        validator = EncodingValidator(n_features=self.n_pca)
        return validator.validate_all(function_code, self.template_family)
    
    def _evaluate_encoding(self, encoding_func, X_train, X_test, y_train, y_test) -> dict:
        """Evaluate an encoding function"""
        # Convert string to function if needed
        if isinstance(encoding_func, str):
            # Create safe execution environment
            # Capture the string value in the closure properly
            func_string = encoding_func
            namespace = {'np': np, 'numpy': np, 'range': range, 'len': len, 'max': max, 'min': min}
            
            def angle_func(x):
                namespace['x'] = x
                return eval(func_string, {"__builtins__": {}}, namespace)
            
            encoding_func = angle_func
        
        # Build circuit - use config's n_qubits (10), not n_pca which can be 80
        from config import CONFIG
        n_qubits = CONFIG['quantum']['n_qubits']
        circuit_builder = QuantumCircuitBuilder(n_qubits=n_qubits)
        circuit = circuit_builder.build_circuit([encoding_func], entanglement="linear")
        
        # Compute kernels with subsampling for speed
        kernel_computer = QuantumKernel()
        # For large datasets, subsample to max 2000 for kernel computation
        max_kernel_samples = 2000 if X_train.shape[0] > 2000 else None
        K_train = kernel_computer.compute_kernel_matrix(circuit, X_train, subsample=max_kernel_samples)
        K_test = kernel_computer.compute_kernel_matrix(circuit, X_train, X_test, subsample=max_kernel_samples)
        
        # Train SVM
        svm_trainer = QuantumSVMTrainer(C=1.0)
        svm_trainer.train(K_train, y_train)
        
        # Evaluate
        classification_metrics = svm_trainer.evaluate(K_test, y_test)
        
        # Compute additional metrics
        metrics_collector = MetricsCollector()
        complexity_metrics = metrics_collector.compute_complexity_metrics(
            n_qubits=n_qubits,  # Use actual circuit qubits, not PCA dims
            depth=circuit_builder.get_circuit_depth([encoding_func]),
            n_gates=circuit_builder.get_gate_count([encoding_func])
        )
        kernel_metrics = metrics_collector.compute_kernel_metrics(K_train, y_train)
        
        return {
            "classification": classification_metrics,
            "complexity": complexity_metrics,
            "kernel_quality": kernel_metrics
        }
    
    def _compare_results(self, baseline: dict, llm_generated: dict) -> dict:
        """Compare baseline vs LLM results"""
        baseline_acc = baseline['classification']['accuracy']
        llm_acc = llm_generated['classification']['accuracy']
        
        improvement = llm_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        return {
            "baseline_accuracy": baseline_acc,
            "llm_accuracy": llm_acc,
            "absolute_improvement": improvement,
            "relative_improvement_pct": improvement_pct,
            "llm_is_better": llm_acc > baseline_acc
        }
    
    def _print_summary(self, results: dict):
        """Print experiment summary"""
        print("EXPERIMENT SUMMARY")
        
        print(f"\nDataset: {results['dataset']}")
        print(f"Template: {results['template_family']}")
        print(f"PCA components: {results['n_pca']}")
        print(f"Samples: {results['n_train']} train, {results['n_test']} test")
        
        comparison = results['comparison']
        print(f"\nRESULTS:")
        print(f"  Baseline:      {comparison['baseline_accuracy']:.4f}")
        print(f"  LLM-generated: {comparison['llm_accuracy']:.4f}")
        print(f"  Improvement:   {comparison['absolute_improvement']:+.4f} ({comparison['relative_improvement_pct']:+.2f}%)")
        
        if comparison['llm_is_better']:
            print("\n LLM-generated encoding OUTPERFORMS baseline!")
        else:
            print("\n LLM-generated encoding underperforms baseline")
        
        print(f"\n Total time: {results['total_time']:.2f} seconds")
    
    def save_results(self, output_dir: str = "results"):
        """Save results to JSON file"""
        Path(output_dir).mkdir(exist_ok=True)
        
        filename = f"{output_dir}/{self.dataset_name}_{self.template_family}_pca{self.n_pca}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f" Results saved to: {filename}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run quantum ML experiment on single dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--template', type=str, default='linear', choices=['linear', 'polynomial', 'global_stats', 'pca_mix'])
    parser.add_argument('--pca_dims', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--n_train', type=int, default=500, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')
    parser.add_argument('--use_mock', action='store_true', help='Use mock LLM instead of Claude')
    
    args = parser.parse_args()
    
    experiment = SingleDatasetExperiment(
        dataset_name=args.dataset,
        template_family=args.template,
        n_pca=args.pca_dims,
        n_train=args.n_train,
        n_test=args.n_test
    )
    
    results = experiment.run(use_mock_llm=args.use_mock)
    experiment.save_results()
    
   