import numpy as np
from typing import Dict
import time


class MetricsCollector:
    """Collect and report comprehensive metrics"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name: str, value: float):
        """Add a single metric"""
        self.results[name] = value
    
    def compute_complexity_metrics(self, n_qubits: int, depth: int, n_gates: int) -> Dict:
        """
        Compute circuit complexity metrics
        
        Args:
            n_qubits: Number of qubits
            depth: Circuit depth
            n_gates: Total gate count
            
        Returns:
            Complexity metrics
        """
        return {
            "n_qubits": n_qubits,
            "circuit_depth": depth,
            "gate_count": n_gates,
            "qubit_depth_product": n_qubits * depth
        }
    
    def compute_kernel_metrics(self, K: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compute kernel quality metrics
        
        Args:
            K: Kernel matrix
            y: Labels
            
        Returns:
            Kernel quality metrics
        """
        # Kernel target alignment
        y_matrix = y.reshape(-1, 1) == y.reshape(1, -1)
        Y_ideal = y_matrix.astype(float)
        alignment = np.sum(K * Y_ideal) / (np.linalg.norm(K, 'fro') * np.linalg.norm(Y_ideal, 'fro'))
        
        # Condition number (numerical stability)
        try:
            cond_number = np.linalg.cond(K)
        except:
            cond_number = np.inf
        
        # Effective rank (diversity of features)
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) > 0:
            entropy = -np.sum((eigenvalues / eigenvalues.sum()) * np.log(eigenvalues / eigenvalues.sum() + 1e-10))
            effective_rank = np.exp(entropy)
        else:
            effective_rank = 0
        
        return {
            "kernel_alignment": alignment,
            "condition_number": cond_number,
            "effective_rank": effective_rank,
            "mean_kernel_value": np.mean(K),
            "std_kernel_value": np.std(K)
        }
    
    def generate_report(self, dataset_name: str, template_family: str, 
                       classification_metrics: Dict, complexity_metrics: Dict,
                       kernel_metrics: Dict, execution_time: float) -> Dict:
        """
        Generate comprehensive report
        
        Returns:
            Full report dictionary
        """
        report = {
            "dataset": dataset_name,
            "template_family": template_family,
            "classification": classification_metrics,
            "complexity": complexity_metrics,
            "kernel_quality": kernel_metrics,
            "execution_time_seconds": execution_time
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Pretty print report"""
        print("\n" + "="*80)
        print(f"EVALUATION REPORT: {report['dataset']} - {report['template_family']}")
        print("="*80)
        
        print("\n📊 Classification Performance:")
        for metric, value in report['classification'].items():
            print(f"  {metric:20s}: {value:.4f}")
        
        print("\n🔧 Circuit Complexity:")
        for metric, value in report['complexity'].items():
            print(f"  {metric:20s}: {value}")
        
        print("\n🎯 Kernel Quality:")
        for metric, value in report['kernel_quality'].items():
            if isinstance(value, float):
                print(f"  {metric:20s}: {value:.4f}")
            else:
                print(f"  {metric:20s}: {value}")
        
        print(f"\n⏱️  Execution Time: {report['execution_time_seconds']:.2f} seconds")
        print("="*80 + "\n")


# Test
if __name__ == "__main__":
    collector = MetricsCollector()
    
    # Dummy metrics
    classification = {"accuracy": 0.92, "f1_score": 0.91}
    complexity = collector.compute_complexity_metrics(10, 5, 25)
    
    K_test = np.random.rand(50, 50)
    K_test = (K_test + K_test.T) / 2
    y_test = np.random.randint(0, 2, 50)
    
    kernel = collector.compute_kernel_metrics(K_test, y_test)
    
    report = collector.generate_report(
        "MNIST", "linear", classification, complexity, kernel, 120.5
    )
    
    collector.print_report(report)