import numpy as np
import ast
import re
from typing import Tuple, Optional, List


class EncodingValidator:
    """Validate angle encoding functions for safety and correctness"""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.allowed_functions = {
            'np', 'numpy', 'sum', 'mean', 'std', 'min', 'max', 'clip', 'pi'
        }
    
    def validate_syntax(self, code_string: str) -> Tuple[bool, Optional[str]]:
        """Check if code is valid Python expression"""
        try:
            ast.parse(code_string, mode='eval')
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
    
    def validate_safety(self, code_string: str) -> Tuple[bool, Optional[str]]:
        """Check for malicious code patterns"""
        dangerous_patterns = [
            r'import\s+',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__\w+__',
            r'open\s*\(',
            r'file\s*\(',
            r'subprocess',
            r'os\.',
            r'sys\.'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code_string):
                return False, f"Unsafe pattern detected: {pattern}"
        
        return True, None
    
    def validate_output_range(self, code_string: str, n_test: int = 10) -> Tuple[bool, Optional[str]]:
        """Test that output is in [0, 2π] for random inputs"""
        try:
            # Create safe namespace with built-ins for list comprehensions
            namespace = {
                'np': np,
                'numpy': np,
                'range': range,
                'len': len,
                'max': max,
                'min': min,
                'x': None
            }
            
            angle_samples = []
            
            # Test with random inputs
            for _ in range(n_test):
                test_input = np.random.rand(self.n_features)
                namespace['x'] = test_input
                
                result = eval(code_string, {"__builtins__": {}}, namespace)
                
                # Handle both scalar and list outputs
                if isinstance(result, (list, np.ndarray)):
                    # List of angles - validate each
                    for angle in result:
                        if not isinstance(angle, (int, float, np.number)):
                            return False, f"Output element must be numeric, got {type(angle)}"
                        if angle < 0 or angle > 2 * np.pi + 0.1:  # Small tolerance
                            return False, f"Output angle {angle:.3f} outside [0, 2π]"
                    angle_samples.append(np.array(result))
                elif isinstance(result, (int, float, np.number)):
                    # Scalar output
                    if result < 0 or result > 2 * np.pi + 0.1:  # Small tolerance
                        return False, f"Output {result:.3f} outside [0, 2π]"
                    angle_samples.append([result] * 10)  # Treat as 10 identical angles
                else:
                    return False, f"Output must be scalar or list, got {type(result)}"
            
            # Check diversity: angles should vary across qubits
            if angle_samples:
                avg_std = np.mean([np.std(angles) for angles in angle_samples])
                if avg_std < 0.3:
                    return False, f"Low angle diversity (std={avg_std:.3f}). All qubits too similar - try feature mixing!"
            
            return True, None
            
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def validate_template_compliance(self, code_string: str, template_family: str) -> Tuple[bool, Optional[str]]:
        """Check if function matches declared template family"""
        
        if template_family == "linear":
            # Should only have x[i] terms, no x[i]*x[j]
            if re.search(r'x\[\d+\]\s*\*\s*x\[\d+\]', code_string):
                return False, "Linear template should not have interaction terms"
        
        elif template_family == "polynomial":
            # Should have x[i]*x[j] terms
            if not re.search(r'x\[\d+\]\s*\*\s*x\[\d+\]', code_string):
                return False, "Polynomial template should have interaction terms"
        
        elif template_family == "global_stats":
            # Should have mean() or std()
            if not (re.search(r'mean\(', code_string) or re.search(r'std\(', code_string)):
                return False, "Global stats template should use mean() or std()"
        
        elif template_family == "pca_mix":
            # Should only use first 4 features
            indices = re.findall(r'x\[(\d+)\]', code_string)
            if indices and max(map(int, indices)) >= 4:
                return False, "PCA mix template should only use first 4 components"
        
        return True, None
    
    def validate_all(self, code_string: str, template_family: str) -> Tuple[bool, List[str]]:
        """Run all validation checks"""
        errors = []
        
        # Syntax check
        valid, error = self.validate_syntax(code_string)
        if not valid:
            errors.append(error)
            return False, errors  # Stop if syntax is invalid
        
        # Safety check
        valid, error = self.validate_safety(code_string)
        if not valid:
            errors.append(error)
        
        # Output range check
        valid, error = self.validate_output_range(code_string)
        if not valid:
            errors.append(error)
        
        # Template compliance check
        valid, error = self.validate_template_compliance(code_string, template_family)
        if not valid:
            errors.append(error)
        
        return len(errors) == 0, errors


# Test
if __name__ == "__main__":
    validator = EncodingValidator(n_features=10)
    
    # Test valid linear
    code1 = "np.clip(0.8 * x[0] + 0.2 * x[1], 0, 2*np.pi)"
    valid, errors = validator.validate_all(code1, "linear")
    print(f"Linear test: {valid}, errors: {errors}")
    
    # Test invalid (malicious)
    code2 = "import os; os.system('rm -rf /')"
    valid, errors = validator.validate_all(code2, "linear")
    print(f"Malicious test: {valid}, errors: {errors}")
    
    # Test invalid (out of range)
    code3 = "10.0 * x[0]"  # Will exceed 2π
    valid, errors = validator.validate_all(code3, "linear")
    print(f"Range test: {valid}, errors: {errors}")
