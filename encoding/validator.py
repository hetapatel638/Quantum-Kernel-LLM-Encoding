"""
Level 3: Encoding Validator
Validate generated encoding formulas
"""

import numpy as np
import re
from typing import Tuple, Optional


class EncodingValidator:
    """Validate generated encoding formulas"""
    
    @staticmethod
    def validate_encoding(code: str, X_sample: np.ndarray) -> Tuple[bool, str]:
        """
        Validate encoding formula
        
        Returns:
            (is_valid, error_message)
        """
        if not code or not isinstance(code, str):
            return False, "Code must be non-empty string"
        
        # Clean code: remove common issues
        code = code.strip()
        if code.startswith("return "):
            code = code[7:]  # Remove 'return' statement
        
        # Check 1: Valid Python syntax
        try:
            compile(code, '<string>', 'eval')
        except SyntaxError as e:
            return False, f"Invalid syntax: {str(e)}"
        
        # Check 2: Can execute with sample data
        try:
            for row in X_sample[:3]:
                result = eval(code, {"x": row, "np": np})
                if result is None:
                    return False, "Output is None"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
        
        # Check 3: Output type is numeric (array or scalar)
        try:
            for row in X_sample[:3]:
                result = eval(code, {"x": row, "np": np})
                # Accept scalars, arrays, or anything numeric
                if isinstance(result, (int, float, np.ndarray)):
                    continue
                else:
                    return False, f"Output type {type(result)} is not numeric"
        except Exception as e:
            return False, f"Type check error: {str(e)}"
        
        # Check 4: Output produces reasonable values
        try:
            test_vals = []
            for row in X_sample[:5]:
                result = eval(code, {"x": row, "np": np})
                if isinstance(result, np.ndarray):
                    test_vals.extend(result.flatten())
                else:
                    test_vals.append(float(result))
            
            test_vals = np.array(test_vals)
            
            # Check for NaN or Inf
            if np.any(np.isnan(test_vals)) or np.any(np.isinf(test_vals)):
                return False, "Output contains NaN or Inf values"
            
            # Warn if completely outside [0, 2π] but don't fail
            # (circuit will clip them)
            if np.mean(np.abs(test_vals)) > 10:
                print(f"  ⚠️  Warning: angles very large (mean={np.mean(test_vals):.2f}), will be clipped")
        
        except Exception as e:
            return False, f"Value check error: {str(e)}"
        
        return True, "Valid"
    
    @staticmethod
    def extract_coefficients(code: str) -> Optional[np.ndarray]:
        """Try to extract linear coefficients from code (best effort)"""
        # Look for patterns like 0.8*x[0] + 0.2*x[1]
        pattern = r'([\d.]+)\s*\*\s*x\[(\d+)\]'
        matches = re.findall(pattern, code)
        
        if not matches:
            return None
        
        coeffs = {}
        for coeff_str, idx_str in matches:
            idx = int(idx_str)
            coeff = float(coeff_str)
            coeffs[idx] = coeff
        
        if not coeffs:
            return None
        
        max_idx = max(coeffs.keys())
        result = np.zeros(max_idx + 1)
        for idx, coeff in coeffs.items():
            result[idx] = coeff
        
        return result
