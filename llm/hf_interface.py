import json
import numpy as np
from typing import Dict, Any, List

# Import anthropic only when needed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMInterface:
    """
    Backward-compatible interface for existing pipeline.
    Uses Claude API if available, otherwise falls back to mock functions.
    """
    def __init__(self, model_name: str = None, use_local: bool = True):
        self.model_name = model_name
        self.use_local = use_local
        
        # Try to use Claude if API key is available
        import os
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.use_claude = True
            print("Using Claude Haiku API for encoding generation...")
        else:
            self.use_claude = False
            if not ANTHROPIC_AVAILABLE:
                print("anthropic library not installed, using mock LLM...")
            else:
                print("No ANTHROPIC_API_KEY found, using mock LLM...")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> str:
        """Generate encoding function using Claude or mock"""
        if self.use_claude:
            return self._generate_with_claude(prompt, temperature)
        else:
            return self._generate_mock(prompt)
    
    def _generate_with_claude(self, prompt: str, temperature: float) -> str:
        """Use Claude Haiku to generate high-quality encoding"""
        
        # Enhanced prompt with clear requirements
        enhanced_prompt = f"""{prompt}

OBJECTIVE:
Generate a quantum encoding function that performs better than the baseline.

BASELINE PERFORMANCE:
- Simple encoding: theta_i = pi * x[i]
- Accuracy: 68-83% depending on dataset
- Your encoding needs to achieve at least 2-10% improvement

REQUIREMENTS:
1. Output must be a Python list with exactly 10 angles [theta_0, theta_1, ..., theta_9]
2. Format: [expression for i in range(10)]
3. All angles must be in range [0, 2*pi] using np.clip(..., 0, 2*np.pi)
4. Each qubit should get different angle values (diversity is important)
5. Do not just copy the baseline theta_i = pi * x[i]

NOTE: Circuit now uses multi-layer rotations (RX, RY, RZ) with entanglement.
This means simpler encodings can work better since circuit adds complexity.

STRATEGIES:

Strategy 1 - Amplitude Encoding:
Use power scaling to compress dynamic range.
Formula: theta_i = pi * x[i]**alpha where alpha in [0.6, 0.9]
Works by giving more resolution to small values.

Example:
[np.clip(np.pi * x[i%len(x)]**0.75, 0, 2*np.pi) for i in range(10)]

Strategy 2 - Phase Shift Encoding:
Add structured phase differences between qubits.
Formula: theta_i = pi * x[i] + beta * pi * (i / n_qubits)
Creates interference patterns that help classification.

Example:
[np.clip(np.pi * x[i%len(x)] + 0.4*np.pi*(i/10), 0, 2*np.pi) for i in range(10)]

Strategy 3 - Weighted Feature Encoding:
Emphasize important features, de-emphasize noise.
Formula: theta_i = pi * x[i] * weight_i where early features get higher weight

Example:
[np.clip(np.pi * x[i%len(x)] * (1.0 if i < 5 else 0.5), 0, 2*np.pi) for i in range(10)]

Strategy 4 - Differential Encoding:
Encode local differences between adjacent features.
Formula: theta_i = pi * (x[i] + gamma * (x[i+1] - x[i]))

Example:
[np.clip(np.pi * (x[i%len(x)] + 0.3*(x[(i+1)%len(x)] - x[i%len(x)])), 0, 2*np.pi) for i in range(10)]

IMPORTANT - DO NOT GENERATE:
- theta_i = pi * x[i] (this is just the baseline)
- Anything using np.mean(x) or np.std(x) (makes all qubits the same)
- theta_i = c * x[i] where c is just a different constant (still baseline)
- Same expression for all i values (no diversity)

TIPS:
- Use different feature indices like x[i], x[(i+3)%len(x)], x[(i+7)%len(x)]
- Keep coefficient weights summing to around 1.0
- Quadratic terms should use smaller weights (0.1 to 0.3)
- Make sure angles are different across qubits

OUTPUT FORMAT (return valid JSON without markdown):
{{
  "function": "[np.clip(...your expression..., 0, 2*np.pi) for i in range(10)]",
  "explanation": "Brief description of your approach"
}}

Generate an encoding function that improves over baseline:"""
        
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                temperature=max(0.95, temperature),  # Force high creativity
                messages=[{"role": "user", "content": enhanced_prompt}]
            )
            
            response_text = message.content[0].text
            
            # Clean markdown formatting if present
            response_text = response_text.strip()
            response_text = response_text.replace("```json", "").replace("```", "")
            response_text = response_text.strip()
            
            # Try to parse as JSON
            try:
                parsed = json.loads(response_text)
                return json.dumps(parsed)
            except:
                # Extract JSON if embedded in text
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    return json_match.group()
                else:
                    print(f"Warning: Could not parse Claude response, using fallback")
                    return self._generate_mock(prompt)
                    
        except Exception as e:
            print(f"Claude API error: {e}")
            print("Falling back to mock generation...")
            return self._generate_mock(prompt)
    
    def _generate_mock(self, prompt: str) -> str:
        """Fallback mock generation - returns vector-valued function for multi-qubit encoding"""
        # Parse template family from prompt
        if "linear" in prompt.lower():
            return json.dumps({
                "function": "[np.clip(np.pi * x[i % len(x)], 0, 2*np.pi) for i in range(10)]",
                "explanation": "Per-qubit linear encoding: θᵢ = π·xᵢ"
            })
        elif "polynomial" in prompt.lower():
            return json.dumps({
                "function": "[np.clip(np.pi * (x[i % len(x)] + 0.1 * x[i % len(x)]**2), 0, 2*np.pi) for i in range(10)]",
                "explanation": "Per-qubit polynomial encoding with quadratic term"
            })
        else:
            # Global stats
            return json.dumps({
                "function": "[np.clip(np.pi * (0.7 * x[i % len(x)] + 0.3 * np.mean(x)), 0, 2*np.pi) for i in range(10)]",
                "explanation": "Per-qubit mix of local feature and global mean"
            })
    
    def parse_json_response(self, response: str) -> dict:
        """Parse JSON response"""
        try:
            return json.loads(response)
        except:
            return None


class ClaudeLightweightFiller:
    """
    Uses Claude Haiku for parameter filling
    
    Why Claude Haiku:
    - Small model (efficient)
    - Cheap (~$1/million tokens)
    - Good at structured tasks
    - Reliable output formatting
    
    Cost comparison:
    - Sakka: 3 LLMs (o3-mini + GPT-4o + GPT-4o-mini) ~$5/trial
    - Ours: 1 LLM (Claude Haiku) ~$0.10/trial
    - Savings: 50× cheaper
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-haiku-20240307"  # Lightweight model
        self.call_count = 0
        self.total_cost = 0.0
    
    def fill_template(
        self, 
        template_name: str,
        template_spec: Dict,
        dataset_context: Dict,
        performance_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Fill template parameters using Claude Haiku
        """
        # Build structured prompt
        prompt = self._build_prompt(
            template_name, 
            template_spec, 
            dataset_context,
            performance_history
        )
        
        # Call Claude
        response = self._call_claude(prompt)
        
        # Parse response
        params = self._parse_response(response, template_spec)
        
        # Track metrics
        self.call_count += 1
        self.total_cost += self._estimate_cost(prompt, response)
        
        return params
    
    def _build_prompt(
        self,
        template_name: str,
        template_spec: Dict,
        dataset_context: Dict,
        performance_history: List[Dict]
    ) -> str:
        """
        Build structured prompt for Claude
        
        Key strategy: Very explicit instructions
        """
        
        prompt = f"""You are a quantum computing expert designing feature maps.

TASK: Fill numerical parameters for the {template_spec['name']} template.

DATASET INFORMATION:
- Name: {dataset_context['name']}
- Features: {dataset_context['n_features']} (after PCA reduction)
- Classes: {dataset_context['n_classes']}
- Feature range: [{dataset_context['feature_min']:.3f}, {dataset_context['feature_max']:.3f}]
- Sample size: {dataset_context['n_samples']}

TEMPLATE SPECIFICATION:
- Name: {template_spec['name']}
- Formula: {template_spec['formula']}
- Theory: {template_spec.get('theory', 'N/A')}

PARAMETERS TO FILL:
"""
        
        # Add each parameter with constraints
        for param_name, param_info in template_spec['parameters'].items():
            prompt += f"\n{param_name}:"
            prompt += f"\n  Description: {param_info['description']}"
            prompt += f"\n  Range: {param_info['range']}"
            if 'constraint' in param_info:
                prompt += f"\n  Constraint: {param_info['constraint']}"
            if 'dimension' in param_info:
                prompt += f"\n  Dimension: {param_info['dimension']}"
        
        # Add performance guidance if available
        if performance_history:
            prompt += f"\n\nPERFORMANCE HISTORY:"
            
            # Show best performing templates
            sorted_history = sorted(
                performance_history, 
                key=lambda x: x.get('test_acc', 0), 
                reverse=True
            )
            
            prompt += "\nTop 3 performers:"
            for i, trial in enumerate(sorted_history[:3], 1):
                prompt += f"\n  {i}. {trial['template']}: {trial['test_acc']:.4f}"
                prompt += f"\n     Parameters: {trial['params']}"
            
            # Template-specific history
            template_trials = [
                t for t in performance_history 
                if t['template'] == template_name
            ]
            
            if template_trials:
                avg_acc = np.mean([t['test_acc'] for t in template_trials])
                best_trial = max(template_trials, key=lambda x: x['test_acc'])
                
                prompt += f"\n\n{template_name} history:"
                prompt += f"\n  Average: {avg_acc:.4f}"
                prompt += f"\n  Best: {best_trial['test_acc']:.4f}"
                prompt += f"\n  Best params: {best_trial['params']}"
                prompt += f"\n  → TIP: Try values close to best parameters"
        
        # Output format instructions (CRITICAL for reliable parsing)
        prompt += """

OUTPUT FORMAT (CRITICAL - Follow Exactly):
Return ONLY a valid JSON object. No explanations, no markdown formatting.

"""
        
        # Show example based on template type
        if template_name == "linear":
            prompt += """Example for linear template:
{"b0": 0.785, "b1": 3.14159}

"""
        elif template_name == "polynomial":
            prompt += """Example for polynomial template:
{"w_matrix": [[0.1, -0.05, 0.08], [-0.05, 0.12, -0.03], [0.08, -0.03, 0.15]], "alpha": 1.0}

"""
        elif template_name == "global_state":
            prompt += """Example for global_state template:
{"a": [0.5, 0.4, 0.6, 0.7, 0.3, 0.8, 0.5, 0.6, 0.4, 0.7], "G": 2.5}

"""
        elif template_name == "pca_weighted":
            prompt += """Example for pca_weighted template:
{"v": [0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03]}

"""
        
        prompt += "\nNow generate the parameters as JSON:"
        
        return prompt
    
    def _call_claude(self, prompt: str) -> str:
        """
        Call Claude Haiku API
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,  # Some creativity
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"Claude API call failed: {e}")
            raise
    
    def _parse_response(
        self, 
        response: str, 
        template_spec: Dict
    ) -> Dict[str, Any]:
        """
        Parse Claude's response
        
        Claude is good at following JSON format,
        but we still validate carefully
        """
        # Remove any markdown formatting
        text = response.strip()
        text = text.replace("```json", "").replace("```", "")
        text = text.strip()
        
        # Parse JSON
        try:
            params = json.loads(text)
        except json.JSONDecodeError as e:
            # Fallback: try to extract JSON from text
            import re
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse response: {response}\nError: {e}")
        
        # Validate and normalize
        params = self._validate_params(params, template_spec)
        
        return params
    
    def _validate_params(
        self, 
        params: Dict, 
        template_spec: Dict
    ) -> Dict:
        """
        Validate parameters and clip to valid ranges
        """
        validated = {}
        
        for param_name, param_info in template_spec['parameters'].items():
            if param_name not in params:
                # Use default if available
                if 'default' in param_info:
                    validated[param_name] = param_info['default']
                else:
                    raise ValueError(f"Missing required parameter: {param_name}")
            else:
                value = params[param_name]
                
                # Handle arrays
                if isinstance(value, (list, np.ndarray)):
                    value = np.array(value)
                    
                    # Check dimension if specified
                    if 'dimension' in param_info:
                        expected_dim = param_info['dimension']
                        if isinstance(expected_dim, tuple):
                            # Matrix
                            if value.shape != expected_dim:
                                # Try to reshape or pad/trim
                                value = np.array(value).flatten()[:np.prod(expected_dim)]
                                value = value.reshape(expected_dim)
                        else:
                            # Vector
                            if len(value) != expected_dim:
                                # Pad or trim
                                if len(value) < expected_dim:
                                    value = np.pad(value, (0, expected_dim - len(value)))
                                else:
                                    value = value[:expected_dim]
                    
                    # Clip to range
                    if 'range' in param_info:
                        vmin, vmax = param_info['range']
                        value = np.clip(value, vmin, vmax)
                    
                    validated[param_name] = value
                    
                else:
                    # Scalar
                    if 'range' in param_info:
                        vmin, vmax = param_info['range']
                        value = float(np.clip(value, vmin, vmax))
                    
                    validated[param_name] = float(value)
        
        return validated
    
    def _estimate_cost(self, prompt: str, response: str) -> float:
        """
        Estimate Claude Haiku cost
        
        Pricing (as of Dec 2024):
        - Input: $0.25 per million tokens
        - Output: $1.25 per million tokens
        """
        # Rough estimate: 1 token ≈ 4 characters
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        
        input_cost = (input_tokens / 1_000_000) * 0.25
        output_cost = (output_tokens / 1_000_000) * 1.25
        
        return input_cost + output_cost
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            'total_calls': self.call_count,
            'estimated_cost': self.total_cost,
            'avg_cost_per_call': self.total_cost / max(1, self.call_count)
        }

if __name__ == "__main__":
    print("CLAUDE HAIKU LLM INTERFACE DEMO")
   
    API_KEY = input("\nEnter your Anthropic API key: ").strip()
    
    # Create interface
    llm = ClaudeLightweightFiller(api_key=API_KEY)
    
    # Load template
    from template_library import get_template
    template_spec = get_template("linear")
    
    # Mock dataset context
    dataset_context = {
        'name': 'MNIST',
        'n_features': 80,
        'n_samples': 1000,
        'n_classes': 10,
        'feature_min': 0.0,
        'feature_max': 1.0,
        'feature_variance': 0.15
    }
    
    # Test without history
    print("\nTest 1: Fill LINEAR template (no history)")
    
    params = llm.fill_template(
        template_name="linear",
        template_spec=template_spec,
        dataset_context=dataset_context,
        performance_history=[]
    )
    
    print(f"✓ Claude filled parameters:")
    print(f"  b0 = {params['b0']:.4f}")
    print(f"  b1 = {params['b1']:.4f}")
    
    # Test with history
    print("\n\nTest 2: Fill LINEAR template (with history)")
    
    performance_history = [
        {
            'template': 'linear',
            'params': params,
            'test_acc': 0.945,
            'trial': 1
        }
    ]
    
    params2 = llm.fill_template(
        template_name="linear",
        template_spec=template_spec,
        dataset_context=dataset_context,
        performance_history=performance_history
    )
    
    print(f"Claude refined parameters:")
    print(f"b0 = {params2['b0']:.4f} (was {params['b0']:.4f})")
    print(f"b1 = {params2['b1']:.4f} (was {params['b1']:.4f})")
    
    # Show stats
    print("\n\nLLM USAGE STATS")
    stats = llm.get_stats()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Estimated cost: ${stats['estimated_cost']:.4f}")
    print(f"Avg cost/call: ${stats['avg_cost_per_call']:.4f}")
