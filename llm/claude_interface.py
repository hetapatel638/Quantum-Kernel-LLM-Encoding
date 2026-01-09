"""
Level 2: Claude LLM Interface
Call Claude API to generate encoding formulas with fallback models
"""

import os
from typing import Tuple
from anthropic import Anthropic
from config import CONFIG


class ClaudeInterface:
    """Interface to Claude API for encoding generation with fallbacks"""
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic()
        self.api_key = api_key
        self.primary_model = CONFIG["llm"]["model"]
        self.fallback_models = CONFIG["llm"].get("fallback_models", [])
    
    def generate_encoding(self, prompt: str) -> Tuple[str, str]:
        """
        Call Claude to generate encoding formula with automatic fallback
        
        Returns:
            (encoding_code, explanation)
        """
        # Try primary model first
        models_to_try = [self.primary_model] + self.fallback_models
        
        for model in models_to_try:
            try:
                print(f"  [Attempting with {model}]")
                message = self.client.messages.create(
                    model=model,
                    max_tokens=CONFIG["llm"]["max_tokens"],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response_text = message.content[0].text
                
                # Extract Python code from response
                if "```python" in response_text:
                    code = response_text.split("```python")[1].split("```")[0].strip()
                elif "```" in response_text:
                    code = response_text.split("```")[1].split("```")[0].strip()
                else:
                    code = response_text.strip()
                
                print(f"  ✓ Success with {model}")
                return code, response_text
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ {model} failed: {error_msg[:80]}")
                if model == models_to_try[-1]:  # Last model
                    raise ValueError(f"All Claude models failed. Last error: {error_msg}")
