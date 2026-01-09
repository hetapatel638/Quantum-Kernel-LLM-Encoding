#!/usr/bin/env python3
"""Real-time experiment monitor - no sleep, just polling"""

import subprocess
import sys
import time

terminal_id = "384264aa-6048-4f77-826f-478f56badb51"

print("Monitoring experiment progress...")
print("Press Ctrl+C to stop monitoring\n")

last_lines = []
iteration = 0

try:
    while True:
        # Get terminal output
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        # Check if experiment is still running
        is_running = "run_single_dataset.py" in result.stdout
        
        # Try to read from results file
        try:
            with open("results/mnist_linear_pca40.json", "r") as f:
                import json
                data = json.load(f)
                if 'comparison' in data:
                    print("\n" + "="*70)
                    print("EXPERIMENT COMPLETE!")
                    print("="*70)
                    print(f"Baseline:  {data['baseline']['classification']['accuracy']:.4f}")
                    print(f"Claude:    {data['llm_generated']['classification']['accuracy']:.4f}")
                    print(f"Improvement: {data['comparison']['relative_improvement_pct']:.2f}%")
                    print(f"Status: {data['status']}")
                    print("="*70)
                    break
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        
        # Show progress indicator
        iteration += 1
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][iteration % 10]
        status = "Running" if is_running else "Finishing"
        print(f"\r{spinner} {status}... ({iteration}s elapsed)", end='', flush=True)
        
        # If not running and no results, experiment may have failed
        if not is_running and iteration > 30:
            print("\n\n⚠ Experiment stopped but no results found. Check for errors.")
            break
        
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user")
