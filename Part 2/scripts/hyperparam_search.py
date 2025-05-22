"""
This script will be used for hyperparameter tuning.

- Define a list of hyperparameter combinations (e.g., as a list of dicts).
- For each combination, train a model using train.py logic.
- Log results, metrics, and best checkpoints for each run.
- Optionally, automate grid or random search.

To be implemented after initial training pipeline is validated.

"""

import itertools
import json
import subprocess
import os
from datetime import datetime

# Define the grid of hyperparameters to search
param_grid = {
    "lr": [0.001, 0.0005],
    "dropout_p": [0.3, 0.5],
    "batch_size": [32, 64],
    "num_epochs": [10],  # Keep short for search
}

# Generate all combinations
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Output folder for search results
search_dir = os.path.join(
    "..", "logs", "hyperparam_search_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(search_dir, exist_ok=True)

summary = []

for i, combo in enumerate(combinations):
    print(f"Running combo {i + 1}/{len(combinations)}: {combo}")
    # Save config for this run
    config_path = os.path.join(search_dir, f"config_{i + 1}.json")
    with open(config_path, "w") as f:
        json.dump(combo, f, indent=2)
    # Run train.py as a subprocess with environment variables for hyperparams
    env = os.environ.copy()
    for k, v in combo.items():
        env[f"HP_{k.upper()}"] = str(v)
    result = subprocess.run(
        ["python", os.path.join("..", "scripts", "train.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    # Save stdout/stderr
    with open(os.path.join(search_dir, f"run_{i + 1}_output.txt"), "w") as f:
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)
    # Optionally, parse and summarize results (e.g., best val acc from output)
    # Here, just log the config and return code
    summary.append({"config": combo, "returncode": result.returncode})

# Save summary
with open(os.path.join(search_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Hyperparameter search complete. Results in {search_dir}")
