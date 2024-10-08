from datetime import datetime
from itertools import product
import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"/python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_competing_probs.sh"

LOG_DIR = "logs/competing_probs"
os.makedirs(LOG_DIR, exist_ok=True)

model_size = "s"
noise_fracs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
max_concurrent_jobs = 10


## shared parameters
project_name = "mislabel_detection_probabilities"


commands = []

for dataset_name, noise_type, noise_frac, method, seed in product(
    DATASET_NAMES, NOISE_TYPES, noise_fracs, COMPETING_METHODS, SEEDS
):
    
    if noise_type != "symmetric" and noise_frac == 0.6:
        continue
    
    project_name_str = f" --project_name '{project_name}' "
    dataset_name_str = f" --dataset_name '{dataset_name}' "
    noise_type_str = f" --noise_type '{noise_type}' "
    noise_frac_str = f" --noise_frac {noise_frac} "
    noise_seed_str = f" --noise_seed {seed} "
    method_str = f" --method '{method}' "
    feature_str = f" --features '{FEATURES}' "


    command_str = (
        script_str
        + dataset_name_str
        + noise_type_str
        + noise_frac_str
        + noise_seed_str
        + method_str
        + feature_str
        + project_name_str
        + " --wandb_log_artifacts "
    )

    if noise_type == "confidence":
        command_str += f" --noise_conf_model_size {model_size}"

    commands.append(command_str)


print(f"Total commands: {len(commands)}")


commands = commands[1:]


# Write the shell script to handle job execution
with open(shell_script_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"mkdir -p {LOG_DIR}\n")
    f.write(f"MAX_CONCURRENT_JOBS={max_concurrent_jobs}\n")
    f.write(f"current_jobs=0\n\n")

    # Function to run a command
    for i, cmd in enumerate(commands):
        log_file = os.path.join(
            LOG_DIR, f"cmd_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        f.write(f"echo 'Running command: {cmd}'\n")
        f.write(f"echo 'Logging to: {log_file}'\n")
        f.write(f"{cmd} > {log_file} 2>&1 &\n")
        f.write("current_jobs=$((current_jobs + 1))\n")

        # Check if the max concurrent jobs limit is reached
        f.write("if [ $current_jobs -ge $MAX_CONCURRENT_JOBS ]; then\n")
        f.write("    wait -n\n")  # Wait for any job to finish
        f.write("    current_jobs=$((current_jobs - 1))\n")
        f.write("fi\n\n")

    # Wait for all remaining jobs to finish
    f.write("wait\n")
    f.write("echo 'All jobs completed.'\n")

# Make the shell script executable
os.chmod(shell_script_path, 0o755)

print(f"Shell script '{shell_script_path}' has been created.")
