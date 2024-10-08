from itertools import product
import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *

script_str = f"python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_fit_frac_ablation.sh"

LOG_DIR = "logs/data_complexity"
os.makedirs(LOG_DIR, exist_ok=True)


dataset_names = ["cifar10", "cifar100"]
noise_types = ["symmetric", "asymmetric"]
noise_fracs = np.array([0.1, 0.2, 0.3])
project_name = "reconstructor_fit_frac_auroc"


project_name_str = f" --project_name '{project_name}' "

method_str = f" --method '{METHOD}' "
feature_str = f" --features '{FEATURES}' "
n_workers_str = f" --recon_n_workers {N_WORKERS} "

fit_fracs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


commands = []

for dataset_name, noise_frac, fit_frac, noise_type in product(
    dataset_names, noise_fracs, fit_fracs, noise_types
):

    dataset_name_str = f" --dataset_name '{dataset_name}' "
    fit_frac_str = f" --recon_fit_frac '{fit_frac}' "
    noise_frac_str = f" --noise_frac {noise_frac} "
    noise_type_str = f" --noise_type '{noise_type}' "

    if dataset_name == "cifar100" and fit_frac == 0.01:
        ## too few samples
        continue

    command_str = (
        script_str
        + dataset_name_str
        + noise_type_str
        + noise_frac_str
        + method_str
        + feature_str
        + project_name_str
        + fit_frac_str
        + n_workers_str
    )

    commands.append(command_str)


print(f"Total commands: {len(commands)}")


# Write the shell script to handle job execution without parallelization
with open(shell_script_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"mkdir -p {LOG_DIR}\n\n")

    for i, cmd in enumerate(commands):
        log_file = generate_log_file_name(LOG_DIR, cmd, i)
        f.write(f"echo 'Running command: {cmd}'\n")
        f.write(f"echo 'Logging to: {log_file}'\n")
        f.write(f"{cmd} > {log_file} 2>&1\n\n")

os.chmod(shell_script_path, 0o755)

print(f"Shell script '{shell_script_path}' has been created.")
