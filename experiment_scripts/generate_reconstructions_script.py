from itertools import product
import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_reconstructions_main.sh"

LOG_DIR = "logs/main_mislabel_comp_recon"
os.makedirs(LOG_DIR, exist_ok=True)


noise_fracs = np.round(np.arange(0.05, 0.31, 0.05), 2)
noise_fracs = np.insert(noise_fracs, 0, 0.01)
noise_fracs = np.insert(noise_fracs, 0, 0.005)
noise_fracs = np.insert(noise_fracs, 0, 0.0)


project_name = "mislabel_detection_method_comparison"
fit_frac = 1.0


commands = []

for dataset_name, noise_type, noise_frac, seed in product(
    DATASET_NAMES, NOISE_TYPES, noise_fracs, SEEDS
):
    if not is_valid_noise(dataset_name, noise_type, noise_frac):
        continue

    project_name_str = f" --project_name '{project_name}' "
    dataset_name_str = f" --dataset_name '{dataset_name}' "
    noise_type_str = f" --noise_type '{noise_type}' "
    noise_frac_str = f" --noise_frac {noise_frac} "
    noise_seed_str = f" --noise_seed {seed} "
    method_str = f" --method '{METHOD}' "
    feature_str = f" --features '{FEATURES}' "
    n_workers_str = f" --recon_n_workers {N_WORKERS} "

    fit_frac_str = f" --recon_fit_frac '{fit_frac}' "

    command_str = (
        script_str
        + dataset_name_str
        + noise_type_str
        + noise_frac_str
        + noise_seed_str
        + method_str
        + feature_str
        + project_name_str
        + fit_frac_str
        + " --compute_f1_optimal "
        + n_workers_str
    )

    if noise_type == "confidence":
        for model_size in CONFIDENT_NOISE_MODEL_SIZE:
            command = command_str + f" --noise_conf_model_size {model_size}"
            commands.append(command)
    else:
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

