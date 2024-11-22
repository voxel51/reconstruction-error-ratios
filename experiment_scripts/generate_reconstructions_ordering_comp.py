import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"/home/ubuntu/miniconda3/envs/py_311/bin/python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_ordering.sh"

LOG_DIR = "logs/error_ordering"
os.makedirs(LOG_DIR, exist_ok=True)


noise_type = "symmetric"
noise_frac = 0.0
noise_seed = 13
fit_frac = 1.0


features = [
    "clip-vit-base-patch32",
    # "clip-vit-large-patch14",
    # "dinov2-small",
    "dinov2-base",
    # "dinov2-large",
    # "dinov2-giant",
]

project_name = "reconstructor_error_ratio_ordering"


project_name_str = f" --project_name '{project_name}' "
noise_type_str = f" --noise_type '{noise_type}' "
noise_frac_str = f" --noise_frac {noise_frac} "
noise_seed_str = f" --noise_seed {noise_seed} "
method_str = f" --method '{METHOD}' "
feature_str = f" --features '{features}' "
n_workers_str = f" --recon_n_workers {N_WORKERS} "

fit_frac_str = f" --recon_fit_frac '{fit_frac}' "

commands = []


for feature in features:
    for dataset_name in DATASET_NAMES:
        dataset_name_str = f" --dataset_name '{dataset_name}' "
        feature_str = f" --features '{feature}' "


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
            + n_workers_str
            + " --wandb_log_artifacts "
            + " --recon_store_error_array "
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

