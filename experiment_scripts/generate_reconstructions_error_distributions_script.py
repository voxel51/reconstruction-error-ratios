import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_reconstructions_error_distributions.sh"

LOG_DIR = "logs/error_distributions"
os.makedirs(LOG_DIR, exist_ok=True)


dataset_name = "cifar10"
noise_type = "symmetric"
noise_frac = 0.0
noise_seed = 13
fit_frac = 1.0


# feature = "clip-vit-base-patch32"
features = "dinov2-base"

project_name = "reconstructor_error_distributions"


project_name_str = f" --project_name '{project_name}' "
dataset_name_str = f" --dataset_name '{dataset_name}' "
noise_type_str = f" --noise_type '{noise_type}' "
noise_seed_str = f" --noise_seed {noise_seed} "
method_str = f" --method '{METHOD}' "
feature_str = f" --features '{features}' "
n_workers_str = f" --recon_n_workers {N_WORKERS} "

fit_frac_str = f" --recon_fit_frac '{fit_frac}' "

commands = []

noise_fracs = np.round(np.arange(0.0, 0.61, 0.05), 2)


for noise_frac in noise_fracs:
    noise_frac_str = f" --noise_frac {noise_frac} "


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
        + " --recon_compute_2d_umap "
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

