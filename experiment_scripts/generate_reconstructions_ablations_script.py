import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"/home/ubuntu/miniconda3/envs/py_311/bin/python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_reconstruction_ablations.sh"

LOG_DIR = "logs/reconstructions_ablations_final"
os.makedirs(LOG_DIR, exist_ok=True)

noise_frac = 0.0
project_name = "reconstructor_ablations_final"
noise_type = "symmetric"
seed = 13
N_WORKERS = 20

# dataset_names = ["cifar10", "cifar100", "DescribableTextures", "StanfordDogs", "RESISC45"]
# dataset_names = ["FGVC-Aircraft", "deep-weeds", "EuroSAT"]
dataset_names = ["RESISC45"]

# n_neighbors = [2, 4, 8, 16, 32, 64, 128, 256, 512]
n_neighbors = [256, 512]

parametric_reconstruction_loss_weights = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
# parametric_reconstruction_loss_weights = [0.001, 0.01]


commands = []

project_name_str = f" --project_name '{project_name}' "
noise_type_str = f" --noise_type '{noise_type}' "
noise_frac_str = f" --noise_frac {noise_frac} "
noise_seed_str = f" --noise_seed {seed} "
method_str = f" --method '{METHOD}' "
feature_str = f" --features '{FEATURES}' "
n_workers_str = f" --recon_n_workers {N_WORKERS} "

fit_frac = 1.0
fit_frac_str = f" --recon_fit_frac '{fit_frac}' "

base_command_str = (
    script_str
    + noise_type_str
    + noise_frac_str
    + noise_seed_str
    + method_str
    + feature_str
    + project_name_str
    + n_workers_str
    + fit_frac_str
)



for dataset_name in dataset_names:
    dataset_name_str = f" --dataset_name '{dataset_name}' "
    for n_neighbor in n_neighbors:
        command_str = (
            base_command_str
            + dataset_name_str
            + f" --recon_n_neighbors {n_neighbor} --wandb_notes 'n_neighbors={n_neighbor}'"
        )
        commands.append(command_str)

    # for pw in parametric_reconstruction_loss_weights:
    #     command_str = (
    #         base_command_str
    #         + dataset_name_str
    #         + f" --recon_parametric_reconstruction_loss_weight {pw} --wandb_notes 'parametric_reconstruction_loss_weight={pw}'"
    #     )
    #     commands.append(command_str)



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
