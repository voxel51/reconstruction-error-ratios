import os

import numpy as np

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"/home/ubuntu/miniconda3/envs/py_311/bin/python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_reconstruction_ablations.sh"

LOG_DIR = "logs/reconstructions_ablations"
os.makedirs(LOG_DIR, exist_ok=True)

noise_frac = 0.0
project_name = "reconstructor_ablations"
dataset_name = "DescribableTextures"
noise_type = "symmetric"
seed = 13

# n_components = [2, 4, 10]
# n_components = [2, 4, 10, 20, 40, 80, 100, 200, 400]

# hidden_dims = [[8], [16], [32], [64], [128], [256], [1024], [2048]]
# hidden_dims = [[8], [16]]

n_neighbors = [2, 4, 8, 16, 32, 64, 160]
repulsion_strengths = [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

parametric_reconstruction_loss_weights = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

dropout_rates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

regs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# spreads = [5, 10, 15, 20, 25]
spreads = [1.0, 3.0, 5.0, 7.0, 9.0, 10.0, 20.0]

commands = []

project_name_str = f" --project_name '{project_name}' "
dataset_name_str = f" --dataset_name '{dataset_name}' "
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
    + dataset_name_str
    + noise_type_str
    + noise_frac_str
    + noise_seed_str
    + method_str
    + feature_str
    + project_name_str
    + n_workers_str
    + fit_frac_str
)


for spread in spreads:
    min_dist = np.round(spread * 24.0/25.0, 2)
    command_str = (
        base_command_str
        + f" --recon_spread {spread} --recon_min_dist {min_dist} --wandb_notes 'spread={spread}'"
    )
    commands.append(command_str)


# for n_neighbor in n_neighbors:
#     command_str = (
#         base_command_str
#         + f" --recon_n_neighbors {n_neighbor} --wandb_notes 'n_neighbors={n_neighbor}'"
#     )
#     commands.append(command_str)

# for pw in parametric_reconstruction_loss_weights:
#     command_str = (
#         base_command_str
#         + f" --recon_parametric_reconstruction_loss_weight {pw} --wandb_notes 'parametric_reconstruction_loss_weight={pw}'"
#     )
#     commands.append(command_str)


# for n_component in n_components:
#     command_str = (
#         base_command_str
#         + f" --recon_n_components {n_component} --wandb_notes 'n_components={n_component}'"
#     )
#     commands.append(command_str)


# for dropout_rate in dropout_rates:
#     command_str = (
#         base_command_str
#         + f" --recon_dropout {dropout_rate} --wandb_notes 'dropout={dropout_rate}'"
#     )
#     commands.append(command_str)

# for reg in regs:
#     command_str = (
#         base_command_str
#         + f" --recon_reg_strength {reg} --wandb_notes 'recon_reg_strength={reg}'"
#     )
#     commands.append(command_str)

# for repulsion_strength in repulsion_strengths:
#     command_str = (
#         base_command_str
#         + f" --recon_repulsion_strength {repulsion_strength} --wandb_notes 'repulsion_strength={repulsion_strength}'"
#     )
#     commands.append(command_str)


# for hidden_dim in hidden_dims:
#     command_str = (
#         base_command_str
#         + f" --recon_hidden_dims {' '.join(map(str, hidden_dim))} --wandb_notes 'hidden_dims={hidden_dim}'"
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
