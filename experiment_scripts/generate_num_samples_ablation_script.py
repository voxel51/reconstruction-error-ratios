from itertools import product
import os

try:
    from COMMON import *
except:
    from .COMMON import *


script_str = f"python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_data_complexity_num_samples.sh"

LOG_DIR = "logs/data_complexity"
os.makedirs(LOG_DIR, exist_ok=True)

NOISE_TYPE = "symmetric"
NOISE_FRAC = 0.0
project_name = "reconstructor_dataset_complexity"


project_name_str = f" --project_name '{project_name}' "
noise_type_str = f" --noise_type '{NOISE_TYPE}' "
noise_frac_str = f" --noise_frac {NOISE_FRAC} "
method_str = f" --method '{METHOD}' "
feature_str = f" --features '{FEATURES}' "
n_workers_str = f" --recon_n_workers {N_WORKERS} "

fit_samples_per_class = [20, 40, 60, 83, 100, 120, 140, 160, 180, 200]

commands = []

for dataset_name, fit_spc in product(DATASET_NAMES, fit_samples_per_class):
    
    dataset_name_str = f" --dataset_name '{dataset_name}' "
    fit_samples_str = f" --recon_fit_samples_per_class {fit_spc} "

    command_str = (
        script_str
        + dataset_name_str
        + noise_type_str
        + noise_frac_str
        + method_str
        + feature_str
        + project_name_str
        + fit_samples_str
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

