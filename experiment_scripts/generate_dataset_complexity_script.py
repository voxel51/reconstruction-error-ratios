import os

try:
    from COMMON import *
except:
    from .COMMON import *

script_str = f"python test_methods.py "

# Path to the shell script file
shell_script_path = "experiment_scripts/run_data_complexity.sh"

LOG_DIR = "logs/data_complexity"
os.makedirs(LOG_DIR, exist_ok=True)

NOISE_TYPE = "symmetric"
NOISE_FRAC = 0.0


feature = "dinov2-small" # or "clip-vit-base-patch32" or "resnet50-imagenet" ...
project_name = "reconstructor_dataset_complexity"


project_name_str = f" --project_name '{project_name}' "
noise_type_str = f" --noise_type '{NOISE_TYPE}' "
noise_frac_str = f" --noise_frac {NOISE_FRAC} "
method_str = f" --method '{METHOD}' "
feature_str = f" --features '{feature}' "
n_workers_str = f" --recon_n_workers {N_WORKERS} "
fit_frac = 1.0
fit_frac_str = f" --recon_fit_frac '{fit_frac}' "

commands = []

for dataset_name in DATASET_NAMES:
    
    dataset_name_str = f" --dataset_name '{dataset_name}' "
    
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

