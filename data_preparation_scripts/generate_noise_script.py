from datetime import datetime
from itertools import product
import os

import numpy as np

script_str = f"python data_preparation_scripts/noise.py "


DATASET_NAMES = [
    "caltech101",
    "caltech256",
    "cifar10",
    "cifar100",
    "CUB-200-2011",
    "deep-weeds",
    "DescribableTextures",
    "EuroSAT",
    "fashion-mnist",
    "FGVC-Aircraft",
    "Food101",
    "MIT-Indoor-Scenes",
    "mnist",
    "OxfordFlowers102",
    "places",
    "RESISC45",
    "StanfordDogs",
    "SUN397",
]

SEEDS = [13, 42, 51]
CIFAR10_HUMAN_LIMIT = 0.172
CIFAR100_HUMAN_LIMIT = 0.402
ASYMMETRIC_NOISE_LIMIT = 0.491
CONFIDENT_NOISE_LIMIT = 0.491
# CONFIDENT_NOISE_MODEL_SIZE = ["n", "s", "m"]
CONFIDENT_NOISE_MODEL_SIZE = ["s"]


NOISE_TYPES = ["symmetric", "asymmetric", "confidence"]


noise_fracs = np.round(np.arange(0.05, 0.31, 0.05), 2)
noise_fracs = np.insert(noise_fracs, 0, 0.01)
noise_fracs = np.insert(noise_fracs, 0, 0.005)


commands = []

for dataset_name, noise_type, noise_frac, seed in product(DATASET_NAMES, NOISE_TYPES, noise_fracs, SEEDS):
    if noise_type == "human":
        if dataset_name == "cifar10":
            if noise_frac > CIFAR10_HUMAN_LIMIT:
                continue
        elif dataset_name == "cifar100":
            if noise_frac > CIFAR100_HUMAN_LIMIT:
                continue
        else:
            continue
    elif noise_type == "asymmetric":
        if noise_frac > ASYMMETRIC_NOISE_LIMIT:
            continue

    elif noise_type == "confidence":
        if noise_frac > CONFIDENT_NOISE_LIMIT:
            continue

    command_str = f"--dataset_name {dataset_name} --noise_type {noise_type} --noise_frac {noise_frac} --seed {seed}"
    if noise_type == "confidence":
        for model_size in CONFIDENT_NOISE_MODEL_SIZE:
            command = command_str + f" --conf_model_size {model_size}"
            commands.append(command)
    else:
        commands.append(command_str)


print(f"Total commands: {len(commands)}")

commands = [script_str + cmd for cmd in commands]

LOG_DIR = "noise_gen_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def generate_log_file_name(cmd, index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd_name = cmd.split()[0].split("/")[-1]
    return os.path.join(LOG_DIR, f"{cmd_name}_{index}_{timestamp}.log")

# Path to the shell script file
shell_script_path = "data_preparation_scripts/generate_noise.sh"

# Write the shell script to handle job execution without parallelization
with open(shell_script_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"mkdir -p {LOG_DIR}\n\n")

    for i, cmd in enumerate(commands):
        log_file = generate_log_file_name(cmd, i)

        ## only echo if multiple of 10, and echo command number
        if i % 10 == 0:
            f.write(f"echo 'Command {i}/{len(commands)}'\n")
        f.write(f"{cmd} > {log_file} 2>&1\n\n")

os.chmod(shell_script_path, 0o755)

print(f"Shell script '{shell_script_path}' has been created.")
