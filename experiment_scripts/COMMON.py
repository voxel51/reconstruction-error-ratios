import os
from datetime import datetime

SEEDS = [13, 42, 51]
N_WORKERS = 20
MAX_CONCURRENT_JOBS = 10
FEATURES = "clip-vit-large-patch14"
METHOD = "reconstruction"
COMPETING_METHODS = ["zero_shot", "confident_learning"]
NOISE_TYPES = ["symmetric", "asymmetric", "confidence"]

DATASET_NAMES = [
    "EuroSAT",
    "deep-weeds",
    "caltech101",
    "caltech256",
    "cifar10",
    "cifar100",
    "DescribableTextures",
    "fashion-mnist",
    "FGVC-Aircraft",
    "Food101",
    "MIT-Indoor-Scenes",
    "mnist",
    "OxfordFlowers102",
    "RESISC45",
    "StanfordDogs",
    "SUN397",
    "CUB-200-2011",
    "places",
    "imagenet",
]

MEDMNIST_DATASETS = [
    "bloodmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "retinamnist",
    "tissuemnist",
]

dataset_num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "StanfordDogs": 120,
    "EuroSAT": 10,
    "OxfordFlowers102": 102,
    "DescribableTextures": 47,
    "MIT-Indoor-Scenes": 67,
    "CUB-200-2011": 200,
    "FGVC-Aircraft": 100,
    "Food101": 101,
    "mnist": 10,
    "fashion-mnist": 10,
    "places": 365,
    "SUN397": 362,
    "deep-weeds": 9,
    "RESISC45": 45,
    "imagenet": 1000,
    "bloodmnist": 8,
    "chestmnist": 2,
    "dermamnist": 7,
    "octmnist": 4,
    "organamnist": 11,
    "organcmnist": 11,
    "organsmnist": 11,
    "pathmnist": 9,
    "retinamnist": 5,
    "tissuemnist": 8,
    "caltech101": 102,
    "caltech256": 257,
}


    

CIFAR10_HUMAN_LIMIT = 0.172
CIFAR100_HUMAN_LIMIT = 0.402
ASYMMETRIC_NOISE_LIMIT = 0.491
CONFIDENT_NOISE_LIMIT = 0.491
CONFIDENT_NOISE_MODEL_SIZE = ["s"]


def is_valid_noise(dataset_name, noise_type, noise_frac):
    if noise_type == "human":
        if dataset_name == "cifar10":
            if noise_frac > CIFAR10_HUMAN_LIMIT:
                return False
        elif dataset_name == "cifar100":
            if noise_frac > CIFAR100_HUMAN_LIMIT:
                return False
        else:
            return False
    elif noise_type == "asymmetric":
        if noise_frac > ASYMMETRIC_NOISE_LIMIT:
            return False

    elif noise_type == "confidence":
        if noise_frac > CONFIDENT_NOISE_LIMIT:
            return False

    return True


def generate_log_file_name(log_dir, cmd, index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd_name = cmd.split()[0].split("/")[-1]
    return os.path.join(log_dir, f"{cmd_name}_{index}_{timestamp}.log")
