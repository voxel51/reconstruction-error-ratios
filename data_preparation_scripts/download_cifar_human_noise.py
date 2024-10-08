"""
CIFAR-10 and CIFAR-100 human noise download script

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
| Noisy Labels from https://github.com/UCSC-REAL/cifar-10-100n
"""

import argparse
import io
import requests
import numpy as np

import torch

common_path = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/"

cifar10_n_labels_url = common_path + "data/CIFAR-10_human.pt"
cifar100_n_labels_url = common_path + "data/CIFAR-100_human.pt"


def _load_torch_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Load the content into a BytesIO object and use torch.load()
        file_content = io.BytesIO(response.content)
        return torch.load(file_content, weights_only=False)
    else:
        print(f"Failed to retrieve file. Status code: {response.status_code}")
        return None

def download_cifar10_human_noise():
    cifar10_n_labels = _load_torch_file(cifar10_n_labels_url)['random_label1']
    np.save("data/cifar10_y_noisy_human.npy", cifar10_n_labels)

def download_cifar100_human_noise():
    cifar100_n_labels = _load_torch_file(cifar100_n_labels_url)['noisy_label']
    np.save("data/cifar100_y_noisy_human.npy", cifar100_n_labels)


def download_human_labels(dataset_name):
    if dataset_name == "cifar10":
        download_cifar10_human_noise()
    else:
        download_cifar100_human_noise()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    args = parser.parse_args()

    if args.dataset == "cifar10":
        download_cifar10_human_noise()
    else:
        download_cifar100_human_noise()


if __name__ == "__main__":
    main()