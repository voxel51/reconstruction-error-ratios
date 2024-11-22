#!/bin/bash
dataset_names=(
  "OxfordFlowers102" "cifar10" "cifar100" "caltech101" "caltech256" "CUB-200-2011" "deep-weeds" "DescribableTextures" "EuroSAT" "fashion-mnist" "FGVC-Aircraft" "Food101" "MIT-Indoor-Scenes" "mnist" "places" "RESISC45" "StanfordDogs" "SUN397"
)
# dataset_names=(
#  "chestmnist" "dermamnist" "octmnist" "organamnist" "origancmnist" "origansmnist" "pathmnist" "retinamnist" "tissuemnist"
# )

for dataset_name in "${dataset_names[@]}"; do
  /home/ubuntu/miniconda3/envs/py_311/bin/python data_preparation_scripts/generate_embeddings.py --dataset_name $dataset_name
done
