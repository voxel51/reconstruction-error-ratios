import argparse
import numpy as np
from sklearn.cluster import KMeans

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
    "imagenet_subset1M": 1000,
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

ZOO_DATASETS = [
    "cifar10",
    "cifar100",
    "caltech256",
    "mnist",
    "fashion-mnist",
    "places",
    "caltech101",
]
NONZOO_DATASETS = [
    "EuroSAT",
    "StanfordDogs",
    "CUB-200-2011",
    "OxfordFlowers102",
    "SUN397",
    "FGVC-Aircraft",
    "DescribableTextures",
    "Food101",
    "MIT-Indoor-Scenes",
    "deep-weeds",
    "imagenet",
]
HF_DATASETS = [
    "RESISC45",
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
ALL_DATASETS = ZOO_DATASETS + NONZOO_DATASETS + HF_DATASETS #+ MEDMNIST_DATASETS


def _load_features(dataset_name, features):
    X = np.load(f"data/{dataset_name}_X_{features}.npy")
    return X


def _load_gt_labels(dataset_name):
    y = np.load(f"data/{dataset_name}_y_gt.npy")
    return y

def _get_num_classes(dataset_name):
    return len(np.unique(_load_gt_labels(dataset_name)))


def _cluster_features(X, num_clusters, cluster_method):
    if cluster_method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(X)
        return kmeans.labels_
    else:
        raise ValueError(f"Unknown cluster method {cluster_method}")


def _store_clusters(dataset_name, features, num_clusters, cluster_method):
    if num_clusters is None:
        num_clusters = _get_num_classes(dataset_name)
    
    X = _load_features(dataset_name, features)
    labels = _cluster_features(X, num_clusters, cluster_method)
    np.save(
        f"data/{dataset_name}_clusters_{features}_{num_clusters}_{cluster_method}.npy",
        labels,
    )


def generate_clusters(dataset_name, features, num_clusters, cluster_method):
    _store_clusters(dataset_name, features, num_clusters, cluster_method)


cluster_nums = [5, 20, 30, 40, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--features", type=str, default=None)
    # parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--cluster_method", type=str, default="kmeans")
    args = parser.parse_args()

    # dataset_name = args.dataset_name
    features = args.features
    # num_clusters = args.num_clusters
    cluster_method = args.cluster_method

    for dataset_name in ALL_DATASETS:
        nc = dataset_num_classes[dataset_name]
        cluster_nums = [nc - 1, nc + 1, int(0.9 * nc), int(1.1 * nc)]
        for cluster_num in cluster_nums:
            print(f"Generating clusters for {dataset_name} with {cluster_num} clusters")
            generate_clusters(dataset_name, features, cluster_num, cluster_method)

    # for cluster_num in cluster_nums:
    #     for dataset_name in ALL_DATASETS:
    #         print(f"Generating clusters for {dataset_name} with {cluster_num} clusters")
    #         generate_clusters(dataset_name, features, cluster_num, cluster_method)

    # if dataset_name is not None:
    #     generate_clusters(dataset_name, features, num_clusters, cluster_method)
    # else:
    #     for dataset_name in ALL_DATASETS:
    #         print(f"Generating clusters for {dataset_name}")
    #         generate_clusters(dataset_name, features, num_clusters, cluster_method)
