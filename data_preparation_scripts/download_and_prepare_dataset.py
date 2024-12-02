import argparse
import os
import requests
import tarfile
from zipfile import ZipFile

import numpy as np

import fiftyone as fo
import fiftyone.utils.random as four
import fiftyone.zoo as foz

FO_DOWNLOAD_DIR = fo.config.default_dataset_dir
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
ALL_DATASETS = ZOO_DATASETS + NONZOO_DATASETS + HF_DATASETS + MEDMNIST_DATASETS


## check if huggingface_hub is installed
def _has_huggingface_hub():
    try:
        import huggingface_hub

        return True
    except ImportError:
        return False


def download_dataset(dataset_name):
    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if dataset_name in ZOO_DATASETS:
        if dataset_name == "places":
            for sample in dataset.iter_samples(autosave=True, progress=True):
                label = sample.ground_truth.label
                ## reformat for Ultralytics training and CLIP embedding
                sample["ground_truth"].label = label[3:].replace("/", "-")
        else:
            dataset = foz.load_zoo_dataset(dataset_name)
        dataset.persistent = True

        if dataset_name == "fashion-mnist":
            # replace T-shirt/top with T-shirt
            for sample in dataset.iter_samples(autosave=True, progress=True):
                if sample.ground_truth.label == "T-shirt/top":
                    ## reformat for Ultralytics training and CLIP embedding
                    sample.ground_truth.label = "T-shirt"
        if "cifar" in dataset_name or "mnist" in dataset_name:
            return dataset

    if dataset_name == "EuroSAT":
        dataset = download_EuroSAT()
    elif dataset_name == "StanfordDogs":
        dataset = download_StanfordDogs()
    elif dataset_name == "CUB-200-2011":
        dataset = download_CUB_200_2011()
    elif dataset_name == "OxfordFlowers102":
        dataset = download_OxfordFlowers102()
    elif dataset_name == "SUN397":
        dataset = download_SUN397()
    elif dataset_name == "FGVC-Aircraft":
        dataset = download_FGVC_Aircraft()
    elif dataset_name == "DescribableTextures":
        dataset = download_DescribableTextures()
    elif dataset_name == "Food101":
        dataset = download_Food101()
    elif dataset_name == "MIT-Indoor-Scenes":
        dataset = download_mit_indoor_scenes()
    elif dataset_name == "deep-weeds":
        dataset = download_deep_weeds()
    elif dataset_name == "RESISC45":
        dataset = download_resisc45()
    elif "mnist" in dataset_name:
        dataset = download_medmnist_dataset(dataset_name)
    create_splits(dataset)
    return dataset


def create_splits(dataset):
    ## random split 90/10
    tags = dataset.distinct("tags")
    dataset.untag_samples(tags)
    train, test = four.random_split(dataset, [0.9, 0.1])
    train.tag_samples("train")
    test.tag_samples("test")


def download_hf_basic(repo_id, dataset_name):
    if not _has_huggingface_hub():
        raise ValueError("huggingface_hub is not installed. Please install it.")
    from fiftyone.utils.huggingface import load_from_hub

    dataset = load_from_hub(
        repo_id,
        format="parquet",
        classification_fields="label",
        batch_size=100,
        num_workers=4,
        name=dataset_name,
    )
    dataset.persistent = True
    dataset.rename_sample_field("label", "ground_truth")

    return dataset


def download_resisc45():
    return download_hf_basic("jonathan-roberts1/NWPU-RESISC45", "RESISC45")


def download_medmnist_dataset(dataset_name):
    if not _has_huggingface_hub():
        raise ValueError("huggingface_hub is not installed. Please install it.")
    from PIL import Image

    from huggingface_hub import hf_hub_download

    filepath = hf_hub_download(
        repo_id="albertvillanova/medmnist-v2", filename="data/bloodmnist"
    )
    download_dir = os.path.join(FO_DOWNLOAD_DIR, "medmnist")
    filepath = os.path.join(download_dir, f"{dataset_name}.npz")
    download_dir = os.path.join(download_dir, dataset_name)

    os.makedirs(download_dir, exist_ok=True)

    npz = np.load(filepath, allow_pickle=True)

    train_images = npz["train_images"]
    train_labels = npz["train_labels"]
    val_images = npz["val_images"]
    val_labels = npz["val_labels"]
    test_images = npz["test_images"]
    test_labels = npz["test_labels"]

    dataset = fo.Dataset(name=dataset_name, persistent=True)
    samples = []

    def _process_split(images, labels, split):
        for idx, (image, label) in enumerate(zip(images, labels)):
            filepath = os.path.join(download_dir, f"{split}_{idx}.png")
            image = (image * 255).astype(np.uint8)

            Image.fromarray(image).save(filepath)
            sample = fo.Sample(
                filepath=filepath, ground_truth=fo.Classification(label=str(label[0]))
            )
            samples.append(sample)

    _process_split(train_images, train_labels, "train")
    _process_split(val_images, val_labels, "val")
    _process_split(test_images, test_labels, "test")

    dataset.add_samples(samples)
    return dataset


def download_EuroSAT():
    if "EuroSAT" in fo.list_datasets():
        return fo.load_dataset("EuroSAT")

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "EuroSAT")
    url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"

    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)
    zip_path = os.path.join(download_dir, "EuroSATRGB.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    path = os.path.join(download_dir, "EuroSAT_RGB")

    dataset = fo.Dataset.from_dir(
        path, dataset_type=fo.types.ImageClassificationDirectoryTree
    )
    dataset.name = "EuroSAT"
    dataset.persistent = True
    return dataset


def download_StanfordDogs():
    if "StanfordDogs" in fo.list_datasets():
        return fo.load_dataset("StanfordDogs")

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "StanfordDogs")
    url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"

    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)
    tar_path = os.path.join(download_dir, "StanfordDogs.tar")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    path = os.path.join(download_dir, "Images")
    dataset = fo.Dataset.from_dir(
        path, dataset_type=fo.types.ImageClassificationDirectoryTree
    )
    dataset.name = "StanfordDogs"
    dataset.persistent = True

    for sample in dataset.iter_samples(progress=True, autosave=True):
        label = sample.ground_truth.label
        label = "-".join(label.split("-")[1:])
        label = label.replace("_", " ")
        sample.ground_truth.label = label
    return dataset


def download_CUB_200_2011():
    url = (
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    )
    download_dir = os.path.join(FO_DOWNLOAD_DIR, "CUB_200_2011")

    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)
    tar_path = os.path.join(download_dir, "CUB_200_2011.tgz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    path = os.path.join(download_dir, "CUB_200_2011", "images")
    dataset = fo.Dataset.from_dir(
        path, dataset_type=fo.types.ImageClassificationDirectoryTree
    )
    dataset.name = "CUB-200-2011"
    dataset.persistent = True

    for sample in dataset.iter_samples(progress=True, autosave=True):
        label = sample.ground_truth.label
        label = label.split(".")[1].replace("_", " ")
        sample.ground_truth.label = label
    return dataset


def download_OxfordFlowers102():
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    download_dir = os.path.join(FO_DOWNLOAD_DIR, "OxfordFlowers102")

    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)
    tar_path = os.path.join(download_dir, "102flowers.tgz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    path = os.path.join(download_dir, "jpg")
    dataset = fo.Dataset.from_images_dir(path)
    dataset.name = "OxfordFlowers102"
    dataset.persistent = True

    set_id_mat_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    set_id_mat_path = os.path.join(download_dir, "setid.mat")
    response = requests.get(set_id_mat_url)
    with open(set_id_mat_path, "wb") as f:
        f.write(response.content)

    image_labels_url = (
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    )
    image_labels_path = os.path.join(download_dir, "imagelabels.mat")
    response = requests.get(image_labels_url)
    with open(image_labels_path, "wb") as f:
        f.write(response.content)

    from scipy.io import loadmat

    labels = loadmat(image_labels_path)["labels"][0]
    labels = [fo.Classification(label=str(label)) for label in labels]
    dataset.set_values("ground_truth", labels)

    return dataset


def download_SUN397():
    url = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "SUN397")
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)

    tar_path = os.path.join(download_dir, "SUN397.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    dataset = fo.Dataset(name="SUN397")

    path = os.path.join(download_dir, "SUN397")
    from glob import glob

    subdirs = glob(os.path.join(path, "*"))

    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue
        dataset.add_dir(subdir, dataset_type=fo.types.ImageClassificationDirectoryTree)

    dataset.persistent = True
    return dataset


def download_FGVC_Aircraft():
    url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "FGVC-Aircraft")
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)

    tar_path = os.path.join(download_dir, "fgvc-aircraft-2013b.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    images_dir = os.path.join(download_dir, "fgvc-aircraft-2013b", "data", "images")
    dataset = fo.Dataset.from_images_dir(images_dir)

    dataset.name = "FGVC-Aircraft"
    dataset.persistent = True

    images_variant_trainval_path = os.path.join(
        download_dir, "fgvc-aircraft-2013b", "data", "images_variant_trainval.txt"
    )
    images_variant_test_path = os.path.join(
        download_dir, "fgvc-aircraft-2013b", "data", "images_variant_test.txt"
    )

    with open(images_variant_trainval_path, "r") as f:
        lines = f.readlines()

    mapping = {
        line.split(" ")[0]: " ".join(line.split(" ")[1:]).strip() for line in lines
    }

    with open(images_variant_test_path, "r") as f:
        lines = f.readlines()

    mapping.update(
        {line.split(" ")[0]: " ".join(line.split(" ")[1:]).strip() for line in lines}
    )

    for sample in dataset.iter_samples(autosave=True, progress=True):
        variant = mapping[sample.filename.split(".")[0]].replace("/", "-")
        sample["ground_truth"] = fo.Classification(label=variant)

    return dataset


def download_DescribableTextures():
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "DescribableTextures")
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)

    tar_path = os.path.join(download_dir, "dtd-r1.0.1.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    images_dir = os.path.join(download_dir, "dtd", "images")
    dataset = fo.Dataset.from_dir(
        images_dir, dataset_type=fo.types.ImageClassificationDirectoryTree
    )

    dataset.name = "DescribableTextures"
    dataset.persistent = True

    return dataset


def download_Food101():
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "Food101")
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)

    tar_path = os.path.join(download_dir, "food-101.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    images_dir = os.path.join(download_dir, "food-101", "images")
    dataset = fo.Dataset.from_dir(
        images_dir, dataset_type=fo.types.ImageClassificationDirectoryTree
    )

    dataset.name = "Food101"
    dataset.persistent = True

    return dataset


def download_mit_indoor_scenes():
    if "MIT-Indoor-Scenes" in fo.list_datasets():
        return fo.load_dataset("MIT-Indoor-Scenes")

    url = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "MIT-Indoor-Scenes")
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)

    tar_path = os.path.join(download_dir, "indoorCVPR_09.tar")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(download_dir)

    images_dir = os.path.join(download_dir, "Images")
    dataset = fo.Dataset.from_dir(
        images_dir, dataset_type=fo.types.ImageClassificationDirectoryTree
    )

    dataset.name = "MIT-Indoor-Scenes"
    dataset.persistent = True

    return dataset


def download_deep_weeds():
    if "deep-weeds" in fo.list_datasets():
        return fo.load_dataset("deep-weeds")

    download_dir = os.path.join(FO_DOWNLOAD_DIR, "deep-weeds")
    images_path = os.path.join(download_dir, "images.zip")
    import eta.core.web as etaw

    # Download the file
    etaw.download_google_drive_file(
        "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj", path=images_path
    )
    print("Download complete.")

    with ZipFile(images_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    labels_url = (
        "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"
    )

    # # Send a GET request to the URL
    response = requests.get(labels_url)
    labels_file = os.path.join(download_dir, "labels.csv")

    # Save the content to a local file
    with open(labels_file, "wb") as file:
        file.write(response.content)

    import pandas as pd

    labels = pd.read_csv(labels_file)

    dataset = fo.Dataset(name="deep-weeds", persistent=True)
    samples = []
    for _, row in labels.iterrows():
        filename = row.Filename
        species = row.Species
        filepath = os.path.join(download_dir, filename)
        sample = fo.Sample(
            filepath=filepath, ground_truth=fo.Classification(label=species)
        )
        samples.append(sample)

    dataset.add_samples(samples)

    return dataset


def prepare_dataset(dataset):
    if not dataset.has_field("clip-vit-base-patch32"):
        from generate_embeddings import generate_clip_b32_embeddings

        generate_clip_b32_embeddings(dataset)

    ## store X, y
    train = dataset.match_tags("train").exists("clip-vit-base-patch32")
    X = np.array(train.values("clip-vit-base-patch32"))
    y = np.array(train.values("ground_truth.label"))
    class_names = dataset.distinct("ground_truth.label")
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_to_idx[label] for label in y])

    np.save(f"data/{dataset.name}_X.npy", X)
    np.save(f"data/{dataset.name}_y_gt.npy", y)


def download_and_prepare(dataset_name):
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = download_dataset(dataset_name)

    prepare_dataset(dataset)
    return


def store_y_for_features(dataset_name, features):
    dataset = fo.load_dataset(dataset_name)
    train = dataset.match_tags("train").exists(features)

    y = np.array(train.values("ground_truth.label"))
    class_names = dataset.distinct("ground_truth.label")
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_to_idx[label] for label in y])

    np.save(f"data/{dataset_name}_y_{features}_gt.npy", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## allow any of ALL_DATASETS
    parser.add_argument(
        "--dataset_name", type=str, default="all", choices=ALL_DATASETS + ["all"]
    )
    args = parser.parse_args()

    if args.dataset_name is not "all":
        download_and_prepare(args.dataset_name)
    else:
        for dataset_name in ALL_DATASETS:
            download_and_prepare(dataset_name)
