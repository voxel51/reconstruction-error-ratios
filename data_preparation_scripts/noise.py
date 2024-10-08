"""
Label noise generation methods

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
| Symmetric and asymmetric noise generation methods adapted from 
| https://github.com/UCSC-REAL/SimiFeat
"""

import argparse
import os

import numpy as np
import fiftyone as fo

from DEFAULTS import *


def noisify_labels(
    dataset_name=None, seed=None, noise_fracs=None, noise_type=None, **kwargs
):
    y_gt = _load_labels(dataset_name)
    num_classes = len(np.unique(y_gt))

    seed = seed if seed is not None else np.random.randint(1000)
    random_state = np.random.RandomState(seed)

    noise_func_kwargs = {
        "y_train": y_gt,
        "random_state": random_state,
        "num_classes": num_classes,
    }

    if noise_type == "symmetric":
        noise_func = noisify_multiclass_symmetric
    elif noise_type == "asymmetric":
        noise_func = noisify_multiclass_asymmetric
    elif noise_type == "human":
        noise_func = noisify_human
        y_noisy = _load_noisy_human_labels(dataset_name)
        noise_func_kwargs["y_noisy"] = y_noisy
    elif noise_type == "confidence":
        keys = list(kwargs.keys())
        for k in keys:
            v = kwargs[k]
            if k.startswith("conf_"):
                kwargs[k.replace("conf_", "")] = v
        noise_func = noisify_confidence
        y_noisy = _load_noisy_confidence_labels(dataset_name=dataset_name, **kwargs)
        noise_func_kwargs["y_noisy"] = y_noisy

    for noise_frac in noise_fracs:
        noise_func_kwargs["noise_frac"] = noise_frac


        noise_str = f"{noise_frac}_{noise_type}"
        if noise_type == "confidence":
            noise_str = f"{noise_str}_yolov8{kwargs.get('conf_model_size', 's')}_cls"
        seed_str = f"_{seed}" if seed is not None else ""
        y_noisy_filepath = f"data/{dataset_name}_y_noisy_{noise_str}{seed_str}.npy"

        if os.path.exists(y_noisy_filepath):
            continue

        cpt_labels = noise_func(**noise_func_kwargs)
        np.save(y_noisy_filepath, cpt_labels)


def noisify_multiclass_symmetric(
    y_train=None, noise_frac=None, random_state=None, num_classes=None, **kwargs
):
    """
    Adapted from https://github.com/UCSC-REAL/SimiFeat/blob/main/data/utils.py
    """
    P = np.ones((num_classes, num_classes))
    n = noise_frac
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1.0 - n
        for i in range(1, num_classes - 1):
            P[i, i] = 1.0 - n
        P[num_classes - 1, num_classes - 1] = 1.0 - n

        y_train_noisy = _multiclass_noisify(y_train, P, random_state=random_state)
        y_train = y_train_noisy

    return y_train


def noisify_multiclass_asymmetric(
    y_train=None, noise_frac=None, random_state=None, num_classes=None, **kwargs
):
    """mistakes:
    Adapted from https://github.com/UCSC-REAL/SimiFeat/blob/main/data/utils.py
    """
    P = np.eye(num_classes)
    n = noise_frac

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1.0 - n, n
        for i in range(1, num_classes - 1):
            P[i, i], P[i, i + 1] = 1.0 - n, n
        P[num_classes - 1, num_classes - 1], P[num_classes - 1, 0] = 1.0 - n, n

        y_train_noisy = _multiclass_noisify(y_train, P, random_state=random_state)
        y_train = y_train_noisy

    return y_train


def noisify_human(
    y_train=None, y_noisy=None, noise_frac=None, random_state=None, **kwargs
):
    error_inds = np.where(y_train != y_noisy)[0]
    num_errors = len(error_inds)
    max_error_frac = num_errors / len(y_train)

    if noise_frac > max_error_frac:
        raise ValueError(
            f"Cannot introduce {noise_frac} noise, max possible noise: {max_error_frac}"
        )

    num_errors_to_flip = int(noise_frac * len(y_train))
    ## randomly select num_errors_to_flip indices from error_inds
    flip_inds = random_state.choice(error_inds, num_errors_to_flip, replace=False)

    y_train_noisy = np.copy(y_train)
    y_train_noisy[flip_inds] = y_noisy[flip_inds]

    return y_train_noisy


def noisify_confidence(
    y_train=None, y_noisy=None, noise_frac=None, random_state=None, **kwargs
):
    """
    Every label in y_noisy is incorrect by construction, so just sample from y_noisy
    with probability noise_frac to introduce noise.
    """

    y_train_noisy = np.copy(y_train)
    num_errors = int(noise_frac * len(y_train))
    flip_inds = random_state.choice(np.arange(len(y_train)), num_errors, replace=False)
    y_train_noisy[flip_inds] = y_noisy[flip_inds]

    return y_train_noisy


def _store_y_gt_labels(dataset_name):
    sample_collection = _load_train_split(dataset_name)
    class_names = sample_collection.distinct("ground_truth.label")
    label2id = {label: i for i, label in enumerate(class_names)}

    labels = sample_collection.values("ground_truth.label")
    y_train = np.array([label2id[l] for l in labels])

    y_gt_filepath = f"data/{dataset_name}_y_gt.npy"
    np.save(y_gt_filepath, y_train)


def _load_labels(dataset_name):
    y_gt_filepath = f"data/{dataset_name}_y_gt.npy"
    if not os.path.exists(y_gt_filepath):
        _store_y_gt_labels(dataset_name)
    return np.load(y_gt_filepath)


def _load_noisy_human_labels(dataset_name):
    y_noisy_filepath = f"data/{dataset_name}_y_noisy_human.npy"
    if not os.path.exists(y_noisy_filepath):
        from download_cifar_human_noise import download_human_labels

        download_human_labels(dataset_name)
    return np.load(y_noisy_filepath)


def _format_kwargs(kwargs):
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k.replace("conf_", "")] = v
    return new_kwargs

def _generate_confidence_labels(dataset_name, y_noisy_filepath, **kwargs):
    from yolov8cls import train_classifier

    model_size = kwargs.get("conf_model_size", "s")

    train = _load_train_split(dataset_name)

    class_names = train._dataset.distinct("ground_truth.label")
    label2id = {label: i for i, label in enumerate(class_names)}

    model = train_classifier(dataset_name=dataset_name, **kwargs)
    label_field = f"yolov8{model_size}_cls_predictions"
    train.apply_model(model, label_field=label_field)
    

    gt_labels = np.array(train.values("ground_truth.label"))
    pred_labels = np.array(train.values(f"{label_field}.label"))
    pred_logits = np.array(train.values(f"{label_field}.logits"))

    y_noisy = []

    for idx in range(len(gt_labels)):
        if pred_labels[idx] != gt_labels[idx]:
            y_noisy.append(label2id[pred_labels[idx]])
        else:
            logits = pred_logits[idx]
            # set the predicted label to the 2nd highest confidence
            logits[label2id[gt_labels[idx]]] = -np.inf
            y_noisy.append(np.argmax(logits))

    y_noisy = np.array(y_noisy)

    np.save(y_noisy_filepath, y_noisy)


def _load_noisy_confidence_labels(dataset_name=None, **kwargs):
    model_size = kwargs.get("model_size", "s")
    y_noisy_filepath = (
        f"data/{dataset_name}_y_noisy_confidence_yolov8{model_size}_cls.npy"
    )
    if not os.path.exists(y_noisy_filepath):
        conf_kwargs = _format_kwargs(kwargs)
        _generate_confidence_labels(dataset_name, y_noisy_filepath, **conf_kwargs)

    return np.load(y_noisy_filepath)


def _load_train_split(dataset_name):
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
    else:
        from download_and_prepare_dataset import download_dataset

        dataset = download_dataset(dataset_name)

    return dataset.match_tags("train").exists("clip-vit-base-patch32")


def _multiclass_noisify(y, P, random_state=None):
    """
    Adapted from https://github.com/UCSC-REAL/SimiFeat/blob/main/data/utils.py
    """
    # Ensure random_state is a RandomState instance
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    new_y = np.copy(y)
    for idx in range(len(y)):
        i = y[idx]
        # Ensure P[i, :] is a 1-D array
        flipped = random_state.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument(
        "--noise_type",
        type=str,
        required=True,
        choices=["symmetric", "asymmetric", "human", "confidence"],
    )
    parser.add_argument(
        "--noise_fracs",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    parser.add_argument("--seed", type=int, default=42)

    ## Confidence-based noise generation
    parser.add_argument("--conf_model_size", type=str, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--conf_image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--conf_epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    kwargs = vars(args)
    noisify_labels(**kwargs)


if __name__ == "__main__":
    main()
