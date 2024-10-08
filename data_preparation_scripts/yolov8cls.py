"""
Ultralytics YOLOv8*-cls model training script
for generating confidence-based noise labels for a dataset.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|

Requires `ultralytics` and `fiftyone>=0.25.0` to be installed.
"""
import argparse
import os
import tempfile
import torch
from ultralytics import YOLO
import fiftyone as fo

try:
    from DEFAULTS import *
except:
    from .DEFAULTS import *


import wandb

wandb.require("core")


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_classifier(
    dataset_name=None,
    model_size=DEFAULT_MODEL_SIZE,
    image_size=DEFAULT_IMAGE_SIZE,
    epochs=DEFAULT_EPOCHS,
    project_name="mislabel_confidence_noise",
    gt_field="ground_truth",
    train_split=None,
    test_split=None,
    **kwargs
):
    from ultralytics import settings
    settings.update({"wandb": False})
    if dataset_name:
        dataset = fo.load_dataset(dataset_name)
        train = dataset.match_tags("train")
        test = dataset.match_tags("test")
    else:
        train = train_split
        test = test_split

    if model_size is None:
        model_size = "s"
    elif model_size not in ["n", "s", "m", "l", "x"]:
        raise ValueError("model_size must be one of ['n', 's', 'm', 'l', 'x']")

    splits_dict = {
        "train": train,
        "val": test,
        "test": test,
    }

    data_dir = tempfile.mkdtemp()

    for key, split in splits_dict.items():
        split_dir = os.path.join(data_dir, key)
        os.makedirs(split_dir)
        split.export(
            export_dir=split_dir,
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field=gt_field,
            export_media="symlink",
        )

    # Load a pre-trained YOLOv8 model for classification
    model = YOLO(f"yolov8{model_size}-cls.pt")

    # Train the model
    model.train(
        data=data_dir,  # Path to the dataset
        epochs=epochs,  # Number of epochs
        imgsz=image_size,  # Image size
        device=get_torch_device(),
        project=project_name,
    )

    return model



def main():

    if fo.__version__ < "0.25.0":
        raise ValueError("Please upgrade to the latest version of FiftyOne")
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--project_name", type=str, default="mislabel_confidence_noise")
    args = parser.parse_args()

    train_classifier(
        dataset_name=args.dataset_name,
        model_size=args.model_size,
        image_size=args.image_size,
        epochs=args.epochs,
        project_name=args.project_name,
    )


if __name__ == "__main__":
    main()