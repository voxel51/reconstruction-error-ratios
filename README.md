# Class-wise Autoencoders Measure Classification Difficulty and Detect Label Mistakes



([Jacob Marks](https://github.com/jacobmarks)\*, [Brent A. Griffin](https://github.com/griffbr), [Jason J. Corso](https://github.com/jasoncorso)) @ [Voxel51](https://voxel51.com)

\* Corresponding author


<figure>
  <img src="./assets/dataset_difficulty.png" alt="Dataset difficulty scores for CIFAR-10, CIFAR-100, and ImageNet. The scores are computed using the RER framework, which measures the difficulty of classifying each sample in the dataset." style="width:100%">
    <figcaption>Dataset difficulty scores for CIFAR-10, CIFAR-100, and ImageNet. The scores are computed using the RER framework with CLIP ViT-L/14 features, which measures the difficulty of classifying each sample in the dataset.
    </figcaption>
</figure>



This repository contains the code for the paper *Class-wise Autoencoders Measure Classification Difficulty and Detect Label Mistakes*.

Reconstruction Error Ratios (RERs) provide a simple, fast, and flexible framework for analyzing visual classification datasets at the sample, class, and entire dataset level. This repo contains the code to compute RERs on your dataset and reproduce the experiments in the paper.


## Installation

First, clone the repository:

```bash
git clone https://github.com/voxel51/rers.git
```

Then, install the dependencies:

```bash
cd rers
pip install -r requirements.txt
```

To reproduce the experiments in the paper, you will also need to install the Weights and Biases Python client:

```bash
pip install wandb
```

Additionally, to generate confidence-based noise, you will need  `ultralytics>=0.8.0` and `fiftyone>=0.25.0`. You can install these with:

```bash
pip install ultralytics>=0.8.0 fiftyone>=0.25.0
```


## Usage

To compute a dataset difficulty score or detect potential label mistakes, you first need to prepare your dataset. If your dataset is in an image classification directory structure like

```bash
dataset/
    class1/
        image1.jpg
        image2.jpg
        ...
    class2/
        image1.jpg
        image2.jpg
        ...
    ...
```

then you can run the following command, replacing `\path\to\dataset` with the path to the dataset directory and `my_dataset` with the name of the dataset:

```bash
python data_preparation_scripts/prepare_from_directory.py --data_dir '\path\to\dataset' --dataset_name my_dataset
```

This will create a Fiftyone dataset named `my_dataset`, which you can load and visualize in the Fiftyone App with:

```python
import fiftyone as fo
dataset = fo.load_dataset("my_dataset")
session = fo.launch_app(dataset)
```

To compute the RERs for the dataset, run:

```bash
python run.py --dataset_name my_dataset
```

If you refresh the FiftyOne App, you will then see the mistakenness score for each sample stored in the `mistakenness` field of the dataset, as well as a boolean `mistake` field indicating whether the sample is a potential label mistake.

You can filter the dataset to only show the potential label mistakes with:

```python
from fiftyone import ViewField as F
mistake_view = dataset.match(F("mistake") == True)
session.view = mistake_view
```

You can sort by the mistakenness score with:

```python
sorted_view = mistake_view.sort_by("mistakenness", reverse=True)
session.view = sorted_view
```

where `reverse=True` will sort in descending order, or `reverse=False` will sort in ascending order.

You can extract the mistakenness values for the dataset with:

```python
import numpy as np
mistakenness = np.array(dataset.values("mistakenness"))
```

## 📚 Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{marks2024classwise,
  title={Class-wise Autoencoders Measure Classification Difficulty and Detect Label Mistakes},
  author={Marks, Jacob and Griffin, Brent A and Corso, Jason J},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```

You may also want to check out our open-source toolkit, [FiftyOne](https://voxel51.com/fiftyone), which provides a powerful interface for exploring, analyzing, and visualizing datasets for computer vision and machine learning.