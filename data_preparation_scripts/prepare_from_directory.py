import argparse

import fiftyone as fo

from generate_embeddings import generate_clip_b32_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    args = parser.parse_args()

    dn = args.dataset_name
    data_dir = args.data_dir

    dataset = fo.Dataset.from_dir(
        data_dir, dataset_type=fo.types.ImageClassificationDirectoryTree
    )
    dataset.persistent = True
    dataset.name = dn
    generate_clip_b32_embeddings(dataset)
