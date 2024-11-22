import argparse
import gc
import os

import fiftyone as fo
import fiftyone.zoo as foz

import numpy as np
import torch
from transformers import Dinov2Model, CLIPModel

device = "cuda"
batch_size = 64
torch_dtype = torch.float16

clip_uris = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]

dino_uris = [
    "facebook/dinov2-base",
    "facebook/dinov2-small",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
]

resnet50_uris = ["resnet50-imagenet-torch"]


all_uris = clip_uris + dino_uris + resnet50_uris

def _to_field_name(model_uri):
    fn = model_uri.split("/")[-1]
    if '-torch' in fn:
        return fn.replace("-torch", "")
    return fn


def _free_memory(model):
    gc.collect()
    torch.cuda.empty_cache()
    del model



def _get_dino_embeddings_kwargs(model_uri):
    model = Dinov2Model.from_pretrained(model_uri, torch_dtype=torch.float16)
    model = model.to(device=device, dtype=torch_dtype)

    return {
        "model": model,
        "batch_size": batch_size,
    }


def _get_clip_embeddings_kwargs(model_uri):
    ## if torch>=2.1.1, attn_implementation="sdpa" is available    
    torch_version = torch.__version__
    if torch_version >= "2.1.1":
        model = CLIPModel.from_pretrained(
            model_uri, torch_dtype=torch.float16, attn_implementation="sdpa"
        )
    else:
        model = CLIPModel.from_pretrained(
            model_uri, torch_dtype=torch.float16
        )

    model = model.to(device)

    return {
        "model": model,
        "batch_size": batch_size,
    }


def get_fo_embeddings_kwargs(model_uri):
    model = foz.load_zoo_model(model_uri, device=device)

    return {
        "model": model,
    }


def get_embeddings_kwargs(model_uri):
    if "dinov2" in model_uri:
        return _get_dino_embeddings_kwargs(model_uri)
    elif "clip" in model_uri:
        return _get_clip_embeddings_kwargs(model_uri)
    else:
        return get_fo_embeddings_kwargs(model_uri)


def run_model(model_uri, sample_collection):
    dn = sample_collection._dataset.name
    embeddings_kwargs = get_embeddings_kwargs(model_uri)
    model = embeddings_kwargs.get("model")
    embeddings_field = _to_field_name(model_uri)

    sample_collection.compute_embeddings(embeddings_field=embeddings_field, **embeddings_kwargs)
    _free_memory(model)

    ## store the embeddings
    X = np.array(sample_collection.exists(embeddings_field).values(embeddings_field))
    np.save(f"data/{dn}_X_{embeddings_field}.npy", X)

    y = np.array(sample_collection.exists(embeddings_field).values("ground_truth.label"))
    class_names = sample_collection.distinct("ground_truth.label")
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.array([label_to_idx[label] for label in y])
    np.save(f"data/{dn}_y_{embeddings_field}_gt.npy", y)



def generate_embeddings(dataset_name):
    sample_collection = fo.load_dataset(dataset_name).match_tags("train")

    for model_uri in all_uris:
        run_model(model_uri, sample_collection)


def get_class_name_embeddings(dataset_name=None, features=None, **kwargs):
    if features is None or features == None:
        features = "clip-vit-base-patch32"
    if "clip" not in features:
        raise ValueError("Only CLIP features are supported for now")
    
    filename = f"data/{dataset_name}_{features}_class_name_embs.npy"
    if os.path.exists(filename):
        return np.load(filename)
    
    if "base" in features:
        model_uri = "openai/clip-vit-base-patch32"
    elif "large" in features:
        model_uri = "openai/clip-vit-large-patch14"
    else:
        raise ValueError("Unknown model uri")
    
    kwargs = _get_clip_embeddings_kwargs(model_uri)
    model = kwargs.get("model")

    dataset = fo.load_dataset(dataset_name)
    class_names = dataset.distinct("ground_truth.label")
    prompts = [f"A photo of a {cn}" for cn in class_names]

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_uri)
    text_features = []

    for prompt in prompts:
        inputs = processor(text=prompt, return_tensors="pt")
        with torch.no_grad():
            text_features.append(
                model.base_model.get_text_features(
                    **inputs.to(device)
                ).cpu().numpy()
            )

    text_features = np.array(text_features)

    ## free memory
    _free_memory(model)

    np.save(filename, text_features)

    return text_features


def generate_clip_b32_embeddings(dataset):
    model_uri = "openai/clip-vit-base-patch32"
    run_model(model_uri, dataset)

def generate_clip_b32_class_embeddings(dataset_name):
    features = "clip-vit-base-patch32"
    get_class_name_embeddings(dataset_name=dataset_name, features=features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    ## Choices include cifar10, cifar100, EuroSAT, StanfordDogs, ...
    # but you can add your own as well, as long as:
    # - the dataset is in fiftyone
    # - it is an image classification dataset with noisy GT labels in an
    # - fo.Classifications label field with name `ground_truth`

    # Computing embeddings requires CUDA visible device

    parser.add_argument("--store_class_embeddings", action="store_true")

    args = parser.parse_args()

    dataset_name = args.dataset_name

    generate_embeddings(dataset_name)

    if args.store_class_embeddings:
        generate_clip_b32_class_embeddings(dataset_name)


if __name__ == "__main__":
    main()
