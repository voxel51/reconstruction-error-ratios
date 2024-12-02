import argparse
from math import ceil
import os
import logging
from contextlib import redirect_stdout
import warnings

# Set up logging
logging.basicConfig(
    filename="run_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

warnings.filterwarnings("ignore")


## Restrict the number of threads used by numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppresses all logging except fatal errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import numpy as np

import fiftyone as fo
from fiftyone import ViewField as F

from models.DEFAULTS import *
from models.reconstruction import UMAPAutoEncoderModel

def _format_kwargs(kwargs):
    recon_keys = [
        "recon_fit_frac",
        "recon_fit_samples_per_class",
        "recon_n_workers",
        "recon_reg_strength",
        "recon_n_components",
        "recon_parametric_reconstruction_loss_weight",
        "recon_hidden_dims",
        "recon_n_neighbors",
        "recon_skip_multiprocessing",
        "recon_estimate_probs",
    ]
    for key in recon_keys:
        if key[6:] not in kwargs:
            continue
        kwargs[key] = kwargs.pop(key[6:])

    return kwargs


def estimate_classification_difficulty(chi_avg, embeddings_field):
    ## get min and max values for classification difficulty (empirical estimates)

    if embeddings_field == "clip-vit-large-patch14":
        min_val = 0.76
        max_val = 1.04
    elif embeddings_field == "clip-vit-base-patch32":
        min_val = 0.84
        max_val = 1.01
    elif "dinov2" in embeddings_field:
        min_val = 0.88
        max_val = 1.06
    elif "resnet50" in embeddings_field:
        min_val = 0.93
        max_val = 1.3
    
    ## normalize chi_avg to be between 0 and 1
    clf_diff = (chi_avg - min_val) / (max_val - min_val)
    return np.clip(clf_diff, 0, 1)


def load_data(**kwargs):
    ## Option 1: Load from a FiftyOne dataset
    dataset_name = kwargs.get("dataset_name", None)

    if dataset_name is not None:
        embeddings_field = kwargs.get("embeddings_field", "clip-vit-base-patch32")

        dataset = fo.load_dataset(dataset_name)
        X = np.array(dataset.exists(embeddings_field).values(embeddings_field))
        y = np.array(dataset.exists(embeddings_field).values("ground_truth.label"))
        class_names = dataset.distinct("ground_truth.label")
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        y = np.array([label_to_idx[label] for label in y])

        return X, y


def run_model(X, y, **kwargs):
    model = UMAPAutoEncoderModel(X, y, **kwargs)
    results_dict = model.detect_label_errors()

    keys = ["threshold", "eta_est", "chi_avg", "mistakenness", "mistakenness_probs"]
    results = {key: results_dict[key] for key in keys if key in results_dict}
    return results


def estimate_time_to_completion(y, **kwargs):
    ## ~ fixed time for loading data, preprocessing, etc.
    ## training time depends minimally on the number of samples
    ## and linearly on the number of classes divided by the number of workers

    ## This expression is a very rough estimate based on empirical observations
    num_classes = len(np.unique(y))
    num_samples = len(y)
    num_samples_per_class = num_samples / num_classes

    num_workers = kwargs.get("n_workers", os.cpu_count() - 1)

    if num_classes > 500 and num_samples >= 1E7:
        num_workers = min(num_workers, 4)
        print("Reducing number of workers to 4")
        print("Lots of classes and samples, this may take a while")
        est_time = None

    est_time = 5 + 10*ceil(num_classes/num_workers) * (1 + 0.001 * num_samples_per_class**1.08)
    
    estimate_probs = kwargs.get("estimate_probs", False)
    if estimate_probs:
        est_time *= 2
    return est_time ## seconds


def _gen_run_name(dataset, kwargs):
    run_prefix = "rers"
    embed_field = kwargs.get("embeddings_field", "clip-vit-base-patch32")
    run_name = f"{run_prefix}_{embed_field}"
    if run_name in dataset.list_runs():
        for i in range(2, 100):
            run_name = f"{run_prefix}_{embed_field}_{i}"
            if run_name not in dataset.list_runs():
                break
    return run_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--embeddings_field", type=str, default="clip-vit-base-patch32")
    parser.add_argument("--mistakenness_field", type=str, default="mistakenness")


    ### Reconstruction
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--fit_frac", type=float, default=None)
    group.add_argument("--fit_samples_per_class", type=int, default=None)

    parser.add_argument("--n_workers", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--reg_strength", type=float, default=DEFAULT_REG_STRENGTH)
    parser.add_argument("--n_components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--n_neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--skip_multiprocessing", action="store_true")
    parser.add_argument("--estimate_probs", action="store_true")

    args = parser.parse_args()
    kwargs = _format_kwargs(vars(args))

    X, y = load_data(**kwargs)
    est_time = estimate_time_to_completion(y, **kwargs)
    if est_time is not None:
        print(f"Estimated computation time: {np.round(est_time, -1)} seconds")

    # Open the log file and redirect intermediate output
    with open("run_model.log", "a") as log_file:
        with redirect_stdout(log_file):
            logging.info("Starting model run")
            res = run_model(X, y, **kwargs)
            logging.info("Model run complete")

    eta = res.get("eta_est", None)
    print(f"Estimated noise: {eta:.2f}")

    chi_avg = res.get("chi_avg", None)
    embeddings_field = args.embeddings_field
    clf_diff = estimate_classification_difficulty(chi_avg, embeddings_field)
    print(f"Estimated classification difficulty: {clf_diff:.2f}")

    mistakenness = res.get("mistakenness", None)
    threshold = res.get("threshold", None)

    is_mistaken = mistakenness > threshold

    num_above_threshold = np.sum(is_mistaken)
    print(f"{num_above_threshold} potential label mistakes detected")

    ## store results in the dataset
    print("Storing results in dataset")
    dataset_name = args.dataset_name
    dataset = fo.load_dataset(dataset_name)

    ## create a custom run in FiftyOne
    config = fo.RunConfig(**kwargs)
    run_name = _gen_run_name(dataset, kwargs)
    dataset.register_run(run_name, config)

    class_names = dataset.distinct("ground_truth.label")
    class_scores = {}
    for label in class_names:
        sample_collection = dataset.match_tags("train").match(F("ground_truth.label") == label)
        class_scores[label] = sample_collection.mean(mistakenness_field)

    results = fo.RunResults(
        dataset, 
        config, 
        run_name,
        eta_est=eta,
        chi_avg=chi_avg,
        threshold=threshold,
        classification_difficulty=clf_diff,
        class_chis=class_scores,
    )
    dataset.save_run_results(run_name, results)
    
    print("Results stored in run:", run_name)
    print(f"You can access via `results = dataset.load_run_results({run_name})`")


    mistakenness_field = args.mistakenness_field
    if not dataset.has_sample_field(mistakenness_field):
        dataset.add_sample_field(mistakenness_field, fo.FloatField)
    dataset.set_field(mistakenness_field, res["mistakenness"])
    dataset.save()
    print("Mistakenness values stored in field:", mistakenness_field)

    if not dataset.has_sample_field("mistaken"):
        dataset.add_sample_field("mistaken", fo.BooleanField)
    dataset.set_field("mistaken", is_mistaken)
    dataset.save()

    if args.recon_estimate_probs:
        mistakenness_probs = res.get("mistakenness_probs", None)
        mistakenness_probs_field = "mistakenness_probs"
        if not dataset.has_sample_field(mistakenness_probs_field):
            dataset.add_sample_field(mistakenness_probs_field, fo.FloatField)
        dataset.set_field(mistakenness_probs_field, mistakenness_probs)
        dataset.save()
        print("Mistakenness probabilities stored in field: ", mistakenness_probs_field)



if __name__ == "__main__":
    main()
