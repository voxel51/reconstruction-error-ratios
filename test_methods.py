import argparse
from copy import copy
import os

## Restrict the number of threads used by numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from evaluation import evaluate_model
from models.DEFAULTS import *

import wandb

wandb.require("core")

PROJECT_NAME = "rers"


def load_data(**kwargs):
    dataset_name = kwargs.get("dataset_name", DEFAULT_DATASET_NAME)
    noise_frac = kwargs.get("noise_frac", None)
    noise_type = kwargs.get("noise_type", None)

    features = kwargs.get("features", DEFAULT_FEATURES)
    if features != DEFAULT_FEATURES:
        X_filepath = f"data/{dataset_name}_X_{features}.npy"
        y_gt_filepath = f"data/{dataset_name}_y_{features}_gt.npy"
    else:
        X_filepath = f"data/{dataset_name}_X.npy"
        y_gt_filepath = f"data/{dataset_name}_y_gt.npy"


    prefix = f"data/{dataset_name}_y_noisy_"
    seed = kwargs.get("noise_seed", None)
    suffix = f"_{seed}.npy" if (seed is not None and seed != None) else ".npy"

    if noise_frac == 0.0:
        y_noisy_filepath = y_gt_filepath
    elif noise_type == "human" and (noise_frac is None or noise_frac == None):
        y_noisy_filepath = f"{prefix}{noise_type}{suffix}"
    elif noise_type == "confidence":
        model_size = kwargs.get("noise_conf_model_size", "s")
        y_noisy_filepath = f"{prefix}{noise_frac}_{noise_type}_yolov8{model_size}_cls{suffix}"
    else:
        y_noisy_filepath = f"{prefix}{noise_frac}_{noise_type}{suffix}"

    X = np.load(X_filepath)

    y_gt = np.load(y_gt_filepath)
    y_noisy = np.load(y_noisy_filepath)

    return X, y_gt, y_noisy


def get_model(model_name):
    if model_name == "confident_learning":
        from models.confident_learning import ConfidentLearningModel

        return ConfidentLearningModel
    elif model_name == "simifeat":
        from models.simifeat import SimiFeatModel

        return SimiFeatModel
    elif model_name == "reconstruction":
        from models.reconstruction import UMAPAutoEncoderModel

        return UMAPAutoEncoderModel
    elif model_name == "zero_shot":
        from models.zero_shot import ZeroShotModel

        return ZeroShotModel
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_model(model_name, X, y_noisy, **kwargs):
    model = get_model(model_name)
    if model_name == "zero_shot":
        from data_preparation_scripts.generate_embeddings import (
            get_class_name_embeddings,
        )

        class_name_embs = get_class_name_embeddings(**kwargs)
        kwargs["class_name_embs"] = class_name_embs

    model = model(X, y_noisy, **kwargs)
    results_dict = model.detect_label_errors()
    return results_dict


def run_and_evaluate(**kwargs):
    config = _create_config(kwargs)

    notes = kwargs.get("wandb_notes", "Comparing different methods for mislabel detection")

    project_name = kwargs.get("project_name", PROJECT_NAME)
    wandb.init(
        project=project_name,
        notes=notes,
        config=config,
    )

    X, y_gt, y_noisy = load_data(**kwargs)

    if kwargs.get("wandb_log_artifacts", False):
        _log_initial_artifacts(**kwargs)

    method = kwargs.get("method")
    results_dict = run_model(method, X, y_noisy, **kwargs)
    y_pred = results_dict["y_pred"]
    threshold = results_dict["threshold"]
    mistakenness = results_dict["mistakenness"]
    mistakenness_probs = results_dict.get("mistakenness_probs", None)

    if kwargs.get("wandb_log_artifacts", False):
        _log_final_artifacts(results_dict)

    noise_frac = kwargs.get("noise_frac", None)
    if noise_frac != 0.0:

        res = evaluate_model(
            y_true=y_gt,
            y_noisy=y_noisy,
            y_pred=y_pred,
            mistakenness_scores=mistakenness,
            mistakenness_probs=mistakenness_probs,
            threshold=threshold,
            **kwargs,
        )

        for k, v in res.items():
            wandb.run.summary[k] = v

    for k, v in results_dict.items():
        if k in ["y_pred", "mistakenness", "mistakenness_probs"]:
            continue
        wandb.run.summary[k] = v

    wandb.finish()


def _create_config(kwargs):
    config = copy(kwargs)
    config.pop("project_name")
    config.pop("wandb_log_artifacts")
    config.pop("wandb_notes")

    method = kwargs.get("method")
    for k, _ in kwargs.items():
        if k.startswith(f"sf_") and method != "simifeat":
            config.pop(k)
        if k.startswith(f"cl_") and method != "confident_learning":
            config.pop(k)
        if k.startswith(f"recon_") and method != "reconstruction":
            config.pop(k)

    train_clf_flag = kwargs.get("train_clf", False)
    if k.startswith(f"train_clf_") and not train_clf_flag:
        config.pop(k)

    noise_type = kwargs.get("noise_type")
    if noise_type != "confidence":
        config.pop("noise_conf_model_size")
    return config


def _log_initial_artifacts(**kwargs):
    ## Log y_gt, y_noisy as artifacts
    api = wandb.Api(timeout=29)

    y_gt_artifact_name = f"y_gt_{kwargs['dataset_name']}"
    if not api.artifact_exists(y_gt_artifact_name):
        y_gt_artifact = wandb.Artifact(y_gt_artifact_name, type="data")
        y_gt_artifact.add_file(f"data/{kwargs['dataset_name']}_y_gt.npy")
        wandb.run.log_artifact(y_gt_artifact)

    if kwargs['noise_frac'] == 0.0:
        return

    if kwargs["noise_type"] == "human" and kwargs['noise_frac'] is None:
        noise_str = "human"
    elif kwargs["noise_type"] == "confidence":
        noise_str = f"{kwargs['noise_frac']}_{kwargs['noise_type']}_yolov8{kwargs['noise_conf_model_size']}_cls"
    else:
        noise_str = f"{kwargs['noise_frac']}_{kwargs['noise_type']}"

    seed_str = f"_{kwargs['noise_seed']}" if kwargs["noise_seed"] is not None else ""

    y_noisy_artifact_name = f"y_noisy_{kwargs['dataset_name']}_{noise_str}{seed_str}"
    if not api.artifact_exists(y_noisy_artifact_name):
        y_noisy_artifact = wandb.Artifact(y_noisy_artifact_name, type="data")
        y_noisy_artifact.add_file(
            f"data/{kwargs['dataset_name']}_y_noisy_{noise_str}{seed_str}.npy"
        )
        wandb.run.log_artifact(y_noisy_artifact)


def _log_mistakenness(mistakenness):
    uuid = wandb.run.name
    filename = f"mistakenness_{uuid}.npy"
    np.save(filename, mistakenness)
    artifact = wandb.Artifact(f"mistakenness_{uuid}", type="data")
    artifact.add_file(filename)
    wandb.run.log_artifact(artifact)

    ## clean up
    os.remove(filename)


def _log_probs(probs):
    # PDF of the mistake probability
    uuid = wandb.run.name
    filename = f"p_mistake_estimated_{uuid}.npy"
    np.save(filename, probs)
    artifact = wandb.Artifact(f"p_mistake_estimated_{uuid}", type="data")
    artifact.add_file(filename)
    wandb.run.log_artifact(artifact)

    ## clean up
    os.remove(filename)


def _log_final_artifacts(results_dict):
    ## save artifacts ##
    mistakenness = results_dict["mistakenness"]
    mistakenness_probs = results_dict.get("mistakenness_probs", None)

    if mistakenness_probs is not None:
        _log_probs(mistakenness_probs)

    _log_mistakenness(mistakenness)

    artifacts = results_dict.get("artifacts", {})
    for artifact_name, artifact in artifacts.items():
        uuid = wandb.run.name
        filename = f"{artifact_name}_{uuid}.npy"
        np.save(filename, artifact)
        artifact = wandb.Artifact(f"{artifact_name}_{uuid}", type="data")
        artifact.add_file(filename)
        wandb.run.log_artifact(artifact)

        ## clean up
        os.remove(filename)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name", type=str, default=PROJECT_NAME
    )  # wandb project name
    parser.add_argument("--wandb_log_artifacts", action="store_true")
    parser.add_argument("--wandb_notes", type=str, default=None)
    parser.add_argument("--compute_f1_optimal", action="store_true")
    parser.add_argument("--train_clf", action="store_true")
    parser.add_argument("--train_clf_model_size", type=str, default="s")

    parser.add_argument(
        "--method",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "confident_learning", "simifeat", "zero_shot"],
    )
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--features", type=str, default="clip-vit-base-patch32")

    parser.add_argument("--noise_frac", type=float, default=None)
    parser.add_argument(
        "--noise_type",
        type=str,
        required=True,
        choices=["symmetric", "asymmetric", "human", "confidence"],
    )
    parser.add_argument("--noise_conf_model_size", type=str, default="s")
    parser.add_argument("--noise_seed", type=int)

    ### Reconstruction
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--recon_fit_frac", type=float, default=None)
    group.add_argument("--recon_fit_samples_per_class", type=int, default=None)

    parser.add_argument(
        "--recon_reg_strength", type=float, default=DEFAULT_REG_STRENGTH
    )
    parser.add_argument("--recon_n_components", type=int, default=DEFAULT_N_COMPONENTS)
    parser.add_argument("--recon_dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--recon_n_epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--recon_batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--recon_parametric_reconstruction_loss_weight",
        type=float,
        default=DEFAULT_PARAMETRIC_RECONSTRUCTION_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--recon_hidden_dims",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_DIMS,
    )
    parser.add_argument("--recon_lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--recon_metric", type=str, default=DEFAULT_METRIC)
    parser.add_argument("--recon_n_neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--recon_spread", type=float, default=None)
    parser.add_argument("--recon_min_dist", type=float, default=None)
    parser.add_argument("--recon_a", type=float, default=None)
    parser.add_argument("--recon_b", type=float, default=None)
    parser.add_argument(
        "--recon_repulsion_strength", type=float, default=DEFAULT_REPULSION_STRENGTH
    )
    parser.add_argument("--recon_gamma1", type=float, default=DEFAULT_GAMMA1)
    parser.add_argument("--recon_gamma2", type=float, default=DEFAULT_GAMMA2)
    parser.add_argument("--recon_gamma3", type=float, default=DEFAULT_GAMMA3)
    parser.add_argument("--recon_n_workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--recon_skip_multiprocessing", action="store_true")
    parser.add_argument("--recon_estimate_probs", action="store_true")
    parser.add_argument("--recon_store_error_array", action="store_true")
    parser.add_argument("--recon_compute_2d_umap", action="store_true")

    ### Confident Learning
    parser.add_argument(
        "--cl_classifier_arch", type=str, default=DEFAULT_CL_CLASSIFIER_ARCH
    )
    parser.add_argument("--cl_max_iter", type=int, default=DEFAULT_CL_MAX_ITER)

    ### SimiFeat
    parser.add_argument(
        "--sf_selection_cutoff", type=float, default=DEFAULT_SF_SELECTION_CUTOFF
    )

    args = parser.parse_args()

    ## Validation: human noise only for CIFAR10 and CIFAR100
    if args.noise_type == "human" and args.dataset_name not in ["cifar10", "cifar100"]:
        raise ValueError(f"Human noise is only supported for CIFAR10 and CIFAR100")

    run_and_evaluate(**vars(args))


if __name__ == "__main__":
    main()
