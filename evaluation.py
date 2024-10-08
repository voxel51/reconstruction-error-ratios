import os

## Restrict the number of threads used by numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from sklearn.metrics import f1_score
from data_preparation_scripts.yolov8cls import train_classifier


def evaluate_model(
    y_noisy=None,
    y_true=None,
    mistakenness_scores=None,
    mistakenness_probs=None,
    threshold=None,
    train_clf=False,
    compute_f1_optimal=False,
    **kwargs,
):
    """
    Evaluate a model for detecting label errors.

    Parameters
    ----------
    y_noisy : np.array
        Noisy labels.
    y_true : np.array
        True labels.
    mistakenness_scores : np.array
        Scores indicating the likelihood that a sample is corrupted, higher is more likely.
    mistakenness_probs : np.array
        Probabilities indicating the likelihood that a sample is corrupted, higher is more likely.
        If None, and mistakenness_scores values are normalized, mistakenness_scores is used.
    threshold : float
        The threshold above which a sample is considered corrupted.
    """
    is_corrupt_mask = y_noisy != y_true

    f1 = compute_f1_score(is_corrupt_mask, mistakenness_scores, threshold)

    auroc = compute_auroc(is_corrupt_mask, mistakenness_scores)
    f1_weighted = compute_f1_weighted_score(
        is_corrupt_mask, mistakenness_scores, threshold, mistakenness_probs
    )

    metrics = {
        "f1_at_threshold": f1,
        "auroc": auroc,
    }

    if compute_f1_optimal:
        best_f1, best_threshold = compute_optimal_f1_score(
            is_corrupt_mask, mistakenness_scores
        )
        metrics["f1_optimal"] = best_f1
        metrics["optimal_threshold"] = best_threshold

    if f1_weighted is not None:
        metrics["f1_weighted"] = f1_weighted

        ## compute change in f1
        delta_f1 = f1_weighted - f1
        metrics["delta_f1"] = delta_f1
        delta_f1_nomalized = delta_f1 / (1 - f1)
        metrics["delta_f1_normalized"] = delta_f1_nomalized

    for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(
            is_corrupt_mask, mistakenness_scores, k
        )

    for k in [100, int(np.sum(is_corrupt_mask))]:
        lift = compute_lift_at_k(is_corrupt_mask, mistakenness_scores, k)
        if k == 100:
            metrics["lift@100"] = lift
        else:
            metrics["lift@num_errors"] = lift

    if train_clf:
        classif_metrics = compute_classification_metrics(
            y_true, y_noisy, mistakenness_scores, threshold, **kwargs
        )
        metrics.update(classif_metrics)

    return metrics


def compute_classification_metrics(
    y_true, y_noisy, mistakenness_scores, threshold, **kwargs
):
    import fiftyone as fo
    from fiftyone import ViewField as F

    dataset_name = kwargs.get("dataset_name")
    dataset = fo.load_dataset(dataset_name)
    model_size = kwargs.get("train_clf_model_size", "s")

    # Create a copy to work on
    new_dataset = dataset.clone()
    train = new_dataset.match_tags("train")
    test = new_dataset.match_tags("test")

    def compute_accuracy(train_split, test_split, gt_field, pred_field):
        model = train_classifier(
            train_split=train_split,
            test_split=test_split,
            model_size=model_size,
        )
        test_split.apply_model(model, label_field=pred_field)
        res = test_split.evaluate_classifications(
            gt_field, pred_field, eval_key=f"{pred_field}_eval"
        )
        return res.metrics()

    gt_label_field = "ground_truth"
    noisy_label_field = "ground_truth_noisy"
    cleaned_label_field = "ground_truth_cleaned"

    clean_model_metrics = compute_accuracy(
        train, test, "ground_truth", "clean_predictions"
    )

    classes = dataset.distinct(f"{gt_label_field}.label")

    ## Train a classifier on the noisy labels
    noisy_labels = [classes[idx] for idx in y_noisy]
    new_dataset.clone_sample_field(gt_label_field, noisy_label_field)
    train.set_values(f"{noisy_label_field}.label", noisy_labels)
    train.save()

    noisy_model_metrics = compute_accuracy(
        train, test, noisy_label_field, "noisy_predictions"
    )

    ## Threshold the mistakenness scores
    pred_is_corrupt_mask = mistakenness_scores > threshold
    new_dataset.add_sample_field("pred_is_corrupt", fo.BooleanField)
    train.set_values("pred_is_corrupt", pred_is_corrupt_mask)

    thresholded_train = train.match(~F("pred_is_corrupt"))
    thresholded_model_metrics = compute_accuracy(
        thresholded_train, test, gt_label_field, "thresholded_predictions"
    )

    ## Clean the dataset
    new_dataset.clone_sample_field(noisy_label_field, cleaned_label_field)

    y_true_cleaned = y_true[pred_is_corrupt_mask]
    true_labels_cleaned = [classes[idx] for idx in y_true_cleaned]
    train.set_values(f"{cleaned_label_field}.label", true_labels_cleaned)
    train.save()

    cleaned_model_metrics = compute_accuracy(
        train, test, gt_label_field, "cleaned_predictions"
    )

    # Clean up
    fo.delete_dataset(new_dataset.name)

    return {
        "clean_model_metrics": clean_model_metrics,
        "noisy_model_metrics": noisy_model_metrics,
        "thresholded_model_metrics": thresholded_model_metrics,
        "cleaned_model_metrics": cleaned_model_metrics,
    }


def compute_ndcg_at_k(is_corrupt_mask, rankings, k):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at k.

    Parameters
    ----------
    is_corrupt_mask : np.array
        Boolean mask indicating whether each sample is corrupted.
    rankings : np.array
        Rankings of the samples.
    k : int
        The number of top-ranked samples to consider.
    """
    dcg = 0
    idcg = 0
    num_hits = np.sum(is_corrupt_mask)

    sorted_rankings = np.argsort(rankings)[::-1]
    sorted_is_corrupt_mask = is_corrupt_mask[sorted_rankings]

    for i in range(min(k, num_hits)):
        dcg += sorted_is_corrupt_mask[i] / np.log2(i + 2)
        idcg += 1 / np.log2(i + 2)

    return dcg / idcg


def compute_lift_at_k(is_corrupt_mask, rankings, k):
    """
    Compute the Lift at k.

    Parameters
    ----------
    is_corrupt_mask : np.array
        Boolean mask indicating whether each sample is corrupted.
    rankings : np.array
        Rankings of the samples.
    k : int
        The number of top-ranked samples to consider.
    """

    num_total_errors = np.sum(is_corrupt_mask)
    print(f"Total errors: {num_total_errors}")

    sorted_rankings = np.argsort(rankings)[::-1]
    sorted_is_corrupt_mask = is_corrupt_mask[sorted_rankings]

    num_hits_in_k = np.sum(sorted_is_corrupt_mask[:k])
    percent_hits_in_k = num_hits_in_k / k
    print(f"Percent hits in top {k}: {percent_hits_in_k}")

    percent_errors_in_dataset = num_total_errors / len(is_corrupt_mask)
    print(f"Percent errors in dataset: {percent_errors_in_dataset}")

    lift = percent_hits_in_k / percent_errors_in_dataset
    return lift


def compute_f1_score(is_corrupt_mask, mistakenness_scores, threshold, **kwargs):
    """
    Compute the F1 score for detecting corrupted samples.

    Parameters
    ----------
    is_corrupt_mask : np.array
        Boolean mask indicating whether each sample is corrupted.
    mistakenness_scores : np.array
        Scores indicating the likelihood that a sample is corrupted, higher is more likely.
    threshold : float
        The threshold above which a sample is considered corrupted.
    """
    pred_is_corrupt_mask = mistakenness_scores > threshold
    return f1_score(is_corrupt_mask, pred_is_corrupt_mask)


def compute_f1_weighted_score(
    is_corrupt_mask, mistakenness_scores, threshold, mistakenness_probs
):
    """
    Compute the weighted F1 score for detecting corrupted samples.

    Parameters
    ----------
    is_corrupt_mask : np.array
        Boolean mask indicating whether each sample is corrupted.
    mistakenness_scores : np.array
        Scores indicating the likelihood that a sample is corrupted, higher is more likely.
    threshold : float
        The threshold above which a sample is considered corrupted.
    mistakenness_probs : np.array
        Probabilities indicating the likelihood that a sample is corrupted, higher is more likely.
    """
    if mistakenness_probs is None:
        _min, _max = np.min(mistakenness_scores), np.max(mistakenness_scores)
        if _min < 0 or _max > 1:
            return None
        mistakenness_probs = mistakenness_scores

    pred_is_corrupt_mask = mistakenness_scores > threshold
    pos_pred_conf = (mistakenness_probs - threshold) / (1 - threshold)
    neg_pred_conf = (threshold - mistakenness_probs) / threshold

    weighted_tp = np.sum(is_corrupt_mask * pred_is_corrupt_mask * pos_pred_conf)
    weighted_fp = np.sum((1 - is_corrupt_mask) * pred_is_corrupt_mask * pos_pred_conf)
    weighted_fn = np.sum(is_corrupt_mask * (1 - pred_is_corrupt_mask) * neg_pred_conf)

    precision = weighted_tp / (weighted_tp + weighted_fp)
    recall = weighted_tp / (weighted_tp + weighted_fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_optimal_f1_score(is_corrupt_mask, mistakenness_scores):
    """
    Compute the optimal F1 score for detecting corrupted samples.

    Parameters
    ----------
    is_corrupt_mask : np.array
        Boolean mask indicating whether each sample is corrupted.
    mistakenness_scores : np.array
        Scores indicating the likelihood that a sample is corrupted, higher is more likely.
    """
    best_threshold = None
    best_f1 = 0

    _min, _max = np.min(mistakenness_scores), np.max(mistakenness_scores)

    for threshold in np.linspace(_min, _max, 1000):
        f1 = compute_f1_score(is_corrupt_mask, mistakenness_scores, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_f1, best_threshold


def compute_auroc(is_corrupt_mask, mistakenness_scores):
    """
    Compute the Area Under the Receiver Operating Characteristic (AUROC) curve.

    Parameters
    ----------
    is_corrupt_mask : np.array
        Boolean mask indicating whether each sample is corrupted.
    mistakenness_scores : np.array
        Scores indicating the likelihood that a sample is corrupted, higher is more likely.
    """
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(is_corrupt_mask, mistakenness_scores)
