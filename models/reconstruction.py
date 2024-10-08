from .base_model import BaseModel

import logging
import os
import warnings

import numpy as np
import wandb

wandb.require("core")

## Restrict the number of threads used by numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
# Suppress the specific Keras warning
warnings.filterwarnings(
    "ignore", category=UserWarning, message="`build()` was called on layer"
)
# Suppress logging warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)

tf.config.optimizer.set_jit(False)
tf.config.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "GPU")

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from tensorflow.keras import regularizers  # type: ignore
from umap.parametric_umap import ParametricUMAP
from umap import UMAP

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde, kurtosis, skew

from .DEFAULTS import *

NUM_RETRIES = 2
CHI0_CUTOFF = 0.85

keras_fit_kwargs = {
    "callbacks": [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=5 * 10**-3,
            patience=5,
            verbose=1,
        )
    ]
}


class UMAPAutoEncoderModel(BaseModel):
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)
        self.kwargs = format_kwargs(kwargs)

    def detect_label_errors(self):
        results_dict = fit_and_reconstruct(self.X, self.y, **self.kwargs)
        return results_dict
    

def chi_star_ansatz1(chi0, eta_est, gamma1, gamma2):
    return 1.01 * chi0 ** (-gamma1 / (1 + gamma2 * eta_est))


def chi_star_ansatz2(chi_rand, eta_est, gamma3):
    return 1.0 + (1.0 / chi_rand - 1.0) / (1.0 + gamma3 * eta_est)


def estimate_threshold(chi0=None, eta_est=None, chi_max=None, chi_rand=None, **kwargs):
    if chi_max < 1.12:
        gamma3 = kwargs.get("gamma3", DEFAULT_GAMMA3)
        chi_star = chi_star_ansatz2(chi_rand, eta_est, gamma3)
        return max(1.01, chi_star)
    else:
        gamma1 = kwargs.get("gamma1", DEFAULT_GAMMA1)
        gamma2 = kwargs.get("gamma2", DEFAULT_GAMMA2)
        return chi_star_ansatz1(chi0, eta_est, gamma1, gamma2)


def format_kwargs(kwargs):
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k.replace("recon_", "")] = v
    return new_kwargs


def _construct_decoder(
    input_dim=None, hidden_dims=None, dropout=None, reg_strength=None, n_components=None
):
    layers = [tf.keras.layers.InputLayer(shape=(n_components,))]

    for dim in hidden_dims[::-1]:
        layers.append(
            tf.keras.layers.Dense(
                units=dim,
                activation="relu",
                kernel_regularizer=(
                    regularizers.l2(reg_strength) if reg_strength else None
                ),
            )
        )
        layers.append(tf.keras.layers.Dropout(dropout))

    layers.append(tf.keras.layers.Dense(units=input_dim, activation="sigmoid"))

    decoder = tf.keras.Sequential(layers)
    return decoder


def _construct_encoder(
    input_dim=None, hidden_dims=None, dropout=None, reg_strength=None, n_components=None
):
    layers = [tf.keras.layers.InputLayer(shape=(input_dim,))]

    for dim in hidden_dims:
        layers.append(
            tf.keras.layers.Dense(
                units=dim,
                activation="relu",
                kernel_regularizer=(
                    regularizers.l2(reg_strength) if reg_strength else None
                ),
            )
        )
        layers.append(tf.keras.layers.Dropout(dropout))

    layers.append(
        tf.keras.layers.Dense(
            units=n_components,
            kernel_regularizer=regularizers.l2(reg_strength) if reg_strength else None,
        )
    )

    encoder = tf.keras.Sequential(layers)
    return encoder


def create_feature_embedder(X_train, **kwargs):
    n_components = kwargs.get("n_components", DEFAULT_N_COMPONENTS)
    reg_strength = kwargs.get("reg_strength", DEFAULT_REG_STRENGTH)
    dropout = kwargs.get("dropout", DEFAULT_DROPOUT)
    n_epochs = kwargs.get("n_epochs", DEFAULT_N_EPOCHS)
    batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
    parametric_reconstruction_loss_weight = kwargs.get(
        "parametric_reconstruction_loss_weight",
        DEFAULT_PARAMETRIC_RECONSTRUCTION_LOSS_WEIGHT,
    )
    hidden_dims = kwargs.get("hidden_dims", DEFAULT_HIDDEN_DIMS)
    metric = kwargs.get("metrics", DEFAULT_METRIC)

    spread = kwargs.get("spread", None)
    min_dist = kwargs.get("min_dist", None)
    a = kwargs.get("a", None)
    b = kwargs.get("b", None)

    n_neighbors = kwargs.get("n_neighbors", DEFAULT_N_NEIGHBORS)
    repulsion_strength = kwargs.get("repulsion_strength", DEFAULT_REPULSION_STRENGTH)
    lr = kwargs.get("lr", DEFAULT_LR)

    input_dim = X_train.shape[1]

    encoder = _construct_encoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        reg_strength=reg_strength,
        n_components=n_components,
    )
    decoder = _construct_decoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        reg_strength=reg_strength,
        n_components=n_components,
    )

    pumap_kwargs = {
        "encoder": encoder,
        "decoder": decoder,
        "n_components": n_components,
        "dims": (input_dim,),
        "parametric_reconstruction": True,
        "autoencoder_loss": True,
        "keras_fit_kwargs": keras_fit_kwargs,
        "parametric_reconstruction_loss_fcn": tf.keras.losses.MeanSquaredError(),
        "parametric_reconstruction_loss_weight": parametric_reconstruction_loss_weight,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "metric": metric,
        "n_neighbors": n_neighbors,
        "repulsion_strength": repulsion_strength,
        "learning_rate": lr,
        "verbose": False,
    }
    if _has_kwarg(a) and _has_kwarg(b):
        pumap_kwargs["a"] = a
        pumap_kwargs["b"] = b
    else:
        pumap_kwargs["spread"] = spread
        pumap_kwargs["min_dist"] = min_dist

    embedder = ParametricUMAP(**pumap_kwargs)

    for i in range(3):
        try:
            _ = embedder.fit(X_train)
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            if i == 2:
                raise e
    return embedder


def fit_and_run_embedder_for_class(c, X_c, X_train, kwargs):
    """
    Fit a feature embedder on the given class and return the reconstruction
    errors on the training set.
    """

    embedder = create_feature_embedder(X_c, **kwargs)
    X_train_recon = embedder.inverse_transform(embedder.transform(X_train))

    train_recon_errors = np.linalg.norm(X_train - X_train_recon, axis=1)
    return c, train_recon_errors


def _has_kwarg(kwarg):
    return kwarg and kwarg is not None and kwarg != None


def _get_fit_X_y(X_train, y_train, **kwargs):
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_samples = len(y_train)

    fit_frac = kwargs.get("fit_frac", None)
    fit_samples_per_class = kwargs.get("fit_samples_per_class", None)

    if _has_kwarg(fit_frac):
        n_fit_samples = int(fit_frac * n_samples)

        fit_inds = np.random.choice(n_samples, n_fit_samples, replace=False)
        X_train_fit = X_train[fit_inds]
        y_train_fit = y_train[fit_inds]
    elif _has_kwarg(fit_samples_per_class):
        ## randomly pick fit_samples_per_class samples from each class
        all_class_fit_inds = []
        for c in range(n_classes):
            class_inds = np.where(y_train == c)[0]
            n_samples_in_class = len(class_inds)
            class_fit_inds = np.random.choice(
                n_samples_in_class, fit_samples_per_class, replace=False
            )
            all_class_fit_inds.extend(class_inds[class_fit_inds])
        all_class_fit_inds = sorted(all_class_fit_inds)
        X_train_fit = X_train[all_class_fit_inds]
        y_train_fit = y_train[all_class_fit_inds]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    return X_train_fit, y_train_fit


def compute_class_distances(X, y):
    ## for each class, choose 10 samples
    ## compute the pairwise distances and average
    ## results in a n_classes x n_classes matrix

    classes = np.unique(y)
    n_classes = len(classes)

    dist_matrix = np.zeros((n_classes, n_classes))

    for i, c1 in enumerate(classes):
        c1_inds = np.where(y == c1)[0]
        c1_inds = np.random.choice(c1_inds, 10)
        for j, c2 in enumerate(classes):
            c2_inds = np.where(y == c2)[0]
            c2_inds = np.random.choice(c2_inds, 10)
            # use cosine distance
            from scipy.spatial.distance import cdist

            dist_matrix[i, j] = np.mean(cdist(X[c1_inds], X[c2_inds], metric="cosine"))

    ## average dist for each class without the GT class
    avg_dist = np.mean(dist_matrix, axis=1)
    ## remove the GT class dist and reaverage
    avg_dist = (avg_dist * n_classes - np.diag(dist_matrix)) / (n_classes - 1)

    return dist_matrix, avg_dist


def _partial_fit_and_reconstruct(X_train, y_train, **kwargs):
    use_multiprocessing = not kwargs.get("skip_multiprocessing", False)
    num_workers = kwargs.get("n_workers", DEFAULT_N_WORKERS)

    X_train_fit, y_train_fit = _get_fit_X_y(X_train, y_train, **kwargs)

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_samples = len(y_train)

    ### Fit scaler and embedders based on train set
    scaler = MinMaxScaler()
    _ = scaler.fit(X_train_fit)

    X_train_normalized = scaler.transform(X_train).clip(0, 1)
    X_train_fit_normalized = scaler.transform(X_train_fit).clip(0, 1)

    ## get normalized features for each class
    class2features = {}
    for c in classes:
        class2features[c] = X_train_fit_normalized[y_train_fit == c]

    args = [(c, class2features[c], X_train_normalized, kwargs) for c in classes]

    if not use_multiprocessing:
        embedders = []
        for arg in args:
            embedders.append(fit_and_run_embedder_for_class(*arg))
    else:
        # Use multiprocessing to fit embedders in parallel
        import multiprocessing

        num_workers = min(num_workers, n_classes)
        with multiprocessing.Pool(processes=num_workers) as pool:
            embedders = pool.starmap(fit_and_run_embedder_for_class, args)

    ## Create arrays to store the reconstruction errors
    error_array = np.zeros((n_classes, n_samples))

    for c, c_err in embedders:
        error_array[c, :] = c_err

    ## Get GT reconstruction errors
    gt_recon_error = error_array[y_train, np.arange(n_samples)]

    mask = np.zeros_like(error_array, dtype=bool)
    mask[y_train, np.arange(n_samples)] = True
    masked_array = error_array.copy()
    masked_array[mask] = np.inf
    min_other_recon_error = np.min(masked_array, axis=0)

    ## get the mean other class reconstruction error
    avg_other_recon_error = np.mean(error_array, axis=0)
    ## remove the GT reconstruction error and reaverage
    avg_other_recon_error = (avg_other_recon_error * n_classes - gt_recon_error) / (
        n_classes - 1
    )

    min_recon_error = np.min(error_array, axis=0)
    avg_without_min = np.mean(error_array, axis=0)
    avg_without_min = (avg_without_min * n_classes - min_recon_error) / (n_classes - 1)
    chi0 = np.mean(np.divide(gt_recon_error, avg_without_min))

    class_distances, avg_class_distances = compute_class_distances(X_train, y_train)

    return {
        "chi0": chi0,
        "error_array": error_array,
        "gt_recon_error": gt_recon_error,
        "min_other_recon_error": min_other_recon_error,
        "class_distances": class_distances,
        "avg_class_distances": avg_class_distances,
    }


def _has_ab_kwargs(kwargs):
    a, b, spread, min_dist = (
        kwargs.get("a", None),
        kwargs.get("b", None),
        kwargs.get("spread", None),
        kwargs.get("min_dist", None),
    )

    if _has_kwarg(a) and _has_kwarg(b):
        return True
    elif _has_kwarg(spread) and _has_kwarg(min_dist):
        return True
    return False


def _fit(X_train, y_train, **kwargs):
    has_ab_flag = _has_ab_kwargs(kwargs)
    if has_ab_flag:
        partial_results = [partial_fit_and_reconstruct(X_train, y_train, **kwargs)]
    else:
        n_reconstructions = kwargs.get("n_reconstructions", DEFAULT_N_RECONSTRUCTIONS)
        if n_reconstructions > 3:
            logging.warning(
                "Number of reconstructions is greater than 3 but only 3 defaults are supported."
            )
            n_reconstructions = 3
        partial_results = []
        for i in range(n_reconstructions):
            kwargs_copy = kwargs.copy()
            kwargs_copy["a"] = DEFAULT_AB_PAIRS[i][0]
            kwargs_copy["b"] = DEFAULT_AB_PAIRS[i][1]
            partial_results.append(
                partial_fit_and_reconstruct(X_train, y_train, **kwargs_copy)
            )

    partial_result = max(partial_results, key=lambda x: x["chi0"])
    return partial_result


def estimate_transition_prob_matrix(y_gt, y_pred, n_classes):
    """
    Estimate the transition probability matrix (for errors only) from the
    reconstruction errors.
    """

    ## get the transition matrix
    transition_matrix = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            ## get the indices where the GT class is i and the predicted class is j
            inds = np.where((y_gt == i) & (y_pred == j))[0]
            transition_matrix[i, j] = len(inds) / len(y_gt[y_gt == i])

    ## normalize the rows
    for i in range(len(transition_matrix)):
        if np.sum(transition_matrix[i]) == 0:
            transition_matrix[i] = np.ones_like(transition_matrix[i]) / (n_classes - 1)
            transition_matrix[i][i] = 0
        else:
            transition_matrix[i] = transition_matrix[i] / np.sum(transition_matrix[i])

    return transition_matrix


def concentration_index(probabilities, n_classes, threshold=0.95):
    """
    Calculate the concentration index of the given probabilities.

    The concentration index is defined as the fraction of samples that contain
    a threshold (e.g., 95%) of the probability mass.

    Normalized between 0 and 1, where 0 indicates that all samples have the same
    probability and 1 indicates that only one sample has a non-zero probability.
    """
    n = len(probabilities)

    # Sort probabilities in descending order
    sorted_probs = np.sort(probabilities)[::-1]

    # Calculate the cumulative sum
    cumsum = np.cumsum(sorted_probs)

    # Find the number of samples that contain threshold (e.g., 95%) of the probability mass
    n_samples = np.argmax(cumsum >= threshold) + 1

    # Calculate the concentration index
    concentration = (1 - (n_samples / n)) ** 2
    ## Normalize the concentration index by the number of classes
    return concentration ** np.log10(n_classes)


def estimate_psi(y_gt, y_pred, n_classes):
    T = estimate_transition_prob_matrix(y_gt, y_pred, n_classes)
    concentration = np.mean(
        np.array([concentration_index(T[i], n_classes) for i in range(n_classes)])
    )
    return concentration


def _compute_normalized_entropy(probs):
    n = len(probs)
    s = 0
    for p in probs:
        if p == 0:
            continue
        s += p * np.log2(p)
    return -s / np.log2(n)


def estimate_entropy(y_gt, y_pred, n_classes):
    T = estimate_transition_prob_matrix(y_gt, y_pred, n_classes)
    entropy = np.mean([_compute_normalized_entropy(T[i]) for i in range(n_classes)])
    return entropy


def fit_and_reconstruct(X_train, y_train, **kwargs):
    partial_result = _fit(X_train, y_train, **kwargs)

    chi0 = partial_result["chi0"]
    error_array = partial_result["error_array"]
    gt_recon_error = partial_result["gt_recon_error"]
    min_other_recon_error = partial_result["min_other_recon_error"]

    n_classes = len(np.unique(y_train))

    ## get index of the class with the minimum reconstruction error
    y_pred = np.argmin(error_array, axis=0)

    ## compute distances
    class_distances = partial_result["class_distances"]
    avg_class_distances = partial_result["avg_class_distances"]
    T = estimate_transition_prob_matrix(y_train, y_pred, n_classes)

    ## for each class, get the average distance weighted by the transition matrix
    weighted_avg_dist = np.sum(T * class_distances, axis=1)
    avg_all = np.mean(avg_class_distances)
    avg_weighted = np.mean(weighted_avg_dist)
    phi = (avg_weighted / avg_all) ** 2

    recon_error_ratio = gt_recon_error / min_other_recon_error

    rer_gt_1 = recon_error_ratio[recon_error_ratio > 1]
    rer_lt_1 = recon_error_ratio[recon_error_ratio <= 1]

    ## variance of the ratio
    chi_var = np.var(recon_error_ratio)
    chi_gt_1_var = np.var(rer_gt_1)
    chi_lt_1_var = np.var(rer_lt_1)

    ## kurtoisis of the ratio
    chi_kurtosis = kurtosis(recon_error_ratio)
    chi_gt_1_kurtosis = kurtosis(rer_gt_1)
    chi_lt_1_kurtosis = kurtosis(rer_lt_1)

    ## skewness of the ratio
    chi_skew = skew(recon_error_ratio)
    chi_gt_1_skew = skew(rer_gt_1)
    chi_lt_1_skew = skew(rer_lt_1)

    entropy = estimate_entropy(y_train, y_pred, n_classes)

    eta_max = np.sum(recon_error_ratio > 1) / len(y_train)

    delta_gt = np.mean(gt_recon_error)
    delta_best = estimate_delta_best(error_array)
    delta2 = estimate_delta2(error_array)

    chi_rand = estimate_chi_rand(error_array)
    chi_max = np.max(recon_error_ratio)
    chi2 = delta2 / delta_best
    delta_rand = delta_best / chi_rand


    ## Estimate eta and chi0

    chi_avg = np.mean(recon_error_ratio)

    psi = estimate_psi(y_train, y_pred, n_classes)
    eta_est = estimate_eta(chi0, chi_rand)

    chi_star = estimate_threshold(
        chi0=chi0,
        eta_est=eta_est,
        chi_max=chi_max,
        chi_rand=chi_rand,
        **kwargs,
    )

    artifacts = {}

    return_dict = {
        "y_pred": y_pred,
        "mistakenness": recon_error_ratio,
        "eta_est": eta_est,
        "chi_star": chi_star,
        "threshold": chi_star,
        "chi0": chi0,
        "chi_rand": chi_rand,
        "chi_max": chi_max,
        "chi_avg": chi_avg,
        "eta_max": eta_max,
        "psi": psi,
        "phi": phi,
        "delta_gt": delta_gt,
        "delta_best": delta_best,
        "delta2": delta2,
        "delta_rand": delta_rand,
        "chi_var": chi_var,
        "chi_kurt": chi_kurtosis,
        "chi_skew": chi_skew,
        "chi_gt_1_var": chi_gt_1_var,
        "chi_gt_1_kurt": chi_gt_1_kurtosis,
        "chi_gt_1_skew": chi_gt_1_skew,
        "chi_lt_1_var": chi_lt_1_var,
        "chi_lt_1_kurt": chi_lt_1_kurtosis,
        "chi_lt_1_skew": chi_lt_1_skew,
        "entropy": entropy,
    }

    if kwargs.get("estimate_probs", False):
        p_mistake_func, p_r_x, p_r_x_given_mistake = construct_mistake_pdf(
            recon_error_ratio, error_array, y_train, eta_est
        )

        return_dict["p_mistake_max"] = p_mistake_func(chi_max)
        return_dict["p_mistake_star"] = p_mistake_func(chi_star)
        return_dict["p_mistake_min"] = p_mistake_func(np.min(recon_error_ratio))

        probs = np.array([p_mistake_func(r) for r in recon_error_ratio])


    else:
        probs = None

    return_dict["mistakenness_probs"] = probs

    if kwargs.get("store_error_array", False):
        artifacts["error_array"] = error_array

    if kwargs.get("compute_2d_umap", False):
        reducer = UMAP(n_components=2)
        z = reducer.fit_transform(X_train)
        artifacts["umap_2d_points"] = z

    return_dict["artifacts"] = artifacts

    return return_dict


def partial_fit_and_reconstruct(X_train, y_train, **kwargs):
    for i in range(NUM_RETRIES):
        try:
            return _partial_fit_and_reconstruct(X_train, y_train, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            if i == NUM_RETRIES - 1:
                raise e

    wandb.log({"error": str(e)})
    wandb.finish()


def estimate_eta(chi0, chi_rand):
    return (chi0 - chi_rand) / (1 - chi_rand)


def estimate_delta2(error_array):
    ## get second minimum
    return np.mean(np.partition(error_array, 1, axis=0)[1])


def estimate_delta_best(error_array):
    return np.mean(np.min(error_array, axis=0))


def estimate_chi_rand(error_array):
    min_errors = np.min(error_array, axis=0)

    n_classes = error_array.shape[0]

    avg_other_recon_error = np.mean(error_array, axis=0)
    ## remove the GT reconstruction error and reaverage
    avg_other_recon_error = (avg_other_recon_error * n_classes - min_errors) / (
        n_classes - 1
    )

    return np.mean(np.divide(min_errors, avg_other_recon_error))


class PRXConstructor:
    def __init__(self):
        pass

    def base_density_func(self, x):
        return self.kde((x - self._min) / (self._max - self._min))

    def construct_p_r_x(self, recon_error_ratio):
        ## Kernel density estimation
        ## with reflection at right boundary
        _min, _max = np.min(recon_error_ratio), np.max(recon_error_ratio)
        rers_normalized = (recon_error_ratio - _min) / (_max - _min)

        right_reflected = 2 - rers_normalized

        kde = gaussian_kde(np.concatenate([rers_normalized, right_reflected]))

        self.kde = kde
        self._min = _min
        self._max = _max
        return self.base_density_func

    def construct_p_r_x_given_mistake(self, error_array, y_train):
        num_samples = len(y_train)

        ## randomly choose new labels for each sample
        classes = np.unique(y_train)
        y_train_prime = np.array(
            [np.random.choice(np.delete(classes, y), 1)[0] for y in y_train]
        )

        ## get GT reconstruction errors
        gt_prime_recon_error = error_array[y_train_prime, np.arange(num_samples)]

        mask = np.zeros_like(error_array, dtype=bool)
        mask[y_train_prime, np.arange(num_samples)] = True
        masked_array = error_array.copy()
        masked_array[mask] = np.inf
        min_other_prime_recon_error = np.min(masked_array, axis=0)

        rer_prime = gt_prime_recon_error / min_other_prime_recon_error

        return self.construct_p_r_x(rer_prime)


class BasePMistakeConstructor:
    def __init__(self):
        pass

    def base_p_mistake(self, r):
        ## Bayes rule
        p = self.p_r_x_given_mistake(r) * self.eta_est / (self.p_r_x(r)[0])
        if np.isnan(p):
            return 0.0
        return p.clip(0, 1)

    def construct_base_p_mistake(self, p_r_x, p_r_x_given_mistake, eta_est):
        self.p_r_x = p_r_x
        self.p_r_x_given_mistake = p_r_x_given_mistake
        self.eta_est = eta_est
        return self.base_p_mistake


def construct_mistake_pdf(
    recon_error_ratio,
    error_array,
    y_train,
    eta_est,
    use_multiprocessing=True,
    n_workers=None,
    **kwargs,
):
    if not use_multiprocessing:
        p_r_x = PRXConstructor().construct_p_r_x(recon_error_ratio)
        p_r_x_given_mistake = PRXConstructor().construct_p_r_x_given_mistake(
            error_array, y_train
        )

    else:
        import multiprocessing

        with multiprocessing.Pool(processes=2) as pool:
            p_r_x = pool.apply(PRXConstructor().construct_p_r_x, (recon_error_ratio,))
            p_r_x_given_mistake = pool.apply(
                PRXConstructor().construct_p_r_x_given_mistake, (error_array, y_train)
            )

    base_p_mistake = BasePMistakeConstructor().construct_base_p_mistake(
        p_r_x, p_r_x_given_mistake, eta_est
    )

    ## get maximum value and argmax of the mistake probability
    max_p_mistake = None
    max_r = None
    if not use_multiprocessing:
        for r in recon_error_ratio:
            p = base_p_mistake(r)
            if max_p_mistake is None or p > max_p_mistake:
                max_p_mistake = p
                max_r = r
    else:
        if n_workers is None:
            n_workers = DEFAULT_N_WORKERS
        with multiprocessing.Pool(processes=n_workers) as pool:
            max_p_mistake = max(pool.map(base_p_mistake, recon_error_ratio))
            max_r = recon_error_ratio[
                np.argmax(pool.map(base_p_mistake, recon_error_ratio))
            ]

    def p_mistake(r):
        if r > max_r:
            return max_p_mistake
        else:
            return base_p_mistake(r)

    return p_mistake, p_r_x, p_r_x_given_mistake
