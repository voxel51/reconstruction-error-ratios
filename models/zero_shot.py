"""
Zero-Shot model for label error detection.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
| 
"""

from .base_model import BaseModel

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import torch
import torch.nn as nn

from .DEFAULTS import *


def format_kwargs(kwargs):
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k.replace("zs_", "")] = v
    return new_kwargs


def _get_class_logits(text_features, image_features):
    # source: https://github.com/openai/CLIP/blob/main/README.md
    image_features = torch.tensor(image_features, dtype=torch.float32)
    text_features = torch.tensor(text_features, dtype=torch.float32)

    image_features = image_features / image_features.norm(
        dim=1, keepdim=True
    )
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    return logits_per_image.detach().numpy()

def _compute_mistakenness(logits, m):
    # constrain m to either 1 (incorrect) or -1 (correct)
    m = m * -2.0 + 1.0

    c = -1.0 * entropy(softmax(np.asarray(logits)))
    mistakenness = (m * np.exp(c) + 1.0) / 2.0

    return mistakenness


class ZeroShotModel(BaseModel):
    def __init__(self, X, y, class_name_embs=None, **kwargs):
        """Zero-Shot model for label error detection.

        class_name_embs (np.ndarray): an array of shape (n_classes, emb_dim)
            containing the embedding for "A photo of a <class_name>" for each
            class in the dataset

        Cosine similarity is used to compute the similarity between the class
        name embeddings and feature embeddings. y_j is predicted to be 
        incorrect if the similarity between X_j and the class name embedding
        class_name_embs[y_j] is not the highest similarity for X_j.
        
        """
        super().__init__(X, y, **kwargs)
        self.kwargs = format_kwargs(kwargs)
        self.class_name_embs = class_name_embs

    def detect_label_errors(self):
        logits = _get_class_logits(self.X, self.class_name_embs)
        y_pred = np.argmax(logits, axis=0).flatten()

        pred_is_correct = y_pred == self.y
        pred_is_correct = pred_is_correct.flatten()

        mistakenness = _compute_mistakenness(logits, pred_is_correct)
        mistakenness = mistakenness.flatten()
        threshold = 0.5

        return {
            "y_pred": y_pred,
            "mistakenness": mistakenness,
            "threshold": threshold,
        }

