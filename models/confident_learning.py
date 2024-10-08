"""
Confident Learning model for label error detection.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
| 
"""

from .base_model import BaseModel

import numpy as np

from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression as LogReg

from .DEFAULTS import *


def format_kwargs(kwargs):
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k.replace("cl_", "")] = v
    return new_kwargs


def _get_classifier(
    classifier_arch=DEFAULT_CL_CLASSIFIER_ARCH, max_iter=DEFAULT_CL_MAX_ITER, **kwargs
):
    if classifier_arch != "logistic":
        raise ValueError("Only logistic regression is currently supported")
    return LogReg(max_iter=max_iter)


class ConfidentLearningModel(BaseModel):
    def __init__(self, X, y, **kwargs):
        """Confident Learning model for label error detection.
        https://docs.cleanlab.ai/stable/cleanlab/classification.html#cleanlab.classification.CleanLearning
        """
        super().__init__(X, y, **kwargs)
        self.kwargs = format_kwargs(kwargs)
        self.clf = _get_classifier(**self.kwargs)

    def detect_label_errors(self):
        cl = CleanLearning(clf=self.clf)
        cl.fit(self.X, self.y)

        probs = cl.predict_proba(self.X)
        label_issues = cl.find_label_issues(labels=self.y, pred_probs=probs)

        y_pred = np.array(label_issues["predicted_label"])
        label_quality = np.array(label_issues["label_quality"])

        pred_label_errors = np.where(np.array(label_issues["is_label_issue"]))[0]

        mistakenness = 1 - label_quality
        threshold = np.min(mistakenness[pred_label_errors])

        return {
            "y_pred": y_pred,
            "mistakenness": mistakenness,
            "threshold": threshold,
        }
