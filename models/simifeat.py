"""
SimiFeat model for label error detection.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
| Adapted from https://github.com/UCSC-REAL/SimiFeat
"""

from .base_model import BaseModel
from .DEFAULTS import *

import os
import numpy as np

import sys
import os

o_path = os.getcwd()
sys.path.append(o_path)

import numpy as np
import torch

from docta.apis import DetectLabel, Diagnose
from docta.core.report import Report
from docta.datasets import CustomizedDataset
from docta.utils.config import Config


def format_kwargs(kwargs):
    new_kwargs = {}
    for k, v in kwargs.items():
        new_kwargs[k.replace("sf_", "")] = v
    return new_kwargs


class SimiFeatModel(BaseModel):
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)

        self.kwargs = format_kwargs(kwargs)

        index = np.arange(X.shape[0])
        self.dataset = CustomizedDataset(X, label=y, index=index, preprocess=None)

        self.cfg = self._build_config(**kwargs)

        self.selection_cutoff = kwargs.get(
            "sf_selection_cutoff", DEFAULT_SF_SELECTION_CUTOFF
        )

    def _build_config(self, **kwargs):
        num_classes = len(np.unique(self.y))

        cfg = Config.fromfile("models/simifeat_config.py")
        cfg["num_classes"] = num_classes
        cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cfg

    def detect_label_errors(self):
        # initialize report
        report = Report()

        # diagnose labels
        estimator = Diagnose(self.cfg, self.dataset, report=report)
        estimator.hoc()

        # label error detection
        detector = DetectLabel(self.cfg, self.dataset, report=report)
        detector.detect()

        y_pred = np.copy(self.y)

        mistakenness = np.zeros_like(self.y, dtype=float)

        label_curation = np.array(report.curation["label_curation"])

        sel = label_curation[:, 2] > self.selection_cutoff
        y_pred[label_curation[sel, 0].astype(int)] = label_curation[sel, 1].astype(int)

        mistakenness[label_curation[:, 0].astype(int)] = label_curation[:, 2]

        return {
            "y_pred": y_pred,
            "mistakenness": mistakenness,
            "threshold": self.selection_cutoff,
        }
