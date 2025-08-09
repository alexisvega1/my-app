#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from sklearn.metrics import adjusted_rand_score

def ari_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    a = labels_true.reshape(-1)
    b = labels_pred.reshape(-1)
    return float(adjusted_rand_score(a, b))
