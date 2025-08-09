#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

def synapse_precision_recall(gt_points: np.ndarray, pred_points: np.ndarray, tol: float = 2.0) -> dict:
    # naive nearest-neighbor matching within tolerance
    if gt_points.size == 0 and pred_points.size == 0:
        return {"precision": 1.0, "recall": 1.0}
    if pred_points.size == 0:
        return {"precision": 0.0, "recall": 0.0}
    matched = 0
    used = set()
    for g in gt_points:
        dists = np.linalg.norm(pred_points - g, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= tol and j not in used:
            matched += 1
            used.add(j)
    precision = matched / max(1, len(pred_points))
    recall = matched / max(1, len(gt_points))
    return {"precision": float(precision), "recall": float(recall)}
