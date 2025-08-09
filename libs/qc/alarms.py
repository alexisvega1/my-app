#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

def merge_alarm(mem_normals: np.ndarray, threshold: float = 0.8) -> float:
    # Heuristic: high average absolute dot-products across boundaries indicates merges
    # mem_normals: (N,3) normalized membrane normals along candidate crossings
    if mem_normals.size == 0:
        return 0.0
    dots = np.abs(np.sum(mem_normals[:-1] * mem_normals[1:], axis=1))
    return float((dots > threshold).mean())

def seam_consistency(pred: np.ndarray, seam_slices) -> float:
    # Compare labels across seam; return inconsistency rate
    inconsistencies = 0
    total = 0
    for a,b in seam_slices:
        la = pred[a]
        lb = pred[b]
        total += la.size
        inconsistencies += (la != lb).sum()
    return float(inconsistencies / max(1,total))
