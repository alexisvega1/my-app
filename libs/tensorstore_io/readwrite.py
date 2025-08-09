#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

try:
    import tensorstore as ts
except Exception:  # pragma: no cover
    ts = None

DEFAULT_CHUNK = (64,64,64)

def write_precomputed(path: str, volume: np.ndarray, voxel_size_nm=(8,8,8), chunk=DEFAULT_CHUNK, halos=(0,0,0)) -> Dict[str, Any]:
    if ts is None:
        raise RuntimeError("tensorstore not available")
    spec = {
        'driver': 'n5',
        'kvstore': {'driver': 'file', 'path': path},
        'metadata': {
            'dimensions': ['z','y','x'],
            'voxel_size_nm': voxel_size_nm,
            'halos': halos,
            'chunk_size': chunk,
        }
    }
    arr = ts.open(spec, create=True, dtype=volume.dtype, shape=volume.shape, chunk_layout={'grid_origin':[0,0,0],'write_chunk':list(chunk)}).result()
    arr[...] = volume
    return {'path': path, 'shape': volume.shape, 'dtype': str(volume.dtype)}

def read_precomputed(path: str) -> np.ndarray:
    if ts is None:
        raise RuntimeError("tensorstore not available")
    spec = {'driver':'n5','kvstore':{'driver':'file','path':path}}
    arr = ts.open(spec).result()
    return np.array(arr)
