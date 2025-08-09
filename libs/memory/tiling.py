#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math

@dataclass
class TilingConfig:
    safety: float = 0.8
    min_tile: Tuple[int,int,int] = (64,64,64)
    max_tile: Tuple[int,int,int] = (512,512,512)

def cube_root(n: float) -> float:
    return n ** (1.0/3.0)

def clamp_tile(tile: Tuple[int,int,int], cfg: TilingConfig) -> Tuple[int,int,int]:
    tz = max(cfg.min_tile[0], min(cfg.max_tile[0], tile[0]))
    ty = max(cfg.min_tile[1], min(cfg.max_tile[1], tile[1]))
    tx = max(cfg.min_tile[2], min(cfg.max_tile[2], tile[2]))
    return (int(tz), int(ty), int(tx))

def auto_tile_for_model(model_bytes_per_voxel: float, free_vram_bytes: float, cfg: TilingConfig = TilingConfig()) -> Tuple[int,int,int]:
    """Compute near-cubic tile size to fit within VRAM budget.
    model_bytes_per_voxel should include params+activations factor for inference.
    """
    budget = free_vram_bytes * cfg.safety
    voxels_budget = max(1.0, budget / max(model_bytes_per_voxel, 1.0))
    side = int(cube_root(voxels_budget))
    tile = (side, side, side)
    return clamp_tile(tile, cfg)

def estimated_memory_for_tile(tile: Tuple[int,int,int], model_bytes_per_voxel: float) -> float:
    z,y,x = tile
    return float(z*y*x) * model_bytes_per_voxel
