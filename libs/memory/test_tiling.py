#!/usr/bin/env python3
from libs.memory.tiling import auto_tile_for_model, estimated_memory_for_tile, TilingConfig

def test_auto_tile_bounds():
    cfg = TilingConfig()
    tile = auto_tile_for_model(16.0, 16*(1024**3), cfg)
    assert cfg.min_tile[0] <= tile[0] <= cfg.max_tile[0]

def test_memory_estimate_monotonic():
    m1 = estimated_memory_for_tile((64,64,64), 16.0)
    m2 = estimated_memory_for_tile((128,128,128), 16.0)
    assert m2 > m1
