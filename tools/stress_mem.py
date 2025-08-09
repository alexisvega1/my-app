#!/usr/bin/env python3
import itertools
import traceback
import numpy as np
from libs.memory.tiling import auto_tile_for_model, estimated_memory_for_tile, TilingConfig

def stress(model_bpv=16.0, free_vram_gb=16.0):
    cfg = TilingConfig()
    free_vram_bytes = free_vram_gb * (1024**3)
    failures = 0
    cases = [(bpv, vram) for bpv in (8.0, 12.0, 16.0, 24.0) for vram in (8.0, 12.0, 16.0, 24.0)]
    for bpv, vram in cases:
        try:
            tile = auto_tile_for_model(bpv, vram*(1024**3), cfg)
            mem = estimated_memory_for_tile(tile, bpv)
            assert mem <= vram*(1024**3)*cfg.safety*1.2
        except Exception:
            failures += 1
            traceback.print_exc()
    print(f"stress: failures={failures} cases={len(cases)}")
    return failures

if __name__ == "__main__":
    rc = stress()
    if rc != 0:
        raise SystemExit(1)
