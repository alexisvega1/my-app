---
name: Ray chunked inference + deterministic merge
about: Distributed inference with halos and byte-identical stitching
labels: distributed, inference, high-priority
---

## Scope
- pjit/pmap shard rules (N,C,Z,Y,X)
- Ray task graph for chunked inference
- Boundary ID reconciliation (deterministic)

## Deliverables
- services/segmenter/infer_ray.py
- libs/chunks/tiler.py
- libs/chunks/merge_ids.py

## Acceptance
- 1024^3 < 30 min on 1x A100
- 8 GPUs → ≥6× speedup; 4 nodes → ≥3× vs 1 node
- Same ROI on different nodes → byte-identical outputs

## Gotchas
- Pin H2D staging; prefetch next chunk
- Stash halos to avoid seams
