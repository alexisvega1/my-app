# Ray chunked inference + deterministic merge

## Scope
- pjit/pmap shard rules (N,C,Z,Y,X)
- Ray task graph for chunked inference across nodes
- Boundary ID reconciliation at chunk edges (deterministic)

## Deliverables
- `services/segmenter/infer_ray.py`
- `libs/chunks/tiler.py` + `libs/chunks/merge_ids.py`
- `tools/bench/infer_1024_cube.py`
- Determinism test: same ROI twice → identical content hash

## Definition of Done
- 1024³ inference < 30 minutes on 1×A100
- 8 GPUs → speedup ≥6×; 4 nodes → speedup ≥3× vs 1 node
- Re-run same ROI on different nodes → byte-identical outputs

## Gotchas
- Pin host-to-device staging; prefetch next chunk
- Stash halos to avoid seams; trim on write
