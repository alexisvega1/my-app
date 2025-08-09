# FFN++ (Flax) with topology loss & context

## Scope
- FFN++ (3D conv/residual, GroupNorm, bf16)
- Frontier/seed policy (uncertainty)
- Global-context assist (downsampled)
- Topology-aware loss (boundary contrastive / membrane-respect)
- U-Net baseline + watershed postprocess

## Deliverables
- `services/segmenter/models_ffn.py`
- `services/segmenter/train_jax.py`
- `libs/losses/topology.py` (+ tests)
- `configs/segmenter/{ffn_pp.yaml,unet.yaml}`
- `tools/bench/train_256_cubes.py`

## Definition of Done
- Trains on 256³ patches; logs `VI_total` on held-out cubes
- Mixed precision (bf16) enabled; throughput (Mvox/s) logged
- Checkpoint/restore works; fixed seed → deterministic weights hash
- Unit tests pass for topology losses on synthetic shapes

## Gotchas
- XLA conv memory spikes → add warm-up step
- Prefer bf16 on TPU/A100; set deterministic XLA flags
- Use GroupNorm (freeze BN) for small batch per device
