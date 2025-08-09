---
name: FFN++ (Flax) with topology loss & context
about: Implement FFN++ with GroupNorm, bf16, context path, seed policy, topology-aware loss
labels: model, training, high-priority
---

## Scope
- FFN++ (3D conv/residual, GroupNorm, bf16)
- Frontier/seed policy (uncertainty)
- Global-context assist (downsampled)
- Topology-aware loss (boundary contrastive / membrane-respect)
- U-Net baseline + watershed postprocess

## Deliverables
- services/segmenter/models_ffn.py
- services/segmenter/train_jax.py
- libs/losses/topology.py (+ tests)

## Acceptance
- Trains on 256^3 patches; VI_total baseline on held-out
- Mixed precision enabled; throughput (Mvox/s) logged
- Checkpoint/restore deterministic with fixed seed

## Gotchas
- XLA conv memory spikes → warm-up step
- Prefer bf16 on TPU/A100
- Use GroupNorm for small batch per device
