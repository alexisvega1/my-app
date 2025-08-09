---
name: TensorStore Precomputed IO
about: Read/write helpers (precomputed, mips, halos) with unit tests
labels: infra, io
---

## Scope
- Precomputed read/write
- Mip pyramid and halos metadata
- Round-trip tests

## Deliverables
- libs/tensorstore_io/readwrite.py
- libs/tensorstore_io/test_roundtrip.py

## Acceptance
- 512^3 round-trip; mips/halos correct in metadata
- Unit tests pass in CI

## Gotchas
- Keep precomputed chunk sizes consistent across mips
- Store voxel size; document axes order
