# TensorStore Precomputed IO

## Scope
- TensorStore read/write helpers (precomputed/N5), mips, halos
- Round-trip tests and metadata verification

## Deliverables
- `libs/tensorstore_io/readwrite.py`
- `libs/tensorstore_io/test_roundtrip.py`

## Definition of Done
- 512³ round-trip write/read; mips & halos metadata verified
- Tests green in CI; documented voxel size & axes order

## Gotchas
- Keep chunk sizes consistent across mips
- Ensure file permissions and disk space in runners
