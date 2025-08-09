---
name: QC metrics & CI gate
about: Implement VI, ARI, synapse P/R, alarms, and CI regression gate
labels: qc, ci
---

## Scope
- Metrics: VI_split/merge, ARI, synapse P/R
- Alarms: merge-alarm heuristic, seam-consistency
- CI gate: tiny cubes baseline, fail on regression

## Deliverables
- libs/metrics/{vi.py, ari.py, syn_pr.py}
- libs/qc/alarms.py
- services/qc/report.py (HTML/JSON + Prometheus)
- .github/workflows/ci.yml (QC job)

## DoD
- CI computes metrics on tiny cubes; baseline snapshot stored
- Regression > threshold fails CI
- Dashboard/report shows VI, PR, alarms by dataset/version

## Gotchas
- Remap labels with Hungarian before VI
