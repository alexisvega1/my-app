# 24-Month "Mouse-Connectome-Ready" Game Plan

everything that must ship on the software side so a whole-brain mouse reconstruction is achievable by mid-2027

⸻

## 0. Guiding KPI Targets

| KPI | 2025-Q3 Baseline | 2026-Q2 Goal | 2027-Q2 "Go/No-Go" |
|---|---|---|---|
| Automated VI (bits, lower = better) | 0.22 on cortex cube | 0.12 | ≤ 0.09 |
| Manual corrections per mm³ | 12.5 | 4 | ≤ 1 |
| End-to-end throughput | 5 M vox/s/GPU | 25 M | ≥ 60 M |
| Proofreading queue 95-th pct wait | 48 h | 12 h | ≤ 2 h |
| Replay-fine-tune latency | — | 4 h batch | 30 min streaming |

If any "go" metric slips, escalate to Program Steering Committee within 48 h.

⸻

## 1. Architecture snapshot

Imaging GCP Bucket ─▶ Chunk-Ingest ◀────────────┐
                    │                            │
                    ▼                            │
      +────────────Segmentation Workers (K8s GPU pods)─────────────+
      │  FFN-v2-Inception  │  SegCLR-Lite  │  Morphology CNN (syn) │
      +────────────┬───────┴───────────────┴───────────────┬───────+
                   │                                       │
                   ▼                                       ▼
          Uncertainty Map (entropy)         Morphology anomaly score
                   │                                       │
                   └──────┬────────────────────────────────┘
                          ▼
              Proofreading Task Queue (Firestore)
                          │
                Human UI  (Neuroglancer hub)
                          │ edits
                          ▼
            LoRA Continual-Learner Service  (GPU/TPU)
                          │  ▲
                          │  │ new adapters
                          ▼  │
            Segmentation-Weights Registry  ◀─────────────┘
                          │
                          ▼
       Telemetry Exporter →  Prometheus → Grafana Dashboard


⸻

## 2. Phased Timeline (deliverable-oriented)

| Month | Milestone & Must-deliver Software | Owners |
|---|---|---|
| M0-M2 | MVP Pipeline• FFN-v2-Inception plugin runs on 8-GPU node, CPU fallback.• Synthetic dataset CI + unit tests in GitHub Actions.• Prometheus exporter with VI, edge-error, throughput gauges.• CLI defaults to dummy data (demo works on any laptop). | Seg Core / Dev Infra |
| M3-M5 | Uncertainty-Triggered Proofreading v1• Entropy thresholding, Firestore stub, Neuroglancer jump-links.• Proofreader Chrome extension with merge/split hotkeys.• Metrics: queue depth + per-task latency. | Proofreading / UX |
| M6 (Q4-25) | First "closing-the-loop" demo on 0.1 mm³ of real cortex.Automated → flag → human fix → LoRA fine-tune → VI drop ≥ 15 %. | All teams |
| M7-M9 | Distributed inference on Ray or KubeFlow;peak 2 Tvox/day with 64×A100.FFN-v2 plugin supports half-precision & tiled seeding.Chunk prefetcher + LRU SSD cache. | Seg Core / Platform |
| M10-M12 | Morphology-anomaly detector (NEURD-like) surfaces merge errors missed by entropy.Adapter Fusion to blend LoRA adapters.Dashboard v2 with per-region drill-downs. | Analytics |
| M13 (Q2-26) | Public "alpha" release of pipeline spec & viewer (internal dataset).External lab can reproduce full stack via Docker Compose. | Dev Rel |
| M14-M18 | Auto-synapse & connectivity export integrated (graph DB).Proofreading crowd mode (Eyewire-style micro-tasks).Replay buffer → continual learner latency cut to < 1 h. | Connectome Apps |
| M19-M21 | Dress-rehearsal on 10 mm³ (≈ 10 % of mouse).Must sustain KPIs under load; failure post-mortems. | All |
| M22-M24 | Mouse-scale rollout• Eight regional shards processed in parallel.• Live VI & queue metrics on exec dashboard.• Weekly adapter pushes; gating on validation VI. | Program finish |


⸻

## 3. Immediate backlog (next two sprints)

| P | Acceptance Test | ETA |
|---|---|---|
| 1 | Implement InceptionBlock3D & uncertainty head (done). | Forward pass under 80 ms @ 64³ on A100. | Week +1 |
| 1 | Entropy calc + Firestore enqueue in proofreading.py. | Top 1 % entropy voxels generate ≤ 10 tasks per 64³ chunk. | Week +1 |
| 1 | LoRAContinualLearner.fit(patches) streaming mode. | 50 patch batch, 3 epochs < 2 min on single GPU. | Week +2 |
| 1 | Prometheus endpoint gauges: vi, edge_error, throughput, queue_depth. | Curl returns 4 metrics, Grafana board imports. | Week +2 |

P2 items: Ray executor, half-precision autotest, CAVE versioning shim.

⸻

## 4. Resourcing

| Role | FTE | Notes |
|---|---|---|
| Deep-learning engineer (FFN/SegCLR) | 3 | CUDA & mixed-precision expertise |
| Proofreading & UI dev | 2 | React + Neuroglancer overlay |
| ML infra (Ray/K8s + monitoring) | 2 | GPU cluster, Prometheus, CI |
| Continual-learning researcher | 1 | PEFT, adapter fusion |
| Data engineering / storage | 1.5 | CloudVolume, SSD cache, CAVE |
| PM / Tech writer | 0.5 | docs, specs, external demos |

Budget: ≈ $1.5 M/yr engineering, plus ≈ $300 k/yr GPU cloud (64×A100 spot, 40 GB).

⸻

## 5. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Imaging backlog slower than compute | Idle GPU $ | Parallelism parameter auto-tunes to imaging pace; stub synthetic volumes keep pipeline "hot." |
| VI plateau even after LoRA updates | Miss accuracy KPI | Introduce SegCLR encoder swap via ModelRegistry; escalate to joint imaging/AI root-cause task-force. |
| Proofreader burnout | Queue explosion | Entropy threshold auto-raises when queue > K; crowd micro-task fallback. |
| GPU spot eviction | Throughput dip | Ray "checkpoint & resume" + half-precision fall-back on CPU for slack fill. |


⸻

## 6. Interview-ready sound-bite

"Our 24-month roadmap delivers an uncertainty-routed, LoRA-fine-tuning FFN-v2 pipeline that should push automated VI to < 0.09 while scaling to 60 M vox/s per GPU—keeping proofreading wait below 2 hours. That makes a full mouse connectome feasible by mid-2027."

Drop this game plan into your Cursor repo's README_next_steps.md and treat each milestone as an issue/PR epic. Your software stack will be "mouse-ready" well within 24 months. 