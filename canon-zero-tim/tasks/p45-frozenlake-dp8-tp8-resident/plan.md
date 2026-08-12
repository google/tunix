# Plan

## Outcome

Create a separate production-shaped FrozenLake full/eval carrier on the same
64-chip v5p slice using DP8xTP8 and device-resident Adam state. Preserve the
global optimization contract (32 prompts, 8 generations, 256 trajectories,
learning rate `1e-6`, 450 steps) and the local canonical logprob shape M256.
Do not mutate or promote the existing DP16xTP4 carrier-debug evidence.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P45.1 | DP8xTP8 workload, profile, recipe/adapter generalization, and isolated full/eval renderer entries | focused Python and renderer tests pass; old DP16xTP4 renders remain contract compatible | completed |
| P45.2 | Local static and render admission | generated manifests attest DP8xTP8, M2048/M256, 32 local trajectories, resident optimizer, evaluation selection, and hard safety gates | completed |
| P45.3 | 64-chip full/eval target run | first real update reports device-resident state, optimizer H2D/D2H zero, finite update, HBM headroom, online W&B, and continued training; no OOM | pending |

## Decisions

- Confirmed: P41 ran one real Qwen3-8B resident update on DP1xTP4 without OOM, but peak HBM left only 4.52 GiB per chip and did not admit the production default.
- Confirmed: current P33 full/eval manifests force DP16xTP4 and pinned-host optimizer offload.
- Decision: introduce a new DP8xTP8 resident carrier instead of changing the existing P33 debug or full entries.
- Decision: preserve global batch, generations, learning rate, sequence limits, step budget, and evaluation cadence. Set the topology-derived global trajectory microbatch to DP8 so every rank still contributes one trajectory per fixed group; this produces 32 ordered gradient groups instead of the DP16 carrier's 16.
- Decision: full-training alignment remains warning-only for observed A/B/C drift, but non-finite values, invalid placement, failed transaction, replica mismatch, and OOM remain hard failures.
- Hypothesis: TP8 materially increases per-chip HBM reserve and resident placement removes optimizer host transfers; end-to-end speedup remains unproven because TP8 communication and 32 rank-local gradient groups may offset part of the gain.
