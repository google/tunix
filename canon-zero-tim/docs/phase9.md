# Phase 9 — DP16xTP4 real-model initialization boundary

Status: **local implementation and exact-image CPU gates PASS; 64-device materialization NOT RUN**
Date: 2026-08-07

## Goal

Promote the bounded Pathways/operator/toy-update admission to the first real Qwen3-8B state
boundary without accidentally starting training. The target gate must materialize the actor,
AdamW state and FP32 accumulator on a topology-aware `(16,4)` mesh and then stop.

## Implemented boundary

- Qwen3 weights are TP-sharded and replicated over the dedicated DP axis.
- The production transaction is fixed at 32 prompts, 8 generations, 256 global trajectories
  and 16 trajectories per DP rank.
- The grouped adapter exposes 16 rank-major reverse groups over global M4096/local M256.
- `model-init-only` materializes the exact 36-layer Qwen3-8B actor state in FP32, AdamW state in
  pinned-host memory, and the FP32 gradient accumulator on device.
- The materialization state is deterministic zero-valued structure. It does not load a
  checkpoint or execute forward, backward, update, W&B networking, or training.

## Local gates

```text
P32 reducer/inventory tests on 64 forced CPU devices: 17/17 PASS
Grouped segmented adapter frozen-image tests:          24/24 PASS
Qwen3-8B abstract inventory:                           PASS
Model-init classifier/profile/tiny materializer:       16/16 PASS
Python/Bash syntax and diff checks:                     PASS
```

The tiny materializer is only a construction smoke test. The full Qwen3-8B local evidence is
abstract-state inventory; no local command allocates 131 GB of replicated host state.

## Target gate

Run a fresh Attempt 0 with `CANON_MODE=model-init-only` and
`cluster/profiles/qwen3-8b-dp16-tp4-model-init.env`. The classifier requires 399 actor leaves,
799 optimizer leaves, 399 accumulator leaves, zero DP-sharded leaves, exact memory kinds, full
64-device coverage and zero execution/commit counters.

Until that artifact exists, model initialization remains `TARGET NOT RUN`. Forward,
backward-no-commit, fixed DP16 reduction, optimizer commit and training remain later phases.

## Rollback

Unset the dedicated mode/profile or revert only the additive Phase-9 probe, classifier, cluster
step and documentation. The original admission profile continues to refuse training, and no
production default is changed.
