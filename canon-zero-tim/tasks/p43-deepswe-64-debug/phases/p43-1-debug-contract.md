# P43.1 — 64-chip Qwen3-8B debug contract

## Scope

Add a separate, default-off DeepSWE bring-up profile for one 64-device
`4x4x4` slice. The physical slice is divided into two host-complete
32-device roles; rollout and trainer are each DP4xTP8. The existing P34
production and P39 Qwen3-32B pilot contracts remain unchanged.

## Signed debug geometry

- Model: `Qwen/Qwen3-8B`
- Global prompt batch: 4
- Generations per prompt: 4
- Global trajectories per step: 16
- Local trajectories per DP rank: 4
- Prompt limit: 4096 tokens
- Response limit: 4096 tokens
- Maximum turns: 5
- Stages: `rollout-only`, `one-update`, `three-update`
- Optimizer state: resident for the first debug launch
- Dataset seed: 42, using the real gold-filtered R2E-Gym dataset

## Artifact contract

When `CANON_P43_DEEPSWE_DEBUG=1`, every complete training rollout batch must
produce, before any optimizer update:

1. One atomic `batch-<step>.trajectories.jsonl.gz` containing 16 readable
   post-environment trajectories. Each record retains prompt-group and sample
   identities, status, raw final reward, solve label, advantage, conversation,
   tool/environment trace fields, and policy version.
2. One fsync'd row in `batch_metrics.jsonl` with trajectory solve ratio and
   prompt-group counts for all-solved, all-failed, mixed, and incomplete
   groups, plus advantage-activity counts and status/reward histograms.
3. One fsync'd `run_manifest.json` recording the artifact schema, solve
   definition, source commit, stage, model, topology, batch geometry, and
   fixed dataset seed.

R2E-Gym currently exposes a scalar final reward rather than a separate
boolean verdict. Therefore P43 uses the explicit diagnostic definition
`r2egym_final_reward_eq_1`: a complete trajectory is solved exactly when its
finite raw final reward equals `1.0`. Positive non-binary rewards are retained
and counted separately, never promoted to solved.

## Exit gates

- Contract unit tests accept exactly the P43 geometry and reject model,
  topology, batch, stage, and production-mode drift.
- Renderer tests prove an immutable source SHA, digest-pinned image, Qwen3-8B
  command, DP4xTP8 roles, 16 trajectories, fixed stage, and dedicated PVC
  artifact paths.
- Artifact unit tests cover all-solved, all-failed, mixed, incomplete, and
  non-binary cases and round-trip the compressed trajectory file.
- Adjacent P34/P39 CPU tests remain green.

## Remote evidence gate

Another agent launches the exact published SHA in this order:
`rollout-only`, `one-update`, `three-update`. A stage is promoted only after
its classifier and artifact inspection pass. The rollout-only stage must exit
before backward/update; one-update must record one finite commit; three-update
must record three sequential finite commits.
