# P35 Pathways envelope discriminator handoff

Updated: 2026-08-09 UTC

## Current verdict

Do not rerun the old FrozenLake alignment job unchanged. Its first action-only boundary is exact
and its second is red, but the returned data does not distinguish scheduler packing/metadata from
wrapper/program context.

The valid r18 rates are byte fractions, not token mismatch rates:

- GSM8K: 153,089 differing bytes out of 766,608 action bytes, 20.0%.
- FrozenLake: 28,161 differing bytes out of 118,776 action bytes, 23.7%.

The next report schema records differing action elements explicitly. Do not divide byte counts by
token counts.

## What is and is not proven

Proven:

- action-only `S_decode == S_prefill` in both returned r18 workloads;
- `S_prefill != T_old` in both returned r18 workloads;
- GSM8K had a scheduler M contract error;
- FrozenLake already had serving and adapter canonical local M256, so M alone is insufficient;
- generic Pathways `jit(f)` versus `jit(value_and_grad(f)).primal` can drift without TP reduction.

Not proven:

- that generic THIRDPROG drift causes the current forward-only boundary;
- that F4 is ineffective for the production Qwen boundary;
- that packing, page metadata or wrapper context is the unique carrier;
- that any training/backward boundary is green on this platform.

## Required next target experiment

One source-pinned 64-chip run must produce three pre-backward arms in the same process:

1. A: native serving rescore with dynamic packing.
2. B: native serving rescore with one sequence per DP rank and canonical local M256.
3. C: current adapter rescore with the same sequence and canonical local M256.

Before comparing values, fail closed unless the run attests:

- identical model-leaf fingerprints and policy version;
- identical selected action token IDs and action/validity masks;
- expected DP16xTP4 mesh shape and device order;
- canonical local M256 in B and C;
- positions, context lengths, page/block tables and cache initialization for each arm;
- exactly one completed A/B/C measurement row.

The classifier is mechanical:

| A vs B | B vs C | Verdict |
|---|---|---|
| red | exact | packing/metadata carrier |
| exact | red | wrapper/program-context carrier |
| red | red | both carriers |
| exact | exact | pre-backward envelope issue removed; advance to actual-model THIRDPROG |

Any missing arm is `INCONCLUSIVE`. A red earlier contract makes all value comparisons VOID.

## If B vs C is red

Run an in-process exact-input replay. Feed identical engine leaves, IDs, positions, attention
metadata and initialized caches to the direct `runner.model_fn` entry and adapter wrapper. Record
hashes and selected target statistics only; do not serialize full weights or caches. Bisect raw
target logits, log-normalizers and per-layer hidden checkpoints until the first divergence is
identified.

## If the three arms are exact

Only then enable the actual Qwen training forward and compare `T_old` with the primal returned by
the real `value_and_grad` program. The generic way-count probe is a warning, not a substitute for
this production-model gate.

## Operator instructions

At this commit the P35 target producer is **NOT RUN and not yet admitted**. Do not render or apply
a target manifest until the task state says P35.2 is locally complete and names the exact runner
and expected artifact paths. The fail-closed classifier is
`canon-zero-tim/tests/p35_envelope/classify_envelope.py`; it already rejects missing arms, red
contracts, an unobserved negative control and inconsistent hashes/counts. Continue using the
existing r18 logs as immutable input.

The native serving B-arm primitive now exists as
`VllmRollout.get_grouped_prefill_rescore_logps` with an RL-cluster passthrough. It has an
exact-image complete-group control and rejects partial groups. It is not wired into the workload
yet. Two admission gaps remain: observing the actual serving page/block metadata, and attesting
exact equality between the trainer anchor mapped leaves and the live engine leaves.

The existing P18 TPU-runner capture can provide the A/B page/block metadata without a new callback.
Its engine weight fingerprint is only a checksum and must not be promoted to an exact
trainer-versus-engine equality gate. Implementation details and the remaining producer steps are
in `tasks/p35-envelope-discriminator/phases/p35-2-three-arm-producer.md`.

## Rollback

Leave all P35 environment switches unset. Do not alter precision, canonical M, DP/TP geometry,
sampling, loss, fixed reductions, VJP, optimizer semantics, W&B or Hugging Face configuration.
Preserve all red and inconclusive artifacts.
