# P35 Pathways envelope discriminator handoff

Updated: 2026-08-09 UTC

## Current verdict

Do not rerun the old FrozenLake alignment job unchanged. Its first action-only boundary is exact
and its second is red, but the returned data does not distinguish scheduler packing/metadata from
wrapper/program context.

The valid r18 rates are byte fractions, not token mismatch rates:

- GSM8K: 153,089 differing bytes out of 766,608 action bytes, 20.0%.
- FrozenLake: 28,161 differing bytes out of 118,776 action bytes, 23.7%.

GSM8K r19 fixed the serving M contract, but the red boundary was effectively unchanged. M is no
longer a live load-bearing hypothesis for this boundary. The next report records differing action
elements explicitly. Do not divide byte counts by token counts.

Attempt r21 is a failed pre-measurement run. It completed rollout, then the native reference
forward rejected response 64 because Splash query block 256 did not divide sequence length 1088.
It produced no P35 report or classification. The next attempt must use the unique response cap 256;
do not rerun r21 or relax this contract.

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
2. B: native serving rescore of the exact rank-strided 16-row C group containing the current first
   A-C mismatch, with one sequence per DP rank and canonical local M256.
3. C: current adapter rescore for those same source rows and canonical local M256.

Before comparing values, fail closed unless the run attests:

- identical model-leaf fingerprints and policy version;
- identical selected action token IDs and action/validity masks;
- expected DP16xTP4 mesh shape and device order;
- canonical local M256 in B and C;
- positions, context lengths, page/block tables and cache initialization for each arm;
- exactly one completed A/B/C measurement row;
- a direct A-C red that reproduces the production boundary before classifying A-B or B-C.

The classifier is mechanical:

| A vs B | B vs C | Verdict |
|---|---|---|
| red | exact | packing/metadata carrier |
| exact | red | adapter-envelope carrier; run exact-input replay before naming program context |
| red | red | both carriers |
| exact | exact | reproduction failure/inconclusive |

Any missing arm is `INCONCLUSIVE`. Exact A-B and exact B-C with red A-C violates bitwise
transitivity and is also `INCONCLUSIVE`. A red earlier contract makes all value comparisons VOID.

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

The producer, classifier and renderer are locally complete, but the target is **NOT RUN**. Fetch
the reviewed target branch, resolve its concrete 40-hex SHA, and render exactly one source-pinned
JobSet:

```bash
git fetch origin yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse origin/yuxzhang/canon-zero-tim)"
python3 canon-zero-tim/cluster/render_p35_jobset.py \
  --source-commit "$SOURCE_SHA" \
  --run-id r22 \
  --output /tmp/canon-p35-gsm8k-envelope-r22.yaml
kubectl apply --dry-run=server \
  -f /tmp/canon-p35-gsm8k-envelope-r22.yaml
```

Do not apply until the server-side dry run passes and the operator confirms the source SHA. The
target is GSM8K Qwen3-1.7B, DP16xTP4, response 256, max step 1, no commit, Attempt 0. It intentionally
terminates before backward. A valid return contains:

- exactly one `[CANON_P35] REPORT_COMPLETE ... STOP_BEFORE_BACKWARD` marker;
- `p35_envelope.json` with schema version 2;
- compact `p35_metadata_*.json/.npz` A/B records;
- `p35_envelope.classification.json` with `measurement_verdict=COMPLETE`;
- raw log and SHA-256 values for every returned artifact.

The postflight accepts only the expected diagnostic exit code 1. Missing evidence, any other exit
code, stale output paths or an inconclusive classifier are failures. No production training,
backward or optimizer update is part of this run.

## Rollback

Leave all P35 environment switches unset. Do not alter precision, canonical M, DP/TP geometry,
sampling, loss, fixed reductions, VJP, optimizer semantics, W&B or Hugging Face configuration.
Preserve all red and inconclusive artifacts.
