# P33 r17 recovery handoff

Updated: 2026-08-09 UTC

This handoff admits exactly two fresh, source-pinned target jobs:

1. GSM8K `full`, to verify the tied-embedding endpoint repair through a real update.
2. FrozenLake `alignment-short`, to classify the two value boundaries before paying for a
   full-length segmented reverse.

Do not submit FrozenLake `full` or the older full-length `backward-no-commit` job from this
handoff.

## Why r17 stopped

The archived GSM8K r17 log reached the first segmented reverse and failed because Qwen3-1.7B ties
its output projection to the token embedding. The published endpoint repair uses
`embed_tokens.decode` and combines the input-embedding and output-projection cotangents once in a
fixed order. A target pass still requires the live tied marker, all alignment rows, 200 committed
updates and the terminal classifier.

The archived FrozenLake r17 log completed all 16 forward groups and the full 36-layer reverse,
then stopped before the first DP reduction. The admitted mesh names its DP16 axis `data`, while
the generic reducer defaulted to `dp`. The candidate keeps the reducer fail-closed and makes the
Qwen adapter pass `dp_axis=data` explicitly.

The old FrozenLake endpoint metric was nonzero, but the old reporter ran only after backward and
therefore never identified the first red boundary. The new pre-backward gate compares:

- `S_decode` versus `S_prefill`;
- `S_prefill` versus `T_old`.

It writes `pre_alignment.jsonl` and aborts before backward on any differing byte. The short stage
preserves Qwen3-8B, DP16xTP4, 32 prompts x 8 generations, local M256/global M4096, precision,
sampling, fixed reductions and VJP2. It changes only the diagnostic response cap from 2048 to 512
and the FrozenLake environment horizon from 5 to 2. It is diagnostic evidence, not a convergence
recipe.

## Render only the two admitted manifests

Start from a clean target branch and use a fresh run id:

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="r18"
OUT="/tmp/p33-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

GSM="$OUT/jobset-p33-gsm8k-full.yaml"
FL="$OUT/jobset-p33-frozenlake-alignment-short.yaml"
test -f "$GSM"
test -f "$FL"
kubectl apply --dry-run=server -f "$GSM"
kubectl apply --dry-run=server -f "$FL"
kubectl apply -f "$GSM"
kubectl apply -f "$FL"
```

Do not apply the generated directory. It also contains workloads that this handoff does not
admit. Change `RUN_ID` rather than editing or overwriting a generated manifest.

## FrozenLake decision tree

The short job must print one `[CANON_ALIGN_PRE]` line before any segmented reverse.

| First observation | Classification | Next action |
|---|---|---|
| No pre-alignment line or JSON row | `INCONCLUSIVE` | Locate the last completed producer/rescore stage; do not rerun unchanged. |
| `S_decode_vs_S_prefill` differs | `FAIL: boundary 1` | Freeze the selected rows and compare decode versus engine-prefill positions, cache/page state and row ownership. |
| Boundary 1 is zero and `S_prefill_vs_T_old` differs | `FAIL: boundary 2` | Compare trainer token order, validity/action masks, positions and DP row mapping. |
| Both pre-boundaries are zero | Pre-backward value gate passes | Continue the same job; require `gradient_reducer_ready dp_axis=data`, 16 fixed reductions, the full third boundary and the no-commit classifier. |

If a pre-boundary is red, the nonzero process exit is an expected fail-closed numerical verdict,
not an infrastructure failure. Preserve the raw log and `pre_alignment.jsonl`; do not interpret
downstream values because backward did not run.

## Required live evidence

GSM8K must include:

```text
[P28.G5C] TIED_EMBEDDING_HEAD on
[CANON_ALIGN_PRE]
[CANON_P33_DP16] update_step_committed
[P33.RUN] VERDICT PASS
```

FrozenLake must include:

```text
[CANON_ALIGN_PRE]
[P33.DP16] gradient_reducer_ready dp_axis=data dp_size=16
[CANON_P33_DP16] backward_no_commit verdict=PASS
[P33.RUN] VERDICT PASS
```

For each JobSet archive the complete head log, `pre_alignment.jsonl`, `alignment.jsonl`, update
report and classifier JSON with SHA-256 values before deleting the JobSet. An exit code, reward,
Pearson correlation or Pallas trace is not a numerical pass.

## Rollback

Leave P33 admission disabled, or additively revert only this recovery commit. Preserve the r17
logs and all newly returned target artifacts. Do not change precision, canonical M, DP/TP
geometry, sampling, loss, gradient reduction, optimizer semantics, W&B or Hugging Face settings
as part of this recovery.
