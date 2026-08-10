# P38 target handoff

## Purpose

P38.2 separates two observed flag-on `S_decode_vs_S_prefill` signatures. GSM8K is a tail-aval
candidate; FrozenLake contains a `0.10390` maximum difference and requires upstream/multi-turn
localization. Both are pre-backward diagnostics. Neither may commit an optimizer update or be
described as a full-training run.

## Proven locally

- The alignment test suite passes 21/21 in `tunix_frozenlake_image:vllm-tpu0.25.0`.
- The complete P33 CPU gate passes, including a deliberately failed workload whose pre-alignment
  JSON and SHA survive in stdout.
- The existing hard gate is unchanged: any nonzero pre-backward boundary still exits nonzero.
- A signed GSM8K DP1xTP4 direct-attached run observed 11,340 action tokens with
  `S_decode_vs_S_prefill=0/45360 bytes` and
  `S_prefill_vs_T_old=0/45360 bytes`; the classifier verdict is
  `LOCAL_NOT_REPRODUCED`.
- A production-shape canonical-tail control compared 38,895,616 f32 elements
  across two outer JIT programs with zero differences and detected an injected
  one-bit negative control.
- A model-free DP1xTP4 aval probe ran the live sampling transform at M16/M256
  and the live canonical scorer at M256/M256. Its transform HLO digests were
  different but all five numerical comparisons were exact. This is
  `MODEL_FREE_NOT_REPRODUCED`, not a target fix.
- A synthetic multi-turn mismatch now records turn index, action-run offset,
  completion and sequence chunk coordinates, logical KV prefix length, and
  distance to the next M256 boundary. The complete CPU gate passed.

## Not proven

- No 64-chip P38 run has been launched.
- The r35 A-B carrier has not been localized to a specific token, operator, or proxy flag effect.
- `T_old_vs_T_current` remains unmeasured in the flag-on production model because r35 stopped
  before backward.

The one-host result is a construction gate, not evidence that r35 was repaired.
Its immutable local artifacts are:

- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2/pre_alignment.jsonl`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`
- `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`

## Source and render the model-free target probe

Run only after the P38 patch has been reviewed, committed, and pushed with explicit approval.
Use a clean `yuxzhang/canon-zero-tim` worktree and replace `p38a0` if that run id already exists.

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="p38a0"
TARGET="/tmp/jobset-p38-aval-$RUN_ID.yaml"
python3 canon-zero-tim/cluster/render_p38_aval_jobset.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output "$TARGET"
kubectl apply --dry-run=server -f "$TARGET"
```

The dry run must pass before resource allocation. Do not apply the rendered directory and do not
queue `gsm8k-full` or `frozenlake-full` in this phase.

## Stage 1 target run: model-free aval discriminator

The external operator may apply exactly one manifest after confirming resource
approval. It uses no model, workload, backward, optimizer, checkpoint, or W&B:

```bash
kubectl apply -f "$TARGET"
```

Require Attempt 0, zero restarts, the source commit printed by the renderer, and the proxy
`XLA_FLAGS=--xla_allow_excess_precision=false` contract. A failed numerical gate is an expected
diagnostic outcome; do not restart it automatically.

Return the complete head-pod stdout plus the durable
`CANON_P38_AVAL_REPORT`. The report must contain five completed comparisons,
the registered DP16xTP4 shape table (transform M16/M4096, score M256/M4096),
sharding specs, HLO digests, and a one-element negative control. Missing fields
make the run inconclusive. A fully exact model-free result does not prove the
production boundary; it advances to Stage 2.

## Stage 2 target runs: both production boundaries

After Stage 1 is classified, render the existing no-commit production probes
from the same source commit. Do not substitute one workload for the other:

```bash
RUN_ID="p38prod0"
OUT="/tmp/p38-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

GSM="$OUT/jobset-p33-gsm8k-alignment-short.yaml"
FL="$OUT/jobset-p33-frozenlake-alignment-short.yaml"
kubectl apply --dry-run=server -f "$GSM"
kubectl apply --dry-run=server -f "$FL"
```

With separate resource approval, apply only `GSM` and `FL`. Both stop before
backward and optimizer commit. GSM8K tests the low-amplitude tail candidate.
FrozenLake independently tests the `0.10390` multi-turn signature and must
emit turn, action-run, M256 chunk, and logical-KV coordinates for every
reported mismatch.

## Evidence to return

Archive the complete head-pod stdout and report its SHA. The raw log must contain:

- `[CANON_ALIGN_PRE_JSON]` with both boundary records;
- `[CANON_ALIGN_PRE_EVIDENCE]` with the report SHA;
- on failure, `[CANON_PRE_ALIGN_ARTIFACT]` and every
  `[CANON_PRE_ALIGN_ARTIFACT_JSON]` row;
- the exact source commit, Attempt 0 marker, proxy XLA environment, mesh order, local canonical
  row count, `N_action`, and workload exit code.

For every mismatch, preserve coordinate, token id, exact A/B bits, XOR, byte offsets, ULP
distance, and absolute delta. The report is inconclusive if a target line is missing; absence is
not equality.

## Pre-registered verdict

- A-B nonzero and B-C zero: P38.2b reproduces the GSM8K serving carrier; classify the transform,
  score, and implied-normalizer fields before selecting a repair. FrozenLake is still required.
- A-B zero and B-C zero: GSM8K did not reproduce the sparse r35 carrier. This is not proof of a
  fix; P38.2c FrozenLake remains independently required.
- B-C nonzero, an invalid shape, missing evidence, source drift, a retry, or an infrastructure
  disconnect: the numerical result is not admitted.

FrozenLake evidence must additionally identify the turn index, assistant-run offset, canonical
chunk index, logical KV prefix length, and whether the mismatch is adjacent to a turn or M256
boundary. A tail-only repair is not admitted for a `0.10390` upstream signature.

No tolerance, report-only committing mode, old-logprob recomputation, precision change, or
optimizer commit is authorized by this handoff.
