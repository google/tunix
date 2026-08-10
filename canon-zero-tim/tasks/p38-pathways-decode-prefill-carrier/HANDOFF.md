# P38 target handoff

## Purpose

P38.2 reproduces the remaining flag-on `S_decode_vs_S_prefill` carrier with durable per-element
evidence. It is a pre-backward diagnostic. It must not commit an optimizer update and must not be
described as a full-training run.

## Proven locally

- The alignment test suite passes 18/18 in `tunix_frozenlake_image:vllm-tpu0.25.0`.
- The complete P33 CPU gate passes, including a deliberately failed workload whose pre-alignment
  JSON and SHA survive in stdout.
- The existing hard gate is unchanged: any nonzero pre-backward boundary still exits nonzero.

## Not proven

- No 64-chip P38 run has been launched.
- The r35 A-B carrier has not been localized to a specific token, operator, or proxy flag effect.
- `T_old_vs_T_current` remains unmeasured in the flag-on production model because r35 stopped
  before backward.

## Source and render

Run only after the P38 patch has been reviewed, committed, and pushed with explicit approval.
Use a clean `yuxzhang/canon-zero-tim` worktree and replace `p38a0` if that run id already exists.

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="p38a0"
OUT="/tmp/p38-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

TARGET="$OUT/jobset-p33-gsm8k-alignment-short.yaml"
kubectl apply --dry-run=server -f "$TARGET"
```

The dry run must pass before resource allocation. Do not apply the rendered directory and do not
queue `gsm8k-full` or `frozenlake-full` in this phase.

## Target run

The external operator may apply exactly one manifest after confirming resource approval. The
GSM8K diagnostic preserves the signed 1024-token response shape but caps execution at one
no-commit alignment transaction:

```bash
kubectl apply -f "$TARGET"
```

Require Attempt 0, zero restarts, the source commit printed by the renderer, and the proxy
`XLA_FLAGS=--xla_allow_excess_precision=false` contract. A failed numerical gate is an expected
diagnostic outcome; do not restart it automatically.

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

- A-B nonzero and B-C zero: P38.2 reproduces the serving carrier; classify its positions and bit
  patterns before selecting a repair. FrozenLake is not required for reproduction.
- A-B zero and B-C zero: GSM8K did not reproduce the sparse r35 carrier. This is not proof of a
  fix; run only the rendered FrozenLake `alignment-short` manifest next.
- B-C nonzero, an invalid shape, missing evidence, source drift, a retry, or an infrastructure
  disconnect: the numerical result is not admitted.

No tolerance, report-only committing mode, old-logprob recomputation, precision change, or
optimizer commit is authorized by this handoff.
