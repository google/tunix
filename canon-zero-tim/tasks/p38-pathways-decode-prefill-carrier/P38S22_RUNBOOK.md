# P38s22 lm-head algorithm discriminator runbook (historical; do not relaunch)

P38s22 has run. The current operator card is
`P38S22_OFFSITE_AUDIT_RUNBOOK.md`, which performs a zero-TPU read-only audit of
the immutable source. Do not execute the launch instructions below again.

This is a 64-TPU (`DP16xTP4`) FrozenLake diagnostic. It is not a full training
run and it is not P38s21 terminal recapture. Read
`phases/p38-2w-lm-head-program-discriminator.md` first.

## Preconditions

- The source commit must be the exact user-approved published SHA containing
  P38.2w.
- The real-weight one-host command must have returned either
  `ALGORITHM_ELIMINATES_OPERATOR_DRIFT` or
  `BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE`, with its one-bit negative equal
  to one.
- The checkout and output directory must be clean/new.
- Do not edit the rendered YAML.

```bash
set -euo pipefail
git fetch origin yuxzhang/canon-zero-tim
SOURCE_COMMIT="<USER_APPROVED_FULL_SHA>"
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
```

## Render and inspect

```bash
set -euo pipefail
RUN_ID=p38s22
OUT="/tmp/p38-serving-$RUN_ID"
test ! -e "$OUT"

python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only \
  --max-concurrency 256 \
  --lm-head-algo

YAML="$OUT/jobset-p38-serving-stock.yaml"
grep -Fq 'canon.zero-tim/lm-head-algo: "1"' "$YAML"
grep -Fq 'name: CANON_MM_ALGO' "$YAML"
grep -A1 -F 'name: CANON_MM_ALGO' "$YAML" | grep -Fq 'value: "1"'
grep -Fq 'name: CANON_MM_ALGO_PRESET' "$YAML"
grep -A1 -F 'name: CANON_MM_ALGO_PRESET' "$YAML" \
  | grep -Fq 'value: BF16_BF16_F32'
grep -Fq 'name: CANON_PALLAS_ALL_PROJ' "$YAML"
! grep -Fq 'name: CANON_P38_SEAM_OBSERVER' "$YAML"
! grep -Fq 'name: CANON_P38_TAIL_OBSERVER' "$YAML"
! grep -Fq 'name: CANON_P38_TERMINAL_DISCRIMINATOR' "$YAML"
kubectl apply --dry-run=server -f "$YAML"
```

Stop if any assertion fails. In particular, do not add terminal observers to
this run; P38s21 already paid for localization and the current question needs
only the production A/B/C endpoint.

## Launch once

```bash
kubectl apply -f "$YAML"
```

Do not reuse an old run id, allow a restart, enable prefix cache, enable
backward/evaluation, change concurrency, or inject any other numerical flag.

## Required return

Return one small, self-contained Attempt-0 bundle:

- rendered YAML and SHA256;
- complete head log from byte zero through exit;
- all three pre-alignment JSON records and their capsules;
- `ROUND_COMPLETE` receipts and root `COLLECTED`/`COMPLETE`;
- source commit; and
- a `SHA256SUMS` covering every returned file.

The head log must contain the `CANON_MM_ALGO` PATHTRACE, three
`PRECHECK_ROUND_COMPLETE` markers, zero backward, zero optimizer commits, and
controlled exit 42. Missing data is `INCONCLUSIVE`, not permission to infer a
result from a truncated log.

## Interpretation

- A-B exact in all three rounds and B-C exact: candidate causal repair; next
  run is strict FrozenLake backward-no-commit, not full training.
- A-B red and B-C exact: existing algorithm preset rejected; implement a
  dedicated fixed-tile Pallas lm_head.
- B-C red or canonical projection/PATHTRACE missing: invalid intervention.
- Any incomplete marker/SHA/Attempt-0 contract: infrastructure inconclusive.

Rollback is simply to omit `--lm-head-algo`; the default profile remains
unchanged.
