# P33 FrozenLake r15 recovery handoff

Updated: 2026-08-09 UTC

This is the current handoff for the next agent operating the 64-device Pathways cluster. It is
intentionally narrower than `P33_QUEUE.md`: do not launch either full FrozenLake training or an
unchanged copy of r15 before this diagnostic is classified.

## Current verified state

- Target branch: `yuxzhang/canon-zero-tim`.
- The source used by r15 was `4fd7e137`.
- r15 reached all 36 promoted Qwen3-8B forward layers and then failed before backward and before
  any optimizer commit.
- Its endpoint report was nonzero:
  `logp_diff=(0.00768, 0.31682)` and `pearson=0.99859`. Pearson correlation is not a bitwise gate.
- r15 emitted no `[CANON_ALIGN]` record and no alignment JSONL row. Therefore it did not measure
  which of the three semantic boundaries was first red.
- Commit `b3d8e278` repairs the control-flow error that stopped r15. FrozenLake now uses
  `sampler_is="token"`; the `sampler_is=None` exception is restricted to exactly GSM8K.
- The later r16 artifact is GSM8K-only. It reached backward and then failed in Pathways-only
  `device.memory_stats()` telemetry; commit `0ac8a7cb` makes that telemetry best-effort. It does
  not classify or replace the missing FrozenLake backward-no-commit run.
- The sampler repair does not claim to repair the numerical mismatch. In a zero-TIM run its
  token-level weights are an exact no-op: `w = r = w*r = 1`, with zero clip and TIS hits.
- The package already contains the three-boundary reporter. Commit `b3d8e278` does not add a new
  numerical print path; it allows the existing reporter to execute.

Archived input:

```text
canon-zero-tim/debug_logs/p33_r15_frozenlake_canary_backward_pass.raw.log
```

Rollback for the repair is an additive revert of `b3d8e278`. Never delete or replace the r15
log.

## The only admitted next target run

Run one fresh, source-pinned FrozenLake `backward-no-commit` JobSet. Do not apply the generated
directory and do not launch FrozenLake full training from this handoff.

Start from a clean target branch:

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim
git merge-base --is-ancestor b3d8e278 HEAD

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="r17flbwd"
OUT="/tmp/p33-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"

FROZENLAKE_BWD="$OUT/jobset-p33-frozenlake-backward-no-commit.yaml"
test -f "$FROZENLAKE_BWD"
kubectl apply --dry-run=server -f "$FROZENLAKE_BWD"
kubectl apply -f "$FROZENLAKE_BWD"
```

`RUN_ID` must be changed if that name has already been rendered or submitted. Do not edit the
generated YAML after rendering.

Find the exact JobSet and head Job without assuming an autoscaled device-id range:

```bash
JOBSET="$(python3 - "$FROZENLAKE_BWD" <<'PY'
import pathlib
import sys
import yaml

document = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
print(document["metadata"]["name"])
PY
)"
HEAD_JOB="$(kubectl get jobs \
  -l jobset.sigs.k8s.io/jobset-name="$JOBSET" \
  -l jobset.sigs.k8s.io/replicatedjob-name=pathways-head \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl logs -f "job/$HEAD_JOB" -c jax-tpu
```

The launch must remain Attempt 0 with zero JobSet, head and worker retries. A retry is a new
execution context and cannot be combined with the first attempt's evidence.

## Mandatory evidence capture

Archive the complete head log before deleting any JobSet. Do not store the only copy under
`/tmp`.

```bash
mkdir -p evidence/p33
RAW="evidence/p33/${JOBSET}.raw.log"
kubectl logs "job/$HEAD_JOB" -c jax-tpu > "$RAW"
sha256sum "$RAW"
grep -aE \
  '^\[entrypoint\] JOBSET_ATTEMPT|^\[run\] (cmd|log|exit)|^\[CANON_ALIGN\]|^\[P33.RUN\] (VERDICT|JSON)|AlignmentGateError|logp_diff=' \
  "$RAW"
```

Return all of the following to the reviewing agent:

1. target branch full SHA and generated JobSet name;
2. complete raw head log and its SHA-256;
3. persisted `alignment.jsonl`, if it exists, and its SHA-256;
4. persisted update report, if it exists, and its SHA-256;
5. the final `[P33.RUN] VERDICT`/`[P33.RUN] JSON` rows, if emitted;
6. the last 100 log lines when no `[CANON_ALIGN]` row exists.

Do not report the run as green from an exit code, rollout reward, Pearson correlation, or the
presence of Pallas/F4 traces alone.

## How to classify the result

The current reporter runs after the real `value_and_grad` call. A completed backward-no-commit
transaction must produce 16 alignment records. A fail-closed numerical red may produce only the
first red record. No alignment record means no semantic boundary was measured.

| First observation | Classification | Next code investigation |
|---|---|---|
| No `[CANON_ALIGN]` and no JSONL row | `INCONCLUSIVE` | Locate the last completed stage. If it stopped in `value_and_grad`, add a separate pre-backward report for the first two boundaries before another expensive retry. |
| `S_decode_vs_S_prefill` is nonzero | `FAIL: boundary 1` | Freeze tokens and selected rows; compare decode versus engine-prefill positions, page tables, cache state, sampling metadata, row ownership and processed logprob. |
| Boundary 1 is zero; `S_prefill_vs_T_old` is nonzero | `FAIL: boundary 2` | Compare trainer context construction: token order, `completion_valid_mask`, action mask, positions, DP rank/row mapping and the full-vocabulary selected rows. |
| Boundaries 1 and 2 are zero; `T_old_vs_T_current` is nonzero | `FAIL: boundary 3` | Isolate the plain-forward versus `value_and_grad` third program; inspect fusion/remat/layout and the excess-precision contract. |
| All three boundaries are zero for 16/16 records | Candidate numerical pass | Require exact `w/r/w*r`, zero clip/TIS, finite nonzero gradients, exact fixed-DP replicas, unchanged state, zero commits, clean postflight and the terminal classifier before promotion. |

`differing_bytes` is only a SAME/DIFFERS predicate. Use `max_abs` and the first mismatch to
describe a red boundary; never rank two failures by saturated byte count.

## Stop rules

- Do not launch FrozenLake full training after any missing or red boundary.
- Do not rerun the same source and configuration unchanged after an inconclusive result.
- Do not change precision, canonical M, DP/TP geometry, sampling, loss, gradient reduction or
  optimizer semantics as part of this diagnostic.
- Do not use token sampler-IS, clipping or TIS to declare the mismatch repaired. The pass
  condition remains bitwise equality with exact unit ratios.
- Do not interpret downstream numbers from a hard-gate-red run.

The GSM8K full JobSet is an independent workload and is not classified by this FrozenLake
handoff. Its execution policy remains in `P33_QUEUE.md`.
