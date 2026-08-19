# P38h fixed-lm-head backward-no-commit runbook

This is the only current operator card for P38.2h. It launches one 64-TPU
FrozenLake `DP16xTP4` actual-model backward transaction with the fixed Pallas
lm-head enabled. Evaluation, prefix cache, warning-only alignment, diagnostic
observers, optimizer commit, and checkpoint writes are disabled.

The target is not another P38 forward precheck. It uses the ordinary P33
`backward-no-commit` path, executes all 16 local gradient groups and the DP
reducer, and then proves that model, optimizer, accumulator, reference, and
train-step state did not change. Read
`phases/p38-2h-fixed-lm-head-backward-no-commit.md` completely before running.

Do not hand-edit the rendered YAML, environment, run ID, topology, workload
command, or evidence records.

The historical `p38h1` Attempt 0 at source `957876b3` is not reusable: it
completed the reverse groups but stopped before the compact return because its
post-backward checker misclassified the intentional optimizer skip. Use only a
new user-approved source SHA that contains the no-commit attestation repair and
its focused regression test.

## Operator input

The user supplies one published 40-character `SOURCE_COMMIT` containing this
runbook. Work from a clean detached checkout of exactly that commit.

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_40_HEX_SHA>"
WORKTREE="/tmp/canon-zero-tim-p38h-${SOURCE_COMMIT:0:12}"
OUT="/tmp/p38h-render-${SOURCE_COMMIT:0:12}"
LAUNCH_RETURN="/tmp/p38h-launch-${SOURCE_COMMIT:0:12}"

git fetch origin yuxzhang/canon-zero-tim
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test ! -e "$WORKTREE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
```

## Render, preflight, and launch once

The checked-in launcher refuses a dirty checkout or reused output path,
validates the exact source and semantic YAML, and performs a server dry-run
before apply.

```bash
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/launch_p38h_backward.sh \
  --source-commit "$SOURCE_COMMIT" \
  --output-dir "$OUT" \
  --return-dir "$LAUNCH_RETURN" \
  --apply
```

The preflight must print `P38H_SEMANTIC_PREFLIGHT_PASS`. The one JobSet is
`canon-p38h-fl-bwd-p38h1-${SOURCE_COMMIT:0:8}` with `maxRestarts: 0`.

## Preserve the complete attempt-0 head log

Wait for the attempt to terminate. Return the complete `jax-tpu` head log from
byte zero, not grep excerpts and not a worker log.

```bash
JOBSET="canon-p38h-fl-bwd-p38h1-${SOURCE_COMMIT:0:8}"
kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=$JOBSET" \
  -o name | tee "$LAUNCH_RETURN/pods.txt"

HEAD_POD="pod/<EXACT_HEAD_POD_NAME>"
kubectl logs "$HEAD_POD" -c jax-tpu > "$LAUNCH_RETURN/head.full.log"
test -s "$LAUNCH_RETURN/head.full.log"
```

If the cluster uses a different discovery label, use `kubectl get pods -A`
only to identify the exact head pod. Do not reconstruct or concatenate logs.

## Mechanically build the compact return

The successful runtime prints SHA-bound base64 copies of the three P33
records to stdout after the official in-pod classifier passes. The collector
decodes those records, verifies each SHA, reruns the official P33 classifier,
checks finite/nonzero gradients and zero state mutation, then seals one small
return directory. No GCS access is required.

```bash
FINAL_RETURN="/tmp/p38h-final-${SOURCE_COMMIT:0:12}"
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/collect_p38h_backward_return.sh \
  --source-commit "$SOURCE_COMMIT" \
  --head-log "$LAUNCH_RETURN/head.full.log" \
  --launch-dir "$LAUNCH_RETURN" \
  --output-dir "$FINAL_RETURN"

(cd "$FINAL_RETURN" && sha256sum -c RETURN_SHA256SUMS --quiet)
python3 -m json.tool "$FINAL_RETURN/verdict.json"
```

Return the complete `$FINAL_RETURN` directory. Do not return only
`verdict.json`, edit evidence, invent a conclusion, commit, or push.

## Mechanical decision table

| Result | Meaning | Next action |
|---|---|---|
| `P38H_FIXED_LM_HEAD_BACKWARD_NO_COMMIT_PASS` | forward boundaries, actual-model gradients, DP transaction, and zero-mutation gates passed | admit a separately reviewed full-training candidate |
| launcher/collector refuses | source, runtime receipt, alignment, gradient, no-commit, or transport contract is incomplete/red | `INCONCLUSIVE` or numerical reject according to the returned raw log; do not train |

A PASS does not prove optimizer commit, checkpoint/resume, quality,
performance, or a full campaign.

## Background-free operator prompt

> Read `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38H_BACKWARD_RUNBOOK.md`
> completely. You are execution-only. Use the exact user-supplied 40-character
> source SHA, run the checked-in launcher once, preserve the complete
> attempt-0 head log, then run the checked-in collector. Return the entire
> compact final directory including `RETURN_SHA256SUMS`. Do not edit code,
> YAML, env, logs, JSON, verdicts, docs, Git history, or cloud objects; do not
> commit or push.
