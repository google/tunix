# P38s23r3 fixed-lm-head three-round target runbook

This is the only current operator card for the P38.2x forward discriminator.
It launches one 64-TPU FrozenLake diagnostic (`DP16xTP4`, concurrency 256,
three frozen rounds) with the fixed-tile Pallas lm-head enabled. It performs
zero backward and zero optimizer commits. P38s23r2 is historical: its first
round was bitwise exact, but the old full-forensics snapshot occupied the only
durability worker and starved the round seal beyond 900 seconds.

P38s23r3 uses `round-alignment-v1`: every completed round durably uploads only
the complete run log at that instant, its one round-scoped alignment record,
an inventory, and a mismatch capsule only when A-B is red. Periodic live
snapshots and KV/seam/tail/terminal observers are forbidden. Exact rounds
intentionally have no mismatch capsule.

Read `phases/p38-2x-fixed-tile-pallas-lm-head.md` first. Do not hand-edit the
rendered YAML, env, run ID, GCS path, prefix-cache setting, concurrency, or
observer set.

## Operator input

The user supplies one published 40-character `SOURCE_COMMIT` containing this
runbook and the P38s23r3 scripts. Work only from a clean detached checkout of
that exact commit.

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_40_HEX_SHA>"
WORKTREE="/tmp/canon-zero-tim-p38s23r3-${SOURCE_COMMIT:0:12}"
OUT="/tmp/p38-serving-p38s23r3-${SOURCE_COMMIT:0:12}"
LAUNCH_RETURN="/tmp/p38-launch-p38s23r3-${SOURCE_COMMIT:0:12}"

git fetch origin yuxzhang/canon-zero-tim
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test ! -e "$WORKTREE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
```

## One-command render, preflight, and launch

The script validates the exact source, performs semantic YAML assertions and a
server dry-run before applying. It refuses dirty checkouts and existing output
directories.

```bash
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/launch_p38s23r3.sh \
  --source-commit "$SOURCE_COMMIT" \
  --output-dir "$OUT" \
  --return-dir "$LAUNCH_RETURN" \
  --apply
```

The final pre-launch line must include
`P38S23R3_SEMANTIC_PREFLIGHT_PASS`. The rendered JobSet must be
`canon-p38-fl-stock-p38s23r3-${SOURCE_COMMIT:0:8}` and must have
`maxRestarts: 0`.

## Preserve the complete attempt-0 head log

Do not return grep excerpts. After the attempt exits, list the JobSet pods,
identify the single head pod, and save the complete `jax-tpu` log from byte 0:

```bash
JOBSET="canon-p38-fl-stock-p38s23r3-${SOURCE_COMMIT:0:8}"
kubectl get pods \
  -l "jobset.sigs.k8s.io/jobset-name=$JOBSET" \
  -o name | tee "$LAUNCH_RETURN/pods.txt"

# Set this to the one head pod printed above; do not guess or use a worker pod.
HEAD_POD="pod/<EXACT_HEAD_POD_NAME>"
kubectl logs "$HEAD_POD" -c jax-tpu > "$LAUNCH_RETURN/head.full.log"
test -s "$LAUNCH_RETURN/head.full.log"
```

If the label query differs on the cluster, `kubectl get pods -A | grep
"$JOBSET"` may be used only to discover the exact pod name. The returned log
must still be the complete attempt-0 `jax-tpu` head log.

## One-command evidence download and classification

The collector derives the immutable GCS prefix from the source SHA. It
downloads all three round archives plus root markers, verifies every manifest
and deterministic archive, checks all seven fixed-lm-head receipts and all
three ACKs, and writes the verdict mechanically.

```bash
FINAL_RETURN="/tmp/p38-final-p38s23r3-${SOURCE_COMMIT:0:12}"
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/collect_p38s23r3_return.sh \
  --source-commit "$SOURCE_COMMIT" \
  --head-log "$LAUNCH_RETURN/head.full.log" \
  --launch-dir "$LAUNCH_RETURN" \
  --output-dir "$FINAL_RETURN"

(cd "$FINAL_RETURN" && sha256sum -c RETURN_SHA256SUMS --quiet)
python3 -m json.tool "$FINAL_RETURN/verdict.json"
```

Return the complete `$FINAL_RETURN` directory. It is intentionally compact;
do not copy the full GCS tree, edit JSON, invent a hand verdict, commit, or
push.

## Mechanical decision table

| `verdict.json.status` | Meaning | Next branch-side action |
|---|---|---|
| `P38S23R3_FORWARD_EXACT_PASS` | A-B and B-C are exact in all 3 sealed rounds | candidate causal forward repair; prepare strict backward-no-commit |
| `P38S23R3_FIXED_LM_HEAD_INSUFFICIENT` | at least one finite A-B red remains; B-C exact | reject fixed lm-head as sufficient; reopen the residual tail interval |
| collector refuses | source, log, archive, B-C, receipt, or durability contract is incomplete | `INCONCLUSIVE`; do not interpret numerics |

Neither accepted status proves backward, optimizer, full training, or
performance.

## Background-free operator prompt

> Read `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S23R3_RUNBOOK.md`
> completely. You are an execution-only operator. Use the exact 40-character
> SHA supplied by the user, run the checked-in launch script once, preserve the
> complete attempt-0 head log, then run the checked-in collector. Return the
> entire compact final directory with `RETURN_SHA256SUMS`. Do not edit code,
> YAML, env, evidence, verdicts, docs, Git history, or GCS objects; do not
> commit or push.
