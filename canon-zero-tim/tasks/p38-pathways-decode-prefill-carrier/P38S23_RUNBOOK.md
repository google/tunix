# P38s23 fixed-tile lm-head target runbook

> **HISTORICAL / DO NOT EXECUTE.** Source `32caa773` stopped during vLLM
> warmup because this version admitted only M16/M256 and omitted request bucket
> M32. It produced no numerical round. Use `P38S23R2_RUNBOOK.md` instead.

This is the historical operator card for P38s23. It launched one 64-TPU FrozenLake
diagnostic (`DP16xTP4`, concurrency 256, three frozen rounds) with
`CANON_P38_FIXED_LM_HEAD=1` as the only numerical change from the known-red
stock envelope. It performs zero backward and zero optimizer commits.

Read `phases/p38-2x-fixed-tile-pallas-lm-head.md` first. Do not add env values
by hand, enable the old algorithm arm, attach seam/terminal observers, enable
prefix cache, or reuse a rendered YAML.

## Precondition

Use the exact user-approved published full SHA. The current local one-host
receipt is `artifacts/p38_2x_fixed_lm_head_onehost_0818.md`; if executable
files change during publication review, rerun that gate before launch.

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_FULL_SHA_CONTAINING_P38_2X>"
RUN_ID=p38s23
WORKTREE="/tmp/canon-zero-tim-$RUN_ID-${SOURCE_COMMIT:0:12}"
OUT="/tmp/p38-serving-$RUN_ID-${SOURCE_COMMIT:0:12}"
RETURN="/tmp/p38-return-$RUN_ID-${SOURCE_COMMIT:0:12}"

git fetch origin yuxzhang/canon-zero-tim
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test ! -e "$WORKTREE"
test ! -e "$OUT"
test ! -e "$RETURN"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p "$RETURN"
printf '%s\n' "$SOURCE_COMMIT" > "$RETURN/source_commit.txt"
```

## Render and launch

```bash
python3 canon-zero-tim/cluster/render_p38_serving_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --stock-only \
  --max-concurrency 256 \
  --fixed-lm-head | tee "$RETURN/render.txt"

STOCK="$OUT/jobset-p38-serving-stock.yaml"
test -s "$STOCK"
test ! -e "$OUT/jobset-p38-serving-unified.yaml"
cp "$STOCK" "$RETURN/rendered-stock.yaml"
python3 - "$STOCK" <<'PY'
import sys
import yaml

doc = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]
labels = spec["metadata"]["labels"]
container = spec["spec"]["containers"][0]
env = {entry["name"]: str(entry.get("value", "")) for entry in container["env"]}
args = [str(arg) for arg in container.get("args", [])]

assert labels.get("canon.zero-tim/fixed-lm-head") == "1", labels
assert env.get("CANON_P38_FIXED_LM_HEAD") == "1", env.get("CANON_P38_FIXED_LM_HEAD")
assert "CANON_MM_ALGO" not in env
assert all(not name.startswith(("CANON_P38_SEAM", "CANON_P38_TAIL", "CANON_P38_TERMINAL")) for name in env)
assert any("--max_concurrency=256" in arg for arg in args), args
print("P38S23_RENDER_SEMANTIC_PASS")
PY

kubectl apply --dry-run=server -f "$STOCK" | tee "$RETURN/dry-run.txt"
kubectl apply -f "$STOCK" | tee "$RETURN/apply.txt"
```

## Live admission markers

Before waiting for rollout, require the source/overlay preflight and both
fixed-shape compile receipts. Copy the complete attempt-0 head log to
`$RETURN/head.full.log`; do not return only grep snippets.

```text
[sync] HEAD=<SOURCE_COMMIT>
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=16 fixed_M=256 K=4096 local_N=37984 fixed_N=38144 BM=128 BN=256 BK=256
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=256 fixed_M=256 K=4096 local_N=37984 fixed_N=38144 BM=128 BN=256 BK=256
[CANON_P38] PRECHECK_ROUND_COMPLETE ... exactly three times
backward=0 optimizer_commits=0 in every round
```

Any absent receipt, wrong shape, retry attempt, connection loss, incomplete
round, or missing archive is `INCONCLUSIVE`, not a numerical failure or pass.

## Decision table

| Three sealed rounds | Decision |
|---|---|
| A-B exact in all rounds; B-C exact | candidate causal repair; proceed to P38.2h backward-no-commit |
| A-B red; B-C exact | reject fixed lm-head as sufficient; reopen the remaining terminal interval |
| B-C red or any contract/receipt missing | fail closed as configuration/instrumentation-inconclusive |

## Return contract

Return the complete `$RETURN` directory plus the complete attempt-0 GCS root
bundle or the immutable three round bundles if root postflight is unavailable.
At minimum the return must contain the rendered YAML, render/dry-run/apply
receipts, full attempt-0 head log, source SHA, three round completion markers,
three round manifests, and their deterministic archives. Run `sha256sum` over
every returned regular file and write `RETURN_SHA256SUMS` last. Do not infer or
hand-write a verdict; the branch-side auditor will recompute it.

Stop before commit or push. The operator does not edit code, flags, YAML, GCS
objects, classification, or documentation.

## Background-free operator prompt

> Read `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S23_RUNBOOK.md`
> completely and execute it from the exact full SHA supplied by the user. You
> are an operator only: do not edit code/YAML/env, do not enable other arms,
> and do not interpret the numbers. Return the exact directory and artifacts
> listed under “Return contract”, write `RETURN_SHA256SUMS` last, and stop
> before commit or push.
