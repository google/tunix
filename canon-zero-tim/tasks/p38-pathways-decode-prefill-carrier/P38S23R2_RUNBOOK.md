# P38s23r2 fixed-tile lm-head target runbook

This is the only current operator card for P38.2x2. It launches one 64-TPU
FrozenLake diagnostic (`DP16xTP4`, concurrency 256, three frozen rounds) with
`CANON_P38_FIXED_LM_HEAD=1` as the only numerical change from the known-red
stock envelope. It performs zero backward and zero optimizer commits.

P38s23r1/source `575ef92e` is historical: all request warmups and rollout
passed, but learner rescore M4096 was absent from the lm-head contract. P38s23r2
requires exact request buckets M8/16/32/64/128/256 plus exact learner M4096.
Every request bucket pads to M256; M4096 maps to exactly 16 calls of the same
M256/K4096/N38144 Pallas body. Arbitrary row counts and stock fallback remain
forbidden.

Read `phases/p38-2x-fixed-tile-pallas-lm-head.md` first. Do not edit YAML/env,
enable an old algorithm arm, attach observers, enable prefix cache, or reuse a
previous render.

## Precondition

Use the exact user-approved published full SHA. The current local M4096 receipt
must be named in `HANDOFF.md` and must report zero differences before launch.
If an executable file changes after that receipt, rerun the one-host gate.

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_FULL_SHA_CONTAINING_P38_2X2>"
RUN_ID=p38s23r2
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
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
pod = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]
labels = pod["metadata"]["labels"]
container = pod["spec"]["containers"][0]
env = {x["name"]: str(x.get("value", "")) for x in container["env"]}
args = [str(x) for x in container.get("args", [])]
assert labels.get("canon.zero-tim/fixed-lm-head") == "1", labels
assert env.get("CANON_P38_FIXED_LM_HEAD") == "1", env
assert "CANON_MM_ALGO" not in env
assert all(not n.startswith(("CANON_P38_SEAM", "CANON_P38_TAIL", "CANON_P38_TERMINAL")) for n in env)
assert any("--max_concurrency=256" in x for x in args), args
print("P38S23R2_RENDER_SEMANTIC_PASS")
PY

kubectl apply --dry-run=server -f "$STOCK" | tee "$RETURN/dry-run.txt"
kubectl apply -f "$STOCK" | tee "$RETURN/apply.txt"
```

## Live admission markers

Copy the complete attempt-0 head log to `$RETURN/head.full.log`. The run is
admitted only if it contains all seven compile receipts and exactly three
sealed numerical rounds:

```text
[sync] HEAD=<SOURCE_COMMIT>
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=8 ... chunks=1
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=16 ... chunks=1
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=32 ... chunks=1
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=64 ... chunks=1
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=128 ... chunks=1
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=256 ... chunks=1
[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 semantic_M=4096 ... chunks=16
[CANON_P38] PRECHECK_ROUND_COMPLETE ... exactly three times
backward=0 optimizer_commits=0 in every round
```

Any missing receipt, non-registered M, stock fallback, retry attempt,
connection loss, incomplete round, or incomplete archive is `INCONCLUSIVE`.

## Decision table

| Three sealed rounds | Decision |
|---|---|
| A-B exact in all rounds; B-C exact | candidate causal repair; proceed to P38.2h backward-no-commit |
| A-B red; B-C exact | reject fixed lm-head as sufficient; reopen the remaining terminal interval |
| B-C red or any receipt missing | fail closed as configuration/instrumentation-inconclusive |

## Return contract

Return the complete `$RETURN` directory plus the complete attempt-0 GCS root
bundle or immutable three-round bundles. It must contain the YAML,
render/dry-run/apply receipts, full attempt-0 head log, source SHA, seven
PATHTRACEs, three round completion markers/manifests, deterministic archives,
and a final `RETURN_SHA256SUMS`. Do not write a hand verdict; branch-side tools
will recompute it.

Stop before commit or push. The operator does not edit code, flags, YAML, GCS
objects, classification, or documentation.

## Background-free operator prompt

> Read `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38S23R2_RUNBOOK.md`
> completely and execute it from the exact full SHA supplied by the user. You
> are an operator only: do not edit code/YAML/env and do not interpret the
> numbers. Return every artifact under “Return contract”, write
> `RETURN_SHA256SUMS` last, and stop before commit or push.
