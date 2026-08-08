# P33 three-JobSet queue

This runbook renders and queues three independent, strict Attempt-0 JobSets:

| Queue entry | Model | Stage | Update budget | Purpose |
|---|---|---|---:|---|
| FrozenLake backward-no-commit | Qwen3-8B | `backward-no-commit` | 1 reverse transaction, 0 commits | Exercise the largest rollout/RPA/VJP2/DP16-reducer path without mutating training state. |
| GSM8K full | Qwen3-1.7B | `full` | 200 | Run the signed full GSM8K convergence recipe with online W&B. |
| FrozenLake full | Qwen3-8B | `full` | 450 | Run the signed full FrozenLake recipe with online W&B and periodic evaluation disabled. |

The diagnostic does not gate the two full JobSets. All three can be submitted at once. Kueue
decides whether they run concurrently or wait for separate 64-chip allocations; there is no
cross-JobSet dependency or hidden stage transition.

## What the renderer freezes

The renderer starts from `cluster/jobset-64chip.yaml` and validates every mutation before it
writes output. Each JobSet has:

- `maxRestarts: 0` and head/worker `backoffLimit: 0`;
- one exact 40-character source commit assertion;
- no inherited autoscaling device ids;
- a unique JobSet name, Pathways GCS scratch, state directory, run log, alignment JSONL, update
  report and W&B run name;
- all three P33 admissions enabled, DP16xTP4, global M4096/local M256 and the exact frozen command;
- Kubernetes Secret references for Hugging Face and W&B, with no literal credential;
- FrozenLake periodic evaluation disabled in both the profile and manifest.

Step 90 refuses a pre-existing evidence path before executing the command. A successful child
process is then classified from the immutable update and alignment reports. The final log must
contain one `[P33.RUN] VERDICT PASS` and one `[P33.RUN] JSON ...` line. The JSON includes SHA-256
identities for the run log, update report and alignment report.

## Pull and render

Run these commands only from a clean `yuxzhang/canon-zero-tim` worktree. Never run them from
`main`.

```bash
test "$(git branch --show-current)" = yuxzhang/canon-zero-tim
test -z "$(git status --porcelain)"
git pull --ff-only origin yuxzhang/canon-zero-tim

SOURCE_COMMIT="$(git rev-parse HEAD)"
RUN_ID="a0"
OUT="/tmp/p33-jobsets-$RUN_ID"
python3 canon-zero-tim/cluster/render_p33_jobsets.py \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT"
```

`RUN_ID` is a fresh 1-16 character lowercase DNS component for every attempt. The renderer
refuses to overwrite existing output. The generated YAML is disposable and reproducible; the
raw pod logs and `[P33.RUN] JSON` records are the evidence that must be archived persistently.

Expected render terminal marker:

```text
[P33.JOBSET] VERDICT PASS count=3 source=<40-char-sha> run_id=<run-id>
```

## Validate without allocating resources

The local unit gate verifies exact command parity with `tunix.rl.dp_workloads`, three unique
scratch/state identities, Attempt-0 retry policy, dynamic worker DNS, source pinning and negative
controls:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace" -w /workspace -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
```

If access to the target cluster is already configured, perform an API-schema dry run before the
real apply. This contacts the API server but creates no JobSet:

```bash
kubectl apply --dry-run=server -f "$OUT"
```

Any renderer, unit-gate or server-dry-run failure stops the queue. Do not edit a generated YAML to
bypass it; fix the reviewed base or renderer and use a new `RUN_ID`.

## Queue all three

This is the externally consequential step. Run it only after resource approval:

```bash
kubectl apply -f "$OUT"
```

List the exact generated names and follow each head Job independently:

```bash
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
import yaml

for path in sorted(Path(sys.argv[1]).glob("*.yaml")):
  document = yaml.safe_load(path.read_text())
  print(document["metadata"]["name"])
PY

kubectl get jobsets -l canon.zero-tim/source="${SOURCE_COMMIT:0:8}" -w
```

For one generated JobSet name in `JOBSET`, find and follow the head Job:

```bash
HEAD_JOB="$(kubectl get jobs \
  -l jobset.sigs.k8s.io/jobset-name="$JOBSET" \
  -l jobset.sigs.k8s.io/replicatedjob-name=pathways-head \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl logs -f "job/$HEAD_JOB" -c jax-tpu
```

## Evidence and verdict

Before deleting any JobSet, archive its complete head log outside `/tmp`:

```bash
mkdir -p evidence/p33
kubectl logs "job/$HEAD_JOB" -c jax-tpu \
  > "evidence/p33/${JOBSET}.raw.log"
sha256sum "evidence/p33/${JOBSET}.raw.log"
grep -aE '^\[entrypoint\] JOBSET_ATTEMPT|^\[P33.RUN\] (VERDICT|JSON)' \
  "evidence/p33/${JOBSET}.raw.log"
```

A green JobSet requires all of the following in the same Attempt 0 log:

1. `[entrypoint] JOBSET_ATTEMPT 0 (first attempt)`;
2. image/overlay/source preflight green with the expected source commit;
3. exactly one online W&B attestation;
4. a monotonic-metrics close marker at the expected final step with zero regressions;
5. FrozenLake only: exactly one evaluation-disabled attestation;
6. the expected update count (`1`, `200` or `450`) and 16 alignment records per update;
7. every three-boundary comparison at zero bytes, exact `w=r=w*r=1`, zero clip/TIS hits;
8. finite gradients, fixed DP16 reduction evidence and exact post-reduction replicas;
9. `[P33.RUN] VERDICT PASS ... reasons=[]`.

A missing classifier line, retry, stale evidence rejection, traceback, red boundary or wrong count
is not a partial pass. Preserve the raw log and classify the named JobSet as failed or
inconclusive before changing code.

## Runtime rollback

Do not reapply the JobSet. Leaving all three P33 admission variables at their profile defaults of
`0` restores the fail-closed non-production path. Source rollback is an additive revert of the
renderer/classifier CL; preserve all target logs and do not alter `main`.
