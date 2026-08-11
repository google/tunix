# P33 five-JobSet queue

> **2026-08-11 P38.2d amendment:** the next approved pair is FrozenLake
> `backward-no-commit` plus GSM8K `full`. The renderer enables
> `CANON_GSM8K_ALIGNMENT_WARN_ONLY=1` only for GSM8K full. Finite numerical
> alignment drift is recorded but never stops that campaign. A completed
> classifier is `PASS_WITH_ALIGNMENT_WARNINGS` with
> `claim_level=convergence-only`; it is not a zero-TIM completion claim. Invalid
> shapes, NaN/Inf, reducer/replica errors, optimizer transaction errors, and
> runtime failures remain fatal. Do not apply FrozenLake full.

> **Current r17 recovery:** follow `P33_R17_HANDOFF.md`. It admits only GSM8K `full` and
> FrozenLake `alignment-short`; do not apply the whole rendered directory.

This runbook renders five independent JobSets. Diagnostic and FrozenLake jobs
remain strict Attempt-0 runs. The user-approved GSM8K full campaign alone may
restart from step 0 up to three times after a failed JobSet attempt.

| Queue entry | Model | Stage | Update budget | Purpose |
|---|---|---|---:|---|
| FrozenLake alignment-short | Qwen3-8B | `alignment-short` | 1 diagnostic transaction, 0 commits | Classify the first two value boundaries before an expensive reverse. |
| FrozenLake backward-no-commit | Qwen3-8B | `backward-no-commit` | 1 reverse transaction, 0 commits | Exercise the largest rollout/RPA/VJP2/DP16-reducer path without mutating training state. |
| GSM8K full | Qwen3-1.7B | `full` | 200 | Run the signed full GSM8K convergence recipe with online W&B. |
| FrozenLake full | Qwen3-8B | `full` | 450 | Run the signed full FrozenLake recipe with online W&B and periodic evaluation disabled. |

GSM8K remains independent from the FrozenLake diagnostic. FrozenLake full and the old
full-length backward-no-commit entry remain stopped by `P33_R17_HANDOFF.md` until the short gate
classifies the two pre-backward boundaries.

## What the renderer freezes

The renderer starts from `cluster/jobset-64chip.yaml` and validates every mutation before it
writes output. Each JobSet has:

- GSM8K full: `maxRestarts: 3`; every restart begins again at step 0 because
  checkpointing remains disabled;
- every other entry: `maxRestarts: 0`; all entries retain head/worker
  `backoffLimit: 0`;
- Pathways head and worker Pods both use `priorityClassName: very-high`; the
  renderer rejects either field if it is missing or changed;
- one exact 40-character source commit assertion;
- no inherited autoscaling device ids;
- a unique JobSet name, Pathways GCS scratch, state directory, run log, pre-alignment JSONL,
  alignment JSONL, update report and W&B run name;
- all three P33 admissions enabled, DP16xTP4, global M4096/local M256 and the exact frozen command;
- Kubernetes Secret references for Hugging Face and W&B, with no literal credential;
- FrozenLake periodic evaluation disabled in both the profile and manifest.

For both workloads, vLLM scheduler limits are per DP rank. The frozen commands therefore set
`max_num_batched_tokens=256` and `max_num_seqs=16` (with each recipe's CLI prefix). Under DP16
this is global token capacity 4096 and global sequence capacity 256. With
`MIN_TOKEN_BUCKET=4096`, TPU inference must prepare exactly one global token bucket, `[4096]`,
whose local executable row count is 256.
Long prompts remain supported through chunked prefill; these limits bound one scheduler step,
not the model context length.

Step 90 refuses a pre-existing evidence path before executing the command. A successful child
process is then classified from the immutable update and alignment reports. The final log must
contain one `[P33.RUN] VERDICT` and one `[P33.RUN] JSON ...` line. Strict jobs require `PASS`;
P38.2d GSM8K full requires `PASS_WITH_ALIGNMENT_WARNINGS`. The JSON includes SHA-256 identities
for the run log, update report and alignment report.

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
[P33.JOBSET] VERDICT PASS count=5 source=<40-char-sha> run_id=<run-id>
```

## Validate without allocating resources

Before rendering or applying on each target cluster, verify the cluster-scoped
PriorityClass exists with the signed scheduling policy. This is a read-only
check; the workload never creates or changes a PriorityClass:

```bash
kubectl get priorityclass very-high \
  -o jsonpath='{.metadata.name}{" value="}{.value}{" policy="}{.preemptionPolicy}{"\n"}'
```

The required result is exactly:

```text
very-high value=1000 policy=PreemptLowerPriority
```

Missing or different output stops the launch. Priority reduces preemption by
lower-priority workloads; it does not provide checkpointing or protect against
node maintenance, OOM, or an IFRT session failure.

The local unit gate verifies exact command parity with `tunix.rl.dp_workloads`, three unique
scratch/state identities, the workload-specific retry policy, dynamic worker
DNS, source pinning and negative controls:

```bash
sudo docker run --rm \
  -v "$PWD:/workspace" -w /workspace -e JAX_PLATFORMS=cpu \
  tunix_frozenlake_image:vllm-tpu0.25.0 \
  bash canon-zero-tim/tests/p33_workloads/run_cpu.sh

bash canon-zero-tim/tests/p33_workloads/run_exact_image.sh
```

The exact-image gate must end with
`P33_EXACT_IMAGE_PASS decode_chunk_cases=5 prompt_chunk_cases=5 overlays=2`. For a workload that
schedules 512 decode rows, the live log must additionally show
`canonical_rows=256 chunks=2`. A packed prompt with 2,048 physical rows per DP rank must show
`rows_per_dp=2048 canonical_rows=256 chunks=8`. Both paths reuse the same local-M256 executable;
changing `CANON_LOGPROB_M` to 512 or 2,048 is not an accepted workaround.

The first target startup after this change must also contain:

```text
Prepared token paddings: [4096]
Precompile worker0 backbone --> {'num_tokens': 4096, 'num_reqs': 256}
```

Any prepared token bucket above 4096 or a request capacity above 256 is a contract failure. A
runtime JAX cache miss for a larger backbone shape is also a failure; do not hide it by enabling
`SKIP_JAX_PRECOMPILE`.

If access to the target cluster is already configured, perform API-schema dry
runs for only the two manifests admitted by the active P38.2d handoff. This
contacts the API server but creates no JobSet:

```bash
FL_NO_COMMIT="$OUT/jobset-p33-frozenlake-backward-no-commit.yaml"
GSM_FULL="$OUT/jobset-p33-gsm8k-full.yaml"
kubectl apply --dry-run=server -f "$FL_NO_COMMIT"
kubectl apply --dry-run=server -f "$GSM_FULL"
```

Any renderer, unit-gate or server-dry-run failure stops the queue. Do not edit a generated YAML to
bypass it; fix the reviewed base or renderer and use a new `RUN_ID`.

## Queue selected manifests

This is the externally consequential step. Run it only after resource approval and the active
handoff has named the allowed manifests. Never apply the whole directory merely because all five
manifests rendered successfully.

```bash
FL_NO_COMMIT="${FL_NO_COMMIT:-$OUT/jobset-p33-frozenlake-backward-no-commit.yaml}"
GSM_FULL="${GSM_FULL:-$OUT/jobset-p33-gsm8k-full.yaml}"
test -f "$FL_NO_COMMIT"
test -f "$GSM_FULL"
kubectl apply -f "$FL_NO_COMMIT"
kubectl apply -f "$GSM_FULL"
```

Do not apply the output directory. In particular, P38.2d does not admit
`jobset-p33-frozenlake-full.yaml`.

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

A green diagnostic or FrozenLake JobSet requires all of the following in the
same Attempt 0 log. A restarted GSM8K full attempt is operationally admitted,
but its attempt number must be reported and it is not first-attempt
determinism evidence:

1. `[entrypoint] JOBSET_ATTEMPT 0 (first attempt)`, except for the explicitly
   restartable GSM8K full campaign;
2. image/overlay/source preflight green with the expected source commit;
3. exactly one online W&B attestation;
4. a monotonic-metrics close marker at the expected final step with zero regressions;
5. FrozenLake only: exactly one evaluation-disabled attestation;
6. the expected update count (`1`, `200` or `450`) and 16 alignment records per update;
7. strict jobs: every three-boundary comparison at zero bytes and exact
   `w=r=w*r=1`; GSM8K full under P38.2d: every finite alignment mismatch and
   ratio/clip/TIS observation is retained as a warning with no numerical
   threshold;
8. finite gradients, fixed DP16 reduction evidence and exact post-reduction replicas;
9. strict jobs: `[P33.RUN] VERDICT PASS ... reasons=[]`; GSM8K full under the
   P38.2d amendment: `PASS_WITH_ALIGNMENT_WARNINGS ... reasons=[]` and
   `claim_level=convergence-only`.

A missing classifier line, stale evidence rejection, traceback, red boundary
or wrong count is not a partial pass. Preserve the raw log and classify the
named JobSet as failed or inconclusive before changing code. GSM8K full retries
all nonzero exits, including numerical gate failures; this simple policy does
not distinguish infrastructure failures. It performs no checkpoint restore and
may run at most four complete attempts (initial plus three restarts).

## Runtime rollback

Do not reapply the JobSet. Leaving all three P33 admission variables at their profile defaults of
`0` restores the fail-closed non-production path. Source rollback is an additive revert of the
renderer/classifier CL; preserve all target logs and do not alter `main`.

## Preregistered launch contract: first flag-on full campaigns (2026-08-10)

The GSM8K column below records the earlier strict contract. P38.2d supersedes
only its A/B admission rule; the update budget, W&B, B/C, old/current,
gradient, DP, and optimizer requirements remain unchanged. FrozenLake full is
not admitted by P38.2d.

User-approved launch of both full workloads under the verified proxy-XLA regime. Source must be
pinned at or after the commit that records this contract; both JobSets render from
`render_p33_jobsets.py` with fresh run ids. FrozenLake remains strict Attempt 0;
GSM8K full uses `maxRestarts=3` with no checkpoint and therefore restarts from
step 0. Both retain head `backoffLimit=0`. The step-1 and step-10 report groups are the promotion readouts of the
old ladder, embedded in one launch; a red gate stops the run by itself with base evidence
already persisted.

| | `gsm8k-full` | `frozenlake-full` |
|---|---|---|
| Command contract | renderer spec, unchanged | renderer spec, unchanged (`max_steps=450`) |
| Trajectories/step | 32 prompts x 8 gen = 256 (16 per DP rank) | same |
| Expected reports | `200 x 16 = 3200` | `450 x 16 = 7200` |
| Step-1 readout | 16/16 reports: all boundaries 0 bytes incl. `T_old_vs_T_current` (first flag-on check of the production value-and-grad primal); `w=r=w*r=1`; clip/TIS 0; finite nonzero gradient; commit=sync=1 | same |
| Step-10 readout | commits/syncs exactly `1..10`; 10 distinct policy hashes; zero red reports | same |
| Completion claim ceiling | stable training loop + reward/solve trend; **not convergence** | same |
| Artifacts | W&B (`zero-tim-gsm8k-dp16-tp4`), alignment JSONL, raw log, classifier verdict; evidence-only, no checkpoint | same W&B via profile env; **`CKPT_DIR = None` is hardcoded (train_frozenlake_qwen3.py:580): no model artifact exists at any step count.** Success may not be described as producing a model |
| Wall-clock expectation (UNVERIFIED, from r17/r19 extrapolation) | first step +~40 min JIT; warm 3-5 min/step; ~10-17 h total | warm 8-10 min/step; **~2.5-3 days total** |
| On red | gate stops the run; archive raw log + alignment JSONL + classification before any code change; classify infra failures INCONCLUSIVE, not numerical red |

Both slices may run in parallel on the 256-chip pool. `JOBSET_ATTEMPT` must be 0 in any log used
as evidence; the proxy `STARTUP: env: XLA_FLAGS=--xla_allow_excess_precision=false` line must be
present in both runs or every downstream number is VOID.
