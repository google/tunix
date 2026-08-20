# P38.2y GSM8K fixed-lm-head full-training runbook

This is the only operator card for the next GSM8K campaign. It launches one
200-step Qwen3-1.7B run on 64 TPU (`DP16xTP4`) with:

- fixed-tile Pallas lm-head primal and fixed-order VJP;
- TPU-resident optimizer state;
- P47a prompt-logprob removal;
- P50 batched evidence and batched report;
- A-B warning-only recording, while B-C, gradients, reducer, optimizer, and
  state-transition contracts remain fatal.

P52 `CANON_P28_BATCHED_REVERSE` is intentionally off: its DP16 grouped path is
not implemented and certified. Prefix cache, evaluation, serving observers,
and hand-edited YAML are also out of scope.

P38y6 is a closed bootstrap incident. It paired `data,model` sharding with an
actual `dp,tp` mesh, never loaded the model, and then exhausted retries because
Attempt-0 evidence paths were reused. Do not resume, clone, or reclassify it.
The next valid label is P38y7 from a source SHA containing both the
actual-mesh sharding repair and attempt-scoped GSM8K-full evidence.

## Prerequisite already satisfied

The checked-in one-host gate ran against real Qwen3-1.7B weights on four v5p
chips. Request M=8/16/32/64/128/256 and learner M=4096 all used fixed M256,
K2048, N38144 and were exact. The custom VJP produced exact, repeat-exact,
finite and nonzero dHidden/dWeight, and its negative control fired. This is a
construction gate, not the 64-chip target.

## Operator procedure

The user supplies one published 40-character `SOURCE_COMMIT`. Use a clean
detached checkout of exactly that SHA:

```bash
set -euo pipefail
SOURCE_COMMIT="<USER_APPROVED_40_HEX_REPAIR_SHA>"
RUN_ID="p38y7"
WORKTREE="/tmp/canon-zero-tim-p38y-${SOURCE_COMMIT:0:12}"
OUT="/tmp/p38y-render-${SOURCE_COMMIT:0:12}"
LAUNCH_RETURN="/tmp/p38y-launch-${SOURCE_COMMIT:0:12}"

git fetch origin yuxzhang/canon-zero-tim
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_COMMIT"
test ! -e "$WORKTREE"
git worktree add --detach "$WORKTREE" "$SOURCE_COMMIT"
cd "$WORKTREE"

bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/launch_p38y_gsm8k_full.sh \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --return-dir "$LAUNCH_RETURN" \
  --apply
```

Before `apply`, the script must print all three:

```text
P38Y_PROFILE_PREFLIGHT_PASS resident=1 evidence=1 batched_report=1 batched_reverse=0
P38Y_SHARDING_PREFLIGHT_PASS model_axes=actual_mesh data_axis=actual_mesh restart_evidence=attempt_scoped
P38Y_SEMANTIC_PREFLIGHT_PASS steps=200 topology=DP16xTP4 fixed_lm_head=1 warning_only_ab=1
```

The only applied object is
`canon-p33-gsm8k-full-${RUN_ID}-${SOURCE_COMMIT:0:8}`. Do not apply the other
rendered P33 manifests.

## Monitor and return

Keep all head attempts because the JobSet may restart at most three times.
List exact pod names; do not guess:

```bash
JOBSET="canon-p33-gsm8k-full-${RUN_ID}-${SOURCE_COMMIT:0:8}"
kubectl get jobset "$JOBSET" -o yaml >"$LAUNCH_RETURN/jobset.final.yaml"
kubectl describe jobset "$JOBSET" >"$LAUNCH_RETURN/jobset.describe.txt"
kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=$JOBSET" -o wide \
  >"$LAUNCH_RETURN/pods.txt"
```

For every `pathways-head` pod listed, save the complete `jax-tpu` log from
byte zero as `head-attempt-N.full.log`. Do not return grep excerpts in place
of the full log. Each attempt must contain exactly one matching receipt:

```text
[run] GSM8K_FULL_ATTEMPT_EVIDENCE attempt=N dir=.../attempt-N
```

The bootstrap log must report
`shared_mesh.devices.shape=(16, 4) axis_names=('dp', 'tp')`. Any
`Resource axis: model ... not found in mesh` error is a hard bootstrap failure,
not a retryable numerical result. Finally seal the launch directory:

```bash
(cd "$LAUNCH_RETURN" && find . -maxdepth 1 -type f ! -name RETURN_SHA256SUMS \
  -printf '%f\n' | LC_ALL=C sort | xargs -r sha256sum >RETURN_SHA256SUMS)
(cd "$LAUNCH_RETURN" && sha256sum -c RETURN_SHA256SUMS --quiet)
```

Return the complete `LAUNCH_RETURN` directory or its GCS location plus
`RETURN_SHA256SUMS`.

## Readout

The target is a full-training gate, not another diagnostic-only round:

- 200 completed steps, fixed-lm-head request and M4096 VJP receipts, resident
  optimizer receipt, finite/nonzero gradients, replica-exact reduction, and
  valid optimizer commits are required;
- all steps A=B=C proves the Qwen3-1.7B full-training target;
- finite A-B warnings with exact B-C allow an `alignment-degraded` convergence
  result, not a zero-TIM completion claim;
- B-C, nonfinite, reducer, optimizer, or state-transition red is FAIL;
- eviction or exhausted restarts is INCONCLUSIVE.

Rollback is one reviewed renderer change: set
`CANON_P38_FIXED_LM_HEAD=0` only for `gsm8k-full`. Do not disable the unrelated
P47/P50 performance work or resident optimizer unless separately justified.

## Background-free operator prompt

> Read `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/P38Y_GSM8K_FULL_RUNBOOK.md`
> completely. You are execution-only. Use the user-supplied exact source SHA,
> run the checked-in launcher once with `--apply`, monitor the one named
> JobSet, save every complete head-attempt log, seal the return directory, and
> return it. Do not edit code, YAML, env, docs, evidence, Git history, or GCS
> objects; do not commit or push.
