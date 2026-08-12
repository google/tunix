# Log

## 2026-08-11 UTC — P42.1: register the evaluation restoration plan

- Type: decision
- Fact: FrozenLake evaluation support exists in the learner, but the active P33 workload contract, renderer, preflight, and postflight all require evaluation to be disabled.
- Action: Registered a separate P42 task and pre-registered the evaluation cadence, inventory, policy-version, W&B, and no-mutation gates.
- Command: omitted; this checkpoint records plan-only work.
- Result: No code, cloud resource, training process, credential, commit, or branch was changed.
- Files/artifacts: `state.md`; `plan.md`; `phases/p42-1-restore-evaluation.md`
- Rollback: Remove only this uncommitted task directory if the plan is abandoned; no runtime behavior has changed.
- Next: Implement the default-off evaluation-enabled contract and CPU gates.

## 2026-08-11 UTC — P42.1: implement and locally validate evaluation

- Type: implementation and local evidence
- Action: Added a separate `frozenlake-full-eval` manifest. It enables the
  existing held-out FrozenLake path at policy steps 0, 10, ..., 440, selects
  100 test prompts and eight generations, and leaves the original
  evaluation-disabled manifest unchanged.
- Action: Added explicit online summary metrics under
  `frozenlake_eval/eval/{reward,solve,n,wall_seconds,policy_step}` and a compact
  `[CANON_FROZENLAKE_P42_JSON]` row. Empty or nonfinite reward inventories,
  invalid wall time, and invalid policy steps fail before classification.
- Action: Extended preflight and postflight so evaluation can be enabled only
  for committed FrozenLake full training. The classifier requires exactly 45
  complete 800-reward inventories and 45 summary rows at the registered policy
  steps. Diagnostic/no-commit, GSM8K, missing, duplicate, incomplete, or
  nonfinite controls reject.
- Action: Wrote the operator procedure in
  `../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md` and the cross-session handoff
  in `HANDOFF.md`.
- Command: `sudo docker run --rm -v "$PWD:/workspace:ro" -w /workspace -e
  JAX_PLATFORMS=cpu tunix_frozenlake_image:vllm-tpu0.25.0 bash
  canon-zero-tim/tests/p33_workloads/run_cpu.sh`
- Result: PASS; terminal marker
  `[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1
  p35_stage_probe=1`. The embedded learner suite ran eight tests and the P33
  classifier suite ran fourteen tests.
- Boundary: This is local pinned-image contract evidence. No 64-chip
  evaluation, training update, W&B run, cloud action, commit, push, or target
  numerical verdict occurred.
- Rollback: Select `jobset-p33-frozenlake-full.yaml`, whose evaluation triple
  is `0/1/0`; do not delete the shared learner evaluation path.
- Next: Publish after approval, rerun the gate at the publication SHA, then
  execute the target runbook.

## 2026-08-12 UTC — P42.2b: diagnose target failure and correct the signature contract

- Type: target diagnosis, implementation, and local evidence
- Fact: Archived attempt `p42e2` completed the 800-trajectory step-0
  evaluation and reported `local_M=256 global_M=4096`, proving the previous
  geometry fix. It then failed at the first reducer `finalize()` call with
  `DP rank-local gradient fingerprints are not distinct`, before fixed
  reduction or optimizer commit.
- Finding: the reducer conflated a synthetic observability property with a
  production invariant. Equal rank gradients are legal, and binary-reward
  RLOO can produce exact zero contributions for homogeneous prompt groups.
  The five-float compact signature can also collide for different trees.
- Action: retained strict uniqueness as the reducer default for synthetic
  admission probes, but explicitly disabled it in the production segmented
  adapter. Added signature multiplicity reporting while preserving rank
  cadence, contribution count, fixed tree, finite-gradient, and replica-exact
  gates.
- Command: pinned-image reducer suite with 64 forced CPU devices.
- Result: PASS, 19/19.
- Command: pinned-image canonical adapter suite with 64 forced CPU devices.
- Result: PASS, 36/36.
- Command: `bash canon-zero-tim/tests/p33_workloads/run_cpu.sh` in the pinned
  image.
- Result: PASS; terminal marker `[P33.WORKLOAD] CPU_GATE PASS workloads=2
  p35_postflight=1 p35_stage_probe=1`.
- Boundary: no target reduction or optimizer commit has succeeded with this
  fix. Publication and a separately approved 64-chip retry remain pending.
- Rollback: remove the production `require_distinct_fingerprints=False`
  selection; strict probe behavior itself was never changed.
- Next: publish after approval, rerun the pinned gate at the published SHA,
  and retry the evaluation-enabled manifest through the P42 runbook.
