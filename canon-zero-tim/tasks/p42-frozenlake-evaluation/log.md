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
