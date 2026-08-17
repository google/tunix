# State

- Status: active
- Objective: Run an isolated 64-chip FrozenLake Qwen3-8B DP8xTP8 full-training recipe with device-resident optimizer state, no in-training evaluation, and resumable committed-step checkpoints while preserving the existing DP16xTP4 debug and offload recipes.
- Definition of done: focused workload, renderer, learner, shell, and one-host checkpoint gates pass; the rendered FULL manifest attests DP8xTP8, global batch 32, local M256/global M2048, resident optimizer placement, evaluation cadence 0, online W&B, warning-only alignment, and hard numerical/transaction failures; a fresh target run writes step 10, continues to step 11, and an identical-source resume restores actor/Adam/step and syncs vLLM before rollout.
- Task directory: `canon-zero-tim/tasks/p45-frozenlake-dp8-tp8-resident/`
- Directory state: tracked and published in implementation commit `fae4e67f`
- Current phase: P45.3c — no-eval full training plus checkpoint continuation
- Last verified fact: P45r7's terminal failure is a 300-second in-process
  driver idle timeout caused by streaming evaluation invoking canonical
  prefill rescore with `reset_prefix_cache=True`; optimizer/checkpoint/HBM are
  not the failing stack. The pinned image passes 103 P45 tests, 39 alignment
  tests, two PeftTrainer checkpoint-contract tests, and two focused agentic
  evaluation-schedule tests after the no-eval change. A real one-host v5p DP1xTP4 mechanism gate
  restored sharded model state, device-resident Adam state, step/contract
  metadata exactly and enforced interval 10 plus `LatestN(1)`.
- Next action: render a fresh P45 FULL campaign with a new checkpoint tag,
  prove cadence 0 in the command, run through step 10/11, return a
  cluster-authorized GCS listing, then relaunch the identical source/tag in
  resume mode and require restore plus vLLM weight-sync markers.
- Blockers: the local VM service account gets HTTP 403 on the production GCS
  bucket, so production Step 10 object existence and Pathways restore remain
  target gates. The old P45r7 checkpoint contract is incompatible with the new
  source/cadence and cannot be directly resumed without a reviewed migration.
- Key artifacts: `evidence/p45r7_eval_deadlock_evidence.log`;
  `artifacts/p45r7_step10_eval_deadlock_report.md`;
  `artifacts/p45r7_eval_idle_reset_correction.md`;
  `../../debug_logs/p45_p45r5_frozenlake_resident.raw.log`;
  `../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`; `HANDOFF.md`; `plan.md`;
  `phases/p45-2b-qwen8b-tp8-overlay.md`;
  `phases/p45-3a-gcs-checkpoint-resume.md`;
  `phases/p45-3b-host-memory-hardening.md`; `phases/p45-3-target-run.md`;
  `phases/p45-3c-noeval-checkpoint.md`;
  `evidence/p45_onehost_checkpoint_v5p.txt`
- Updated: 2026-08-17 UTC
