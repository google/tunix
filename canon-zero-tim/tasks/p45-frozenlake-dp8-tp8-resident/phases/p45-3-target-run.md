# P45.3 — 64-chip target run

- Status: pending on P45.3c no-eval checkpoint target gate

## Objective

Run exactly the P45 no-eval FULL manifest on a 64-chip v5p slice and prove that
Qwen3-8B can retain Adam state on TPU across real updates without OOM or
placement drift, write step 10, continue to step 11, and resume that checkpoint
under the identical source/configuration.

## Entry point

Follow `../../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`, then use
`../HANDOFF.md` for the complete evidence return checklist. Dry-run both
generated manifests, apply exactly one, and copy the first update JSON while
the pod is still live.

Before rendering, the exact-image gate must end with
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. At runtime, stop immediately if
the resolved environment does not print
`profile=qwen3-8b-dp8-tp8-frozenlake-resident model_dir=qwen8b_tp8`.

The target manifest must also carry the reviewed P45.3a checkpoint contract.
Do not launch the old checkpoint-disabled P45 manifest after P45.3a begins.
It must explicitly pass `--eval_every_n_steps=0`; the EVAL manifest is
quarantined after P45r7 and is not a valid target for this phase.

## Stop conditions

- TPU OOM or missing/invalid HBM evidence;
- optimizer placement other than `device-resident` before reverse or after
  commit;
- nonzero optimizer H2D/D2H time;
- non-finite gradient/parameter delta;
- invalid optimizer transaction, replica mismatch, topology/bucket drift, or
  accumulator/reference mutation.

Finite alignment drift is warning-only by the requested convergence contract.
It cannot promote the run to a strict zero-TIM claim.
